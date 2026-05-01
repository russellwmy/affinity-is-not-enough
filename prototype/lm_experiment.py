"""
Language model experiment v2: MoE routing at domain boundaries.

Fixes from v1:
  1. Predictor supervision: add MSE(predictor_output, embed[t+1]) to training
     loss — same as anticipatory_routing.py. Without this, predictor gets
     no direct gradient and never learns to anticipate.
  2. Boundary measurement: measure SPECIFICALLY at t=SWITCH_POS-1, the single
     step where input is domain-A but model must predict domain-B.
  3. Structured domains: use repeating patterns (not uniform) so BPC has room
     to improve — models that route correctly get lower perplexity.
  4. N_EXPERTS=2: binary choice forces clear domain specialization signal.

Task: predict next character in a sequence that switches domain at SWITCH_POS.
Domain A: repeating structured pattern from first-half alphabet.
Domain B: repeating structured pattern from second-half alphabet.
The first domain-B character is systematically harder to predict if the
router is still on the domain-A expert at the transition step.
"""

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

SEEDS = [42, 43, 44, 45, 46]   # 5 seeds → mean ± std for paper tables
RESULTS_JSON = os.path.join(os.path.dirname(__file__), "lm_experiment_results.json")

# ── Config ────────────────────────────────────────────────────────────────────
# Non-overlapping character sets with predictable within-domain structure
DOMAIN_A_CHARS = list('abcdefghijklm')   # 13 chars
DOMAIN_B_CHARS = list('nopqrstuvwxyz')   # 13 chars
VOCAB          = DOMAIN_A_CHARS + DOMAIN_B_CHARS
V              = len(VOCAB)
c2i            = {c: i for i, c in enumerate(VOCAB)}

SEQ_LEN    = 64
SWITCH_POS = 32        # domain switches here; tokens 0..31 = A, 32..63 = B
D_MODEL    = 64
N_EXPERTS  = 2         # binary: domain-A expert vs domain-B expert
BATCH      = 256
EPOCHS     = 1500
LR         = 2e-3
BETA_INIT  = 0.9
PRED_COEFF = 0.5       # weight on predictor loss (same as toy experiments)

print(f"Vocab: {V} chars  |  N_EXPERTS={N_EXPERTS}  |  SEQ_LEN={SEQ_LEN}  |  SWITCH@{SWITCH_POS}")
print(f"Domain A: {DOMAIN_A_CHARS[:5]}...  Domain B: {DOMAIN_B_CHARS[:5]}...")
print()


# ── Structured dataset ────────────────────────────────────────────────────────
def sample_domain_segment(domain, length):
    """
    Structured: 3-gram Markov chain within domain charset.
    More predictable than uniform → higher BPC room → clearer differences.
    """
    chars  = DOMAIN_A_CHARS if domain == 0 else DOMAIN_B_CHARS
    n      = len(chars)
    result = [random.randint(0, n - 1)]
    for _ in range(length - 1):
        prev = result[-1]
        # Markov: next char is prev+1 or prev+2 (mod n) with noise
        if random.random() < 0.7:
            result.append((prev + 1) % n)
        elif random.random() < 0.5:
            result.append((prev + 2) % n)
        else:
            result.append(random.randint(0, n - 1))
    return [chars[i] for i in result]


def make_batch(batch):
    x_list, y_list, dom_list = [], [], []
    for _ in range(batch):
        seg_a = sample_domain_segment(0, SWITCH_POS)
        seg_b = sample_domain_segment(1, SEQ_LEN - SWITCH_POS)
        seq   = seg_a + seg_b
        x_tok = [c2i[c] for c in seq[:-1]]
        y_tok = [c2i[c] for c in seq[1:]]
        # domain of the TOKEN BEING PREDICTED at each step t:
        #   t < SWITCH_POS-1: predicting domain-A
        #   t = SWITCH_POS-1: predicting first domain-B token  ← TRANSITION
        #   t >= SWITCH_POS: predicting domain-B
        doms  = ([0] * (SWITCH_POS - 1) +
                 [1] * (SEQ_LEN - SWITCH_POS))  # length = SEQ_LEN-1
        x_list.append(x_tok)
        y_list.append(y_tok)
        dom_list.append(doms)
    return (torch.tensor(x_list),
            torch.tensor(y_list),
            torch.tensor(dom_list))


# ── Routers ───────────────────────────────────────────────────────────────────
class StandardRouter(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.W = nn.Linear(d_model, n_experts)

    def forward(self, x_t, mem, x_next_emb=None):
        return F.softmax(self.W(x_t), dim=-1), mem, None


class BetaRouter(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.W        = nn.Linear(d_model, n_experts)
        self.beta_raw = nn.Parameter(
            torch.full((d_model,), torch.logit(torch.tensor(float(BETA_INIT))).item()))

    def forward(self, x_t, mem, x_next_emb=None):
        beta = torch.sigmoid(self.beta_raw)
        mem  = beta * mem + x_t
        return F.softmax(self.W(mem), dim=-1), mem, None


class BetaAntRouter(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.W         = nn.Linear(d_model, n_experts)
        self.W_pred    = nn.Linear(d_model, n_experts, bias=False)  # additive correction
        self.beta_raw  = nn.Parameter(
            torch.full((d_model,), torch.logit(torch.tensor(float(BETA_INIT))).item()))
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x_t, mem, x_next_emb=None):
        beta  = torch.sigmoid(self.beta_raw)
        mem   = beta * mem + x_t
        x_hat = self.predictor(torch.cat([x_t, mem], dim=-1))
        pred_loss = (F.mse_loss(x_hat, x_next_emb.detach())
                     if x_next_emb is not None else None)
        # Route on β-memory (preserves saturation signal) + predictor correction
        logits = self.W(mem) + self.W_pred(x_hat)
        return F.softmax(logits, dim=-1), mem, pred_loss


# ── MoE Language Model ────────────────────────────────────────────────────────
class MoELM(nn.Module):
    def __init__(self, vocab_size, d_model, n_experts, router):
        super().__init__()
        self.embed    = nn.Embedding(vocab_size, d_model)
        self.router   = router
        self.experts  = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            ) for _ in range(n_experts)
        ])
        self.norm     = nn.LayerNorm(d_model)
        self.head     = nn.Linear(d_model, vocab_size)
        self.n_experts = n_experts

    def forward(self, x, return_gates=False, training_mode=False):
        B, T = x.shape
        emb  = self.embed(x)          # (B, T, D)
        mem  = torch.zeros(B, D_MODEL)

        out_seq, gates_seq = [], []
        total_pred_loss = 0.0
        n_pred_steps    = 0

        for t in range(T):
            x_t      = emb[:, t, :]
            x_next   = emb[:, t+1, :].detach() if (training_mode and t < T-1) else None

            gates, mem, pred_loss = self.router(x_t, mem, x_next)

            if pred_loss is not None:
                total_pred_loss += pred_loss
                n_pred_steps    += 1

            expert_outs = torch.stack(
                [self.experts[e](x_t) for e in range(self.n_experts)], dim=1)  # (B,E,D)
            moe_out = (gates.unsqueeze(-1) * expert_outs).sum(dim=1)
            out_t   = self.norm(x_t + moe_out)

            out_seq.append(out_t)
            if return_gates:
                gates_seq.append(gates.detach())

        out    = torch.stack(out_seq, dim=1)     # (B, T, D)
        logits = self.head(out)                  # (B, T, V)

        avg_pred_loss = total_pred_loss / n_pred_steps if n_pred_steps > 0 else 0.0

        if return_gates:
            return logits, torch.stack(gates_seq, dim=1), avg_pred_loss
        return logits, avg_pred_loss


# ── Training ──────────────────────────────────────────────────────────────────
TRANS_WEIGHT = 5.0   # upweight loss at transition step so anticipation is learned

def train(model, epochs, name):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(epochs):
        x, y, _ = make_batch(BATCH)
        logits, pred_loss = model(x, training_mode=True)

        # Per-token loss, then upweight the transition step
        tok_loss = F.cross_entropy(
            logits.reshape(-1, V), y.reshape(-1), reduction='none'
        ).reshape(BATCH, SEQ_LEN - 1)                    # (B, T)
        weights        = torch.ones(SEQ_LEN - 1)
        weights[SWITCH_POS - 1] = TRANS_WEIGHT           # t=31: predicting first domain-B token
        nll  = (tok_loss * weights.unsqueeze(0)).mean()

        loss = nll + PRED_COEFF * pred_loss if isinstance(pred_loss, torch.Tensor) else nll
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 300 == 0:
            bpc = tok_loss.mean().item() / np.log(2)     # unweighted BPC for logging
            print(f"  [{name}] epoch {epoch+1:4d} | NLL {tok_loss.mean().item():.4f} "
                  f"| BPC {bpc:.4f} | pred_loss {pred_loss.item() if isinstance(pred_loss, torch.Tensor) else 0.0:.4f}")


# ── Evaluation ────────────────────────────────────────────────────────────────
def identify_domain_experts(model, n_batches=20):
    gate_sum = torch.zeros(2, N_EXPERTS)   # [domain, expert]
    counts   = torch.zeros(2)
    with torch.no_grad():
        for _ in range(n_batches):
            x, _, doms = make_batch(256)
            _, gates, _ = model(x, return_gates=True)   # (B, T, n_experts)
            for t in range(gates.shape[1]):
                for d in range(2):
                    mask = (doms[:, t] == d)
                    if mask.any():
                        gate_sum[d] += gates[mask, t, :].sum(0).cpu()
                        counts[d]   += mask.sum().item()
    mean = gate_sum / counts.unsqueeze(1)   # (2, n_experts)
    exp_A = (mean[0] - mean[1]).argmax().item()
    exp_B = (mean[1] - mean[0]).argmax().item()
    return exp_A, exp_B, mean


def evaluate(model, exp_B, n_batches=20):
    """
    BPC_all:          overall BPC
    BPC_transition:   BPC specifically at step SWITCH_POS-1
                      (predicting the FIRST domain-B token from a domain-A input)
    p_correct_trans:  gate probability on domain-B expert at step SWITCH_POS-1
    p_correct_mid:    gate probability on domain-B expert mid-sequence (steps 34-60)
                      as a control — should be high for all models
    """
    bpc_all_list, bpc_t_list, pc_t_list, pc_mid_list = [], [], [], []
    TRANS_STEP = SWITCH_POS - 1   # index into T-1 targets

    with torch.no_grad():
        for _ in range(n_batches):
            x, y, _ = make_batch(512)
            logits, gates, _ = model(x, return_gates=True)  # (B,T,V), (B,T,E)

            tok_loss = F.cross_entropy(
                logits.reshape(-1, V), y.reshape(-1), reduction='none'
            ).reshape(-1, SEQ_LEN - 1)                      # (B, T)

            bpc_all_list.append((tok_loss.mean() / np.log(2)).item())
            bpc_t_list.append((tok_loss[:, TRANS_STEP].mean() / np.log(2)).item())
            pc_t_list.append(gates[:, TRANS_STEP, exp_B].mean().item())
            mid = slice(SWITCH_POS + 2, SWITCH_POS + 20)
            pc_mid_list.append(gates[:, mid, exp_B].mean().item())

    return (np.mean(bpc_all_list), np.mean(bpc_t_list),
            np.mean(pc_t_list),    np.mean(pc_mid_list))


# ── Run across multiple seeds ─────────────────────────────────────────────────
configs = [
    ("Standard MoE", StandardRouter),
    ("β-MoE",        BetaRouter),
    ("β+Ant MoE",    BetaAntRouter),
]

# Per-seed metrics: name → list of dicts, one per seed
seed_metrics = {name: {"bpc_all": [], "bpc_trans": [],
                       "pc_trans": [], "pc_mid": [],
                       "spec": [], "K_needed": []}
                for name, _ in configs}

print("=" * 76)
print(f"LM experiment across {len(SEEDS)} seeds: {SEEDS}")
print("=" * 76)

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'#'*76}")
    print(f"# Seed {seed_idx+1}/{len(SEEDS)}  (torch / random seed = {seed})")
    print(f"{'#'*76}")

    for name, router_cls in configs:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        model = MoELM(V, D_MODEL, N_EXPERTS, router_cls(D_MODEL, N_EXPERTS))
        print(f"\n[seed {seed}] {name}")
        train(model, EPOCHS, name)

        exp_A, exp_B, mean_gate = identify_domain_experts(model)
        spec = (mean_gate[1, exp_B] / (mean_gate[0, exp_B] + 1e-9)).item()
        bpc_a, bpc_t, pc_t, pc_mid = evaluate(model, exp_B)

        K = (np.log(0.01) / np.log(1 - pc_t)) if 0.001 < pc_t < 0.999 \
            else (1.0 if pc_t >= 0.999 else 999.0)
        K = min(K, 999.0)

        seed_metrics[name]["bpc_all"].append(bpc_a)
        seed_metrics[name]["bpc_trans"].append(bpc_t)
        seed_metrics[name]["pc_trans"].append(pc_t)
        seed_metrics[name]["pc_mid"].append(pc_mid)
        seed_metrics[name]["spec"].append(spec)
        seed_metrics[name]["K_needed"].append(K)

        print(f"   → exp_A={exp_A} exp_B={exp_B} spec={spec:.1f}x | "
              f"BPC(all) {bpc_a:.3f} | BPC(trans) {bpc_t:.3f} | "
              f"p_B@trans {pc_t:.3f} | p_B@mid {pc_mid:.3f} | K {K:.1f}")

# ── Aggregate: mean ± std (sample std, ddof=1) ────────────────────────────────
def ms(xs):
    a = np.asarray(xs)
    m = float(a.mean())
    s = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return m, s

summary = {}
for name in seed_metrics:
    s = {}
    for k, v in seed_metrics[name].items():
        m, sd = ms(v)
        s[f"{k}_mean"] = m
        s[f"{k}_std"]  = sd
        s[f"{k}_runs"] = v
    s["n_seeds"] = len(seed_metrics[name]["bpc_all"])
    summary[name] = s

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*76}")
print(f"Aggregate results across {len(SEEDS)} seeds  (mean ± std, sample std)")
print(f"{'─'*76}")
print(f"{'Model':<14} {'BPC(all)':>14} {'BPC(trans)':>16} "
      f"{'p_B@trans':>16} {'K (99%)':>14}")
print("─" * 76)
for name in summary:
    s = summary[name]
    print(f"{name:<14} "
          f"{s['bpc_all_mean']:.3f}±{s['bpc_all_std']:.3f}     "
          f"{s['bpc_trans_mean']:.3f}±{s['bpc_trans_std']:.3f}      "
          f"{s['pc_trans_mean']:.3f}±{s['pc_trans_std']:.3f}     "
          f"{s['K_needed_mean']:.1f}±{s['K_needed_std']:.1f}")

# ── Paired gains over Standard MoE (per seed) ─────────────────────────────────
print(f"\n{'─'*76}")
print("Paired gains over Standard MoE at the transition step  (mean ± std)")
print(f"{'─'*76}")
base_bpc_t = np.asarray(seed_metrics["Standard MoE"]["bpc_trans"])
base_pc_t  = np.asarray(seed_metrics["Standard MoE"]["pc_trans"])
for name in seed_metrics:
    if name == "Standard MoE":
        continue
    dbpc = np.asarray(seed_metrics[name]["bpc_trans"]) - base_bpc_t
    dpc  = np.asarray(seed_metrics[name]["pc_trans"])  - base_pc_t
    print(f"  {name:<14}  ΔBPC(trans) {dbpc.mean():+.3f}±{dbpc.std(ddof=1):.3f}   "
          f"Δp_B@trans {dpc.mean():+.3f}±{dpc.std(ddof=1):.3f}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
with open(RESULTS_JSON, "w") as f:
    json.dump({"config": {
                  "seeds": SEEDS, "epochs": EPOCHS, "batch": BATCH,
                  "d_model": D_MODEL, "n_experts": N_EXPERTS,
                  "seq_len": SEQ_LEN, "switch_pos": SWITCH_POS,
                  "vocab_size": V, "lr": LR,
                  "pred_coeff": PRED_COEFF, "trans_weight": TRANS_WEIGHT},
               "results": summary}, f, indent=2)
print(f"\nSaved aggregated results to: {RESULTS_JSON}")

print(f"""
Key metrics:
  BPC(trans):  bits per character specifically at the domain-switch prediction.
    Lower = model better predicts the first domain-B character from domain-A context.
  p_B@trans:   gate weight on the domain-B-specialised expert at the transition step.
    This is p_correct for the anticipation problem — how much probability is
    assigned to the expert that knows domain B, before domain B is visible in input.
  p_B@mid:     same metric mid-sequence (input already domain-B, easy for all models).
    Should be high for all models — confirms expert identification is correct.
  K_needed:    experts for 99% coverage of correct expert at transition step.

Standard deviation across {len(SEEDS)} seeds gives an honest uncertainty estimate.
The paired-gain table accounts for between-seed variance correctly: same data,
same initialisation order, only the model differs.
""")
