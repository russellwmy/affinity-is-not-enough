"""
Ablation: all 2^3 combinations of {β, Π, anticipation}.

Task: same as anticipatory_routing.py — route on predicted next domain.
Measurements: acc_all, acc_transition (specifically at the domain-switch step).

Eight conditions:
  baseline              — route on x_t
  β only               — LIF membrane h_t = β·h_{t-1} + x_t, route on h_t
  Π only               — route on x_t, modulate by per-expert precision
  Ant only             — stateless predictor x̂=f(x_t), route on x̂
  β+Π                  — LIF membrane + precision modulation
  β+Ant                — stateful predictor x̂=f(x_t, h_t), route on x̂
  Π+Ant                — stateless predictor + precision modulation
  β+Π+Ant (full)       — stateful predictor + precision modulation

Plus oracle as ceiling.

Π in a routing task: track per-expert whether it was the correct routing choice.
Π_i = 1 / (EMA(incorrect_i) + ε), where incorrect_i[t] = 1 if target_t ≠ i.
Experts 0,2 are correct roughly half the time; experts 1,3 are never correct.
→ Π should suppress decoy experts 1,3 and amplify experts 0,2.
"""

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

N_EXPERTS  = 4
N_DOMAINS  = 2
SEQ_LEN    = 12
D_MODEL    = 16
NOISE_STD  = 0.8
TRANSITION = 6
BATCH      = 512
EPOCHS     = 800
LR         = 3e-3
PI_MOM     = 0.95
PI_EPS     = 1e-4
BETA_INIT  = 0.9
PRED_COEFF = 0.5      # weight on predictor MSE loss (matches paper §3.3 and lm_experiment.py)
SEEDS      = [42, 43, 44, 45, 46]   # 5 seeds → mean ± std for paper tables
RESULTS_JSON = os.path.join(os.path.dirname(__file__), "ablation_results.json")


# ── Dataset ───────────────────────────────────────────────────────────────────
def make_batch(batch, seq_len, d_model, transition, noise):
    half    = d_model // 2
    seqs    = torch.randn(batch, seq_len, d_model) * noise
    targets = torch.zeros(batch, seq_len, dtype=torch.long)
    for b in range(batch):
        d_start = torch.randint(0, N_DOMAINS, (1,)).item()
        d_end   = 1 - d_start
        sig_s   = torch.zeros(d_model); sig_s[d_start*half:(d_start+1)*half] = 1.0
        sig_e   = torch.zeros(d_model); sig_e[d_end*half:(d_end+1)*half]     = 1.0
        seqs[b, :transition]  += sig_s
        seqs[b, transition:]  += sig_e
        for t in range(seq_len - 1):
            next_domain = d_start if t + 1 < transition else d_end
            targets[b, t] = next_domain * 2
        targets[b, -1] = d_end * 2
    return seqs, targets


# ── Precision tracker ─────────────────────────────────────────────────────────
class PiTracker:
    def __init__(self, n_experts, mom=PI_MOM, eps=PI_EPS):
        self.mom = mom
        self.eps = eps
        self.var = torch.ones(n_experts) * 0.5   # start neutral
        self.n   = n_experts

    def update(self, targets_t):
        """targets_t: (B,) — correct expert at this step."""
        # incorrect_i = fraction of batch where target ≠ i
        incorrect = torch.zeros(self.n)
        for i in range(self.n):
            incorrect[i] = (targets_t != i).float().mean()
        self.var = self.mom * self.var + (1 - self.mom) * incorrect

    def pi(self):
        return 1.0 / (self.var + self.eps)

    def reset(self):
        self.var = torch.ones(self.n) * 0.5


# ── Unified router (flags control which mechanisms are active) ─────────────────
class AblationRouter(nn.Module):
    def __init__(self, d_model, n_experts, use_beta, use_ant):
        super().__init__()
        self.use_beta = use_beta
        self.use_ant  = use_ant
        self.W        = nn.Linear(d_model, n_experts)

        if use_beta:
            self.beta_raw = nn.Parameter(
                torch.full((d_model,),
                           torch.logit(torch.tensor(float(BETA_INIT))).item()))

        if use_ant:
            in_dim = d_model * 2 if use_beta else d_model
            self.predictor = nn.Sequential(
                nn.Linear(in_dim, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, d_model),
            )

    def forward(self, seq, pi=None):
        """
        seq: (B, T, D)
        pi:  (n_experts,) or None

        Returns:
          logits_all: list of (B, n_experts) logit tensors for steps 0..T-2
          x_hat_all:  list of (B, D) predicted-next-state tensors when use_ant,
                     else None
        """
        B, T, D = seq.shape
        logits_all = []
        x_hat_all  = [] if self.use_ant else None
        h = torch.zeros(B, D)

        for t in range(T - 1):
            x_t = seq[:, t, :]

            if self.use_beta:
                beta = torch.sigmoid(self.beta_raw)
                h    = beta * h + x_t
                ctx  = h
            else:
                ctx = x_t

            if self.use_ant:
                if self.use_beta:
                    inp = torch.cat([x_t, h], dim=-1)
                else:
                    inp = x_t
                x_hat = self.predictor(inp)
                x_hat_all.append(x_hat)
                route_input = x_hat
            else:
                route_input = ctx

            scores = self.W(route_input)

            if pi is not None:
                scores = scores * pi.unsqueeze(0)

            logits_all.append(scores)

        return logits_all, x_hat_all


class OracleRouter(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.W = nn.Linear(d_model, n_experts)

    def forward(self, seq, pi=None):
        B, T, D = seq.shape
        logits_all = []
        for t in range(T - 1):
            x_next = seq[:, t + 1, :]
            scores = self.W(x_next)
            if pi is not None:
                scores = scores * pi.unsqueeze(0)
            logits_all.append(scores)
        return logits_all, None


# ── Training ──────────────────────────────────────────────────────────────────
def train(name, model, use_pi, epochs):
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    tracker = PiTracker(N_EXPERTS) if use_pi else None
    accs_all, accs_trans = [], []

    for epoch in range(epochs):
        seqs, targets = make_batch(BATCH, SEQ_LEN, D_MODEL, TRANSITION, NOISE_STD)
        pi = tracker.pi() if tracker else None

        logits_all, x_hat_all = model(seqs, pi)

        total_loss = 0.0
        all_correct, trans_correct, trans_total = 0, 0, 0

        for t, logits in enumerate(logits_all):
            tgt  = targets[:, t]
            total_loss = total_loss + F.cross_entropy(logits, tgt)

            # Predictor MSE supervision (paper §3.3, λ=0.5) when use_ant
            if x_hat_all is not None:
                x_next = seqs[:, t + 1, :]
                total_loss = total_loss + PRED_COEFF * F.mse_loss(
                    x_hat_all[t], x_next.detach())

            preds = logits.argmax(-1)
            all_correct += (preds == tgt).sum().item()
            if t == TRANSITION - 1:
                trans_correct += (preds == tgt).sum().item()
                trans_total   += BATCH

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        # update Π after gradient step
        if tracker is not None:
            with torch.no_grad():
                for t in range(SEQ_LEN - 1):
                    tracker.update(targets[:, t])

        acc_all   = all_correct / (BATCH * (SEQ_LEN - 1))
        acc_trans = trans_correct / trans_total if trans_total > 0 else 0
        accs_all.append(acc_all)
        accs_trans.append(acc_trans)

        if (epoch + 1) % 200 == 0:
            pi_str = ""
            if tracker:
                p = tracker.pi().numpy()
                pi_str = f" | Π=[{p[0]:.1f},{p[1]:.1f},{p[2]:.1f},{p[3]:.1f}]"
            print(f"  [{name}] epoch {epoch+1} | "
                  f"acc_all {acc_all:.3f} | acc_trans {acc_trans:.3f}{pi_str}")

    return np.mean(accs_all[-50:]), np.mean(accs_trans[-50:])


# ── Run all 8 ablation conditions + oracle, across multiple seeds ─────────────
print("=" * 84)
print("Ablation: 2^3 combinations of {β, Π, anticipation}")
print(f"Task: domain switch at step {TRANSITION}/{SEQ_LEN}. "
      f"Target = expert of next domain.")
print(f"Seeds: {SEEDS}  ({len(SEEDS)} runs per condition → mean ± std)")
print("=" * 84)

conditions = [
    # (name,              use_beta, use_pi, use_ant)
    ("baseline",          False, False, False),
    ("β only",            True,  False, False),
    ("Π only",            False, True,  False),
    ("Ant only",          False, False, True ),
    ("β + Π",             True,  True,  False),
    ("β + Ant",           True,  False, True ),
    ("Π + Ant",           False, True,  True ),
    ("β + Π + Ant (full)",True,  True,  True ),
]

# Per-seed results: name → list of (acc_all, acc_trans), one entry per seed
seed_results = {name: {"all": [], "trans": []} for name, *_ in conditions}
seed_results["oracle"] = {"all": [], "trans": []}

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'#'*84}")
    print(f"# Seed {seed_idx+1}/{len(SEEDS)}  (torch seed = {seed})")
    print(f"{'#'*84}")

    for name, use_beta, use_pi, use_ant in conditions:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\nTraining: {name}")
        model = AblationRouter(D_MODEL, N_EXPERTS, use_beta, use_ant)
        acc_all, acc_trans = train(name, model, use_pi, EPOCHS)
        seed_results[name]["all"].append(acc_all)
        seed_results[name]["trans"].append(acc_trans)

    # oracle
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\nTraining: oracle (routes on x_{{t+1}})")
    oracle = OracleRouter(D_MODEL, N_EXPERTS)
    acc_all, acc_trans = train("oracle", oracle, False, EPOCHS)
    seed_results["oracle"]["all"].append(acc_all)
    seed_results["oracle"]["trans"].append(acc_trans)

# ── Aggregate: mean ± std (sample std, ddof=1) ────────────────────────────────
def mean_std(xs):
    a = np.asarray(xs)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0

base_trans_runs = seed_results["baseline"]["trans"]   # paired baseline per seed

summary = {}
for name in seed_results:
    all_runs   = seed_results[name]["all"]
    trans_runs = seed_results[name]["trans"]
    # Paired gain over baseline (same seed) → cleaner uncertainty than independent diff
    gains      = [t - b for t, b in zip(trans_runs, base_trans_runs)]

    summary[name] = {
        "acc_all_mean":   mean_std(all_runs)[0],
        "acc_all_std":    mean_std(all_runs)[1],
        "acc_trans_mean": mean_std(trans_runs)[0],
        "acc_trans_std":  mean_std(trans_runs)[1],
        "gain_trans_mean": mean_std(gains)[0],
        "gain_trans_std":  mean_std(gains)[1],
        "n_seeds":        len(trans_runs),
        "trans_runs":     trans_runs,
        "all_runs":       all_runs,
    }

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*84}")
print(f"Aggregate results across {len(SEEDS)} seeds  (mean ± std, sample std)")
print(f"{'─'*84}")
print(f"{'Condition':<26} {'acc_all':>16} {'acc_transition':>18} {'gain_trans (paired)':>22}")
print(f"{'─'*84}")
for name in summary:
    s = summary[name]
    marker = " ◄" if s["gain_trans_mean"] > 0.05 else ""
    print(f"{name:<26} "
          f"{s['acc_all_mean']:.3f}±{s['acc_all_std']:.3f}    "
          f"{s['acc_trans_mean']:.3f}±{s['acc_trans_std']:.3f}     "
          f"{s['gain_trans_mean']:+.3f}±{s['gain_trans_std']:.3f}{marker}")

# ── Super-additivity check ────────────────────────────────────────────────────
beta_gain  = np.array([t - b for t, b in zip(seed_results["β only"]["trans"], base_trans_runs)])
ant_gain   = np.array([t - b for t, b in zip(seed_results["Ant only"]["trans"], base_trans_runs)])
ba_gain    = np.array([t - b for t, b in zip(seed_results["β + Ant"]["trans"], base_trans_runs)])
interaction = ba_gain - (beta_gain + ant_gain)

print(f"\n{'─'*84}")
print(f"Super-additivity (paired across seeds):")
print(f"  β only gain:                    {beta_gain.mean():+.3f} ± {beta_gain.std(ddof=1):.3f}")
print(f"  Ant only gain:                  {ant_gain.mean():+.3f} ± {ant_gain.std(ddof=1):.3f}")
print(f"  β+Ant gain:                     {ba_gain.mean():+.3f} ± {ba_gain.std(ddof=1):.3f}")
print(f"  Sum of individual gains:        {(beta_gain+ant_gain).mean():+.3f}")
print(f"  Super-additive interaction:     {interaction.mean():+.3f} ± {interaction.std(ddof=1):.3f}")
print(f"{'─'*84}")

# ── Save JSON for paper writeup ───────────────────────────────────────────────
with open(RESULTS_JSON, "w") as f:
    json.dump({"config": {
                  "seeds": SEEDS, "epochs": EPOCHS, "batch": BATCH,
                  "d_model": D_MODEL, "n_experts": N_EXPERTS,
                  "transition": TRANSITION, "seq_len": SEQ_LEN,
                  "noise_std": NOISE_STD, "lr": LR},
               "results": summary,
               "super_additivity": {
                   "beta_gain_runs":  beta_gain.tolist(),
                   "ant_gain_runs":   ant_gain.tolist(),
                   "ba_gain_runs":    ba_gain.tolist(),
                   "interaction_runs": interaction.tolist(),
                   "interaction_mean": float(interaction.mean()),
                   "interaction_std":  float(interaction.std(ddof=1)),
              }}, f, indent=2)
print(f"\nSaved aggregated results to: {RESULTS_JSON}")

print(f"""
Key predictions (from theory):
  β alone         → gain in acc_transition: β carries domain context across tokens
  Π alone         → small gain: suppresses decoy experts 1,3 (never correct)
  Ant alone       → near-zero gain: stateless predictor can't detect transitions
  β+Ant           → large gain: stateful predictor closes ~74% oracle gap
  β+Π+Ant (full)  → should match or exceed β+Ant (Π adds independent signal)

The β×Ant interaction is the critical test: β+Ant >> β alone AND Ant alone.
The interaction term above quantifies super-additivity per-seed,
giving an honest uncertainty estimate.
""")
