"""
Beta-routing hypothesis: two experiments.

Experiment 1 — Early signal, late noise:
  Domain signal in first 3 tokens only; route at last token.
  Stateless (last token) sees only noise → ~50%.
  Mean pool partially recovers signal. LIF carries it forward.

Experiment 2 — Domain switch:
  Sequence starts as domain A, switches to domain B at step 4.
  Correct routing = CURRENT domain (B at the end).
  Mean pool averages both domains → confusion.
  LIF with appropriate β forgets domain A and tracks domain B.
  This is where temporal memory strictly dominates averaging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)

D_MODEL   = 16
N_EXPERTS = 4
N_DOMAINS = 2
SEQ_LEN   = 8
NOISE_STD = 1.2
BETA      = 0.75      # lower β for switch task — faster forgetting
THRESHOLD = 1.0
BATCH     = 512
EPOCHS    = 500
LR        = 3e-3


# ── Datasets ──────────────────────────────────────────────────────────────────
def domain_signal(d, d_model):
    s = torch.zeros(d_model)
    s[d * (d_model // 2): (d + 1) * (d_model // 2)] = 1.0
    return s


def make_early_signal_batch(batch, seq_len, d_model, signal_steps=3, noise=NOISE_STD):
    """Signal only in first signal_steps tokens. Route at last token."""
    domains = torch.randint(0, N_DOMAINS, (batch,))
    seqs    = torch.randn(batch, seq_len, d_model) * noise
    for b in range(batch):
        sig = domain_signal(domains[b].item(), d_model)
        seqs[b, :signal_steps] += sig
    return seqs, domains * 2


def make_switch_batch(batch, seq_len, d_model, switch_at=4, noise=NOISE_STD):
    """
    Sequence starts as domain A (steps 0..switch_at-1),
    then switches to domain B (steps switch_at..end).
    Correct expert = domain B (the current domain at routing time).
    Mean pool will see ~equal signal from both domains → confused.
    """
    d_start = torch.randint(0, N_DOMAINS, (batch,))
    d_end   = 1 - d_start           # always switch to the other domain
    seqs    = torch.randn(batch, seq_len, d_model) * noise
    for b in range(batch):
        sig_start = domain_signal(d_start[b].item(), d_model)
        sig_end   = domain_signal(d_end[b].item(), d_model)
        seqs[b, :switch_at]   += sig_start
        seqs[b, switch_at:]   += sig_end
    return seqs, d_end * 2          # target = current domain at end


# ── Routers ───────────────────────────────────────────────────────────────────
class StatelessLastToken(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.W = nn.Linear(d_model, n_experts)

    def forward(self, seq):
        return F.softmax(self.W(seq[:, -1, :]), dim=-1)


class StatelessMeanPool(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.W = nn.Linear(d_model, n_experts)

    def forward(self, seq):
        return F.softmax(self.W(seq.mean(dim=1)), dim=-1)


class LIFRouter(nn.Module):
    def __init__(self, d_model, n_experts, beta, threshold):
        super().__init__()
        self.W         = nn.Linear(d_model, n_experts)
        self.beta      = beta
        self.threshold = threshold

    def forward(self, seq):
        B, T, _ = seq.shape
        U = torch.zeros(B, self.W.out_features)
        for t in range(T):
            I = self.W(seq[:, t, :])
            U = self.beta * U + I
            U = U - F.relu(U - self.threshold)
        return F.softmax(U, dim=-1)


class LIFRouterLearnableBeta(nn.Module):
    def __init__(self, d_model, n_experts, beta_init, threshold):
        super().__init__()
        self.W          = nn.Linear(d_model, n_experts)
        self.threshold  = threshold
        self.beta_raw   = nn.Parameter(
            torch.full((n_experts,),
                       torch.logit(torch.tensor(float(beta_init))).item()))

    def forward(self, seq):
        B, T, _ = seq.shape
        beta = torch.sigmoid(self.beta_raw)
        U    = torch.zeros(B, self.W.out_features)
        for t in range(T):
            I = self.W(seq[:, t, :])
            U = beta * U + I
            U = U - F.relu(U - self.threshold)
        return F.softmax(U, dim=-1)

    def learned_betas(self):
        return torch.sigmoid(self.beta_raw).detach().numpy().round(3)


# ── Training ──────────────────────────────────────────────────────────────────
def train(model, make_batch_fn, name, epochs, lr, batch, seq_len, d_model):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    accs = []
    for epoch in range(epochs):
        seqs, targets = make_batch_fn(batch, seq_len, d_model)
        logits = model(seqs)
        loss   = F.cross_entropy(logits, targets)
        opt.zero_grad(); loss.backward(); opt.step()
        acc = (logits.argmax(-1) == targets).float().mean().item()
        accs.append(acc)
        if (epoch + 1) % 100 == 0:
            print(f"  [{name}] epoch {epoch+1} | loss {loss.item():.4f} | acc {acc:.3f}")
    return accs


def run_experiment(name, make_batch_fn, beta):
    print(f"\n{'='*65}")
    print(f"Experiment: {name}")
    print(f"{'='*65}")
    models = {
        "Stateless (last token)": StatelessLastToken(D_MODEL, N_EXPERTS),
        "Stateless (mean pool) ": StatelessMeanPool(D_MODEL, N_EXPERTS),
        f"LIF (fixed β={beta})   ": LIFRouter(D_MODEL, N_EXPERTS, beta, THRESHOLD),
        f"LIF (learned β)       ": LIFRouterLearnableBeta(D_MODEL, N_EXPERTS, beta, THRESHOLD),
    }
    results = {}
    for mname, model in models.items():
        print(f"\nTraining: {mname}")
        accs = train(model, make_batch_fn, mname, EPOCHS, LR, BATCH, SEQ_LEN, D_MODEL)
        results[mname] = {
            "final": accs[-1],
            "mean50": np.mean(accs[-50:]),
        }
        if hasattr(model, 'learned_betas'):
            print(f"  Learned β: {model.learned_betas()}")

    print(f"\n{'─'*65}")
    print(f"{'Model':<35} {'Final':>7} {'Mean50':>8}")
    print(f"{'─'*65}")
    for mname, r in results.items():
        print(f"{mname:<35} {r['final']:>7.3f} {r['mean50']:>8.3f}")
    return results


# ── Run both experiments ──────────────────────────────────────────────────────
r1 = run_experiment(
    "Early signal / late noise  (β=0.90)",
    lambda b, t, d: make_early_signal_batch(b, t, d, signal_steps=3),
    beta=0.90,
)

r2 = run_experiment(
    "Domain switch at step 4    (β=0.75)",
    lambda b, t, d: make_switch_batch(b, t, d, switch_at=4),
    beta=0.75,
)

print(f"\n{'='*65}")
print("Summary: where does temporal memory matter?")
print(f"{'='*65}")
print(f"{'Model':<35} {'Exp1':>6} {'Exp2':>6}")
print(f"{'─'*65}")
models_order = list(r1.keys())
for mname in models_order:
    k2 = [k for k in r2 if k.startswith(mname[:20])]
    v2 = r2[k2[0]]['mean50'] if k2 else 0.0
    print(f"{mname:<35} {r1[mname]['mean50']:>6.3f} {v2:>6.3f}")

print("""
Interpretation:
  Exp1 (early signal): mean pool wins because i.i.d. noise is averaged optimally.
                       LIF carries signal but late noise disrupts membrane.
  Exp2 (domain switch): mean pool fails — it averages both domains, sees neither.
                        LIF with low β forgets old domain and tracks new one.
  → Temporal memory is not always better than averaging.
    It is strictly necessary when the world changes mid-sequence.
    β controls the forgetting rate — must match the timescale of domain change.
""")
