"""
Precision-weighted gating (Π mechanism).

From Friston Box 2:
  Π_i = 1 / σ²_i  (inverse variance of expert i's prediction error)
  Gating: g'_i = g_i · Π_i / Σ_j Π_j

Hypothesis: routing weighted by recent prediction accuracy outperforms
routing weighted by input affinity alone, when experts have heterogeneous
and context-dependent reliability.

Task design:
  - 4 experts. Domain A and domain B inputs.
  - Expert 0: accurate for domain A, noisy for domain B.
  - Expert 2: accurate for domain B, noisy for domain A.
  - Experts 1, 3: decoys — moderate noise regardless of domain.
  - Affinity router: routes by input similarity to expert centroids.
  - Precision router: routes by affinity × Π, where Π = running inverse
    variance of each expert's prediction error on recent tokens.

Two conditions:
  Condition 1 — Static reliability: expert noise levels fixed throughout.
  Condition 2 — Shifting reliability: expert noise levels switch at step 500.
    Expert 0 becomes noisy, expert 2 becomes accurate for domain A.
    Tests whether Π tracks changing reliability online.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)

# ── Config ───────────────────────────────────────────────────────────────────
N_EXPERTS   = 4
N_DOMAINS   = 2
D_MODEL     = 16
D_EXPERT    = 32
BATCH       = 256
STEPS       = 1000
LR          = 3e-3
PI_MOMENTUM = 0.95    # EMA decay for variance estimate
PI_EPS      = 1e-4    # numerical stability


# ── Expert FFNs with controllable output noise ────────────────────────────────
class NoisyExpert(nn.Module):
    def __init__(self, d_model, d_expert, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_expert),
            nn.ReLU(),
            nn.Linear(d_expert, d_out),
        )

    def forward(self, x, noise_std=0.0):
        out = self.net(x)
        if noise_std > 0:
            out = out + torch.randn_like(out) * noise_std
        return out


# ── Precision tracker (non-parametric, online) ────────────────────────────────
class PrecisionTracker:
    """
    Maintains per-expert running variance of prediction error.
    Precision = 1 / variance.
    Updated after each forward pass with observed errors.
    """
    def __init__(self, n_experts, momentum=PI_MOMENTUM, eps=PI_EPS):
        self.n   = n_experts
        self.mom = momentum
        self.eps = eps
        self.var = torch.ones(n_experts)   # start with unit variance → Π=1

    def update(self, errors):
        """errors: (n_experts,) — mean squared error for each expert this step."""
        self.var = self.mom * self.var + (1 - self.mom) * errors

    def precision(self):
        return 1.0 / (self.var + self.eps)

    def reset(self):
        self.var = torch.ones(self.n)


# ── Routers ───────────────────────────────────────────────────────────────────
class AffinityRouter(nn.Module):
    """Standard MoE: routes by input affinity to expert centroids."""
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.W = nn.Linear(d_model, n_experts)

    def forward(self, x, pi=None):
        scores = self.W(x)             # (B, n_experts)
        return F.softmax(scores, dim=-1)


class PrecisionRouter(nn.Module):
    """
    Routes by affinity × Π.
    Π is passed in from the external PrecisionTracker (non-differentiable update).
    Affinity weights are still learned by gradient; Π modulates their effect.
    """
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.W = nn.Linear(d_model, n_experts)

    def forward(self, x, pi):
        scores = self.W(x)             # (B, n_experts)
        # precision-weight the affinity scores before softmax
        scores = scores * pi.unsqueeze(0)
        return F.softmax(scores, dim=-1)


# ── Full MoE model ────────────────────────────────────────────────────────────
class MoEModel(nn.Module):
    def __init__(self, d_model, n_experts, d_expert, d_out, router_cls):
        super().__init__()
        self.router  = router_cls(d_model, n_experts)
        self.experts = nn.ModuleList(
            [NoisyExpert(d_model, d_expert, d_out) for _ in range(n_experts)]
        )
        self.n_experts = n_experts

    def forward(self, x, noise_stds, pi=None):
        """
        x:          (B, d_model)
        noise_stds: list of n_experts floats — noise per expert this step
        pi:         (n_experts,) precision weights, or None for affinity router
        """
        if pi is not None:
            gates = self.router(x, pi)         # (B, n_experts)
        else:
            gates = self.router(x)

        # weighted sum of expert outputs
        expert_outs = torch.stack(
            [self.experts[i](x, noise_stds[i]) for i in range(self.n_experts)],
            dim=1
        )                                      # (B, n_experts, d_out)

        out = (gates.unsqueeze(-1) * expert_outs).sum(dim=1)  # (B, d_out)

        # per-expert errors (for Π update): MSE of each expert vs. target
        # computed externally after calling forward
        return out, gates, expert_outs


# ── Dataset ───────────────────────────────────────────────────────────────────
def make_batch(batch, d_model):
    """
    Domain A: target is a linear function of dims 0..7.
    Domain B: target is a linear function of dims 8..15.
    Inputs are uniform in [-1, 1].
    """
    x       = torch.randn(batch, d_model)
    domains = torch.randint(0, N_DOMAINS, (batch,))
    half    = d_model // 2
    targets = torch.zeros(batch, 1)
    for b in range(batch):
        d = domains[b].item()
        targets[b] = x[b, d * half: (d + 1) * half].mean()
    return x, targets, domains


# ── Noise schedule ────────────────────────────────────────────────────────────
def get_noise_stds(step, shift_at=None):
    """
    Expert reliability:
      Expert 0: low noise domain A, high noise domain B
      Expert 1: medium noise (decoy)
      Expert 2: high noise domain A, low noise domain B
      Expert 3: medium noise (decoy)

    After shift_at (if set): experts 0 and 2 swap reliability.
    """
    if shift_at is not None and step >= shift_at:
        return [1.5, 0.5, 0.1, 0.5]   # expert 2 now reliable, expert 0 noisy
    else:
        return [0.1, 0.5, 1.5, 0.5]   # expert 0 reliable, expert 2 noisy


# ── Training loop ─────────────────────────────────────────────────────────────
def train(model_name, router_cls, steps, lr, batch, d_model, shift_at=None):
    model   = MoEModel(d_model, N_EXPERTS, D_EXPERT, 1, router_cls)
    tracker = PrecisionTracker(N_EXPERTS) if router_cls == PrecisionRouter else None
    opt     = torch.optim.Adam(model.parameters(), lr=lr)

    losses, pi_history = [], []

    for step in range(steps):
        x, targets, domains = make_batch(batch, d_model)
        noise_stds = get_noise_stds(step, shift_at)

        pi = tracker.precision() if tracker else None
        out, gates, expert_outs = model(x, noise_stds, pi)

        loss = F.mse_loss(out, targets)
        opt.zero_grad(); loss.backward(); opt.step()

        # update Π tracker with per-expert prediction errors
        if tracker is not None:
            with torch.no_grad():
                per_expert_err = ((expert_outs - targets.unsqueeze(1)) ** 2
                                  ).mean(dim=(0, 2))   # (n_experts,)
                tracker.update(per_expert_err.detach())
                pi_history.append(tracker.precision().numpy().copy())

        losses.append(loss.item())

        if (step + 1) % 200 == 0:
            pi_str = ""
            if tracker:
                p = tracker.precision().numpy()
                pi_str = f" | Π=[{p[0]:.2f},{p[1]:.2f},{p[2]:.2f},{p[3]:.2f}]"
            print(f"  [{model_name}] step {step+1:4d} | loss {loss.item():.4f}{pi_str}")

    return losses, pi_history


# ── Run experiments ───────────────────────────────────────────────────────────
def run(label, shift_at=None):
    print(f"\n{'='*65}")
    print(f"Condition: {label}")
    if shift_at:
        print(f"  Expert reliability shifts at step {shift_at}:")
        print(f"  Before: expert 0 reliable (noise=0.1), expert 2 noisy (noise=1.5)")
        print(f"  After:  expert 2 reliable (noise=0.1), expert 0 noisy (noise=1.5)")
    else:
        print(f"  Static: expert 0 reliable (noise=0.1), expert 2 noisy (noise=1.5)")
    print(f"{'='*65}")

    results = {}
    for name, cls in [("Affinity router  ", AffinityRouter),
                      ("Precision router ", PrecisionRouter)]:
        print(f"\nTraining: {name}")
        losses, pi_hist = train(name, cls, STEPS, LR, BATCH, D_MODEL, shift_at)
        results[name] = {
            "mean_first100": np.mean(losses[:100]),
            "mean_last100":  np.mean(losses[-100:]),
            "pi_history":    pi_hist,
        }

    print(f"\n{'─'*65}")
    print(f"{'Model':<25} {'Early loss':>12} {'Final loss':>12}")
    print(f"{'─'*65}")
    for name, r in results.items():
        print(f"{name:<25} {r['mean_first100']:>12.4f} {r['mean_last100']:>12.4f}")

    return results


r1 = run("Static reliability")
r2 = run("Shifting reliability", shift_at=500)

# ── Show Π dynamics around the shift ─────────────────────────────────────────
print(f"\n{'='*65}")
print("Π dynamics around reliability shift (step 500)")
print("Expert 0 and 2 swap roles. Does Π track this?")
print(f"{'='*65}")
pi_hist = r2["Precision router "]["pi_history"]
if pi_hist:
    checkpoints = [490, 495, 500, 505, 510, 520, 550, 600]
    print(f"{'Step':>6}  {'Π₀':>8}  {'Π₁':>8}  {'Π₂':>8}  {'Π₃':>8}  {'Π₀>Π₂?':>8}")
    for s in checkpoints:
        if s < len(pi_hist):
            p = pi_hist[s]
            print(f"{s:>6}  {p[0]:>8.3f}  {p[1]:>8.3f}  {p[2]:>8.3f}  {p[3]:>8.3f}"
                  f"  {'YES' if p[0]>p[2] else 'no':>8}")

print("""
Interpretation:
  Static: precision router should converge faster — upweights reliable expert 0
          from the start, downweights noisy expert 2.
  Shifting: after step 500, Π should flip — expert 2 becomes more precise
            than expert 0 as the tracker detects the reliability change.
  The momentum parameter controls adaptation speed:
    High momentum (0.95): slow adaptation — safe but lags on shifts
    Low momentum (0.7):   fast adaptation — responsive but noisy estimates
  This is Friston's precision regulation: the system continuously
  recalibrates which sources of evidence to trust.
""")
