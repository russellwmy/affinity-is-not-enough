"""
Anticipatory routing: route on predicted next state, not current token.

From Friston: active inference selects actions that minimize predicted future
free energy, not current free energy. From Eshraghian: MTP forces representations
to encode trajectory, not just current position. From DeepSeek: MTP is discarded
at inference — this prototype makes it persistent.

Hypothesis: routing on the predicted next token embedding outperforms routing
on the current token when the task requires anticipation — i.e., when the
correct expert for token t depends on token t+1, not token t.

Task design:
  Sequences of tokens from a slowly evolving domain (like a conversation that
  is about to change topic). The expert needed at step t is determined by the
  domain at step t+1 (look-ahead routing).

  Example: legal text transitioning to medical text. At the transition token,
  the best expert is already the medical expert — but the current token still
  looks legal. Only a router that predicts the next token knows to switch.

Three routers:
  1. Current-token router: standard, routes on x_t
  2. Next-token oracle: routes on x_{t+1} directly (upper bound)
  3. Anticipatory router: predicts x̂_{t+1} from x_t, routes on x̂_{t+1}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
N_EXPERTS    = 4
N_DOMAINS    = 2
SEQ_LEN      = 12
D_MODEL      = 16
D_EXPERT     = 32
NOISE_STD    = 0.8
TRANSITION   = 6      # domain switches at this step within each sequence
BATCH        = 512
EPOCHS       = 600
LR           = 3e-3


# ── Dataset ───────────────────────────────────────────────────────────────────
# Each sequence: domain A for steps 0..TRANSITION-1, domain B for steps TRANSITION..end.
# Correct expert at step t = domain of step t+1 (anticipatory).
# At the transition step t=TRANSITION-1: correct expert is domain B's expert,
# but x_t still looks like domain A.

def make_batch(batch, seq_len, d_model, transition, noise):
    half    = d_model // 2
    seqs    = torch.randn(batch, seq_len, d_model) * noise
    targets = torch.zeros(batch, seq_len, dtype=torch.long)  # expert index

    for b in range(batch):
        d_start = torch.randint(0, N_DOMAINS, (1,)).item()
        d_end   = 1 - d_start

        sig_start = torch.zeros(d_model)
        sig_start[d_start * half: (d_start + 1) * half] = 1.0
        sig_end = torch.zeros(d_model)
        sig_end[d_end * half: (d_end + 1) * half] = 1.0

        seqs[b, :transition]   += sig_start
        seqs[b, transition:]   += sig_end

        # target expert at each step = domain of NEXT step
        # steps 0..transition-2: next step is domain A → expert d_start*2
        # step transition-1: next step is domain B → expert d_end*2  ← anticipation
        # steps transition..end-1: next step is domain B → expert d_end*2
        # last step: no next step, use current domain
        for t in range(seq_len - 1):
            next_domain = d_start if t + 1 < transition else d_end
            targets[b, t] = next_domain * 2
        targets[b, -1] = d_end * 2   # last step: current domain

    return seqs, targets


# ── Predictor (lightweight MTP module) ───────────────────────────────────────
class NextTokenPredictor(nn.Module):
    """Predicts x̂_{t+1} from x_t. Trained jointly with the router."""
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x):
        return self.net(x)   # (B, d_model)


# ── Routers ───────────────────────────────────────────────────────────────────
class CurrentTokenRouter(nn.Module):
    """Routes on x_t — standard, no anticipation."""
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.W = nn.Linear(d_model, n_experts)

    def forward(self, x_t, x_next=None):
        return F.softmax(self.W(x_t), dim=-1)


class OracleRouter(nn.Module):
    """Routes on x_{t+1} — upper bound, not achievable at inference."""
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.W = nn.Linear(d_model, n_experts)

    def forward(self, x_t, x_next):
        return F.softmax(self.W(x_next), dim=-1)


class AnticipatoryRouter(nn.Module):
    """
    Routes on x̂_{t+1} predicted from x_t.
    Predictor and router trained jointly — predictor learns to encode
    trajectory-relevant features for routing, not just next-token reconstruction.
    """
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.predictor = NextTokenPredictor(d_model)
        self.W         = nn.Linear(d_model, n_experts)

    def forward(self, x_t, x_next=None):
        x_hat = self.predictor(x_t)
        return F.softmax(self.W(x_hat), dim=-1), x_hat

    def route_only(self, x_t):
        x_hat = self.predictor(x_t)
        return F.softmax(self.W(x_hat), dim=-1)


# ── Training ──────────────────────────────────────────────────────────────────
def train_standard(router_cls, name, epochs, use_next=False):
    router = router_cls(D_MODEL, N_EXPERTS)
    opt    = torch.optim.Adam(router.parameters(), lr=LR)
    accs_all, accs_transition = [], []

    for epoch in range(epochs):
        seqs, targets = make_batch(BATCH, SEQ_LEN, D_MODEL, TRANSITION, NOISE_STD)
        # seqs: (B, T, D), targets: (B, T)

        total_loss = 0.0
        all_correct, trans_correct, trans_total = 0, 0, 0

        for t in range(SEQ_LEN - 1):
            x_t    = seqs[:, t, :]
            x_next = seqs[:, t + 1, :]
            tgt_t  = targets[:, t]

            if use_next:
                logits = router(x_t, x_next)
            else:
                logits = router(x_t, x_next)   # oracle also needs x_next

            loss = F.cross_entropy(logits, tgt_t)
            total_loss += loss

            preds = logits.argmax(-1)
            all_correct += (preds == tgt_t).sum().item()

            # track accuracy specifically at transition step
            if t == TRANSITION - 1:
                trans_correct += (preds == tgt_t).sum().item()
                trans_total   += BATCH

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        acc_all   = all_correct / (BATCH * (SEQ_LEN - 1))
        acc_trans = trans_correct / trans_total if trans_total > 0 else 0
        accs_all.append(acc_all)
        accs_transition.append(acc_trans)

        if (epoch + 1) % 120 == 0:
            print(f"  [{name}] epoch {epoch+1} | acc_all {acc_all:.3f} | acc_transition {acc_trans:.3f}")

    return accs_all, accs_transition


def train_anticipatory(name, epochs):
    router = AnticipatoryRouter(D_MODEL, N_EXPERTS)
    opt    = torch.optim.Adam(router.parameters(), lr=LR)
    accs_all, accs_transition = [], []

    for epoch in range(epochs):
        seqs, targets = make_batch(BATCH, SEQ_LEN, D_MODEL, TRANSITION, NOISE_STD)
        total_loss = 0.0
        all_correct, trans_correct, trans_total = 0, 0, 0

        for t in range(SEQ_LEN - 1):
            x_t    = seqs[:, t, :]
            x_next = seqs[:, t + 1, :]
            tgt_t  = targets[:, t]

            logits, x_hat = router(x_t, x_next)

            # routing loss
            routing_loss = F.cross_entropy(logits, tgt_t)
            # prediction loss: predictor should approximate next token
            pred_loss    = F.mse_loss(x_hat, x_next.detach())
            loss         = routing_loss + 0.5 * pred_loss

            total_loss += loss

            preds = logits.argmax(-1)
            all_correct += (preds == tgt_t).sum().item()

            if t == TRANSITION - 1:
                trans_correct += (preds == tgt_t).sum().item()
                trans_total   += BATCH

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        acc_all   = all_correct / (BATCH * (SEQ_LEN - 1))
        acc_trans = trans_correct / trans_total if trans_total > 0 else 0
        accs_all.append(acc_all)
        accs_transition.append(acc_trans)

        if (epoch + 1) % 120 == 0:
            print(f"  [{name}] epoch {epoch+1} | acc_all {acc_all:.3f} | acc_transition {acc_trans:.3f}")

    return accs_all, accs_transition


# ── Run ───────────────────────────────────────────────────────────────────────
print("=" * 65)
print("Task: anticipatory routing — correct expert at step t is domain of t+1")
print(f"Domain switch at step {TRANSITION}/{SEQ_LEN}. Noise={NOISE_STD}.")
print("=" * 65)
print("\nacc_transition = accuracy specifically at the transition step")
print("This is where anticipation matters most.\n")

configs = [
    ("Current-token router  ", CurrentTokenRouter, False),
    ("Oracle (next token)   ", OracleRouter,        True),
]

results = {}
for name, cls, use_next in configs:
    print(f"Training: {name}")
    a_all, a_trans = train_standard(cls, name, EPOCHS, use_next)
    results[name] = {"mean_all": np.mean(a_all[-50:]),
                     "mean_trans": np.mean(a_trans[-50:])}
    print()

print("Training: Anticipatory router")
a_all, a_trans = train_anticipatory("Anticipatory router   ", EPOCHS)
results["Anticipatory router   "] = {"mean_all": np.mean(a_all[-50:]),
                                      "mean_trans": np.mean(a_trans[-50:])}

print(f"\n{'='*65}")
print(f"{'Model':<25} {'Acc (all)':>12} {'Acc (transition)':>18}")
print(f"{'─'*65}")
for name, r in results.items():
    print(f"{name:<25} {r['mean_all']:>12.3f} {r['mean_trans']:>18.3f}")

print("""
Interpretation:
  Current-token router: acc_transition should be ~0.50 — at the transition
    step the token still looks like domain A but the correct expert is B.
  Oracle: upper bound — routes on the actual next token.
  Anticipatory: should close the gap — predictor learns to detect transitions
    from x_t alone and pre-routes to the upcoming domain's expert.
  The gap (oracle − anticipatory) = cost of prediction uncertainty.
  The gap (anticipatory − current) = gain from anticipation.
""")
