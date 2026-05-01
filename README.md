# Affinity Is Not Enough: Recovering the Free Energy Principle in Mixture-of-Experts

**Author:** Man Yung (Russell) Wong (Independent Researcher)
**Contact:** research@russellwong.io
**Paper:** [paper.pdf](./paper.pdf) — also on arXiv (link to be added after submission)

---

Sparse Mixture-of-Experts (MoE) routing fails at exactly the points where routing matters most — domain transitions where the current token belongs to one distribution and the next belongs to another. In a controlled routing experiment, standard affinity-based routing assigns only $0.006 \pm 0.001$ probability mass to the correct expert at the transition step. We show that three lightweight modifications to the routing gate raise this to $0.748 \pm 0.002$ — a 124× increase, reducing the experts needed for 99% routing coverage from infeasible to a small constant:

- **Temporal memory (β)** — per-expert LIF membrane potential that accumulates routing context across tokens.
- **Precision-weighted gating (Π)** — per-expert inverse variance of recent prediction error, yielding a 31× contrast between reliable and unreliable experts.
- **Anticipatory routing** — next-state predictor conditioned on the β-accumulated hidden state.

The three mechanisms are motivated by Friston's Free Energy Principle and instantiated using LIF dynamics from spiking neural networks. An ablation across all 2³ mechanism subsets reveals a **super-additive β × Ant interaction**: anticipation alone gives nothing ($+0.000 \pm 0.001$), β alone gives modest gain ($+0.295 \pm 0.013$), but combined they close 75% of the oracle gap ($+0.741 \pm 0.002$, exceeding the sum of individual gains by $+0.446 \pm 0.014$). This is a structural finding — a stateless predictor cannot detect approaching domain transitions, because pre-transition tokens are distributionally identical to within-domain tokens.

## Repository contents

```
.
├── paper.pdf            # full paper (27 pages)
├── prototype/           # reference implementations (~200 lines each)
│   ├── beta_routing.py         # β mechanism (paper §4.1)
│   ├── precision_gating.py     # Π mechanism (§4.2)
│   ├── anticipatory_routing.py # anticipatory routing (§4.3)
│   ├── ablation.py             # full 2³ ablation (§4.4)
│   ├── lm_experiment.py        # character-level LM (§4.5)
│   ├── routing_entropy.py      # K-reduction analysis (§3.4)
│   ├── format_tables.py        # emit paper-ready markdown tables
│   ├── pyproject.toml          # uv-managed Python project
│   └── uv.lock
└── LICENSE              # MIT
```

Each prototype script is self-contained — the experimental setup, the model, training loop, and evaluation are in one file. Read the file's docstring for the hypothesis it tests and the expected result.

## Reproducing the experiments

Dependencies: [uv](https://github.com/astral-sh/uv) for Python environment management.

```sh
cd prototype
uv run python ablation.py        # ~50 min on a laptop CPU; 5 seeds × 9 conditions
uv run python lm_experiment.py   # ~2.5 hr; 5 seeds × 3 models
uv run python format_tables.py   # emits paper-ready markdown tables from JSONs
```

Result JSONs (`ablation_results.json`, `lm_experiment_results.json`) and run logs are gitignored — they regenerate on each run.

The two longer scripts dominate runtime. The four other prototypes (`beta_routing.py`, `precision_gating.py`, `anticipatory_routing.py`, `routing_entropy.py`) each run in 1–5 minutes and validate individual mechanisms in isolation.

## Adding the mechanisms to your own MoE

Each mechanism is a small additive modification to a standard routing gate `softmax(W @ x_t)`. Minimal additions (paper §3.5):

**β (temporal memory):**

```python
self.beta_raw = nn.Parameter(torch.full((d_model,), torch.logit(torch.tensor(0.9))))
# at start of each sequence:
h = torch.zeros(B, d_model)
# at each step:
beta = torch.sigmoid(self.beta_raw)
h    = beta * h + x_t
gate = F.softmax(self.W(h), dim=-1)            # was: self.W(x_t)
```

**Π (precision-weighted gating):**

```python
self.var = torch.ones(n_experts) * 0.5         # initial reliability prior
# after each batch (no_grad):
self.var = 0.95 * self.var + 0.05 * mse_per_expert
pi   = 1.0 / (self.var + 1e-4)
gate = F.softmax(self.W(x_t) * pi.unsqueeze(0), dim=-1)
```

**Anticipatory routing (additive form):**

```python
self.W_pred    = nn.Linear(d_model, n_experts, bias=False)
self.predictor = nn.Sequential(nn.Linear(d_model*2, d_model*4), nn.GELU(),
                               nn.Linear(d_model*4, d_model))
# at each step (combined with β above):
x_hat = self.predictor(torch.cat([x_t, h], dim=-1))
gate  = F.softmax(self.W(h) + self.W_pred(x_hat), dim=-1)
# add to training loss:
pred_loss = F.mse_loss(x_hat, x_next.detach())  # weighted at 0.5
```

The full composed gate is `softmax((W·h + W_pred·x̂) ⊙ Π)`. No expert architecture changes required.

## Citation

```bibtex
@article{wong2026affinity,
  author = {Wong, Man Yung Russell},
  title  = {Affinity Is Not Enough: Recovering the Free Energy Principle in Mixture-of-Experts},
  year   = {2026},
  eprint = {arXiv:XXXX.XXXXX},
  url    = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

(arXiv ID will be filled in after submission.)

## License

MIT — see [LICENSE](./LICENSE).

## Acknowledgements

Claude (Anthropic) was used for code implementation, manuscript editing, and literature search. All research direction, hypotheses, experimental design, and interpretation of results are the author's own.
