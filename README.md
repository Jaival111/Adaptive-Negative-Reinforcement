# Adaptive Negative Sample Reinforcement (A-NSR)

> **Adaptive Negative Sample Reinforcement for LLM Reasoning: Dynamically Balancing Correction and Diversity in RLVR**
>
> Yash Ingle · Jaival Chauhan · Ankit Yadav · Sudhakar Mishra
> *Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat, India*

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-blue)](https://arxiv.org)
[![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-orange)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)

---

## Overview

**A-NSR** is a time-adaptive reinforcement learning framework for improving the mathematical reasoning capabilities of Large Language Models (LLMs). It extends Negative Sample Reinforcement (NSR) by replacing the fixed penalty weight in W-REINFORCE with *time-dependent scheduling functions* that dynamically adjust the balance between error correction (NSR) and diversity preservation (PSR) as training progresses.

### The Core Idea

Standard negative reinforcement uses a fixed penalty throughout training — but the model's error patterns evolve significantly during RLVR training. Early on, the model makes many mistakes and needs strong correction. Later, once accuracy improves, an equally aggressive penalty can hurt output diversity. **A-NSR resolves this by adapting the training signal to the current stage of learning.**

```
Early Training  →  Strong NSR (error correction focused)
Late Training   →  Softer NSR + Growing PSR (diversity preservation)
```

This is analogous to **curriculum learning** applied to the reward signal, not the data.

---

## Background

### RLVR Objective

Given a language model with parameters θ, a prompt distribution D, and a deterministic verifier r:

$$\mathcal{L}_{\text{RLVR}}(\theta) = -\mathbb{E}_{x \sim D,\, y \sim \pi_\theta(\cdot|x)} [r(x, y)], \quad r(x,y) \in \{-1, +1\}$$

This decomposes into Positive Sample Reinforcement (PSR) and Negative Sample Reinforcement (NSR):

$$\mathcal{L}_{\text{RLVR}}(\theta) = \mathcal{L}_{\text{PSR}}(\theta) + \mathcal{L}_{\text{NSR}}(\theta)$$

### W-REINFORCE Baseline (Zhu et al., 2025)

$$\mathcal{L}_{\text{W-REINFORCE}}(\theta) = \lambda \mathcal{L}_{\text{PSR}}(\theta) + \mathcal{L}_{\text{NSR}}(\theta), \quad \lambda = 0.1$$

The fixed λ does not account for how the model's error distribution shifts during training.

---

## Method: A-NSR

### Objective

$$\mathcal{L}_{\text{A-NSR}}(\theta; t) = \lambda(t)\,\mathcal{L}_{\text{PSR}}(\theta) + \beta(t)\,\mathcal{L}_{\text{NSR}}(\theta)$$

where `t` is the current training step and `λ(t)`, `β(t)` are positive-valued scheduling functions.

### Scheduling Functions

**Schedule 1 — Exponential Decay for NSR + Linear Increase for PSR** *(recommended)*

$$\beta(t) = \beta_{\min} + (\beta_{\max} - \beta_{\min})\,e^{-\kappa t}$$

$$\lambda(t) = \lambda_{\min} + (\lambda_{\max} - \lambda_{\min})\,\frac{t}{T_{\text{total}}}$$

Strong early correction, smooth transition to balanced updates, softened negative reinforcement in later stages.

**Schedule 2 — Cosine Annealing for NSR**

$$\beta(t) = \beta_{\min} + \frac{1}{2}(\beta_{\max} - \beta_{\min})\left(1 + \cos\!\left(\frac{\pi t}{T_{\text{total}}}\right)\right)$$

A smooth, monotonically decreasing schedule with zero derivative at the endpoints — avoids abrupt transitions.

**Schedule 3 — Performance-Driven Adaptive Weight**

$$\beta(t) = \beta_{\min} + (\beta_{\max} - \beta_{\min})\cdot(1 - \hat{p}_{\text{correct}}(t))$$

β is tied directly to the model's empirical accuracy in the current batch. When many responses are wrong, β stays high; as accuracy improves, β decreases automatically — no schedule tuning required.

### Gradient Analysis

Under adaptive weighting, the effective gradient becomes:

$$\nabla_\theta \mathcal{L}_{\text{A-NSR}} = -\mathbb{E}_{x,y}\left[w(t, r)\,r(x,y)\,\nabla_\theta \log \pi_\theta(y|x)\right]$$

where `w(t, r) = λ(t)` for correct samples and `β(t)` for incorrect ones.

**Key property:** A-NSR reweights the *magnitude* — but not the *direction* — of PSR and NSR gradients. All beneficial properties of NSR (prior-guided redistribution, high-confidence damping, implicit regularization) are preserved.

### Convergence of Effective Weight Ratio (Schedule 1)

$$\frac{\beta(t)}{\lambda(t)} \xrightarrow{t \to T_{\text{total}}} \frac{\beta_{\min}}{\lambda_{\max}}$$

The objective converges to a fixed W-REINFORCE with effective coefficient `λ_eff = λ_max / β_min`, while early stages are NSR-dominated at ratio `β_max / λ_min`.

For default hyperparameters:
- **Early ratio:** 1.5 / 0.05 = **30×** (NSR-dominated)
- **Late ratio:** 0.5 / 0.2 = **2.5×** (balanced)

### Entropy Preservation

$$\left|\frac{dH_t}{dt}\right|_{\text{NSR}} \propto \beta(t) \cdot \pi_{y_t}(1 - \pi_{y_t})$$

Since β(t) decreases, NSR-induced entropy reduction slows over time — preserving output diversity in late training.

---

## Results

Evaluated on **Qwen2.5-Math-1.5B** against W-REINFORCE (Zhu et al., 2025).

### Pass@k on AIME 2025

| Method | @1 | @2 | @4 | @8 | @16 | @32 | @64 | @128 | @256 |
|--------|-----|-----|-----|-----|------|------|------|------|------|
| W-REINFORCE | 0.80 | 1.52 | 2.74 | 4.57 | 6.87 | 9.47 | 12.20 | 14.56 | 16.67 |
| **A-NSR (Ours)** | **0.96** | **1.79** | **3.16** | **5.09** | **7.35** | **9.71** | 11.70 | 12.91 | 13.33 |

### Pass@k on AMC23

| Method | @1 | @2 | @4 | @8 | @16 | @32 | @64 | @128 | @256 |
|--------|-----|-----|-----|-----|------|------|------|------|------|
| W-REINFORCE | 10.25 | 17.57 | 27.27 | 37.96 | 48.46 | 57.92 | 65.10 | 69.34 | 72.50 |
| **A-NSR (Ours)** | **13.34** | **21.87** | **32.17** | **43.10** | **54.22** | **64.29** | **72.55** | **78.76** | **82.50** |

**A-NSR outperforms W-REINFORCE across all Pass@k values on AMC23**, and surpasses it up to Pass@32 on AIME 2025. It is especially strong at low-k settings (k ≤ 32), meaning the model produces correct answers with fewer attempts.

> Note: A-NSR results on MATH500 are unavailable due to compute constraints (evaluation required 22.5M tokens, taking >5 days).

---

## Installation

```bash
git clone https://github.com/<your-username>/A-NSR.git
cd A-NSR
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0
transformers>=4.40
trl>=0.8
accelerate>=0.28
datasets
numpy
```

---

## Usage

### Training with Schedule 1 (Exponential Decay)

```python
from ansr import ANSRTrainer, Schedule1Config

config = Schedule1Config(
    beta_max=1.5,
    beta_min=0.5,
    kappa=0.03,
    lambda_min=0.05,
    lambda_max=0.2,
    total_steps=10000,
)

trainer = ANSRTrainer(
    model="Qwen/Qwen2.5-Math-1.5B",
    config=config,
    schedule="schedule1",  # or "schedule2", "schedule3"
    train_dataset="MATH",
)

trainer.train()
```

### Training with Schedule 3 (Performance-Driven)

```python
from ansr import ANSRTrainer, Schedule3Config

config = Schedule3Config(
    beta_max=1.5,
    beta_min=0.5,
    # beta adapts automatically from batch accuracy
)

trainer = ANSRTrainer(
    model="Qwen/Qwen2.5-Math-1.5B",
    config=config,
    schedule="schedule3",
)
trainer.train()
```

### Evaluation

```python
from ansr import evaluate_passk

results = evaluate_passk(
    model_path="./checkpoints/ansr-final",
    dataset="AIME2025",   # or "AMC23", "MATH"
    n_samples=256,
    temperature=0.6,
    top_p=0.95,
    k_values=[1, 2, 4, 8, 16, 32, 64, 128, 256],
)

print(results)
```

---

## Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Prompt batch size | 1,024 |
| Rollouts per prompt | 8 |
| Mini-batch size | 256 |
| Learning rate | 1e-6 |
| Clip ratio ε | 0.2 |
| Entropy bonus coefficient | 1e-4 |
| Training temperature | 1.0 |
| **A-NSR: β_max** | **1.5** |
| **A-NSR: β_min** | **0.5** |
| **A-NSR: κ (decay rate)** | **0.03** |
| **A-NSR: λ_min** | **0.05** |
| **A-NSR: λ_max** | **0.2** |

Training was run on **2× NVIDIA A100 (80GB VRAM)** for approximately **50–60 hours**.

---

## Prompt Template

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{input}
Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>assistant
```

---

## Project Structure

```
A-NSR/
├── ansr/
│   ├── __init__.py
│   ├── trainer.py          # ANSRTrainer class
│   ├── schedules.py        # Schedule 1, 2, 3 implementations
│   ├── gradients.py        # Token-level gradient analysis utilities
│   └── evaluate.py         # Pass@k evaluation
├── scripts/
│   ├── train_schedule1.sh
│   ├── train_schedule2.sh
│   └── train_schedule3.sh
├── configs/
│   └── default_config.yaml
├── requirements.txt
└── README.md
```

---

## Theoretical Connections

**Curriculum Learning:** A-NSR can be seen as curriculum learning applied to the reward signal rather than the data. The "easy" phase (strong correction of frequent early errors) comes first; the "hard" phase (preserving diversity among mostly-correct outputs) comes later.

**Effective Learning Rate Interpretation:** A-NSR applies different effective learning rates to positive and negative samples:

$$\eta^+_{\text{eff}}(t) = \eta \cdot \lambda(t), \qquad \eta^-_{\text{eff}}(t) = \eta \cdot \beta(t)$$

The ratio ρ(t) = β(t)/λ(t) decreases monotonically under Schedule 1, smoothly shifting from NSR-dominated to PSR-inclusive training.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{ingle2025ansr,
  title     = {Adaptive Negative Reinforcement for LLM Reasoning: Dynamically Balancing Correction and Diversity in RLVR},
  author    = {Ingle, Yash and Chauhan, Jaival and Yadav, Ankit and Mishra, Sudhakar},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```

---

## Related Work

- [W-REINFORCE / NSR](https://arxiv.org) — Zhu et al., NeurIPS 2025 (baseline)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — Guo et al., 2025
- [DAPO](https://arxiv.org/abs/2503.14476) — Yu et al., 2025
- [Qwen2.5-Math](https://arxiv.org/abs/2409.12122) — Yang et al., 2024
- [CW-NSR (companion repo)](https://github.com/Jaival111/Confidence-Weighted-NSR) — Confidence-Weighted NSR

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*For questions or issues, please open a GitHub issue or contact `u23ai035@coed.svnit.ac.in`.*
