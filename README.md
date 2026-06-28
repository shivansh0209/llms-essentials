# BPE Tokenizer
Byte-Pair Encoding (BPE) is a subword tokenization algorithm used by most modern LLMs (like GPT-4 and Llama) to break down raw text into manageable chunks. Instead of splitting text character-by-character (which creates too many tokens) or word-by-word (which fails to handle typos or new words), BPE finds a middle ground.

It starts by treating every individual character as a separate token. Then, it iteratively counts the most frequently occurring adjacent pairs of tokens in a massive text corpus and merges them into a brand-new, single token (for example, combining "t" and "h" into "th"). This merging process repeats until the tokenizer reaches a pre-defined vocabulary size. Because it retains individual characters at its base, BPE can efficiently compress common words into single tokens while still being able to break unfamiliar or rare words down into small, recognizable subword pieces.

# RLHF and DPO

## Part 1 — The Alignment Problem

### What is it?

A language model trained only on next-token prediction learns to imitate the internet. The internet contains everything — helpful answers, toxic rants, misinformation, manipulation. The model has no concept of "good" vs "bad" — it just predicts what token comes next.

**The alignment problem:** How do you make a model that is:
- **Helpful** — actually answers what the user wants
- **Harmless** — doesn't produce dangerous, toxic, or offensive content
- **Honest** — doesn't hallucinate or deceive

You cannot describe "helpful, harmless, honest" in a loss function. You can't write a mathematical equation that captures "this answer is better than that one." Humans can judge it, but humans can't label every possible model output at training scale.

This is the alignment problem — the gap between what the model optimizes (next-token prediction loss) and what we actually want (a model that behaves well).

### Why SFT alone doesn't fully solve it

Supervised Fine-Tuning (SFT) on curated instruction-response pairs helps a lot. But:
- You can only collect a finite number of examples
- For many prompts, there are multiple "correct" answers — SFT doesn't teach the model which is *better*
- SFT teaches the model to imitate good responses, not to understand *why* something is good

RLHF was designed to go further — to directly optimize for human preference, not just imitation.

---

## Part 2 — RLHF (Reinforcement Learning from Human Feedback)

RLHF was the key technique that made ChatGPT, Claude, and Gemini behave well. It was introduced in InstructGPT (OpenAI, 2022).

RLHF has **three distinct phases**. You must understand all three.

---

### Phase 1 — Supervised Fine-Tuning (SFT)

Before any RL, you fine-tune the base pretrained model on a curated dataset of (prompt, good response) pairs. Human annotators write high-quality answers to a diverse set of prompts.

This gives you an **SFT model** — a model that knows the format and style of helpful responses, but hasn't been optimized for preference yet.

The SFT model is the **starting point** for both Phase 2 and Phase 3.

---

### Phase 2 — Reward Model Training

The reward model (RM) is a separate neural network that learns to score model outputs the way humans would.

**How it's trained:**

1. Take a prompt from the dataset.
2. Generate **two or more responses** from the SFT model for the same prompt.
3. Show both responses to a **human annotator** and ask: *which response is better?*
4. Collect thousands of these pairwise comparisons: (prompt, chosen_response, rejected_response).
5. Train the reward model on this preference data.

**The objective:** For any (prompt, response) pair, the reward model should output a scalar score — higher means the response is better. It's trained so that `RM(prompt, chosen) > RM(prompt, rejected)` using a ranking loss.

**Architecture:** The reward model is typically the SFT model with the language model head replaced by a linear layer that outputs a single scalar.

**Key insight:** You only need humans to *compare* responses, not to score them absolutely. Comparison is much easier — even non-experts can reliably say "response A is better than response B."

---

### Phase 3 — RL Fine-Tuning with PPO

Now you use the reward model as a proxy for human judgment and train the LLM using reinforcement learning.

**The setup:**
- **Policy (π):** The LLM you're training. It takes a prompt and generates a response.
- **Environment:** The prompt dataset.
- **Reward signal:** The reward model scores the generated response.
- **Goal:** Update the policy so it generates responses that the reward model scores highly.

**The PPO training loop (one step):**

```
1. Sample a prompt from the dataset
2. The current policy generates a response (this is the "action")
3. The reward model scores the response → scalar reward R
4. Add a KL penalty: R_adjusted = R - β * KL(policy || reference_policy)
5. Use PPO to update the policy weights to increase R_adjusted
6. Repeat
```

**What is the KL penalty?**

Without any constraint, the policy would rapidly find ways to "game" the reward model — generating weird, repetitive text that scores high but is useless. This is called **reward hacking**.

To prevent this, you add a KL divergence penalty between the current policy and the frozen reference policy (the original SFT model). This forces the policy to stay close to the SFT model — it can improve, but can't drift too far.

β is a hyperparameter controlling how strongly the KL penalty is applied.

---

### PPO — Proximal Policy Optimization (Deep Dive)

PPO is the RL algorithm used to update the LLM in Phase 3. You don't need to implement it, but you must understand what it does conceptually.

**The core RL problem:** You want to update the policy to generate higher-reward responses. But if you take too large a gradient step, the policy can collapse — it starts generating garbage. Standard policy gradient methods are unstable.

**What PPO does:** It constrains how much the policy can change in a single update.

**The PPO objective (simplified):**

```
L_PPO = E[ min(r_t * A_t,  clip(r_t, 1-ε, 1+ε) * A_t) ]
```

Where:
- `r_t = π_new(a|s) / π_old(a|s)` — the probability ratio of the new vs old policy taking the same action
- `A_t` — the **advantage** — how much better this action was compared to the average (positive = better than expected, negative = worse)
- `clip(r_t, 1-ε, 1+ε)` — clip the ratio so it can't go too far from 1.0 (typically ε = 0.2)

**Intuition:** If an action had a positive advantage (it was good), increase its probability — but not more than 20%. If an action had a negative advantage (it was bad), decrease its probability — but not more than 20%. This "proximal" constraint is why it's stable.

**In LLM context:** Each "action" is generating a token. The advantage tells the model whether generating this particular token (in this context) led to a higher-than-expected reward from the reward model.

---

### Problems with RLHF

RLHF works, but it has significant practical problems:

| Problem | Description |
|---|---|
| **Complexity** | Three separate training phases, three separate models (SFT, RM, policy). Hard to tune, lots of moving parts. |
| **Instability** | PPO is notoriously difficult to tune. Training often collapses. |
| **Reward hacking** | The policy finds ways to fool the reward model. The RM is only a proxy for human judgment, not the real thing. |
| **Compute cost** | You're running inference through multiple models during training. Expensive. |
| **KL sensitivity** | The β hyperparameter for KL penalty is critical and hard to tune. |
| **Scale** | Requires substantial infrastructure to run stably. |

This is exactly why DPO was invented.

---

## Part 3 — DPO (Direct Preference Optimization)

**Paper:** "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Rafailov et al., Stanford, 2023)

### The Key Insight

RLHF is unnecessarily complex. The researchers showed mathematically that:

> The optimal policy for the RLHF objective can be expressed **directly** in terms of the preference data — without ever explicitly training a reward model or running RL.

In other words, you can skip the reward model and the PPO loop entirely and train the LLM directly on preference pairs. The LLM implicitly becomes its own reward model.

### The Math (Simplified but Complete)

**Step 1 — The RLHF objective:**

RLHF optimizes:
```
max_π E[R(x, y)] - β * KL(π || π_ref)
```

Where R is the reward model score, β is the KL penalty weight, π is the policy, π_ref is the reference (SFT) policy.

**Step 2 — The optimal policy has a closed form:**

It can be shown that the optimal policy satisfying this objective is:
```
π*(y|x) ∝ π_ref(y|x) * exp(R(x,y) / β)
```

This means: the optimal policy is the reference policy re-weighted by the reward, tempered by β.

**Step 3 — Rearrange to express reward in terms of policies:**

From the equation above, you can isolate R:
```
R(x, y) = β * log[π*(y|x) / π_ref(y|x)] + β * log Z(x)
```

Where Z(x) is a partition function (normalizing constant) that cancels out in pairwise comparisons.

**Step 4 — Plug into the Bradley-Terry preference model:**

The Bradley-Terry model says the probability that humans prefer response y_w (winner/chosen) over y_l (loser/rejected) given prompt x is:
```
P(y_w > y_l | x) = σ(R(x, y_w) - R(x, y_l))
```

Where σ is the sigmoid function.

**Step 5 — Substitute the reward expression:**

Since Z(x) cancels in the subtraction:
```
P(y_w > y_l | x) = σ( β * log[π(y_w|x)/π_ref(y_w|x)] - β * log[π(y_l|x)/π_ref(y_l|x)] )
```

**Step 6 — The DPO loss:**

Maximize the log-likelihood of human preferences:
```
L_DPO = -E[(x, y_w, y_l)] [ log σ( β * log[π_θ(y_w|x)/π_ref(y_w|x)] - β * log[π_θ(y_l|x)/π_ref(y_l|x)] ) ]
```

**What this means in plain English:**

The DPO loss trains the model to:
- **Increase** the relative probability of the chosen response (y_w) compared to the reference model
- **Decrease** the relative probability of the rejected response (y_l) compared to the reference model

The ratio `π_θ(y|x) / π_ref(y|x)` acts as an implicit reward. The model learns to be its own reward model.

---

### What DPO Needs

**Input:** A preference dataset of triplets:
```python
{
    "prompt": "What is the capital of France?",
    "chosen": "The capital of France is Paris, which has been the country's capital since...",
    "rejected": "France doesn't really have a capital city per se..."
}
```

**Models needed:**
1. The **policy model** (what you're training) — initialized from SFT checkpoint
2. The **reference model** (frozen) — also the SFT checkpoint

No reward model. No PPO. No RL infrastructure.

---

### DPO vs RLHF — Side by Side

| Aspect | RLHF | DPO |
|---|---|---|
| Phases | 3 (SFT → RM → PPO) | 2 (SFT → DPO) |
| Models during training | SFT model + RM + policy | Policy + frozen reference |
| Requires RL | Yes (PPO) | No |
| Reward model | Explicit, separately trained | Implicit (inside the loss) |
| Training stability | Difficult, PPO collapses | Stable, like supervised learning |
| Compute cost | High (multiple models in loop) | Lower |
| Hyperparameter sensitivity | High (β, PPO clip, GAE λ...) | Low (mainly β) |
| Performance | Strong | Competitive with RLHF, often better |
| Industry adoption (2024+) | Less common | Dominant |

---

### Preference Datasets

Common open-source preference datasets:

- **Anthropic HH-RLHF** — helpfulness and harmlessness comparisons from Claude interactions
- **OpenAI summarization** — TL;DR preference data
- **UltraFeedback** — large-scale GPT-4 scored preference data
- **Orca DPO Pairs** — instruction following preferences

**Format for TRL DPOTrainer:**
```python
dataset = {
    "prompt": [...],     # the input prompt
    "chosen": [...],     # the preferred response
    "rejected": [...]    # the non-preferred response
}
```

---

### The β Hyperparameter — What It Controls

β is the single most important hyperparameter in DPO.

```
L_DPO = -log σ( β * [log π(y_w)/π_ref(y_w) - log π(y_l)/π_ref(y_l)] )
```

- **High β (e.g. 0.5):** Strong penalty for deviating from reference. The policy stays close to the SFT model. Conservative, safe, but limited improvement.
- **Low β (e.g. 0.05):** Weak penalty. The policy can drift further from reference to maximize preference. Can lead to reward hacking or degenerate outputs.
- **Typical range:** 0.05 to 0.5. Start at 0.1.

---

### How to Evaluate DPO Success

After DPO training, the model should:
- Generate more helpful, less harmful responses on your eval set
- Score higher on reward model evaluation (even though you didn't train an RM)
- Show improved win rate in head-to-head comparisons with the SFT base

Tools:
- `trl.RewardTrainer` to compute reward model scores before/after
- LLM-as-judge evaluation (use GPT-4 to compare base vs DPO model)
- Win rate on benchmarks like MT-Bench, AlpacaEval

---

### Variants of DPO (Know These Names for Interviews)

| Variant | Key Difference |
|---|---|
| **IPO** (Identity PO) | Regularizes the loss differently to prevent over-optimization |
| **KTO** (Kahneman-Tversky Optimization) | Doesn't need paired preferences — works with single-response binary labels (good/bad) |
| **ORPO** (Odds Ratio PO) | Combines SFT and DPO into one stage — no separate reference model needed |
| **SimPO** | Removes the reference model entirely, uses response length normalization |

---

## Part 4 — What to Say in an Interview

**Q: What is the alignment problem?**

Pre-training optimizes next-token prediction, which means the model learns to imitate the internet — including harmful, false, and unhelpful content. The alignment problem is making the model behave helpfully, harmlessly, and honestly — properties that can't be captured in a simple loss function. RLHF and DPO are practical solutions to this problem.

**Q: Explain RLHF.**

RLHF has three phases. First, SFT fine-tunes the base model on curated instruction-response pairs. Second, a reward model is trained on human pairwise comparisons to score response quality. Third, PPO uses the reward model as a signal to RL-train the policy, with a KL penalty to prevent it drifting too far from the SFT model. The result is a model that's been directly optimized for human preference, not just token prediction.

**Q: What is the reward model doing exactly?**

It's a version of the SFT model with the LM head replaced by a single scalar output. It's trained on (prompt, chosen, rejected) triplets to assign higher scores to chosen responses. During PPO, it scores every response the policy generates, providing the reward signal.

**Q: What is PPO doing in RLHF?**

PPO is the RL algorithm updating the LLM's weights. It clips the probability ratio between the new and old policy to prevent too-large updates, which would destabilize training. The advantage function tells each token whether it contributed to a better-than-expected or worse-than-expected response.

**Q: Why does RLHF use a KL penalty?**

Without it, the policy rapidly games the reward model — generating text that exploits weaknesses in the RM to get high scores while being useless to humans. The KL penalty forces the policy to stay close to the SFT reference, ensuring it doesn't drift into degenerate territory.

**Q: What is DPO and why is it better than RLHF?**

DPO shows that the RLHF objective has a closed-form optimal solution expressible directly in terms of preference data. This means you can train the LLM directly on (prompt, chosen, rejected) pairs without a separate reward model or any RL. The model implicitly becomes its own reward model. DPO is simpler (two phases instead of three), more stable, cheaper, and achieves competitive or better performance than RLHF.

**Q: What does the β parameter in DPO control?**

It controls how closely the trained policy must stay to the reference (SFT) model. High β means conservative training — the policy won't deviate much from SFT. Low β allows more aggressive preference optimization but risks the policy drifting into degenerate outputs. Typical starting value is 0.1.

**Q: What dataset format does DPO need?**

Triplets of (prompt, chosen_response, rejected_response), where chosen is the human-preferred response and rejected is the less-preferred one. The TRL library's DPOTrainer accepts this format directly.

---

## Summary Diagram

```
BASE MODEL (pretrained)
       │
       ▼
  SFT Phase ──────────────────────────────────────────────┐
  (fine-tune on curated instruction data)                 │
       │                                                  │
       ▼                                                  ▼
RLHF PATH:                                        DPO PATH:
  RM Training                                   Train directly on
  (human pairwise comparisons)                  preference pairs
       │                                               │
       ▼                                               ▼
  PPO Fine-tuning                              ALIGNED MODEL ✓
  (RL loop with KL penalty)
       │
       ▼
  ALIGNED MODEL ✓


RLHF: 3 phases, 3 models, RL required, complex, unstable
DPO:  2 phases, 2 models, no RL, simple, stable
```

---

*This document covers everything a GenAI engineer needs to know about RLHF and DPO for both implementation and interviews.*
