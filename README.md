# 🏦 Classifying Banking Intent from Customer Queries

An end-to-end NLP project that compares a **TF-IDF + MLP baseline** against a **LoRA fine-tuned RoBERTa** transformer for multi-class intent classification on the [Banking77](https://huggingface.co/datasets/PolyAI/banking77) dataset — complete with responsible AI practices including automated **PII redaction and hashing**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [PII Protection](#pii-protection)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Next Steps](#next-steps)

---

## Overview

Customer-facing banking chatbots need to accurately classify what a user wants — routing them to the right service quickly and reliably. This project benchmarks two fundamentally different approaches to that problem:

| Approach | Architecture | Representation |
|---|---|---|
| Baseline | Multi-Layer Perceptron (MLP) | TF-IDF vectors |
| Advanced | Fine-tuned RoBERTa + LoRA | Contextual embeddings |

The project also implements a production-ready **PII protection layer**, a critical requirement when handling real banking customer data.

---

## Dataset

**Banking77** — a benchmark dataset for banking-domain intent detection.

| Property | Value |
|---|---|
| Source | [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) |
| Unique intents | 77 |
| Training queries | 10,003 |
| Testing queries | 3,080 |
| Format | Short natural language customer queries |

**Example:**
```python
{
  'text': 'I am still waiting on my card?',
  'label': 11  # → "card_arrival"
}
```

**Key findings from EDA:**
- Most queries are short (10–20 words), consistent with real chatbot interactions.
- Top recurring intents cluster around **fee complaints**, **transaction errors**, and **balance update issues**.
- Strong keyword overlap across intents (words like `charged`, `account`, `money`) means fine-grained contextual understanding is critical.

---

## Models

### 1. Baseline — TF-IDF + MLP

A fast, lightweight non-contextual classifier to establish a performance floor.

**Pipeline:**
- Text vectorized with `TfidfVectorizer` (unigrams + bigrams, max 50,000 features)
- Labels encoded with `LabelEncoder`
- 3-layer MLP with ReLU activations (hidden size: 512)
- Trained with `CrossEntropyLoss` + `AdamW` (lr=1e-3, weight_decay=1e-4)
- 5 epochs

### 2. Fine-tuned RoBERTa + LoRA

A parameter-efficient transformer fine-tuning approach for superior performance.

**Pipeline:**
- Tokenized with `roberta-base` tokenizer (max length: 256 tokens, dynamic padding)
- LoRA adapters applied to `query`, `key`, and `value` projections in self-attention
- Trained with cosine LR schedule + warmup (10 epochs, lr=2e-4)
- Merged and saved post-training with `merge_and_unload()`

**LoRA Configuration:**

| Hyperparameter | Value |
|---|---|
| Rank (`r`) | 32 |
| Alpha (`lora_alpha`) | 64 |
| Dropout | 0.05 |
| Bias | none |
| Target modules | query, key, value |

---

## Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| MLP (baseline) | 0.8799 | 0.8797 | 0.8797 |
| **LoRA-RoBERTa** | **0.9347** | **0.9347** | **0.9347** |

**LoRA-RoBERTa achieves a +5.48% gain in Macro F1** over the MLP — a meaningful improvement across all 77 intent classes equally.

### Intent-Level Highlights

**Biggest improvements with LoRA-RoBERTa:**

| Intent | MLP F1 | RoBERTa F1 | Gain |
|---|---|---|---|
| `contactless_not_working` | 0.576 | 0.961 | +38.5% |
| `card_not_working` | 0.667 | 0.916 | +24.9% |
| `verify_my_identity` | 0.712 | 0.940 | +22.8% |
| `why_verify_identity` | 0.800 | 0.963 | +16.3% |

> The MLP struggled most with intents requiring contextual disambiguation (e.g., distinguishing `contactless_not_working` from `card_not_working`), where RoBERTa's contextual embeddings excel.

**Observed regressions:** 8 intents showed marginal losses (<3%), all with strong performance (F1 ≥ 0.85) in both models — likely attributable to random variation rather than systematic failure.

### Safe Prediction Function

```python
cleaned_text, predicted_class = predict_text_safe(
    text="My card 4532-1234-5678-9010 was charged twice. Email me at john@email.com",
    model=model,
    tokenizer=tokenizer,
    label_encoder=label_encoder,
    device=device,
    top_k=3,
    hash_identifiers=True
)
```

**Output:**
```
Original input: My card 4532-1234-5678-9010 was charged twice. Email me at john@email.com
Redacted input: My card [CARD NUMBER REDACTED] was charged twice. Email me at [HASH:55f537baf75630a8]

Predicted Customer Intent:
1. transaction_charged_twice: 0.7821
2. extra_charge_on_statement: 0.1453
3. direct_debit_payment_not_recognised: 0.0412
```

---

## Project Structure

```
Classifying-Banking-Intent/
│
├── datasets/
│   ├── banking77_train.csv
│   └── banking77_test.csv
│
├── finetuned_roberta_lora_model/     # Saved model weights + tokenizer
│
├── banking_classification_BERT.ipynb # Main notebook
└── README.md
```
## Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-blue?logo=scikitlearn)
