# Open Source LLMs for Finance Fine-Tuning: Analysis & Comparison

> Last updated: March 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Finance Use Cases](#finance-use-cases)
3. [Model Catalog](#model-catalog)
4. [Comparison Table](#comparison-table)
5. [Fine-Tuning Techniques](#fine-tuning-techniques)
6. [Recommended Stacks by Use Case](#recommended-stacks-by-use-case)
7. [Key Considerations](#key-considerations)

---

## Overview

Fine-tuning open source LLMs for finance is one of the highest-value applications in the industry. Finance demands models that understand domain-specific terminology (EBITDA, Basel III, mark-to-market, SOFR, etc.), can reason over numerical data, comply with regulatory constraints, and produce reliable, factually grounded outputs.

This document catalogues the most relevant open source base models (as of early 2026), evaluates their suitability for finance fine-tuning, and provides a structured comparison to guide model selection.

---

## Finance Use Cases

| Category | Examples |
|---|---|
| **Document Intelligence** | 10-K/10-Q parsing, earnings call summarization, prospectus analysis |
| **Sentiment & Signal** | News/social sentiment for trading, analyst report scoring |
| **Risk & Compliance** | AML/KYC screening, regulatory document Q&A, GDPR/Basel III compliance |
| **Customer Interaction** | Wealth management chatbots, robo-advisory, loan eligibility Q&A |
| **Quantitative Reasoning** | Financial modelling assistance, ratio analysis, forecasting narratives |
| **Fraud Detection** | Transaction narrative analysis, anomaly explanation |
| **Code Generation** | QuantLib scripting, Bloomberg/Refinitiv API automation |

---

## Model Catalog

### 1. LLaMA 3.1 / 3.3 (Meta)

**Sizes:** 8B, 70B, 405B  
**License:** Meta Community License (commercial use permitted up to 700M MAU)

LLaMA 3.x is the current gold standard for finance fine-tuning. The 70B variant hits a sweet spot between capability and deployment cost. The 405B is competitive with GPT-4 on many financial reasoning benchmarks.

**Pros:**
- Excellent multilingual support (LLaMA 3.1+), useful for global finance
- Strong instruction-following after RLHF
- Huge ecosystem: LoRA adapters, quantized versions (GGUF/GPTQ), Axolotl, Unsloth
- Long context window (128K tokens in 3.1) handles full 10-K filings
- Function-calling support for tool-augmented finance agents

**Cons:**
- Large models (70B+) require significant GPU infrastructure
- License restricts certain high-volume commercial deployments
- Numerical reasoning still lags behind specialized math models

---

### 2. Mistral 7B / Mixtral 8×7B / Mistral Large (Mistral AI)

**Sizes:** 7B, 8×7B (MoE ~47B active params), 8×22B  
**License:** Apache 2.0

Mistral 7B punches well above its weight class. Mixtral's Mixture-of-Experts architecture is particularly attractive for finance: lower inference latency with broad capability.

**Pros:**
- Apache 2.0 — fully permissive commercial use
- Mixtral 8×7B achieves GPT-3.5 quality at fraction of the compute
- Sliding window attention handles long documents efficiently
- Great for RAG pipelines over document corpora (EDGAR, Bloomberg)
- Fast inference with vLLM + SGLang

**Cons:**
- 7B base can hallucinate numerical facts under pressure
- MoE models require high memory bandwidth (multiple experts loaded)
- Smaller community fine-tune dataset ecosystem vs. LLaMA

---

### 3. Qwen2.5 / Qwen2.5-Finance (Alibaba / Tongyi)

**Sizes:** 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B  
**License:** Apache 2.0 (most sizes), Qwen License (72B)

Qwen2.5 is arguably the strongest open model for Asian financial markets (CNY instruments, HKEX, SSE). Alibaba has also released **Qwen2.5-Finance**, a domain-adapted variant.

**Pros:**
- Native Chinese + English bilingual capability — critical for cross-border finance
- Qwen2.5-72B competes with GPT-4o on many benchmarks
- Math-specialized variants (Qwen2.5-Math) improve quantitative reasoning
- Strong tool-use and function-calling built in
- Domain adapted finance checkpoint available out-of-the-box

**Cons:**
- 72B license is more restrictive than Apache
- Training data provenance less transparent for compliance-heavy firms
- Smaller Western finance fine-tune community

---

### 4. DeepSeek-V3 / DeepSeek-R1 (DeepSeek AI)

**Sizes:** 7B, 67B, 671B (MoE)  
**License:** MIT (R1), DeepSeek License (V3)

DeepSeek-R1 introduced chain-of-thought reasoning trained via RL, making it exceptional for financial analysis that requires step-by-step numerical reasoning — DCF models, options pricing narratives, etc.

**Pros:**
- State-of-the-art reasoning — outperforms GPT-o1 on math benchmarks
- MIT license on R1 distilled models (1.5B–70B) is highly permissive
- Excellent for structured financial reasoning (valuation, derivatives)
- R1 distillation allows running strong reasoning in small footprints (7B)

**Cons:**
- Full 671B V3 model is impractical to self-host for most firms
- Trained primarily on data with Chinese financial market bias
- R1's "thinking tokens" add latency — unsuitable for real-time trading signals
- DeepSeek License (V3) restricts certain competitive uses

---

### 5. Phi-4 (Microsoft)

**Sizes:** 14B  
**License:** MIT

Microsoft's Phi series prioritizes quality-per-parameter. Phi-4 (14B) achieves near-70B quality on reasoning tasks, making it excellent for resource-constrained deployments (edge banking apps, on-device compliance checks).

**Pros:**
- MIT license — fully open commercial use
- Extremely strong reasoning relative to model size
- Low hardware requirements; runs on a single A100 80GB
- Great for compliance document Q&A with RAG
- Fast fine-tuning iteration cycles due to small size

**Cons:**
- 14B ceiling means it lags on very complex multi-step financial reasoning
- Smaller context window vs. LLaMA 3.1
- Limited multilingual depth (primarily English)
- Smaller fine-tune community

---

### 6. Gemma 2 (Google DeepMind)

**Sizes:** 2B, 9B, 27B  
**License:** Gemma Terms of Use (permissive, commercial allowed)

Gemma 2 benefits from Google's training infrastructure and safety tuning. The 27B model is competitive for enterprise deployment and integrates natively with Google Cloud Vertex AI fine-tuning pipelines.

**Pros:**
- Excellent instruction-following and safety alignment
- Native integration with Vertex AI, TFX, and Google's TPU ecosystem
- Strong performance on document summarization tasks
- 27B is manageable on 2× A100s
- Good multilingual coverage

**Cons:**
- Not Apache 2.0 — custom Gemma license requires compliance review
- Lags behind LLaMA 3.1 70B and DeepSeek on hard reasoning
- Smaller open fine-tune dataset ecosystem
- Tighter content filtering can interfere with certain financial risk discussions

---

### 7. FinBERT (ProsusAI / Yi Yang Lab)

**Sizes:** 110M (BERT-base)  
**License:** Apache 2.0

The gold standard for financial NLP classification tasks. Pre-trained on financial communications corpora (Reuters, Bloomberg, 10-K filings). Not a generative model — purely encoder-based for classification and NER.

**Pros:**
- Purpose-built for finance: superior sentiment accuracy on financial text
- Extremely lightweight — runs on CPU in production
- Battle-tested in production hedge funds and banks since 2019
- Minimal fine-tuning data needed for news sentiment classification
- Fast inference, ideal for real-time market signal pipelines

**Cons:**
- Encoder-only: cannot generate text, summarize, or answer questions
- Limited to classification/NER — not suitable for generative tasks
- Based on BERT (2018 architecture): outdated for complex reasoning

---

### 8. FinGPT / FinLLaMA (AI4Finance Foundation)

**Sizes:** Based on LLaMA 2 7B/13B, Falcon 7B, ChatGLM2  
**License:** Apache 2.0 (framework), base model licenses apply

FinGPT is an open source framework + fine-tuned checkpoints specifically targeting financial applications. It provides a data pipeline (SEC filings, Bloomberg, Yahoo Finance, Reddit WSB) and LoRA-adapted checkpoints.

**Pros:**
- Finance-specific LoRA adapters ready to deploy
- Curated pipeline for continuous financial data ingestion
- Lower barrier to entry: pre-built fine-tuned checkpoints
- Strong community with active finance ML researchers
- Covers diverse tasks: sentiment, summarization, forecasting narratives

**Cons:**
- Built on older base models (LLaMA 2, Falcon) — architecturally behind
- Training data is primarily US market focused
- Community maintained — production reliability variable
- Limited adherence to compliance standards (PII handling, auditability)

---

### 9. Falcon 2 (TII — Technology Innovation Institute)

**Sizes:** 11B, 40B, 180B  
**License:** Apache 2.0 (11B), custom (180B)

UAE-based TII's Falcon models were early open source leaders. Falcon 2 introduces vision capabilities (Falcon 2 VLM) useful for chart/graph understanding in financial reports.

**Pros:**
- Apache 2.0 on Falcon 2 11B
- Multimodal variant can parse financial charts and figures
- Trained on a diverse multilingual corpus
- Solid baseline for Middle Eastern / Gulf financial markets (Arabic support)

**Cons:**
- Has been surpassed by LLaMA 3, Mistral, and Qwen in most benchmarks
- Smaller ecosystem and fewer community fine-tune resources
- The multimodal version is newer and less battle-tested

---

### 10. OpenLLaMA / Pythia / OLMo (EleutherAI / Allen AI)

**Sizes:** 1B – 12B  
**License:** Apache 2.0

These fully open models (including training data and code) are important for **regulated institutions** that require full auditability of the model stack.

**Pros:**
- Fully transparent: training data, code, checkpoints all open
- Ideal for compliance-heavy environments (banking regulators, auditors)
- Apache 2.0 — zero license risk
- OLMo 2 (Allen AI, 2025) shows competitive benchmark performance

**Cons:**
- Significantly behind frontier models in raw capability
- Smaller context windows
- Less fine-tune tooling and community support
- Not suitable for complex reasoning or generation tasks without heavy fine-tuning

---

## Comparison Table

| Model | Org | Params | License | Context Window | Reasoning | Finance NLP | Multilingual | Fine-tune Ease | Deployment Cost | Best Finance Use Case |
|---|---|---|---|---|---|---|---|---|---|---|
| **LLaMA 3.1 70B** | Meta | 70B | Meta Community | 128K | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★★ | High | Full-stack financial AI agent |
| **LLaMA 3.3 70B** | Meta | 70B | Meta Community | 128K | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★★★ | High | RAG over SEC filings, earnings Q&A |
| **Mistral 7B** | Mistral AI | 7B | Apache 2.0 | 32K | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ | ★★★★★ | Low | Sentiment analysis, document triage |
| **Mixtral 8×7B** | Mistral AI | ~47B active | Apache 2.0 | 32K | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★☆ | Medium | Compliance Q&A, report summarization |
| **Qwen2.5 72B** | Alibaba | 72B | Qwen License | 128K | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★☆ | High | Asia-Pacific markets, cross-border finance |
| **Qwen2.5-Finance** | Alibaba | 7B–72B | Apache 2.0 | 128K | ★★★★☆ | ★★★★★ | ★★★★☆ | ★★★★★ | Medium | Finance Q&A, risk summarization |
| **DeepSeek-R1 (distilled 70B)** | DeepSeek | 70B | MIT | 64K | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★★★☆ | High | DCF analysis, options reasoning, quant narratives |
| **DeepSeek-R1 (distilled 7B)** | DeepSeek | 7B | MIT | 64K | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | ★★★★★ | Low | Financial math reasoning, lightweight deployment |
| **Phi-4** | Microsoft | 14B | MIT | 16K | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★★★★ | Low | Compliance checks, on-device banking apps |
| **Gemma 2 27B** | Google | 27B | Gemma ToU | 8K | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★☆ | Medium | Google Cloud-native finance pipelines |
| **FinBERT** | ProsusAI | 110M | Apache 2.0 | 512 tokens | N/A (encoder) | ★★★★★ | ★★☆☆☆ | ★★★★★ | Very Low | Real-time news sentiment, classification |
| **FinGPT** | AI4Finance | 7B–13B | Apache 2.0 | 4K–8K | ★★★☆☆ | ★★★★☆ | ★★☆☆☆ | ★★★★★ | Low | Pre-built finance checkpoints, rapid prototyping |
| **Falcon 2 11B** | TII | 11B | Apache 2.0 | 8K | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | ★★★★☆ | Low | Chart understanding (VLM), Gulf markets |
| **OLMo 2 / Pythia** | Allen AI / EleutherAI | 1B–12B | Apache 2.0 | 4K–8K | ★★☆☆☆ | ★★☆☆☆ | ★★☆☆☆ | ★★★★★ | Very Low | Regulated environments requiring full auditability |

**Rating scale:** ★★★★★ = Excellent | ★★★★☆ = Good | ★★★☆☆ = Adequate | ★★☆☆☆ = Limited | ★☆☆☆☆ = Poor

---

## Fine-Tuning Techniques

For finance, the following techniques are commonly combined:

| Technique | Description | Finance Application |
|---|---|---|
| **LoRA / QLoRA** | Low-rank adapter injection; QLoRA adds 4-bit quantization | Most common; fine-tune 70B on 2×A100s with QLoRA |
| **Full Fine-Tuning** | Update all weights; highest quality, highest cost | Used by banks building proprietary models |
| **RLHF / DPO** | Teach the model to prefer compliant, grounded answers | Reduce hallucination of financial facts/figures |
| **RAG (Retrieval-Augmented)** | External vector DB (EDGAR, Bloomberg, internal docs) | Combines fine-tuned model with live document retrieval |
| **Continued Pre-Training** | Further pre-train on financial corpora before instruction tuning | Best quality for deep domain adaptation |
| **Prompt / Prefix Tuning** | Train soft prompt embeddings only | Lightweight; useful for multiple finance sub-tasks on one base |

### Recommended Data Sources for Finance Fine-Tuning

- **SEC EDGAR** — 10-K, 10-Q, 8-K filings (public, structured)
- **FinPile / RedPajama Finance subset** — curated finance pre-training corpus
- **Financial PhraseBank** — sentiment annotation dataset
- **FLARE Benchmark** — comprehensive finance NLP evaluation suite
- **Bloomberg / Reuters news** — requires licensing for commercial use
- **Earnings call transcripts** — Seeking Alpha, Motley Fool (check ToS)
- **Synthetic data** — GPT-4 generated Q&A pairs over public filings

---

## Recommended Stacks by Use Case

### Real-Time Market Sentiment (High-Throughput)
```
FinBERT (classifier) → Mistral 7B (explanation generator) → Redis cache
```

### Earnings Report Q&A / Investor Relations Bot
```
LLaMA 3.3 70B (QLoRA fine-tuned) + pgvector RAG over EDGAR filings
```

### Regulatory Compliance & Document Review
```
Phi-4 (14B) or Mixtral 8×7B + LangChain + compliance-labeled fine-tune dataset
```

### Quantitative Reasoning / Valuation Models
```
DeepSeek-R1 distilled 70B (strong chain-of-thought) + structured output enforcement
```

### Asia-Pacific / Cross-Border Finance
```
Qwen2.5-Finance 72B (bilingual CN/EN) + cross-border regulatory adapter
```

### Regulated / Auditable Environment (Central Banks, Regulators)
```
OLMo 2 (fully open stack) + fully documented fine-tune pipeline
```

---

## Key Considerations

### Licensing Risk Matrix

| License | Commercial Use | Revenue Restriction | Recommended For |
|---|---|---|---|
| Apache 2.0 | ✅ Unrestricted | ❌ None | All commercial use |
| MIT | ✅ Unrestricted | ❌ None | All commercial use |
| Meta Community | ✅ Up to 700M MAU | ⚠️ Above threshold requires agreement | Most fintechs |
| Qwen License | ✅ Most uses | ⚠️ Review for large-scale deployment | Check 72B specifically |
| Gemma ToU | ✅ Permitted | ⚠️ Prohibited competing with Google AI services | Non-Google competitors |
| DeepSeek License (V3) | ⚠️ Review required | ⚠️ Restrictions on competing services | Internal/research use |

### Hallucination & Financial Fact Accuracy

Financial models must not confabulate numbers. Recommended mitigations:
1. **RAG-first architecture**: always ground outputs in retrieved source documents
2. **DPO fine-tuning** with negative examples that include hallucinated figures
3. **Structured output enforcement** (JSON mode, Outlines, Guidance) for numerical fields
4. **Citation requirements** in system prompt: model must cite source for every claim
5. **Post-generation validation**: regex/rule-based checks on tickers, dates, ISIN codes

### Regulatory & Data Privacy

- Avoid training on **non-anonymized customer data** (GDPR, CCPA, GLBA)
- Document model lineage for **SR 11-7** (Fed/OCC model risk management) compliance
- Implement **output filtering** for MNPI (Material Non-Public Information) risks
- Consider **on-premises deployment** for proprietary trading strategies

### Hardware Requirements Summary

| Model Size | Min GPU (inference) | Min GPU (QLoRA fine-tune) | Approximate Cost/hr (cloud) |
|---|---|---|---|
| 7B (fp16) | 1× A10G 24GB | 1× A100 40GB | ~$1–2/hr |
| 13B (fp16) | 1× A100 40GB | 1× A100 80GB | ~$2–3/hr |
| 14B (fp16) | 1× A100 80GB | 1× A100 80GB | ~$3/hr |
| 70B (fp16) | 2× A100 80GB | 4× A100 80GB | ~$8–16/hr |
| 70B (4-bit GGUF) | 1× A100 80GB | 2× A100 80GB | ~$4–8/hr |
| 405B / 671B | 8×+ H100 | Not practical w/ QLoRA | ~$50+/hr |

---

## Summary Recommendation

| Priority | Model Choice | Rationale |
|---|---|---|
| **Best overall** | LLaMA 3.3 70B (QLoRA) | Largest ecosystem, strong performance, manageable license |
| **Best permissive license** | Mistral / Phi-4 / OLMo 2 | Apache 2.0 or MIT, zero license risk |
| **Best reasoning** | DeepSeek-R1 70B distilled | MIT license, top-tier chain-of-thought for quant tasks |
| **Best for Asia/multilingual** | Qwen2.5-Finance 72B | Native bilingual, finance-adapted checkpoint available |
| **Best lightweight** | Phi-4 14B | MIT, strong reasoning at low hardware cost |
| **Best classification only** | FinBERT | Purpose-built, battle-tested, CPU-deployable |
| **Best for regulated envs** | OLMo 2 / Pythia | Fully auditable open stack |
| **Best rapid prototyping** | FinGPT | Pre-built finance adapters, low setup time |

---

*This document reflects the state of open source LLMs as of March 2026. The space evolves rapidly — re-evaluate model choices every 6 months.*
