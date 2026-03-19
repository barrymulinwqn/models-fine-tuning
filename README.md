# models-fine-tuning
open source models Fine tuning for Finance usage



|  | Mixtral 8×7B | Phi-4 | Llama 3.1 8B | DeepSeek-R1-32B | Qwen2.5-72B |
|---|---|---|---|---| --- |
| **Context** | 32K | 16K | | | |
| **VRAM (QLoRA)** | ~40–48 GB (A100 needed) | ~14–18 GB (single RTX 3090) | | | |
| **Best for** | Long documents, multilingual, high throughput | Reasoning, structured Q&A, tight budgets | | | |
| **Recommended company size** | Mid–Large | Small–Mid | | | |
| **License** |  Apache 2.0 | MIT | | | |
| **Org** | Mistral AI | Microsoft | | | |
| **Cost** |  | Mid-Sized Finance Company (50–500 employees, $10K–$50K budget) | Small Finance Company (< 50 employees, < $5K budget) | | |
| **Finance Fine-Tune Ease** | Medium | Easy | Easy | Medium | Medium |
| **Long_Doc** | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | ★★★★☆ |
| **reasoning** | ★★★☆☆ | ★★★★☆ | ★★★☆☆ | ★★★★★ | ★★★★★ |
|  |  |  |  |  |  |


# Mixtral 8×7B

> **Model:** Mistral AI — Mixtral 8×7B Instruct v0.1 / v0.3

> **License:** Apache 2.0 (fully open-source, commercial use allowed)

> **Architecture:** Sparse Mixture-of-Experts (MoE), 8 experts × 7B — 46.7B total params, ~12.9B active per token

> **Context Window:** 32K tokens

> **Target Audience:** Small to Mid-sized Finance Companies

# Phi-4

> **Model:** Microsoft Phi-4 (14B parameters)

> **License:** MIT (fully open-source, commercial use allowed)

> **Architecture:** Dense Transformer, 14B params, 16K context window

> **Target Audience:** Small to Mid-sized Finance Companies


# Training Dataset Setup

Public datasets (FinanceBench, FinQA, SEC EDGAR, FLARE), private company data guidance, PII scrubbing pipeline (using Presidio), automated Q&A generation using GPT-4o as an annotator, and target dataset size/composition tables.


# Benchmark Testing

FinanceBench, FinQA, FLARE suite coverage, an internal benchmark runner in code, human evaluation rubric (factual accuracy, regulatory compliance, hallucination rate), and A/B rollout strategy.


### Licensing Risk Matrix

| License | Commercial Use | Revenue Restriction | Recommended For |
|---|---|---|---|
| Apache 2.0 | ✅ Unrestricted | ❌ None | All commercial use |
| MIT | ✅ Unrestricted | ❌ None | All commercial use |
| Meta Community | ✅ Up to 700M MAU | ⚠️ Above threshold requires agreement | Most fintechs |
| Qwen License | ✅ Most uses | ⚠️ Review for large-scale deployment | Check 72B specifically |
| Gemma ToU | ✅ Permitted | ⚠️ Prohibited competing with Google AI services | Non-Google competitors |
| DeepSeek License (V3) | ⚠️ Review required | ⚠️ Restrictions on competing services | Internal/research use |


