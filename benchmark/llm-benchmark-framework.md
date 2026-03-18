# Finance LLM Benchmark Framework

> Version: 1.0.0 | Last Updated: March 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Benchmark Toolchain Setup](#2-benchmark-toolchain-setup)
3. [Model Candidates](#3-model-candidates)
4. [Test Dataset Design](#4-test-dataset-design)
5. [Evaluation Dimensions & Metrics](#5-evaluation-dimensions--metrics)
6. [Scoring Rules & Scorecard](#6-scoring-rules--scorecard)
7. [Benchmark Execution Guide](#7-benchmark-execution-guide)
8. [Results Reporting Template](#8-results-reporting-template)
9. [Appendix](#9-appendix)

---

## 1. Executive Summary

### Purpose

This framework provides a **standardized, reproducible methodology** for evaluating and ranking open-source LLMs for financial domain applications — both as base model selection criteria and as fine-tuning quality assessment tools. It addresses four core questions:

| Question | Answer provided by this framework |
|---|---|
| **What to measure** | 8 evaluation dimensions covering the full finance LLM capability spectrum |
| **How to measure** | Standardized metrics with clear computation methods, aligned to academic benchmarks |
| **What data to use** | Curated test datasets spanning financial NLP, numerical reasoning, and domain knowledge |
| **How to score** | Weighted scorecard calibrated per use-case profile with A+–D grading |

### Models Evaluated (v1.0 Cohort)

| Model | Version | Active Params | License | Primary Strength |
|---|---|---|---|---|
| LLaMA | 3.3 70B-Instruct | 70B | Meta Community | General + long-context |
| Mixtral | 8×7B-Instruct-v0.1 | ~13B (MoE) | Apache 2.0 | Speed + RAG pipelines |
| Qwen | 2.5-72B-Instruct | 72B | Qwen License | Asian markets + math |
| DeepSeek | R1-Distill-Llama-70B | 70B | MIT | Multi-step reasoning |
| Phi | 4 (14B) | 14B | MIT | Quality-per-param |
| Gemma | 2-27B-IT | 27B | Gemma ToU | Safety + Google Cloud |

### Key Outputs per Run

- Per-dimension normalized scores (0–100) for every model/checkpoint
- Weighted composite score per use-case profile with letter grade
- Latency & cost efficiency analysis
- Go / Conditional-Go / No-Go recommendation with rationale

---

## 2. Benchmark Toolchain Setup

### 2.1 Infrastructure Requirements

| Profile | GPU Config | Host RAM | Applicable Models |
|---|---|---|---|
| **Minimal** | 1× A100 80GB | 256 GB | ≤ 14B params |
| **Standard** | 2× A100 80GB | 512 GB | 14B – 70B params |
| **Full** | 4× H100 80GB | 1 TB | All 70B+ models with TP=4 |
| **Cloud (on-demand)** | AWS p4d.24xlarge or GCP a3-highgpu-8g | — | Full cohort |

> **Rule:** All models in the same benchmark cohort must be evaluated on identical hardware to make latency comparisons valid.

---

### 2.2 Software Stack

```
inference/
├── vllm >= 0.4.0              # Primary inference backend (tensor-parallel)
├── sglang >= 0.2              # Alternative — faster structured output generation
└── llama.cpp (GGUF)           # Fallback for CPU / edge deployment testing

evaluation/
├── lm-evaluation-harness      # EleutherAI — core standardized harness
├── FinEval                    # Finance-specific academic benchmark tasks
├── ragas                      # RAG faithfulness / relevancy evaluation
└── deepeval                   # LLM-as-judge and custom metric framework

metrics/
├── huggingface/evaluate       # ROUGE, BLEU, BERTScore, Accuracy
├── sacrebleu                  # Reproducible BLEU reference implementation
└── rouge-score / bert-score   # Local metric computation

observability/
├── langfuse                   # Per-request latency tracing, token cost tracking
└── mlflow                     # Experiment tracking, parameter logging, model registry

reporting/
├── pandas                     # Data manipulation, aggregation
├── plotly                     # Radar charts, heatmaps, scatter plots
└── wandb (optional)           # Live dashboard
```

---

### 2.3 Environment Setup

```bash
# Isolated reproducible environment
conda create -n llm-benchmark python=3.11 -y
conda activate llm-benchmark

# Core inference + evaluation
pip install vllm>=0.4.0 lm-eval>=0.4.3 ragas deepeval langfuse mlflow

# Metrics libraries
pip install evaluate sacrebleu bert-score rouge-score nltk

# Reporting
pip install pandas plotly wandb

# Reproducibility: pin all versions
pip freeze > requirements-benchmark.lock
```

---

### 2.4 Model Loading Configuration

```yaml
# benchmark_config.yaml

generation:
  temperature: 0.0          # Greedy decoding — mandatory for reproducibility
  max_new_tokens: 512
  repetition_penalty: 1.0
  seed: 42

models:
  - name: llama-3.3-70b
    path: meta-llama/Llama-3.3-70B-Instruct
    backend: vllm
    dtype: bfloat16
    tensor_parallel: 2
    max_seq_len: 32768

  - name: mixtral-8x7b
    path: mistralai/Mixtral-8x7B-Instruct-v0.1
    backend: vllm
    dtype: bfloat16
    tensor_parallel: 2
    max_seq_len: 32768

  - name: qwen2.5-72b
    path: Qwen/Qwen2.5-72B-Instruct
    backend: vllm
    dtype: bfloat16
    tensor_parallel: 4
    max_seq_len: 32768

  - name: deepseek-r1-70b
    path: deepseek-ai/DeepSeek-R1-Distill-Llama-70B
    backend: vllm
    dtype: bfloat16
    tensor_parallel: 2
    max_seq_len: 32768

  - name: phi-4-14b
    path: microsoft/phi-4
    backend: vllm
    dtype: bfloat16
    tensor_parallel: 1
    max_seq_len: 16384

  - name: gemma-2-27b
    path: google/gemma-2-27b-it
    backend: vllm
    dtype: bfloat16
    tensor_parallel: 2
    max_seq_len: 8192
```

---

## 3. Model Candidates

See [../finance-llm-finetuning-analysis.md](../finance-llm-finetuning-analysis.md) for detailed model profiles. Summary for benchmark reference:

| Model | Architecture | Context Window | Key Differentiator |
|---|---|---|---|
| LLaMA 3.3 70B | Dense transformer | 128K | Best ecosystem; long-context finance |
| Mixtral 8×7B | Sparse MoE | 32K | Low-latency RAG; Apache 2.0 |
| Qwen2.5 72B | Dense transformer | 128K | Chinese markets; built-in finance variant |
| DeepSeek-R1 70B | RL-trained CoT | 64K | Step-by-step numerical reasoning |
| Phi-4 14B | Dense transformer | 16K | Highest quality-per-param; MIT license |
| Gemma 2 27B | Dense transformer | 8K | Strong safety alignment; Vertex AI native |

---

## 4. Test Dataset Design

### 4.1 Dataset Taxonomy

Finance LLM evaluation requires coverage across six dataset categories, each targeting distinct capabilities:

#### Category A — Financial Q&A and Knowledge

| Dataset | Source | Size | Task | License |
|---|---|---|---|---|
| **FinQA** | ACL 2021, S&P 500 annual reports | 8,281 QA pairs | Numerical QA over financial reports (tables + text) | MIT |
| **ConvFinQA** | ACL 2022 | 3,892 conversations | Multi-turn conversational QA over financial tables | MIT |
| **FinBench** | Papers With Code | ~10K questions | Finance knowledge MCQ (CPA, CFA L1–L3, FRM) | Research |
| **TAT-QA** | ACL 2021 | 16,552 QA pairs | Hybrid table + text QA (semi-structured docs) | MIT |
| **DocFinQA** | 2024 | 7,437 QA pairs | Long-document financial QA (full 10-K filings) | Research |

#### Category B — Sentiment & Signal Classification

| Dataset | Source | Size | Task | License |
|---|---|---|---|---|
| **Financial PhraseBank (FPB)** | Malo et al. 2014 | 4,846 sentences | 3-class sentiment: Positive / Neutral / Negative | CC BY-NC-SA |
| **FiQA SA** | FiQA Challenge 2018 | 1,173 items | Sentiment + aspect extraction from microblogs and news | Research |
| **Twitter FinSent** | StockSent 2023 | 25,000 tweets | Directional market sentiment classification | Research |
| **FOMC Statements** | Federal Reserve archive | 216 documents | Hawkish / Dovish / Neutral monetary policy stance | Public Domain |

#### Category C — Document Summarization

| Dataset | Source | Size | Task | License |
|---|---|---|---|---|
| **EDGARCorpus** | SEC EDGAR 10-K sections | 6,900 sections | Extractive + abstractive summarization | Public Domain |
| **EarningsCall** | Refinitiv transcripts | 500 transcripts | Earnings call → executive bullet summary | Proprietary (self-constructed) |
| **ProspectusSumm** | SEC S-1 filings 2019–2024 | 200 documents | IPO risk factor summarization | Public Domain |
| **XBRL Narrative** | XBRL International | 1,200 items | XBRL tag → natural language explanation | Public Domain |

#### Category D — Instruction Following & Structured Output

| Dataset | Source | Size | Task | License |
|---|---|---|---|---|
| **FinInstruct-50K** | Self-curated instruction set | 50,000 prompts | JSON/table extraction from financial text | Internal |
| **IFEval-Finance** | Adapted from IFEval | 500 prompts | Verifiable instruction constraints (format, length, structure) | Apache 2.0 |
| **ToolBench-Finance** | Custom function-calling set | 1,000 prompts | Bloomberg API calls, calculator tool use, SQL generation | Internal |

#### Category E — Compliance, Hallucination & Safety

| Dataset | Source | Size | Task | License |
|---|---|---|---|---|
| **FinHallBench** | Adversarial custom set | 1,500 prompts | Hallucination detection: fabricated tickers, false earnings data | Internal |
| **RegQA** | Basel III, MiFID II, Dodd-Frank texts | 800 QA pairs | Regulatory Q&A accuracy (rule citation, scope, obligations) | Public Domain |
| **FinAdvice-Red** | Red-team adversarial prompts | 500 prompts | Detect inappropriate financial advice; refusal evaluation | Internal |

#### Category F — Code Generation

| Dataset | Source | Size | Task | License |
|---|---|---|---|---|
| **FinCode-Eval** | QuantLib + pandas problems | 400 problems | Python: DCF models, Black-Scholes, bond pricing, duration | Internal |
| **SQLFinBench** | Custom EDGAR schema | 600 queries | SQL over financial schemas (income statement, balance sheet, ratios) | Internal |

---

### 4.2 Dataset Structure & Version Control

```
datasets/
├── category_a_knowledge/
│   ├── finqa/            # Official train/dev/test splits (use test only)
│   ├── convfinqa/
│   ├── finbench/
│   ├── tat_qa/
│   └── docfinqa/
├── category_b_sentiment/
│   ├── fpb/              # Agreed 75/25 test split (sentences_66agree)
│   ├── fiqa_sa/
│   ├── twitter_finsent/
│   └── fomc/
├── category_c_summarization/
│   ├── edgar_corpus/
│   └── prospectus_summ/
├── category_d_instruction/
│   ├── fininstruct_eval/ # 2,000-sample held-out eval subset
│   └── ifeval_finance/
├── category_e_compliance/
│   ├── finhallbench/
│   ├── regqa/
│   └── finadvice_red/
└── category_f_code/
    ├── fincode_eval/
    └── sql_finbench/
```

**Dataset governance rules:**
1. All models evaluated on **identical test splits** — splits frozen before any model evaluation begins (seed = 42)
2. **No training-split contamination** — verify no test items appear in any model's fine-tuning corpus
3. Dataset versions pinned with Git SHA or DVC hash; hash logged in every benchmark run metadata
4. Proprietary / internal datasets anonymized before external sharing or publication
5. Minimum test set size: **400 items per category** to ensure statistical significance (p < 0.05, power = 0.80)

---

### 4.3 Prompt Templates

All tasks use standardized prompt templates to ensure fair cross-model comparison:

```python
PROMPT_TEMPLATES = {
    "finqa": (
        "You are a financial analyst assistant. Answer the question based solely on "
        "the provided context. Show your calculation steps.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    "sentiment": (
        "Classify the sentiment of the following financial text.\n"
        "Respond with exactly one word: Positive, Neutral, or Negative.\n\n"
        "Text: {text}\n\n"
        "Sentiment:"
    ),
    "summarization": (
        "Summarize the following financial document section in 3–5 concise bullet points. "
        "Focus on material facts, key figures, risks, and management guidance.\n\n"
        "Document:\n{document}\n\n"
        "Summary:"
    ),
    "hallucination_probe": (
        "Answer the following question about the company. "
        "If you are not certain, say 'I don't have reliable information on this.'\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    "code_generation": (
        "Write production-quality Python code to solve the following finance problem. "
        "Include type hints, a docstring, and handle edge cases.\n\n"
        "Problem: {problem}\n\n"
        "```python\n"
    ),
    "sql_generation": (
        "Generate a valid SQL query for the following request. "
        "Use only the tables and columns defined in the schema.\n\n"
        "Schema:\n{schema}\n\n"
        "Request: {request}\n\n"
        "SQL:"
    ),
}
```

---

## 5. Evaluation Dimensions & Metrics

### Overview Table

| # | Dimension | Default Weight Range | Primary Metric | Test Categories |
|---|---|---|---|---|
| **D1** | Financial Domain Knowledge | 15–25% | Accuracy (EM + Domain F1) | A, E |
| **D2** | Numerical Reasoning | 15–25% | Exact Match Accuracy | A |
| **D3** | Document Understanding | 10–20% | ROUGE-L + BERTScore F1 | C |
| **D4** | Instruction Following | 10–15% | Prompt-Level Accuracy | D |
| **D5** | Hallucination & Factual Accuracy | 15–20% | Hallucination Rate (HR) | E |
| **D6** | Latency & Throughput | 5–10% | P95 Latency + TPS | All |
| **D7** | Context Window Handling | 5–10% | NIAH Accuracy + Position Bias | A, C |
| **D8** | Code Generation | 5–10% | Pass@1 | F |

---

### D1 — Financial Domain Knowledge

**What it measures:** Model's factual accuracy on finance-specific concepts, definitions, regulations, and market knowledge (CFA / FRM / CPA exam level), plus real-world financial Q&A over company disclosures.

**Test sets:** FinBench MCQ, RegQA, DocFinQA (extractive subset)

| Metric | Formula | Good Threshold |
|---|---|---|
| **Exact Match (EM)** | `# correct answers / # total questions` | ≥ 0.70 |
| **Answer F1** | Token-level F1 between prediction and gold answer | ≥ 0.80 |
| **Domain Sub-Score** | Weighted EM across sub-domains (accounting, derivatives, credit, regulation) | ≥ 0.65 per sub-domain |

```python
from evaluate import load
accuracy = load("accuracy")
score = accuracy.compute(predictions=preds, references=gold_labels)
```

---

### D2 — Numerical Reasoning

**What it measures:** Multi-step arithmetic and logical inference over financial tables and text: DCF calculations, ratio derivations, percentage changes, compound growth, YoY variance analysis.

**Test sets:** FinQA (full test set), TAT-QA, ConvFinQA, custom ratio-analysis prompts

| Metric | Formula | Notes |
|---|---|---|
| **EM Accuracy** | Exact numerical match (±0.01% tolerance) | Primary metric |
| **Program Accuracy** | Correct operator sequence (FinQA DSL programs) | Tests reasoning chain validity |
| **Numeric F1** | Partial credit for correct intermediate steps | For partial-credit evaluation |
| **CoT Quality** | LLM-as-judge (GPT-4o) scores reasoning chain 1–5 | Subjective; secondary only |

> **Tolerance rule:** For currency-denominated answers, responses within ±0.5% of the gold answer are accepted as correct (accounts for rounding differences in multi-step calculations).

---

### D3 — Document Understanding

**What it measures:** Ability to parse, extract, and summarize long-form financial documents including 10-K annual reports, earnings call transcripts, and IPO prospectuses.

**Test sets:** EDGARCorpus summarization, EarningsCall, ProspectusSumm, DocFinQA (summarization subset)

| Metric | Formula | Good Threshold |
|---|---|---|
| **ROUGE-L** | LCS-based F1 between generated and reference summary | ≥ 0.35 |
| **BERTScore F1** | Contextual embedding similarity (deberta-large-mnli) | ≥ 0.88 |
| **METEOR** | Unigram precision/recall with synonym matching | ≥ 0.30 |
| **Coverage Score** | % of key financial entities (figures, names, dates) present in summary | ≥ 0.75 |
| **Factual Consistency** | NLI entailment rate: is summary entailed by source? | ≥ 0.80 |

```python
from evaluate import load
bertscore = load("bertscore")
rouge = load("rouge")

bert_results = bertscore.compute(predictions=summaries, references=refs, lang="en",
                                 model_type="microsoft/deberta-xlarge-mnli")
rouge_results = rouge.compute(predictions=summaries, references=refs)
```

---

### D4 — Instruction Following

**What it measures:** Model's precision in adhering to format constraints (produce JSON, exact field names, fixed output length), multi-step extraction instructions, and structured tool-calling commands in agentic financial workflows.

**Test sets:** IFEval-Finance, FinInstruct eval subset, ToolBench-Finance

| Metric | Formula | Good Threshold |
|---|---|---|
| **Prompt-Level Accuracy** | % of prompts where **all** stated constraints are satisfied | ≥ 0.75 |
| **Instruction-Level Accuracy** | % of individual instruction clauses satisfied | ≥ 0.85 |
| **JSON Validity Rate** | % of JSON responses that parse without error | ≥ 0.95 |
| **Schema Compliance Rate** | % of outputs matching target JSON schema exactly | ≥ 0.80 |
| **Tool Call Accuracy** | % of function calls with correct name + all required parameters | ≥ 0.70 |

---

### D5 — Hallucination & Factual Accuracy

**What it measures:** Tendency to fabricate financial facts — invented stock tickers, incorrect earnings figures, false regulatory citations, non-existent filings. Also measures calibrated uncertainty: does the model admit when it does not know?

**Test sets:** FinHallBench (adversarial), FinAdvice-Red, adversarial probing QA

| Metric | Formula | Direction | Good Threshold |
|---|---|---|---|
| **Hallucination Rate (HR)** | `# hallucinated claims / # total verifiable claims` | Lower is better | ≤ 0.05 |
| **Factual Error Rate (FER)** | `# contradicted facts / # verifiable facts` (cross-referenced vs. EDGAR / Bloomberg Concepts) | Lower is better | ≤ 0.08 |
| **Appropriate Refusal Rate** | % of genuinely-unknowable questions answered with "I don't know" or equivalent | Higher is better | ≥ 0.60 |
| **Calibration Error (ECE)** | Expected Calibration Error on factual true/false questions | Lower is better | ≤ 0.10 |

**Hallucination detection pipeline:**
1. Extract all entity-level claims from model output (tickers, numeric figures, dates, person names, regulatory names)
2. Each claim is verified against a reference database (EDGAR, Bloomberg Concepts, regulatory text)
3. Claims that are unverifiable OR contradict the reference are flagged as hallucinations
4. Reported as a rate per 100 verified claims

---

### D6 — Latency & Throughput

**What it measures:** Production-readiness under realistic finance workloads. Three representative traffic profiles are tested:

| Profile | Scenario | Concurrency | Input / Output Tokens |
|---|---|---|---|
| **Latency-Critical** | Real-time trading signal or compliance alert | 1 | 256 in / 128 out |
| **Interactive** | Wealth advisor chatbot, synchronous loan Q&A | 5 | 512 in / 256 out |
| **Batch** | Overnight 10-K analysis, batch sentiment scoring | 20 | 2048 in / 512 out |

**Test harness:** Custom load test with vLLM OpenAI-compatible server + `locust`

| Metric | Measurement | Latency-Critical Target | Batch Target |
|---|---|---|---|
| **Time-to-First-Token (TTFT)** | ms from request to first streamed token | < 300 ms | < 2,000 ms |
| **P50 Completion Latency** | ms for full response (512 tokens out) | < 2,000 ms | < 12,000 ms |
| **P95 Completion Latency** | ms for full response (95th percentile) | < 5,000 ms | < 25,000 ms |
| **Throughput (TPS)** | output tokens/second per GPU under batch load | — | ≥ 500 TPS |
| **Requests/sec (RPS)** | max sustained RPS before queue depth exceeds 10 | ≥ 3 RPS | ≥ 15 RPS |
| **Peak VRAM Usage** | GB at maximum batch size | < 80 GB | < 160 GB |
| **Estimated Cost / 1M tokens** | USD on on-demand cloud GPU compute | — | — |

---

### D7 — Context Window & Long-Document Handling

**What it measures:** Retrieval accuracy when critical information is embedded inside very long financial documents. Inspired by the "Needle in a Haystack" (NIAH) pattern, adapted for financial scenarios (finding a specific revenue figure in a 150-page 10-K).

**Test sets:** DocFinQA long-context split, custom Long-Context Finance NIAH (purpose-built)

| Metric | Formula | Good Threshold |
|---|---|---|
| **NIAH Accuracy** | % of needle retrievals correct (exact or near-exact match) | ≥ 0.80 |
| **Position Bias Score** | `Δ accuracy = max(acc_start, acc_end) − acc_middle` | ≤ 0.10 (low bias) |
| **Multi-hop Recall F1** | Token-level F1 for QA requiring evidence from ≥2 document sections | ≥ 0.65 |
| **Long-Context Faithfulness** | NLI-based: does answer stay grounded in the long context? | ≥ 0.80 |

**Test context lengths:** 4K, 8K, 16K, 32K, 64K, 128K tokens (model permitting)

---

### D8 — Code Generation (Financial)

**What it measures:** Ability to write correct, executable Python and SQL for financial computations, including quantitative finance functions, XBRL data extraction, and financial schema queries.

**Test sets:** FinCode-Eval (Python), SQLFinBench (SQL)

| Metric | Formula | Good Threshold |
|---|---|---|
| **Pass@1** | % of problems solved correctly on first attempt (execution-based evaluation) | ≥ 0.55 |
| **Pass@3** | % of problems where ≥1 of 3 generated solutions passes all unit tests | ≥ 0.75 |
| **Compilation Rate** | % of outputs that execute without syntax or import errors | ≥ 0.95 |
| **Numerical Correctness** | Output within ±0.01% of reference answer for quantitative problems | ≥ 0.70 |
| **SQL Execution Accuracy** | % of SQL queries which return the correct result set on test tables | ≥ 0.60 |

```python
# Code evaluation via isolated Docker sandbox
def evaluate_code_solution(code: str, test_cases: list[dict], timeout: int = 10) -> dict:
    result = sandbox.run(code=code, tests=test_cases, timeout=timeout)
    return {
        "pass_rate":          result.passed / len(test_cases),
        "compilation":        result.compiled,
        "max_numeric_error":  result.max_numeric_delta,
    }
```

---

## 6. Scoring Rules & Scorecard

### 6.1 Score Normalization

All raw metrics are normalized to a **[0, 100]** scale using min-max normalization against established baselines:

$$\text{Normalized Score}_i = 100 \times \frac{r_i - r_{\min}}{r_{\max} - r_{\min}}$$

Where:
- $r_i$ = raw metric value for model $i$
- $r_{\min}$ = random-baseline or observed worst-case score in cohort
- $r_{\max}$ = theoretical maximum (1.0 for accuracy) OR human expert performance benchmark

**For lower-is-better metrics** (Hallucination Rate, Latency, Calibration Error):

$$\text{Normalized Score}_i = 100 \times \left(1 - \frac{r_i - r_{\text{target\_min}}}{r_{\text{target\_max}} - r_{\text{target\_min}}}\right)$$

---

### 6.2 Dimension Weights by Use Case

Weights are calibrated per use-case profile and must sum to exactly **100%**:

| Dimension | Doc Intelligence | Sentiment & Trading | Risk & Compliance | Customer Advisory | Quant Reasoning | Code Generation |
|---|---|---|---|---|---|---|
| D1 Financial Knowledge | 15% | 15% | **25%** | 20% | 15% | 10% |
| D2 Numerical Reasoning | 10% | 10% | 10% | 10% | **30%** | 15% |
| D3 Document Understanding | **30%** | 10% | 20% | 15% | 10% | 5% |
| D4 Instruction Following | 10% | 10% | 10% | 15% | 10% | 15% |
| D5 Hallucination & Safety | 15% | 15% | **25%** | **20%** | 15% | 5% |
| D6 Latency & Throughput | 5% | **20%** | 5% | 10% | 5% | 5% |
| D7 Context Handling | 10% | 5% | 5% | 5% | 10% | 5% |
| D8 Code Generation | 5% | 15% | 0% | 5% | 5% | **40%** |
| **Total** | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** |

> **Bolded cells** indicate dimensions that are primary differentiators for that use case.

---

### 6.3 Composite Score Formula

$$\text{Composite}(m, u) = \sum_{i=1}^{8} w_i^{(u)} \cdot S_i^{(m)}$$

Where:
- $m$ = model being scored
- $u$ = use-case profile
- $w_i^{(u)}$ = weight of dimension $i$ for use-case $u$ (from table above)
- $S_i^{(m)}$ = normalized score (0–100) of model $m$ on dimension $i$

---

### 6.4 Grade Thresholds

| Grade | Composite Score | Recommendation |
|---|---|---|
| **A+** | 85 – 100 | Production-ready; best-in-class for this use case |
| **A** | 75 – 84 | Production-ready; minor targeted fine-tuning may improve edge cases |
| **B+** | 65 – 74 | Suitable for deployment after use-case-specific fine-tuning |
| **B** | 55 – 64 | Suitable for limited scope; notable capability gaps — evaluate trade-offs carefully |
| **C** | 40 – 54 | Requires substantial fine-tuning or RAG augmentation; high operational risk |
| **D** | < 40 | Not recommended for finance production use in this category |

---

### 6.5 Scorecard Template (Fill After Run)

| Dimension (0–100) | LLaMA 3.3 70B | Mixtral 8×7B | Qwen2.5 72B | DeepSeek-R1 70B | Phi-4 14B | Gemma 2 27B |
|---|---|---|---|---|---|---|
| D1 Financial Knowledge | — | — | — | — | — | — |
| D2 Numerical Reasoning | — | — | — | — | — | — |
| D3 Document Understanding | — | — | — | — | — | — |
| D4 Instruction Following | — | — | — | — | — | — |
| D5 Hallucination & Safety | — | — | — | — | — | — |
| D6 Latency & Throughput | — | — | — | — | — | — |
| D7 Context Handling | — | — | — | — | — | — |
| D8 Code Generation | — | — | — | — | — | — |
| **Composite — Doc Intelligence** | — | — | — | — | — | — |
| **Composite — Sentiment & Trading** | — | — | — | — | — | — |
| **Composite — Risk & Compliance** | — | — | — | — | — | — |
| **Composite — Customer Advisory** | — | — | — | — | — | — |
| **Composite — Quant Reasoning** | — | — | — | — | — | — |
| **Composite — Code Generation** | — | — | — | — | — | — |
| **Overall Grade (primary use case)** | — | — | — | — | — | — |

---

### 6.6 Tie-Breaking Rules

When two models are within **2.0 composite score points** of each other:

1. **Safety first** — Higher D5 (Hallucination & Safety) score wins
2. **Latency second** — Lower P95 completion latency wins
3. **Cost third** — Lower estimated cost per 1M tokens wins
4. **Model size fourth** — Smaller model wins (easier deployment, lower infra cost)

---

## 7. Benchmark Execution Guide

### 7.1 Pre-Run Checklist

```
[ ] All model weights downloaded and checksums verified
[ ] Inference server health-checked (generate one test prompt per model)
[ ] Dataset splits loaded; SHA-256 hashes match pinned versions
[ ] GPU memory headroom ≥ 20% above model's expected footprint
[ ] Temperature = 0.0, seed = 42 confirmed in benchmark_config.yaml
[ ] MLflow experiment initialized; run UUID assigned
[ ] LangFuse project created for latency tracing
[ ] Network isolation confirmed — no external API calls during inference
[ ] Models evaluated SERIALLY (not concurrently) to prevent resource contention
```

---

### 7.2 Execution Commands

```bash
# Step 1: Run inference for each model (serial execution mandatory)
for model in llama-3.3-70b mixtral-8x7b qwen2.5-72b deepseek-r1-70b phi-4-14b gemma-2-27b; do
  python run_benchmark.py \
    --model "${model}" \
    --tasks finqa,convfinqa,finbench,fpb,edgar_summ,ifeval_finance,finhallbench,fincode,sql_finbench \
    --config benchmark_config.yaml \
    --output "./results/${model}/" \
    --seed 42
done

# Step 2: Compute all metrics
python compute_metrics.py \
  --results-dir ./results/ \
  --output ./metrics/metrics_$(date +%Y%m%d).json

# Step 3: Normalize and generate scorecard
python generate_scorecard.py \
  --metrics ./metrics/metrics_$(date +%Y%m%d).json \
  --use-cases all \
  --output ./reports/scorecard_$(date +%Y%m%d).csv

# Step 4: Render dashboard and charts
python render_dashboard.py \
  --scorecard ./reports/scorecard_$(date +%Y%m%d).csv \
  --output ./reports/dashboard_$(date +%Y%m%d).html
```

---

### 7.3 Reproducibility Requirements

Every benchmark run MUST log the following metadata:

```json
{
  "run_id": "uuid-v4",
  "date": "2026-03-17T00:00:00Z",
  "hardware": {
    "gpu": "4x NVIDIA H100 80GB SXM5",
    "cuda_version": "12.3",
    "driver_version": "545.23"
  },
  "software": {
    "vllm": "0.4.2",
    "lm_eval": "0.4.3",
    "python": "3.11.8"
  },
  "dataset_hashes": {
    "finqa_test": "sha256:abc123...",
    "fpb_test": "sha256:def456..."
  },
  "generation_params": {
    "temperature": 0.0,
    "seed": 42,
    "max_new_tokens": 512
  }
}
```

---

## 8. Results Reporting Template

### 8.1 Executive Summary Format

```markdown
## Benchmark Run: YYYY-MM-DD | Run ID: <uuid>

Hardware: 4× H100 80GB | CUDA 12.x | vLLM 0.x.x

### Overall Rankings (Primary Use Case: Risk & Compliance)

| Rank | Model | Composite Score | Grade | Δ vs. #1 |
|---|---|---|---|---|
| 1 | [Model A] | [Score] | [Grade] | — |
| 2 | [Model B] | [Score] | [Grade] | -[Δ] |
| ... |

### Dimension Leaders

| Dimension | Top Model | Score |
|---|---|---|
| D1 Financial Knowledge | [Model] | [Score] |
| D2 Numerical Reasoning | [Model] | [Score] |
| ...

### Key Findings
- [Finding 1: e.g., "DeepSeek-R1 70B leads D2 by 8.3 points but trails on D6 latency by 42%"]
- [Finding 2: ...]
- [Finding 3: ...]

### Recommendation
[Model X] is recommended for [Use Case] because [rationale].
For latency-constrained deployments, [Model Y] offers [tradeoff] at [cost/perf ratio].
```

---

### 8.2 Required Visualizations

| Chart Type | Axes / Dimensions | Purpose |
|---|---|---|
| **Radar / Spider chart** | 8 dimensions per model | Capability profile comparison |
| **Grouped bar chart** | D1–D8 scores, grouped by dimension | Absolute metric comparison |
| **Composite score heatmap** | Models × Use Cases | Go/No-Go decision support |
| **Latency scatter plot** | X: P95 latency, Y: composite score | Speed vs. quality Pareto frontier |
| **Cost-efficiency scatter** | X: cost/1M tokens, Y: composite score | Economic deployment decision |
| **NIAH heatmap** | X: document depth %, Y: context length | Long-context retrieval degradation |

---

## 9. Appendix

### 9.1 Public Benchmark & Dataset References

| Resource | URL | Domain |
|---|---|---|
| FinQA | github.com/czyssrs/FinQA | Financial numerical QA |
| ConvFinQA | github.com/czyssrs/ConvFinQA | Conversational finance QA |
| FinBench | github.com/the-fin-llm/FinBench | Finance knowledge MCQ |
| Financial PhraseBank | huggingface.co/datasets/financial_phrasebank | Sentiment classification |
| TAT-QA | github.com/NExTplusplus/TAT-QA | Hybrid table-text QA |
| FinEval | github.com/SUFE-AIFLM-Lab/FinEval | Chinese finance MCQ |
| lm-evaluation-harness | github.com/EleutherAI/lm-evaluation-harness | Evaluation framework |
| RAGAS | github.com/explodinggradients/ragas | RAG evaluation |
| Open LLM Leaderboard | huggingface.co/spaces/HuggingFaceH4 | General capability reference |

---

### 9.2 LLM-as-Judge Configuration

For subjective evaluations (reasoning chain quality, open-ended summarization quality):

- **Judge model:** GPT-4o or Claude 3.5 Sonnet (external — never use any model under test as its own judge)
- **Scoring rubric:** 1–5 scale with explicitly anchored criteria per dimension
- **Inter-rater reliability:** Cohen's κ ≥ 0.75 required; dual-judge + tiebreaker for disagreements > 1 point
- **Bias mitigation:** Model identities hidden from judge prompt; answer order randomized; system prompt identical across all evaluated outputs

---

### 9.3 Statistical Validity

- **Minimum test set size:** 400 items per category ensures statistical significance at p < 0.05 with power = 0.80 for detecting a 5% accuracy difference
- **Confidence intervals:** Report 95% CI for all primary metrics using bootstrap resampling (n = 1,000)
- **Significance testing:** Use McNemar's test for pairwise accuracy comparisons; report corrected p-values (Bonferroni) for multiple model comparisons

---

### 9.4 Glossary

| Term | Definition |
|---|---|
| **EM** | Exact Match — binary correctness award |
| **ROUGE-L** | Recall-Oriented Understudy for Gisting Evaluation (LCS-variant F1) |
| **BERTScore** | Semantic similarity metric using contextual token embeddings |
| **Pass@k** | Probability that ≥1 of k generated code samples passes all unit tests |
| **NIAH** | Needle-In-A-Haystack — long-context retrieval stress test |
| **IFEval** | Instruction Following Evaluation — verifiable constraint benchmark |
| **HR** | Hallucination Rate |
| **FER** | Factual Error Rate |
| **TPS** | Tokens Per Second (throughput) |
| **TTFT** | Time To First Token (streaming latency) |
| **ECE** | Expected Calibration Error |
| **CoT** | Chain of Thought reasoning |
| **MoE** | Mixture of Experts architecture |
| **TP** | Tensor Parallelism (model sharding across GPUs) |
| **DSL** | Domain-Specific Language (FinQA's arithmetic program representation) |

---

*Framework Version: 1.0.0 | Maintained by: Finance AI Engineering | Last Updated: March 2026*
