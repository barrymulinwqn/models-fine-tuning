# Phi-4 Fine-Tuning Guide for Finance Companies

> **Model:** Microsoft Phi-4 (14B parameters)

> **License:** MIT (fully open-source, commercial use allowed)

> **Architecture:** Dense Transformer, 14B params, 16K context window

> **Target Audience:** Small to Mid-sized Finance Companies

---

## Table of Contents

1. [Model Overview & Fitness for Finance](#1-model-overview--fitness-for-finance)
2. [Step-by-Step Fine-Tuning Guide](#2-step-by-step-fine-tuning-guide)
3. [Training Dataset Setup for Finance](#3-training-dataset-setup-for-finance)
4. [Hardware Requirements](#4-hardware-requirements)
5. [Benchmark Testing](#5-benchmark-testing)
6. [Alternative Models to Consider](#6-alternative-models-to-consider)

---

## 1. Model Overview & Fitness for Finance

### What is Phi-4?

Phi-4 is Microsoft's 14-billion parameter dense language model released in December 2024. Unlike earlier Phi models optimized purely for size efficiency, Phi-4 was trained with a strong emphasis on **reasoning quality**, **math**, and **structured instruction following** — all highly relevant for financial applications.

### Why Phi-4 for Finance?

| Capability | Relevance to Finance |
|---|---|
| Strong chain-of-thought reasoning | Financial analysis, risk scoring, multi-step calculations |
| Math & quantitative reasoning | DCF modeling, ratio analysis, portfolio metrics |
| Instruction following | Report generation, regulatory Q&A, client communication |
| 16K context window | Processing full earnings reports, long contracts |
| MIT License | No licensing overhead for commercial deployment |
| 14B parameters (dense) | Predictable inference cost, easier to optimize |

### Limitations to Be Aware Of

- No native multi-modal capability (no chart/PDF image understanding out of the box)
- Training data cutoff — must be fine-tuned with current market data and regulations
- 16K context may be insufficient for very long legal documents (consider chunking strategies)
- Dense architecture means higher VRAM per parameter compared to Mixture-of-Experts models

---

## 2. Step-by-Step Fine-Tuning Guide

### Recommended Strategy: QLoRA (Quantized Low-Rank Adaptation)

For small/mid-sized finance companies without a dedicated ML infrastructure team, **QLoRA** is the recommended approach. It reduces VRAM requirements by ~4x with negligible quality loss compared to full fine-tuning.

---

### Step 1: Define Your Fine-Tuning Objective

Before writing a single line of code, clearly define **what problem you are solving**:

| Use Case | Fine-Tuning Type |
|---|---|
| Internal financial Q&A chatbot | Instruction fine-tuning (SFT) |
| Earnings call summarizer | Supervised summarization |
| Credit risk narrative generator | Conditional text generation |
| SEC filing analyzer | Extraction + summarization |
| Regulatory compliance assistant | Domain-adapted SFT |
| Trading signal commentary | Structured output fine-tuning |

Choose **one primary use case** for your first fine-tuning iteration. Do not try to cover everything at once.

---

### Step 2: Set Up Your Environment

```bash
# Create a clean Python virtual environment
python -m venv phi4-finance-env
source phi4-finance-env/bin/activate

# Core dependencies
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.45.0
pip install peft==0.12.0          # LoRA / QLoRA
pip install trl==0.9.6            # SFT Trainer
pip install bitsandbytes==0.43.3  # 4-bit quantization
pip install accelerate==0.33.0
pip install datasets==2.20.0
pip install wandb                  # Training monitoring (recommended)
```

---

### Step 3: Load Phi-4 with 4-bit Quantization

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_id = "microsoft/phi-4"

# 4-bit quantization config (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
```

---

### Step 4: Configure LoRA Adapter

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Prepare model for QLoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,               # Rank — higher = more capacity, more VRAM. Start at 16.
    lora_alpha=32,      # Scaling factor (typically 2x rank)
    target_modules=[    # Phi-4 attention projection layers
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: ~0.5–1% of total parameters — very efficient
```

---

### Step 5: Format Your Finance Dataset

Finance data should be formatted as instruction-response pairs. Use the Phi-4 chat template:

```python
def format_finance_sample(sample):
    """
    Format a finance Q&A pair into Phi-4's instruction format.
    """
    return f"""<|system|>
You are an expert financial analyst assistant. Provide accurate, 
concise, and professional financial analysis based on the context provided.
<|end|>
<|user|>
{sample['instruction']}

Context:
{sample['context']}
<|end|>
<|assistant|>
{sample['response']}
<|end|>"""

# Apply formatting to your HuggingFace dataset
formatted_dataset = dataset.map(
    lambda x: {"text": format_finance_sample(x)},
    remove_columns=dataset.column_names
)
```

---

### Step 6: Configure Training with SFTTrainer

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./phi4-finance-adapter",
    
    # Training duration — conservative for small companies
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # Effective batch size = 16
    
    # Learning rate schedule
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    
    # Memory optimization
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    fp16=False,
    bf16=True,
    
    # Logging & saving
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    
    # Context — match your dataset
    max_seq_length=2048,
    
    # Weights & Biases tracking (recommended)
    report_to="wandb",
    run_name="phi4-finance-v1",
    
    # Dataset packing for efficiency
    packing=False,  # Set True if samples are short
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["validation"],
    tokenizer=tokenizer,
    peft_config=lora_config,
)

trainer.train()
```

---

### Step 7: Evaluate and Merge Adapter

```python
# Evaluate on held-out finance test set
results = trainer.evaluate()
print(f"Eval Loss: {results['eval_loss']:.4f}")

# Save the LoRA adapter (small — typically 200–500MB)
trainer.save_model("./phi4-finance-adapter-final")

# Optional: Merge adapter into base model for faster inference
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
merged_model = PeftModel.from_pretrained(base_model, "./phi4-finance-adapter-final")
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained("./phi4-finance-merged")
tokenizer.save_pretrained("./phi4-finance-merged")
```

---

### Step 8: Deploy for Internal Use

For small/mid-sized companies, recommended serving options:

| Option | Best For | Complexity |
|---|---|---|
| **Ollama** (local) | 1–5 internal users, air-gapped | Low |
| **vLLM** (on-prem server) | 5–50 concurrent users | Medium |
| **Azure ML** or **AWS SageMaker** | Cloud-native teams | Medium-High |
| **HuggingFace Inference Endpoints** | Quick production deployment | Low |

```bash
# Example: Serve with vLLM (recommended for production)
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model ./phi4-finance-merged \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --port 8000
```

---

## 3. Training Dataset Setup for Finance

### 3.1 Dataset Categories and Sources

A high-quality finance fine-tuning dataset should cover multiple data types:

#### Tier 1 — Structured Q&A (Highest Priority)

| Source | Description | Format |
|---|---|---|
| **FinanceBench** | 10K+ Q&A over SEC filings | Q&A pairs |
| **FiNER-139** | Financial named entity recognition | Classification |
| **FLARE benchmark datasets** | Financial tasks (NER, sentiment, QA) | Various |
| **FinQA** | Multi-step financial reasoning over tables | Q&A + reasoning |
| **ConvFinQA** | Conversational financial Q&A | Multi-turn dialog |
| **TAT-QA** | Table + text answering in finance | Q&A |

#### Tier 2 — Domain Documents (Bulk Pre-training Signal)

| Source | Description | Volume |
|---|---|---|
| SEC EDGAR full-text | 10-K, 10-Q, 8-K filings | 100K+ documents |
| Earnings call transcripts | Seeking Alpha, Motley Fool | 50K+ transcripts |
| Financial news | Reuters, Bloomberg (where licensed) | Large corpus |
| CFA curriculum materials | Conceptual finance knowledge | Structured text |
| Basel III / IFRS / GAAP docs | Regulatory and accounting standards | Legal/regulatory |

#### Tier 3 — Company-Specific Private Data

This is your **most valuable differentiation**:

- **Internal credit memos** and underwriting decisions
- **Client reports** and investment summaries (anonymized)
- **Internal policies** and compliance SOPs
- **Historical analyst notes** and research
- **CRM Q&A logs** from advisors (anonymized)

> ⚠️ **Privacy Warning:** Always anonymize PII (names, account numbers, SSNs) before including in any training dataset. Consider differential privacy techniques for sensitive financial data.

---

### 3.2 Dataset Construction Pipeline

```
Raw Sources
    │
    ▼
[1] Data Collection
    ├── Scrape/Download public filings (SEC EDGAR API)
    ├── License third-party datasets (FinanceBench, FinQA)
    └── Export internal systems (with PII scrubbing)
    │
    ▼
[2] Data Cleaning
    ├── Remove HTML/XML tags from SEC filings
    ├── Fix encoding issues
    ├── Deduplicate (MinHash or exact hash)
    └── Language filter (English only, unless multilingual target)
    │
    ▼
[3] Quality Filtering
    ├── Perplexity filter (remove low-quality, garbled text)
    ├── Length filter (min 50 tokens, max 2048 tokens per sample)
    └── Domain relevance classifier
    │
    ▼
[4] Instruction Formatting
    ├── Convert documents → Q&A pairs (using GPT-4 as annotator)
    ├── Create summarization tasks (input=full doc, output=summary)
    └── Create extraction tasks (input=doc, output=structured JSON)
    │
    ▼
[5] Dataset Splits
    ├── Train: 80%
    ├── Validation: 10%
    └── Test: 10% (NEVER used during training)
```

---

### 3.3 Automated Q&A Generation from Documents

Use a strong model (GPT-4o or Claude 3.5) to generate instruction pairs from your document corpus:

```python
import openai

def generate_finance_qa(document_chunk: str, n_questions: int = 5) -> list[dict]:
    """
    Use GPT-4o to generate finance Q&A pairs from a document chunk.
    This creates training data for your Phi-4 fine-tune.
    """
    prompt = f"""You are a senior financial analyst. Given the following financial document excerpt, 
generate {n_questions} question-answer pairs that test deep understanding of the content.

Focus on:
- Specific numerical facts and ratios
- Risk factors and their implications  
- Comparison with industry benchmarks
- Regulatory compliance points
- Forward-looking statements and their caveats

Document:
{document_chunk}

Return as JSON array: [{{"question": "...", "answer": "..."}}]"""

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return response.choices[0].message.content

# Process your SEC filing corpus
qa_pairs = []
for filing in sec_filings:
    for chunk in chunk_document(filing, max_tokens=1500):
        pairs = generate_finance_qa(chunk)
        qa_pairs.extend(pairs)
```

---

### 3.4 Target Dataset Size

| Company Size | Training Samples | Estimated Cost (GPT-4o annotation) | Training Time (A100) |
|---|---|---|---|
| Small (< 50 employees) | 5,000 – 20,000 | $50 – $200 | 2–6 hours |
| Mid-sized (50–500) | 20,000 – 100,000 | $200 – $1,000 | 6–24 hours |
| Large (500+) | 100,000+ | $1,000+ | 1–5 days |

---

## 4. Hardware Requirements

### 4.1 Minimum vs. Recommended Configurations

#### For Fine-Tuning (QLoRA — Recommended for Small Companies)

| Configuration | GPU | VRAM | RAM | Storage | Monthly Cloud Cost |
|---|---|---|---|---|---|
| **Minimum** | 1× NVIDIA RTX 3090 | 24 GB | 64 GB | 500 GB SSD | N/A (on-prem) |
| **Recommended (On-prem)** | 1× NVIDIA A100 40GB | 40 GB | 128 GB | 2 TB NVMe | N/A |
| **Comfortable (On-prem)** | 2× NVIDIA A100 80GB | 160 GB | 256 GB | 4 TB NVMe | N/A |
| **Cloud Minimum** | 1× A10G (AWS g5.xlarge) | 24 GB | 16 GB | EFS | ~$1.00/hr |
| **Cloud Recommended** | 1× A100 (AWS p3.2xlarge / GCP a2-highgpu-1g) | 40/80 GB | 64–128 GB | EBS | ~$3–$4/hr |

#### For Full Fine-Tuning (Not Recommended for Small Companies)

| Configuration | GPU | VRAM Required |
|---|---|---|
| Full FP32 | 8× A100 80GB | ~640 GB aggregate |
| Full BF16 + ZeRO-3 | 4× A100 80GB | ~320 GB aggregate |

> **Recommendation for Small/Mid Companies:** Use **QLoRA on a single A100 40GB** or cloud instance. It achieves 90%+ of full fine-tuning quality at 5–10% of the cost.

---

### 4.2 VRAM Breakdown for Phi-4 (14B)

| Component | Full FP16 | QLoRA (4-bit) |
|---|---|---|
| Model weights | ~28 GB | ~7 GB |
| Optimizer states | ~56 GB | ~2 GB (8-bit Adam) |
| Gradients | ~28 GB | ~1 GB (LoRA only) |
| Activations (bs=2) | ~8 GB | ~4 GB |
| **Total** | **~120 GB** | **~14–18 GB** |

This means QLoRA fits comfortably on a **single 24GB RTX 3090** with `gradient_checkpointing=True`.

---

### 4.3 Cloud Provider Quick Reference

#### AWS

```
Training:  p3.2xlarge  (1× V100 16GB)  ~$3.06/hr  — tight but workable
           p3.8xlarge  (4× V100 16GB)  ~$12.24/hr — comfortable
           p4d.24xlarge (8× A100 40GB) ~$32.77/hr — fast iteration
Inference: g5.xlarge   (1× A10G 24GB) ~$1.006/hr — production serving
```

#### Google Cloud

```
Training:  a2-highgpu-1g  (1× A100 40GB) ~$3.67/hr
           a2-highgpu-2g  (2× A100 40GB) ~$7.35/hr
Inference: g2-standard-4  (1× L4 24GB)   ~$0.70/hr
```

#### Azure

```
Training:  Standard_NC6s_v3  (1× V100 16GB) ~$3.06/hr
           Standard_NC24ads_A100_v4 (1× A100 80GB) ~$3.67/hr
```

---

### 4.4 Cost Estimation for a Typical Finance Fine-Tune

| Scenario | Hardware | Duration | Estimated Cost |
|---|---|---|---|
| 10K samples, 3 epochs | 1× A100 40GB (cloud) | ~4 hours | ~$15 |
| 50K samples, 3 epochs | 1× A100 80GB (cloud) | ~18 hours | ~$65 |
| 100K samples, 3 epochs | 2× A100 80GB (cloud) | ~20 hours | ~$145 |

---

## 5. Benchmark Testing

### 5.1 Finance-Specific Benchmarks

After fine-tuning, evaluate against these established benchmarks:

#### Primary Finance Benchmarks

| Benchmark | What It Tests | Metric | Target Score |
|---|---|---|---|
| **FinanceBench** | Q&A over real SEC filings | Accuracy | > 65% (GPT-4 baseline ~74%) |
| **FLARE** | 9 finance NLP tasks | F1 / Accuracy | Task-dependent |
| **FinQA** | Multi-step numerical reasoning | Execution accuracy | > 60% |
| **ConvFinQA** | Conversational financial QA | Execution accuracy | > 55% |
| **FPB (Financial PhraseBank)** | Sentiment classification | Weighted F1 | > 85% |
| **Headline** | News headline classification | Avg F1 | > 82% |
| **NER (FiNER)** | Financial entity recognition | F1 | > 79% |

#### General Benchmarks (Sanity Check — Ensure No Regression)

| Benchmark | What It Tests |
|---|---|
| **MMLU (Finance subset)** | General finance knowledge |
| **GSM8K** | Arithmetic reasoning (must not regress) |
| **HumanEval** | Coding ability (if relevant to use case) |

---

### 5.2 Internal Benchmark Setup

Build a **custom internal benchmark** reflecting your actual business tasks:

```python
# Example: Internal benchmark runner for a credit analysis assistant
import json
from transformers import pipeline

def run_internal_benchmark(model_path: str, test_file: str) -> dict:
    """
    Run your company's internal finance benchmark.
    """
    pipe = pipeline("text-generation", model=model_path, 
                    device_map="auto", torch_dtype="auto")
    
    with open(test_file) as f:
        test_cases = json.load(f)
    
    results = {
        "factual_accuracy": [],
        "numerical_accuracy": [],
        "format_compliance": [],
        "hallucination_rate": [],
    }
    
    for case in test_cases:
        output = pipe(
            case["prompt"],
            max_new_tokens=512,
            do_sample=False,  # Greedy for reproducibility
            temperature=1.0,
        )[0]["generated_text"]
        
        # Evaluate factual accuracy (compare against ground truth)
        results["factual_accuracy"].append(
            evaluate_factual_match(output, case["expected_answer"])
        )
        
        # Evaluate numerical extraction accuracy
        results["numerical_accuracy"].append(
            evaluate_number_extraction(output, case["expected_numbers"])
        )
    
    return {k: sum(v)/len(v) for k, v in results.items()}
```

---

### 5.3 Benchmark Test Categories for Finance

#### Category 1: Factual Retrieval
```
Q: "What was Apple's total revenue in fiscal year 2023 according to the 10-K?"
Expected: Exact figure extraction from provided document
Evaluation: Exact match or within ±1% numerical tolerance
```

#### Category 2: Financial Reasoning
```
Q: "Given revenue of $100M and COGS of $60M, calculate the gross margin 
    and explain whether it's above or below the industry average of 35%."
Expected: Gross margin = 40%, above industry average
Evaluation: Numerical correctness + reasoning chain quality
```

#### Category 3: Regulatory Compliance
```
Q: "Does this investment product description comply with SEC Regulation D 
    requirements for private placements?"
Expected: Yes/No + specific citation of relevant rules
Evaluation: Legal accuracy reviewed by compliance team
```

#### Category 4: Risk Assessment
```
Q: "Identify the top 3 risk factors from this credit memo and rate each 
    on a scale of 1-5."
Expected: Structured risk assessment matching analyst judgment
Evaluation: Inter-rater agreement with human analyst (Cohen's Kappa > 0.6)
```

---

### 5.4 Baseline vs. Fine-Tuned Model Comparison

Always compare fine-tuned Phi-4 against:

1. **Base Phi-4** (zero-shot) — establishes the value-add of fine-tuning
2. **Base Phi-4** (few-shot, 5 examples) — establishes prompting baseline
3. **GPT-4o** (via API) — establishes commercial frontier model ceiling
4. **Previous internal solution** (rule-based, LLM competitor) — business justification

| Model | FinanceBench | Internal Factual | Internal Reasoning | Cost/1K queries |
|---|---|---|---|---|
| GPT-4o (API) | ~74% | ~82% | ~78% | ~$10 |
| Phi-4 Base (zero-shot) | ~55% | ~60% | ~58% | ~$0.50 |
| Phi-4 Base (5-shot) | ~63% | ~70% | ~65% | ~$0.50 |
| **Phi-4 Fine-tuned (target)** | **>68%** | **>75%** | **>72%** | **~$0.50** |

---

## 6. Alternative Models to Consider

### 6.1 Models Better Suited Than Phi-4 for Some Finance Use Cases

#### Option A: Qwen2.5-72B-Instruct (Alibaba) ⭐ Recommended for Mid-Sized Companies

| Attribute | Value |
|---|---|
| Parameters | 72B |
| Context Window | 128K tokens |
| License | Qwen License (commercial for <100M monthly active users) |
| Language Support | Excellent multilingual (important for Asian finance) |
| Math Reasoning | State-of-the-art for open-source |
| Fine-tuning Cost | Higher (2–4× A100 80GB recommended) |

**Why better than Phi-4 for finance:** Significantly larger model with superior factual recall, longer context for processing full annual reports, and stronger multilingual support for firms with global operations. Its 128K context window is transformative for long regulatory document analysis.

```bash
# Serve with vLLM
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768
```

---

#### Option B: Llama 3.1 8B / 70B (Meta) ⭐ Best All-Around for Small Companies

| Attribute | Value |
|---|---|
| Parameters | 8B or 70B |
| Context Window | 128K tokens |
| License | Meta Llama 3.1 License (free for commercial use < 700M MAU) |
| Ecosystem | Largest open-source ecosystem, most fine-tuning resources |
| Fine-tuning Support | Excellent (most tutorials and tools target Llama) |

**Why better than Phi-4 for some cases:** Massive open-source ecosystem, excellent tool use and function calling (important for finance APIs), and the 8B variant is extremely cost-efficient. The 70B model has stronger factual grounding.

---

#### Option C: FinGPT (Finance-Specific) — For Pure NLP Finance Tasks

| Attribute | Value |
|---|---|
| Base Model | Llama 2 / Falcon (varies by variant) |
| Parameters | 7B–13B |
| License | Open-source, research-friendly |
| Training Data | Financial news, SEC filings, earnings calls |
| Best For | Sentiment analysis, financial NER, news summarization |

**Why consider:** Already domain-adapted without fine-tuning. Good starting point if you only need standard finance NLP tasks. However, it lags behind modern models and may require additional fine-tuning for specialized tasks.

---

#### Option D: DeepSeek-R1 / DeepSeek-V3 (DeepSeek AI) — For Reasoning-Heavy Tasks

| Attribute | Value |
|---|---|
| Parameters | 7B to 671B (MoE) |
| Context Window | 128K tokens |
| License | MIT (R1 distills are fully open) |
| Reasoning | Exceptional (SOTA reasoning at open-source level) |
| Fine-tuning | R1-distill-7B/14B/32B variants are highly practical |

**Why consider:** DeepSeek-R1-Distill-14B rivals GPT-4o on reasoning tasks at a fraction of the cost. Excellent for complex financial modeling, multi-step credit analysis, and quantitative reasoning. Strong choice as of 2025–2026.

---

### 6.2 Model Selection Decision Matrix

```
What is your primary use case?
│
├── Document Q&A + Long reports (>16K tokens)
│   └── → Qwen2.5-72B or Llama 3.1 70B (128K context)
│
├── Conversational finance assistant (< 8K context)
│   └── → Phi-4 fine-tuned (excellent quality, lower cost)
│
├── Complex financial reasoning / DCF / modeling
│   └── → DeepSeek-R1-Distill-14B (best reasoning per dollar)
│
├── Multilingual finance (Asian/European markets)
│   └── → Qwen2.5-72B (best multilingual finance model)
│
├── Standard NLP tasks (sentiment, NER, classification)
│   └── → Llama 3.1 8B fine-tuned (cheapest, most efficient)
│
└── Maximum accuracy, cost is secondary
    └── → GPT-4o or Claude 3.5 Sonnet via API
```

---

### 6.3 Total Cost of Ownership Comparison (Annual, Mid-Sized Company)

| Model | Fine-tuning Cost | Inference Hardware | Approximate Annual TCO |
|---|---|---|---|
| GPT-4o (API) | $0 | None | $50K–$200K (usage-based) |
| Phi-4 (fine-tuned, on-prem) | $500–$2K | 1× A100 server ~$25K | ~$30K–$40K |
| Llama 3.1 8B (fine-tuned, on-prem) | $100–$500 | 1× RTX 3090 ~$3K | ~$5K–$8K |
| Qwen2.5-72B (fine-tuned, on-prem) | $1K–$5K | 4× A100 server ~$80K | ~$90K–$100K |
| DeepSeek-R1-14B (fine-tuned, cloud) | $500–$2K | Cloud serving ~$1/hr | ~$15K–$25K |

---

## Summary & Recommendation for Finance Companies

### Small Finance Company (< 50 employees, < $5K budget)
**Recommended:** Fine-tune **Llama 3.1 8B** with QLoRA on 5K–20K internal Q&A pairs. Serve locally with Ollama. Total cost under $2K.

### Mid-Sized Finance Company (50–500 employees, $10K–$50K budget)
**Recommended:** Fine-tune **Phi-4 (14B)** with QLoRA on 20K–100K samples. Deploy on a cloud instance or on-prem A100. Evaluate against DeepSeek-R1-14B for reasoning-heavy workflows.

### Growth-Stage Finance Company (500+ employees, > $50K budget)
**Recommended:** Fine-tune **Qwen2.5-72B** for maximum capability. Pair with a smaller distilled model (Phi-4 or Llama 3.1 8B) for high-volume, lower-complexity queries to optimize cost.

---

*Last updated: March 2026 | Based on model releases through Q1 2026*
