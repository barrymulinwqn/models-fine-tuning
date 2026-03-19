# Mixtral 8×7B Fine-Tuning Guide for Finance Companies

> **Model:** Mistral AI — Mixtral 8×7B Instruct v0.1 / v0.3

> **License:** Apache 2.0 (fully open-source, commercial use allowed)

> **Architecture:** Sparse Mixture-of-Experts (MoE), 8 experts × 7B — 46.7B total params, ~12.9B active per token

> **Context Window:** 32K tokens

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

### What is Mixtral 8×7B?

Mixtral 8×7B is Mistral AI's Sparse Mixture-of-Experts (SMoE) model, releasing in December 2023. It uses **8 expert feed-forward networks** per transformer block, routing each token to **2 experts** at inference time. This architecture delivers performance comparable to a 70B dense model while only activating ~13B parameters per forward pass — achieving excellent throughput at lower inference cost.

### The MoE Architecture Advantage for Finance

```
Input Token
    │
    ▼
┌─────────────────────────────────────────┐
│  Attention Layer (shared, ~7B params)   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Router (Gating Network)                │
│  Selects 2 of 8 specialized experts     │
└─────────────────────────────────────────┘
    │        │
    ▼        ▼
[Expert 3] [Expert 7]  ← Only these 2 activate (~5B params)
    │        │
    └────────┘
         │
         ▼
    Output Token
```

Different experts can implicitly specialize in different domains — this is beneficial when fine-tuning on finance data, as the model can preserve general knowledge in some experts while specializing finance reasoning in others.

### Why Mixtral 8×7B for Finance?

| Capability | Relevance to Finance |
|---|---|
| 32K context window | Process full 10-K annual reports, long contracts, bond prospectuses |
| Strong multilingual support | Multi-market finance across EU, Asia, Americas |
| High throughput (MoE efficiency) | Cost-effective for high-volume queries (trade ops, client portals) |
| Strong code generation | Quantitative finance scripting, spreadsheet automation |
| Apache 2.0 license | Zero licensing risk for commercial finance products |
| Instruction-tuned variant | Direct deployment with finance system prompts |

### Limitations to Be Aware Of

- **Total 46.7B parameters** require significant storage (~90 GB in FP16) even though active compute is ~13B
- MoE models are harder to quantize efficiently than dense models — some quality loss with 4-bit
- Fine-tuning MoE is more complex: routing must be handled carefully to avoid expert collapse
- Not as strong as dense models on pure mathematical reasoning (use DeepSeek-R1 for quant-heavy work)
- Less fine-tuning community tooling compared to Llama-based models

---

## 2. Step-by-Step Fine-Tuning Guide

### Recommended Strategy: QLoRA with Expert-Aware Configuration

For Mixtral 8×7B, QLoRA is still recommended, but the LoRA target modules differ from dense models. You must target modules **within each expert** as well as shared attention layers.

---

### Step 1: Define Your Fine-Tuning Objective

Mixtral 8×7B excels particularly at:

| Use Case | Why Mixtral Fits |
|---|---|
| Long-document analysis (10-K, 10-Q) | 32K native context handles full filings |
| Multi-lingual client reporting | Superior EU/Asian language support |
| Code generation (Python finance scripts) | Strong coding capability |
| Regulatory document Q&A | Context window avoids chunking issues |
| Real-time trade commentary | High throughput MoE architecture |
| Complex instruction following | Instruction-tuned variant available |

Choose your primary use case before proceeding. The dataset structure differs significantly between, e.g., document Q&A and code generation.

---

### Step 2: Set Up Your Environment

```bash
# Create isolated Python environment
python -m venv mixtral-finance-env
source mixtral-finance-env/bin/activate

# Core dependencies — note flash-attn for long context efficiency
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.45.0
pip install peft==0.12.0
pip install trl==0.9.6
pip install bitsandbytes==0.43.3
pip install accelerate==0.33.0
pip install datasets==2.20.0
pip install flash-attn==2.6.3 --no-build-isolation  # Critical for 32K context
pip install scipy                                     # Required by some optimizers
pip install wandb                                     # Training monitoring
```

---

### Step 3: Load Mixtral 8×7B with 4-bit Quantization

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Use the instruction-tuned variant for finance assistants
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",          # NF4 quantization — best for LLMs
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # Required for batch training

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",  # Enable FlashAttention for 32K context
)

print(f"Model loaded. Memory footprint: {model.get_memory_footprint() / 1e9:.1f} GB")
# Expected on 4-bit: ~24-26 GB across devices
```

---

### Step 4: Configure LoRA for the MoE Architecture

This is the most critical difference from dense model fine-tuning. You must target **all expert FFN layers**, not just attention:

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# Mixtral-specific LoRA configuration
# Key: target ALL expert weight matrices (w1, w2, w3) inside each expert block
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    # Mixtral MoE expert layers + attention — this is what differs from dense models
    target_modules=[
        # Attention layers (shared across all tokens)
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # MoE expert feed-forward networks (per expert)
        # Format: model.layers.{N}.block_sparse_moe.experts.{E}.w1/w2/w3
        "w1",   # Gate projection inside experts
        "w2",   # Down projection inside experts
        "w3",   # Up projection inside experts
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# With r=16, targeting all experts: ~1.5-2% of parameters — slightly more than dense models
# but still very memory efficient
```

> **Why target w1/w2/w3?** In Mixtral, each of the 8 experts has its own FFN weights. Fine-tuning only attention layers misses the majority of where domain specialization happens. Targeting expert FFN layers is critical for finance domain adaptation.

---

### Step 5: Format Training Data for Mixtral's Chat Template

Mixtral uses `[INST]...[/INST]` formatting. Do NOT use a different format — it degrades performance significantly:

```python
def format_finance_sample_mixtral(sample: dict) -> str:
    """
    Format finance Q&A pair using Mixtral's instruction template.
    """
    system_prompt = """You are an expert financial analyst and advisor. 
Provide accurate, professional, and regulatory-compliant financial analysis. 
Always cite specific figures when available. Do not speculate beyond the provided data."""
    
    # Mixtral instruction format
    return (
        f"<s>[INST] {system_prompt}\n\n"
        f"Question: {sample['question']}\n\n"
        f"Context:\n{sample['context']} [/INST] "
        f"{sample['answer']}</s>"
    )

# For multi-turn conversation (e.g., client advisory sessions)
def format_multiturn_finance(conversation: list[dict]) -> str:
    """
    Format multi-turn financial conversation.
    """
    formatted = "<s>"
    for i, turn in enumerate(conversation):
        if turn["role"] == "user":
            if i == 0:
                formatted += f"[INST] {turn['content']} [/INST] "
            else:
                formatted += f"[INST] {turn['content']} [/INST] "
        elif turn["role"] == "assistant":
            formatted += f"{turn['content']}</s>"
    return formatted

# Apply to dataset
formatted_dataset = dataset.map(
    lambda x: {"text": format_finance_sample_mixtral(x)},
    remove_columns=dataset.column_names,
)
```

---

### Step 6: Configure Training — With Long Context Considerations

Mixtral's 32K context is a key advantage for finance. Configure accordingly:

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./mixtral-finance-adapter",
    
    # Training
    num_train_epochs=3,
    per_device_train_batch_size=1,         # Lower batch size for long context
    gradient_accumulation_steps=16,        # Effective batch size = 16
    
    # Learning rate — slightly lower than dense models for MoE stability
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    
    # Memory optimization — critical for MoE with 32K context
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",
    bf16=True,
    fp16=False,
    
    # Context length — leverage Mixtral's 32K when needed
    max_seq_length=4096,    # Use 4096 for most finance tasks
    # max_seq_length=16384, # Use for full annual report processing (needs more VRAM)
    
    # Logging
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    evaluation_strategy="steps",
    eval_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # Monitoring
    report_to="wandb",
    run_name="mixtral-8x7b-finance-v1",
    
    # Sequence packing (efficient for variable-length finance documents)
    packing=False,  # Disable if document lengths vary significantly
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

### Step 7: Multi-GPU Training for Large Finance Datasets

For datasets > 50K samples, use multi-GPU with DeepSpeed:

```python
# accelerate_config.yaml — for 2-GPU training
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  zero_optimization:
    stage: 2                    # ZeRO Stage 2 for MoE (Stage 3 can destabilize MoE)
    allgather_partitions: True
    reduce_scatter: True
    overlap_comm: True
  bf16:
    enabled: True
num_processes: 2
```

```bash
# Launch multi-GPU training
accelerate launch --config_file accelerate_config.yaml train_mixtral_finance.py
```

---

### Step 8: Save and Deploy

```python
# Save LoRA adapter
trainer.save_model("./mixtral-finance-adapter-final")
tokenizer.save_pretrained("./mixtral-finance-adapter-final")

# Verify adapter size
import os
adapter_size_mb = sum(
    os.path.getsize(os.path.join("./mixtral-finance-adapter-final", f))
    for f in os.listdir("./mixtral-finance-adapter-final")
) / 1e6
print(f"Adapter size: {adapter_size_mb:.0f} MB")
# Expected: 800MB – 1.5GB (larger than dense due to MoE expert coverage)

# Load for inference
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model = PeftModel.from_pretrained(base_model, "./mixtral-finance-adapter-final")

# Inference example
def finance_query(question: str, context: str) -> str:
    prompt = f"<s>[INST] {question}\n\nContext:\n{context} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,             # Deterministic for finance accuracy
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("[/INST]")[-1].strip()
```

---

## 3. Training Dataset Setup for Finance

### 3.1 Leveraging Mixtral's 32K Context Window in Dataset Design

Mixtral's longer context is a **key differentiator** — design your dataset to leverage it:

```
Dense model (Phi-4, 16K):          Mixtral 8×7B (32K context):
┌──────────────────┐                ┌──────────────────────────────────┐
│ Chunk 1 of filing │  ← Q&A        │ Full filing (up to ~30K tokens)  │ ← Q&A
│ (1-2K tokens)    │               │ with full cross-section context   │
└──────────────────┘                └──────────────────────────────────┘
│ Chunk 2 of filing │
│ (1-2K tokens)    │  ← Context lost between chunks!
└──────────────────┘
```

This means Mixtral can answer questions that require **cross-section reasoning** in a single filing — e.g., comparing risk factors in Part I with financial data in Part II — without losing context.

---

### 3.2 Finance Data Sources

#### Tier 1 — Public Structured Finance Datasets

| Dataset | Description | Size | Access |
|---|---|---|---|
| **SEC EDGAR Full-Text** | 10-K, 10-Q, 8-K filings (1993–present) | Millions of docs | Free API |
| **FinanceBench** | 10K+ Q&A pairs over real SEC filings | 10,231 samples | HuggingFace |
| **FinQA** | Multi-step reasoning over financial tables | 8,281 samples | GitHub |
| **ConvFinQA** | Multi-turn financial conversations | 3,892 samples | GitHub |
| **FLARE Benchmark Suite** | 9 finance NLP tasks | Various | HuggingFace |
| **Financial PhraseBank (FPB)** | Sentiment-labeled financial sentences | 4,845 samples | HuggingFace |
| **Earnings Call Transcripts** | Quarterly earnings call text | 10K+ | Seeking Alpha (licensed) |
| **FOMC Minutes** | Federal Reserve meeting minutes | 200+ documents | Free (federalreserve.gov) |
| **Basel III / MiFID II docs** | Regulatory frameworks | Large | Free |

#### Tier 2 — Long-Document Datasets (Optimal for Mixtral's 32K Context)

These datasets specifically benefit from Mixtral's longer context window:

```python
import requests
from sec_edgar_downloader import Downloader

def download_sec_filings_for_training(
    tickers: list[str], 
    filing_types: list[str] = ["10-K", "10-Q"],
    years: int = 5
) -> list[dict]:
    """
    Download SEC filings and prepare for long-context training.
    Takes advantage of Mixtral's 32K context - no chunking needed for 10-Qs.
    """
    dl = Downloader("YourCompany", "compliance@yourcompany.com")
    training_samples = []
    
    for ticker in tickers:
        for filing_type in filing_types:
            dl.get(filing_type, ticker, limit=years * 4)  # ~4 quarterly filings/year
    
    return training_samples
```

#### Tier 3 — Company-Specific Private Data (Highest Value)

| Data Type | Expected Volume | Privacy Risk | Recommended Treatment |
|---|---|---|---|
| Credit memos & underwriting | 100–10,000 docs | High | Anonymize + differential privacy |
| Client investment reports | 500–50,000 docs | High | Anonymize names, account #s |
| Internal research notes | 200–5,000 docs | Medium | Remove author attribution |
| Compliance SOPs | 10–200 docs | Low | Direct inclusion after review |
| Historical chat/email Q&A | 1,000–100,000 | Very High | Strict PII scrubbing required |
| Product documentation | 50–1,000 docs | Low | Direct inclusion |

---

### 3.3 PII Scrubbing Pipeline (Critical for Finance)

```python
import re
import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Load multilingual NLP model
nlp = spacy.load("en_core_web_lg")
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def scrub_financial_pii(text: str) -> str:
    """
    Remove PII from financial documents before training.
    Finance-specific entities to scrub:
    - Customer names, SSNs, account numbers
    - Tax IDs, passport numbers
    - Specific address information
    - Phone numbers, email addresses
    """
    # Presidio for standard PII
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=[
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", 
            "US_SSN", "CREDIT_CARD", "US_BANK_NUMBER",
            "US_ITIN", "US_PASSPORT",
        ],
    )
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    text = anonymized.text
    
    # Finance-specific pattern scrubbing
    patterns = {
        # Account numbers (various formats)
        r'\b\d{8,17}\b(?=\s*(account|acct))',   '[ACCOUNT_NUMBER]',
        # CUSIP (securities identifier)
        r'\b[0-9A-Z]{9}\b',                      '[CUSIP]',
        # ISIN
        r'\b[A-Z]{2}[0-9A-Z]{10}\b',             '[ISIN]',
        # Routing numbers
        r'\b[0-9]{9}\b(?=\s*(routing|ABA))',      '[ROUTING_NUMBER]',
    }
    
    for pattern, replacement in patterns.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text
```

---

### 3.4 Long-Context Document Preparation (Mixtral-Specific)

Unlike dense models requiring aggressive chunking, Mixtral can process whole documents:

```python
def prepare_long_doc_training_sample(
    filing_text: str,
    tokenizer,
    max_tokens: int = 28000,  # Leave headroom for instruction + output
) -> dict:
    """
    Prepare a full SEC filing for Mixtral's long-context training.
    Key difference from dense model prep: minimal chunking needed.
    """
    # Extract key sections (retain natural document structure)
    sections = {
        "business_overview": extract_section(filing_text, "ITEM 1"),
        "risk_factors": extract_section(filing_text, "ITEM 1A"),
        "financials": extract_section(filing_text, "ITEM 8"),
        "mda": extract_section(filing_text, "ITEM 7"),  # MD&A
    }
    
    # Combine into single context (Mixtral handles this natively)
    full_context = "\n\n".join([
        f"=== {k.upper()} ===\n{v}" 
        for k, v in sections.items() if v
    ])
    
    # Tokenize to check length
    tokens = tokenizer.encode(full_context)
    if len(tokens) > max_tokens:
        # Prioritize: Risk Factors + Financials + MD&A over Business Overview
        full_context = "\n\n".join([
            f"=== {k.upper()} ===\n{v}"
            for k, v in list(sections.items())[1:]  # Skip business overview
            if v
        ])
    
    return {
        "context": full_context,
        "token_count": len(tokenizer.encode(full_context)),
    }
```

---

### 3.5 Target Dataset Composition for Mixtral Finance Fine-Tune

For optimal results, aim for this dataset composition:

```
Dataset Composition (100K samples total):
┌─────────────────────────────────────────────────────┐
│ Long Document Q&A (10-K, 10-Q)      35% — 35,000   │ ← Mixtral's strength
│ Short Q&A / Fact Retrieval          25% — 25,000   │
│ Multi-turn Financial Conversation   15% — 15,000   │
│ Summarization Tasks                 10% — 10,000   │
│ Regulatory Compliance Q&A           10% — 10,000   │
│ Code Generation (Python/SQL)         5% —  5,000   │ ← Finance scripting
└─────────────────────────────────────────────────────┘
```

---

## 4. Hardware Requirements

### 4.1 Key Memory Consideration: MoE Storage vs. Active Compute

Mixtral's memory profile is counterintuitive:

| Metric | Value |
|---|---|
| Total parameters | 46.7B |
| Active parameters per token | ~12.9B (2 of 8 experts) |
| FP16 model size (disk/RAM) | ~93 GB |
| 4-bit quantized size (disk/RAM) | ~26 GB |
| VRAM needed for QLoRA training | ~40–48 GB |
| VRAM needed for inference only | ~24–28 GB (4-bit) |

The MoE architecture means **all 46.7B parameters must be loaded into VRAM** even though only ~13B are used per token. This is Mixtral's primary hardware challenge.

---

### 4.2 Hardware Configurations

#### Fine-Tuning (QLoRA — Recommended for Small/Mid Companies)

| Configuration | GPU | VRAM | Notes | Monthly Cloud Cost |
|---|---|---|---|---|
| **Minimum** | 2× RTX 3090 | 48 GB total | Tight fit with seq_len=2048 | N/A (on-prem) |
| **Recommended** | 1× A100 80GB | 80 GB | Comfortable single-GPU | ~$2.50/hr |
| **Optimal** | 2× A100 80GB | 160 GB | Enables seq_len=16K | ~$5/hr |
| **Fast Iteration** | 4× A100 80GB | 320 GB | seq_len=32K, fast training | ~$13/hr |
| **Budget Cloud** | 2× A10G 24GB | 48 GB | AWS g5.12xlarge ~$5.67/hr | ~$5.67/hr |

#### Full Fine-Tuning (Not Recommended for Most Finance Companies)

| Configuration | VRAM Required | Estimated Cost |
|---|---|---|
| Full BF16 + ZeRO-3 | 8× A100 80GB | ~$25/hr |
| Full BF16 + CPU offload | 4× A100 + 512GB RAM | ~$13/hr |

---

### 4.3 VRAM Breakdown for Mixtral 8×7B (QLoRA)

| Component | Estimated VRAM |
|---|---|
| Model weights (4-bit quantized) | ~26 GB |
| All-expert LoRA adapters | ~3 GB |
| Gradient checkpoints | ~4 GB |
| Optimizer state (8-bit Adam) | ~3 GB |
| Activation memory (bs=1, len=4096) | ~4 GB |
| **Total** | **~40 GB** |

This means a **single A100 80GB** is the recommended sweet spot — comfortable training with headroom for longer sequences.

---

### 4.4 Cloud Instance Reference

#### AWS

```
Best for training (single GPU): 
  ml.p4d.24xlarge (SageMaker)  - 8× A100 40GB, $32.77/hr
  g5.12xlarge                  - 4× A10G 24GB, $5.67/hr  ← Budget option

Best for inferencing:
  g5.4xlarge                   - 1× A10G 24GB, $1.62/hr  ← 4-bit inference only
  p3.8xlarge                   - 4× V100 16GB, $12.24/hr ← Better throughput
```

#### Google Cloud

```
Best for training:
  a2-highgpu-1g    - 1× A100 40GB, $3.67/hr  ← Recommended
  a2-highgpu-2g    - 2× A100 40GB, $7.35/hr
  a2-megagpu-16g   - 16× A100 40GB, $55.7/hr ← For large datasets

Best for inference:
  g2-standard-48   - 4× L4 24GB,   $2.82/hr
```

#### Azure

```
Best for training:
  Standard_NC24ads_A100_v4    - 1× A100 80GB, ~$3.67/hr ← Recommended
  Standard_ND96asr_v4         - 8× A100 40GB, ~$27.20/hr

Best for inference:
  Standard_NV36ads_A10_v5     - 1× A10 24GB,  ~$1.14/hr
```

---

### 4.5 On-Premises Hardware for Mid-Sized Finance Companies

Recommended on-premises investment for a company planning 6+ months of active fine-tuning:

| Component | Specification | Estimated Cost (2025–2026) |
|---|---|---|
| **GPU** | 2× NVIDIA A100 80GB SXM | ~$30,000–$40,000 |
| **CPU** | AMD EPYC 9474F (48 cores) | ~$6,000 |
| **RAM** | 512 GB DDR5 ECC | ~$3,000 |
| **Storage** | 8× 4TB NVMe (RAID-0 for data) | ~$4,000 |
| **Networking** | 100GbE NICs + switch | ~$2,000 |
| **Server chassis** | 4U enterprise chassis | ~$3,000 |
| **Power + cooling** | UPS + rack cooling | ~$5,000 |
| **Total** | | **~$53,000–$63,000** |

> **ROI Note:** At $5/hr cloud cost for 2× A100, the on-premises investment breaks even after approximately **12,000 hours** of training/inference, which a mid-sized company can reach in 2–3 years of active use.

---

### 4.6 Training Time Estimates

| Dataset Size | Epochs | Hardware | Estimated Duration | Cloud Cost |
|---|---|---|---|---|
| 10,000 samples | 3 | 1× A100 80GB | ~8 hours | ~$20 |
| 50,000 samples | 3 | 1× A100 80GB | ~36 hours | ~$90 |
| 100,000 samples | 3 | 2× A100 80GB | ~36 hours | ~$180 |
| 100,000 samples | 3 | 4× A100 80GB | ~20 hours | ~$200 |

---

## 5. Benchmark Testing

### 5.1 Finance-Specific Evaluation Suite

#### Primary Finance Benchmarks

| Benchmark | Evaluates | Metric | Mixtral 8×7B Baseline | Target (Fine-Tuned) |
|---|---|---|---|---|
| **FinanceBench** | SEC filing Q&A | Accuracy | ~58% | > 72% |
| **FinQA** | Multi-step numerical reasoning | Execution Accuracy | ~52% | > 65% |
| **ConvFinQA** | Conversational financial QA | Execution Accuracy | ~50% | > 63% |
| **Financial PhraseBank** | Sentiment classification | Weighted F1 | ~82% | > 90% |
| **FLARE Headline** | News classification | Avg F1 | ~79% | > 87% |
| **NER-Finance (FiNER)** | Entity extraction | F1 | ~76% | > 84% |
| **Long-doc QA (32K)** | Cross-section reasoning | Accuracy | ~60% | > 75% |

#### Mixtral-Specific Advantage: Long-Context Benchmarks

Mixtral has a unique edge for long-context finance tasks. Test with these specifically:

| Test | Description | Evaluation Method |
|---|---|---|
| Full 10-K Q&A | Q&A over complete annual report (~50K words) | Human evaluation + fact check |
| Cross-section reasoning | Questions spanning multiple filing sections | Exact match |
| Bond prospectus analysis | Full prospectus (often 100+ pages after chunking) | Expert review |
| Indenture agreement analysis | Complex legal/financial contracts | Compliance team review |

---

### 5.2 Setting Up Your Evaluation Pipeline

```python
from datasets import load_dataset
from transformers import pipeline
import json
import re

class FinanceBenchEvaluator:
    """
    Evaluate fine-tuned Mixtral on FinanceBench benchmark.
    """
    
    def __init__(self, model_path: str):
        self.pipe = pipeline(
            "text-generation",
            model=model_path,
            device_map="auto",
            torch_dtype="bfloat16",
            model_kwargs={"attn_implementation": "flash_attention_2"},
        )
        self.dataset = load_dataset("PatronusAI/financebench", split="test")
    
    def evaluate(self, n_samples: int = 500) -> dict:
        results = {"correct": 0, "total": 0, "errors": []}
        
        for sample in self.dataset.select(range(n_samples)):
            # Format using Mixtral template
            prompt = (
                f"<s>[INST] You are an expert financial analyst. "
                f"Answer the following question based solely on the provided context.\n\n"
                f"Question: {sample['question']}\n\n"
                f"Context (excerpt from {sample['doc_name']}):\n"
                f"{sample['evidence_text'][:8000]} [/INST]"
            )
            
            output = self.pipe(
                prompt,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.pipe.tokenizer.eos_token_id,
            )[0]["generated_text"]
            
            answer = output.split("[/INST]")[-1].strip()
            
            # Evaluate correctness
            is_correct = self._evaluate_answer(
                predicted=answer,
                expected=sample["answer"],
                question_type=sample.get("question_type", "factual"),
            )
            
            results["total"] += 1
            if is_correct:
                results["correct"] += 1
            else:
                results["errors"].append({
                    "question": sample["question"],
                    "predicted": answer,
                    "expected": sample["answer"],
                })
        
        results["accuracy"] = results["correct"] / results["total"]
        return results
    
    def _evaluate_answer(self, predicted: str, expected: str, question_type: str) -> bool:
        if question_type == "numerical":
            # Extract and compare numbers with 2% tolerance
            pred_nums = re.findall(r'[\d,]+\.?\d*', predicted.replace(",", ""))
            exp_nums = re.findall(r'[\d,]+\.?\d*', expected.replace(",", ""))
            if pred_nums and exp_nums:
                pred_val = float(pred_nums[0])
                exp_val = float(exp_nums[0])
                return abs(pred_val - exp_val) / (exp_val + 1e-8) < 0.02
        else:
            # Keyword overlap for factual questions
            pred_words = set(predicted.lower().split())
            exp_words = set(expected.lower().split())
            overlap = len(pred_words & exp_words) / (len(exp_words) + 1e-8)
            return overlap > 0.6

# Run evaluation
evaluator = FinanceBenchEvaluator("./mixtral-finance-adapter-final")
results = evaluator.evaluate(n_samples=500)
print(f"FinanceBench Accuracy: {results['accuracy']:.2%}")
```

---

### 5.3 Long-Context Quality Testing (Mixtral's Differentiating Benchmark)

Design tests that specifically validate Mixtral's 32K context advantage:

```python
def test_long_context_reasoning(model, tokenizer, filing_path: str):
    """
    Test Mixtral's cross-section reasoning over a full SEC filing.
    This is a unique benchmark that dense models (limited to 16K) struggle with.
    """
    with open(filing_path) as f:
        full_filing = f.read()  # May be 20K–50K tokens
    
    # Test cross-section reasoning (answers require information from multiple sections)
    cross_section_questions = [
        {
            "question": "Based on the risk factors in Part I and the financial data in Part II, "
                       "what is the largest risk that materially impacted revenue this year? "
                       "Provide the revenue impact in dollars.",
            "requires_sections": ["ITEM 1A", "ITEM 8"],
        },
        {
            "question": "Compare the company's stated strategic goals in the Business section "
                       "with the actual performance metrics in MD&A. Did they achieve their goals?",
            "requires_sections": ["ITEM 1", "ITEM 7"],
        },
    ]
    
    for qa in cross_section_questions:
        prompt = (
            f"<s>[INST] Analyze this SEC filing and answer the question. "
            f"You must reference specific data from different sections.\n\n"
            f"Filing:\n{full_filing[:28000]}\n\n"  # Use 28K of Mixtral's 32K window
            f"Question: {qa['question']} [/INST]"
        )
        
        tokens = tokenizer.encode(prompt)
        print(f"Input tokens: {len(tokens)}")  # Should be ~20-30K
        
        # ... generate and evaluate
```

---

### 5.4 Regression Testing — Preventing Catastrophic Forgetting

After finance fine-tuning, verify Mixtral hasn't lost general capabilities:

```python
REGRESSION_TESTS = [
    # Math (finance models must maintain arithmetic)
    {
        "prompt": "Calculate: Revenue $5.7M, COGS $3.2M, OpEx $1.1M. What is EBIT?",
        "expected_contains": ["1.4", "1,400"],
        "category": "arithmetic",
    },
    # Instruction following
    {
        "prompt": "List exactly 3 key financial ratios used in credit analysis.",
        "expected_has_count": 3,
        "category": "instruction_following",
    },
    # General knowledge (should not regress)
    {
        "prompt": "What does Basel III regulate?",
        "expected_contains": ["bank", "capital", "risk"],
        "category": "general_finance_knowledge",
    },
]
```

---

### 5.5 Human Evaluation Rubric

For finance applications, automated metrics are insufficient. Include human evaluation:

| Criterion | Weight | Description |
|---|---|---|
| Factual Accuracy | 40% | Are numbers, dates, and facts correct? |
| Regulatory Compliance | 25% | Does the response avoid giving unlicensed financial advice? |
| Professional Tone | 15% | Is the language appropriate for finance clients? |
| Completeness | 10% | Does the response address all parts of the question? |
| Hallucination Rate | 10% | Are claims supported by the provided context? |

---

### 5.6 A/B Testing in Production

Once deployed, run a structured A/B test before full rollout:

```
Week 1–2: Route 10% of queries to fine-tuned model
Week 3–4: Route 25% of queries to fine-tuned model
Week 5–6: Route 50% of queries (collect feedback)
Week 7+:  Full rollout if metrics meet threshold

Key metrics to track:
├── User satisfaction (thumbs up/down in UI)
├── Query escalation rate (user asks clarification = model unclear)
├── Factual error rate (compliance team spot-checks)
└── Latency (MoE models should be faster than equivalent dense models)
```

---

## 6. Alternative Models to Consider

### 6.1 Direct Competitors to Mixtral 8×7B

#### Option A: Mixtral 8×22B (Mistral AI) — If Budget Allows

| Attribute | Mixtral 8×7B | Mixtral 8×22B |
|---|---|---|
| Total params | 46.7B | 141B |
| Active params | ~12.9B | ~39B |
| Context window | 32K | 64K |
| Quality improvement | Baseline | +15–20% on most tasks |
| VRAM (4-bit inference) | ~26 GB | ~70 GB |

**Recommendation:** If your company handles complex financial instruments (structured products, derivatives) where analytical depth matters, Mixtral 8×22B is the better choice despite higher hardware costs.

---

#### Option B: Command R+ (Cohere) — For RAG-Heavy Finance Systems

| Attribute | Value |
|---|---|
| Parameters | 104B |
| Context Window | 128K tokens |
| Special Feature | Purpose-built for RAG with tool use |
| License | CC-BY-NC (non-commercial), commercial via API |
| Best For | Finance search + retrieval platforms |

**Why consider over Mixtral 8×7B:** If your finance application architecture is heavily RAG-based (retrieving from document stores, compliance databases, market data APIs), Command R+ has first-class RAG support with citation grounding — critical for auditability in regulated finance environments.

---

#### Option C: DeepSeek-V3 / R1-Distill-32B — Best Value Reasoning Model

| Attribute | Value |
|---|---|
| Parameters | 32B (distill) / 671B (full MoE) |
| Context Window | 128K tokens |
| License | MIT (R1-distills) |
| Reasoning | Exceptional — matches GPT-4o on many tasks |
| Fine-tuning | R1-Distill-32B is very practical |

**Why better than Mixtral for quant finance:** DeepSeek's chain-of-thought reasoning significantly outperforms Mixtral 8×7B on multi-step calculations, DCF analysis, option pricing explanations, and complex regulatory interpretation. For quantitative finance teams, this is usually the better open-source choice.

---

#### Option D: Llama 3.1 70B — Best Ecosystem and Tooling

| Attribute | Value |
|---|---|
| Parameters | 70B |
| Context Window | 128K tokens |
| License | Meta Llama 3.1 (commercial, free under 700M MAU) |
| Ecosystem | Largest open-source fine-tuning ecosystem |
| Tool Use | Best-in-class function calling |
| Fine-tuning | Most tutorials, frameworks, and tools available |

**Why consider over Mixtral 8×7B:** The Llama ecosystem is substantially larger. More pre-built finance fine-tuning datasets, adapters, and deployment tools are available for Llama. If your team is less experienced with LLMs, Llama 3.1 70B is significantly easier to work with.

---

### 6.2 Finance-Specific Vertical Models

#### FinGPT (OpenSourceFinAI) — Pre-adapted, Zero Fine-Tuning Needed

```python
# FinGPT is already finance-adapted — just load and use
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("oliverwang15/FinGPT_v33_Llama2_7B_Finetuned")
```

**Best for:** Sentiment analysis, financial NER, news summarization
**Limitation:** Based on older Llama 2 — weaker than modern fine-tuned Mixtral on complex tasks

#### BloombergGPT — Enterprise Finance (Closed Source)

- Trained on 363B finance-specific tokens
- Access via Bloomberg Terminal API only
- Best-in-class for Bloomberg data integration
- **Not viable for self-hosting or fine-tuning**

#### Palmyra-Fin (Writer) — Regulated Finance Focus

- Purpose-built for regulated finance and compliance
- Strong on financial document understanding
- Commercial API access required

---

### 6.3 Final Model Selection Decision Tree

```
Primary Use Case?
│
├── LONG DOCUMENTS (> 16K tokens: full 10-K, bond prospectus)
│   ├── Budget < $30K/year  → Mixtral 8×7B (fine-tuned) ✓
│   └── Budget > $30K/year  → Mixtral 8×22B or Llama 3.1 70B
│
├── COMPLEX REASONING (DCF, options, quantitative analysis)
│   ├── Open-source needed  → DeepSeek-R1-Distill-32B ✓
│   └── API acceptable      → GPT-4o or Claude 3.5 Sonnet
│
├── HIGH-VOLUME QUERIES (> 100K queries/day)
│   ├── Cost is primary     → Mixtral 8×7B (best throughput/cost MoE ratio)
│   └── Quality is primary  → Qwen2.5-72B
│
├── RAG / SEARCH SYSTEM
│   ├── Self-hosted needed  → Llama 3.1 70B with LlamaIndex ✓
│   └── Managed service OK  → Command R+ (Cohere) 
│
├── MULTILINGUAL (EU, Asian markets)
│   └── Qwen2.5-72B (best multilingual finance support)
│
└── STANDARD NLP (sentiment, NER, classification)
    └── Llama 3.1 8B fine-tuned (cheapest, sufficient quality)
```

---

### 6.4 Comprehensive Comparison Table

| Model | Params (Active) | Context | Open Source | Finance Fine-Tune Ease | Long-Doc | Reasoning | TCO (Annual, Mid Company) |
|---|---|---|---|---|---|---|---|
| **Mixtral 8×7B** | 12.9B | 32K | ✅ Apache 2.0 | Medium | ★★★★☆ | ★★★☆☆ | ~$30K–$50K |
| **Phi-4** | 14B | 16K | ✅ MIT | Easy | ★★★☆☆ | ★★★★☆ | ~$25K–$40K |
| **Llama 3.1 8B** | 8B | 128K | ✅ Meta License | Easy | ★★★☆☆ | ★★★☆☆ | ~$5K–$10K |
| **Llama 3.1 70B** | 70B | 128K | ✅ Meta License | Medium | ★★★★☆ | ★★★★☆ | ~$60K–$90K |
| **Qwen2.5-72B** | 72B | 128K | ✅ Qwen License | Medium | ★★★★☆ | ★★★★★ | ~$80K–$110K |
| **DeepSeek-R1-32B** | 32B | 128K | ✅ MIT | Medium | ★★★★☆ | ★★★★★ | ~$40K–$60K |
| **Mixtral 8×22B** | 39B | 64K | ✅ Apache 2.0 | Hard | ★★★★★ | ★★★★☆ | ~$90K–$120K |
| **GPT-4o** | Closed | 128K | ❌ API Only | N/A | ★★★★★ | ★★★★★ | $50K–$300K+ |

---

## Summary & Recommendations

### When to Choose Mixtral 8×7B

**Mixtral 8×7B is the right choice if your company:**

- Processes long financial documents (annual reports, bond prospectuses, indenture agreements) that require 16K–32K context
- Needs cost-efficient high-throughput inference for client-facing applications
- Operates across multiple language markets (EU, Asian finance)
- Requires code generation for financial scripts alongside NLP tasks
- Values the Apache 2.0 license with zero restriction concerns

**Mixtral 8×7B is NOT ideal if your company:**

- Primarily needs short Q&A under 4K context (use Phi-4 or Llama 3.1 8B instead — cheaper)
- Has very limited hardware (< 40 GB VRAM available — Phi-4 runs on 24 GB)
- Needs maximum reasoning depth for quantitative finance (use DeepSeek-R1-32B instead)

### Company Size Guidance

| Company Profile | Recommended Model | Rationale |
|---|---|---|
| Small (< 50 employees, < $10K budget) | Llama 3.1 8B fine-tuned | Lowest TCO, sufficient for basic tasks |
| Small-Mid (analytics-focused, < $30K) | **Phi-4 fine-tuned** | Best reasoning-per-dollar, easy setup |
| Mid (document-heavy, < $60K) | **Mixtral 8×7B fine-tuned** | Long-context advantage justifies cost |
| Mid-Large (multi-language, < $100K) | Qwen2.5-72B or Mixtral 8×22B | Maximum quality open-source |
| Large (> $100K budget) | Multiple specialized models | Route by task to optimize cost/quality |

---

*Last updated: March 2026 | Based on model releases and benchmarks through Q1 2026*
