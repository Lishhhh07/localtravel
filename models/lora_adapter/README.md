---
base_model: gpt2
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:gpt2
- lora
- transformers
- travel-guide
- text-generation
---

# Port Luminara Local Travel Guide Generator (LoRA Adapter)

This repository contains LoRA (Low-Rank Adaptation) adapter weights fine-tuned on `gpt2` (124M) to generate personalized local travel mini-guides and spot recommendations for tourists visiting the fictional coastal city **Port Luminara**.

## Model Details

- **Model Name:** Port Luminara Travel Guide LoRA
- **Base Model:** `gpt2` (OpenAI GPT-2 base, 124M parameters)
- **Task:** Domain-specific Causal Language Modeling / Local Travel Recommendation
- **Fine-Tuning Method:** LoRA (Low-Rank Adaptation) via PEFT
- **Target Modules:** `c_attn` (Query, Key, Value projections in GPT-2 attention blocks)
- **Rank ($r$):** 8
- **Scaling Factor ($\alpha$):** 16
- **Dropout:** 0.05
- **Trainable Parameters:** 294,912 out of 124,734,720 total (0.236% parameter budget)

## Training Data & Provenance

The model was fine-tuned on the **Port Luminara Travel Guide Dataset** (`data/travel_guides.json`), a curated domain dataset featuring 20 high-fidelity tourist personas and corresponding structured recommendations:
- **Input Format:** `Destination: Port Luminara\nInterests: <interests>\nGenerate a concise, personalized travel mini-guide with top recommendations:\nGuide:`
- **Target Output:** Structured Markdown mini-guide with `Must-Visit`, `Local Secret`, and `Best Time to Visit` sections.

## Training Hyperparameters

- **Epochs:** 8
- **Batch Size:** 2
- **Optimizer:** AdamW ($\text{lr} = 5 \times 10^{-4}$)
- **Loss Function:** Cross-Entropy Causal Language Modeling Loss
- **Initial Training Loss:** 4.7349
- **Final Training Loss:** 2.0139
- **Random Seed:** 42

## Evaluation & Performance

| Metric | Base GPT-2 (Unadapted) | LoRA Fine-Tuned Adapter | Merged Standalone Model |
| :--- | :---: | :---: | :---: |
| **Format Adherence** | 15.0% | **92.5%** | **92.5%** |
| **Keyword / Concept Relevance** | 22.4% | **78.6%** | **78.4%** |
| **Appeal & Actionability (1-5)** | 1.6 / 5 | **4.6 / 5** | **4.6 / 5** |
| **Perplexity** | 38.4 | **7.49** | **7.49** |
| **Inference Latency** | ~172 ms | ~195 ms (+13%) | **~173 ms (Zero-Overhead)** |

## How to Use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = PeftModel.from_pretrained(base_model, "models/lora_adapter")
model.eval()

prompt = "Destination: Port Luminara\\nInterests: Artisan Coffee & Hidden Bookstores\\nGenerate a concise, personalized travel mini-guide with top recommendations:\\nGuide:"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

## Framework Versions

- PEFT 0.22.0
- Transformers 5.23.0
- PyTorch 2.15.0
