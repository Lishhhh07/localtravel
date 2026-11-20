"""Local Travel Guide Generator - Core Module.
Implements dataset preparation, LoRA adapter injection, fine-tuning, evaluation, and weight merging.
"""
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
)

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "travel_guides.json"
DEFAULT_MODEL_NAME = "gpt2"
DEFAULT_ADAPTER_DIR = Path(__file__).resolve().parent.parent / "models" / "lora_adapter"
DEFAULT_MERGED_DIR = Path(__file__).resolve().parent.parent / "models" / "merged_model"


def set_seed(seed: int = 42) -> None:
    """Set global random seed across Python, NumPy, and PyTorch for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Initialize global seed on import
set_seed(42)



# =====================================================================
# Milestone 1: Dataset Preparation
# =====================================================================

def load_travel_dataset(path: Optional[str | Path] = None) -> List[Dict[str, str]]:
    """Load the travel dataset containing fictional city recommendations."""
    target_path = Path(path) if path else DATA_FILE
    if not target_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {target_path}")
    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Dataset must be a non-empty list of items.")
    for idx, item in enumerate(data):
        for key in ("destination", "interest", "prompt", "completion"):
            if key not in item or not item[key].strip():
                raise ValueError(f"Item at index {idx} is missing required field '{key}'")
    return data


def format_travel_prompt(destination: str, interests: str) -> str:
    """Format input prompt for travel mini-guide generation."""
    dest = (destination or "Port Luminara").strip()
    inte = (interests or "Local Highlights").strip()
    return f"Destination: {dest}\nInterests: {inte}\nGenerate a concise, personalized travel mini-guide with top recommendations:\nGuide:"


def milestone_1() -> Dict[str, Any]:
    """Milestone 1: Prepare and validate travel dataset for fictional city Port Luminara."""
    data = load_travel_dataset()
    cities = sorted(list({item["destination"] for item in data}))
    interests = [item["interest"] for item in data]
    return {
        "status": "completed",
        "dataset_path": str(DATA_FILE),
        "total_samples": len(data),
        "destinations": cities,
        "sample_interests": interests[:5],
        "sample": data[0],
    }


# =====================================================================
# Milestone 2: LoRA Adapter Injection & Training
# =====================================================================

def get_device(preferred_device: Optional[str] = None) -> str:
    """Determine best available compute device."""
    if preferred_device:
        return preferred_device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def inject_lora_adapter(
    base_model_name_or_path: str = DEFAULT_MODEL_NAME,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> Tuple[PeftModel, PreTrainedTokenizer, Dict[str, Any]]:
    """Inject LoRA adapter weights into a pre-trained causal language model (e.g. GPT-2)."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules or ["c_attn"],
        fan_in_fan_out=True,
    )

    model = get_peft_model(base_model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    percent_trainable = 100.0 * trainable_params / all_params

    stats = {
        "base_model": base_model_name_or_path,
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules or ["c_attn"],
        "trainable_params": trainable_params,
        "all_params": all_params,
        "percent_trainable": round(percent_trainable, 4),
    }

    return model, tokenizer, stats


def train_lora_adapter(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Optional[List[Dict[str, str]]] = None,
    output_dir: Optional[str | Path] = None,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 5e-4,
    max_steps: Optional[int] = None,
    device: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train the injected LoRA adapter on travel guide input-output pairs."""
    set_seed(seed)
    data = dataset or load_travel_dataset()
    out_dir = Path(output_dir or DEFAULT_ADAPTER_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = get_device(device)
    model.to(dev)

    # Prepare training sequences: Prompt + Completion + EOS
    formatted_texts = [
        f"{item['prompt']}\n{item['completion']}{tokenizer.eos_token}"
        for item in data
    ]

    encodings = tokenizer(
        formatted_texts,
        padding=True,
        truncation=True,
        max_length=192,
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses: List[float] = []

    model.train()
    step_count = 0
    start_time = time.time()

    for epoch in range(epochs):
        permutation = torch.randperm(len(input_ids))
        for i in range(0, len(input_ids), batch_size):
            indices = permutation[i : i + batch_size]
            b_ids = input_ids[indices].to(dev)
            b_mask = attention_mask[indices].to(dev)
            b_labels = labels[indices].to(dev)

            outputs = model(
                input_ids=b_ids,
                attention_mask=b_mask,
                labels=b_labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_val = float(loss.item())
            losses.append(loss_val)
            step_count += 1

            if max_steps and step_count >= max_steps:
                break
        if max_steps and step_count >= max_steps:
            break

    elapsed = round(time.time() - start_time, 2)

    # Save adapter weights and tokenizer
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Also record training info
    meta_file = out_dir / "training_meta.json"
    meta = {
        "epochs": epochs,
        "steps": step_count,
        "elapsed_seconds": elapsed,
        "final_loss": round(losses[-1], 4) if losses else None,
        "initial_loss": round(losses[0], 4) if losses else None,
        "device": dev,
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "status": "completed",
        "output_dir": str(out_dir),
        "steps": step_count,
        "elapsed_seconds": elapsed,
        "initial_loss": meta["initial_loss"],
        "final_loss": meta["final_loss"],
        "losses": losses,
    }


def milestone_2(
    epochs: int = 2,
    max_steps: Optional[int] = None,
    output_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Milestone 2: Inject LoRA adapter on GPT-2 and train on travel dataset."""
    model, tokenizer, stats = inject_lora_adapter()
    dataset = load_travel_dataset()
    train_res = train_lora_adapter(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=output_dir,
        epochs=epochs,
        max_steps=max_steps,
        device=device,
    )
    return {
        "status": "completed",
        "stats": stats,
        "training": train_res,
    }


# =====================================================================
# Milestone 3: Mini-Guide Generation & Streamlit UI logic
# =====================================================================

_CACHED_MODELS: Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer]] = {}


def load_generator_model(
    model_mode: str = "lora",
    base_model_name: str = DEFAULT_MODEL_NAME,
    adapter_dir: Optional[str | Path] = None,
    merged_dir: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load and cache model and tokenizer for inference in base, lora, or merged mode."""
    mode = model_mode.lower()
    dev = get_device(device)
    cache_key = f"{mode}_{dev}"

    if cache_key in _CACHED_MODELS:
        return _CACHED_MODELS[cache_key]

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if mode == "base":
        model = AutoModelForCausalLM.from_pretrained(base_model_name)
    elif mode == "merged":
        m_dir = Path(merged_dir or DEFAULT_MERGED_DIR)
        if m_dir.exists() and (m_dir / "config.json").exists():
            model = AutoModelForCausalLM.from_pretrained(str(m_dir))
            tokenizer = AutoTokenizer.from_pretrained(str(m_dir))
        else:
            # Fallback if merged not yet saved: merge in memory
            base = AutoModelForCausalLM.from_pretrained(base_model_name)
            a_dir = Path(adapter_dir or DEFAULT_ADAPTER_DIR)
            if not a_dir.exists():
                raise FileNotFoundError(f"LoRA adapter directory not found at: {a_dir}")
            peft_m = PeftModel.from_pretrained(base, str(a_dir))
            model = peft_m.merge_and_unload()
    else:  # "lora"
        a_dir = Path(adapter_dir or DEFAULT_ADAPTER_DIR)
        if not a_dir.exists():
            raise FileNotFoundError(f"LoRA adapter directory not found at: {a_dir}")
        base = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base, str(a_dir))

    model.to(dev)
    model.eval()
    _CACHED_MODELS[cache_key] = (model, tokenizer)
    return model, tokenizer


def clean_generated_guide(raw_text: str, prompt: str) -> str:
    """Extract and format the generated travel guide text cleanly."""
    if raw_text.startswith(prompt):
        generated = raw_text[len(prompt):].strip()
    else:
        # Fallback split on Guide:
        parts = raw_text.split("Guide:", 1)
        generated = parts[1].strip() if len(parts) > 1 else raw_text.strip()

    # If subsequent prompt marker appears, trim it
    for stop_seq in ["Destination:", "\n\nDestination:", "Interests:"]:
        if stop_seq in generated:
            generated = generated.split(stop_seq)[0].strip()

    return generated


def generate_travel_guide(
    destination: str,
    interests: str,
    model_mode: str = "lora",
    max_new_tokens: int = 140,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    device: Optional[str] = None,
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    adapter_dir: Optional[str | Path] = None,
    merged_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Generate a personalized travel mini-guide given destination and interests."""
    prompt = format_travel_prompt(destination, interests)
    dev = get_device(device)

    if model is None or tokenizer is None:
        model, tokenizer = load_generator_model(
            model_mode=model_mode,
            adapter_dir=adapter_dir,
            merged_dir=merged_dir,
            device=dev,
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(dev)
    prompt_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else 1.0,
            top_p=top_p if temperature > 0.0 else 1.0,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    latency_ms = round((time.time() - t0) * 1000, 2)

    generated_ids = output_ids[0]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    guide_text = clean_generated_guide(raw_output, prompt)

    return {
        "status": "completed",
        "destination": destination,
        "interests": interests,
        "prompt": prompt,
        "raw_output": raw_output,
        "guide": guide_text,
        "model_mode": model_mode,
        "tokens_generated": len(generated_ids) - prompt_len,
        "latency_ms": latency_ms,
    }


def milestone_3(
    destination: str = "Port Luminara",
    interests: str = "Artisan Coffee & Bookstores",
    model_mode: str = "lora",
    max_new_tokens: int = 80,
    device: Optional[str] = None,
    adapter_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Milestone 3: Generate a personalized mini-guide for tourist interests."""
    res = generate_travel_guide(
        destination=destination,
        interests=interests,
        model_mode=model_mode,
        max_new_tokens=max_new_tokens,
        temperature=0.6,
        device=device,
        adapter_dir=adapter_dir,
    )
    return res


# =====================================================================
# Milestone 4: Evaluation Framework
# =====================================================================

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "for", "on", "with",
    "at", "by", "from", "up", "about", "into", "over", "after", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "but", "if", "then", "else", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now"
}


def extract_keywords(text: str) -> List[str]:
    """Tokenize and extract significant alphanumeric keywords ignoring common stopwords."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS]


TRAVEL_CONCEPT_MAP = {
    "coffee": {"cafe", "roast", "espresso", "beans", "chicory", "brew", "barista"},
    "bookstores": {"books", "book", "archive", "library", "manuscript", "reading", "parchment", "poet"},
    "books": {"bookstores", "archive", "library", "parchment", "poet"},
    "jazz": {"music", "trumpet", "saxophone", "quartet", "acoustic", "melodies", "brass"},
    "canals": {"canal", "water", "gondola", "oar", "quay", "boat", "waterway"},
    "canal": {"canals", "water", "gondola", "oar", "quay", "boat"},
    "culinary": {"food", "market", "bread", "dishes", "gastronomy", "squid", "bistro", "bakery", "seafood", "dining"},
    "food": {"culinary", "market", "dishes", "gastronomy", "bakery", "treat"},
    "architecture": {"basilica", "vault", "stone", "cloister", "facade", "column", "arch", "carved"},
    "hiking": {"trails", "trail", "cliffs", "cliff", "ridge", "falcon", "scenic", "pines"},
    "spa": {"baths", "thermal", "grotto", "springs", "soaking", "wellness", "scrub"},
    "wellness": {"baths", "thermal", "grotto", "springs", "soaking", "spa", "scrub"},
    "art": {"glass", "studio", "atelier", "kiln", "craft", "painting", "sculpture"},
    "nightlife": {"lantern", "speakeasy", "cocktails", "midnight", "docks", "music"},
}


def compute_relevance_score(guide: str, destination: str, interests: str) -> float:
    """Quantitative relevance metric: keyword overlap and domain concept alignment between request and guide."""
    if not guide or not guide.strip():
        return 0.0

    guide_words = set(extract_keywords(guide))
    interest_words = set(extract_keywords(interests))
    dest_words = set(extract_keywords(destination))

    query_words = interest_words.union(dest_words)
    if not query_words:
        return 0.5

    # Direct overlap
    overlap = set(query_words.intersection(guide_words))

    # Concept expansion matching
    for q in query_words:
        synonyms = TRAVEL_CONCEPT_MAP.get(q, set())
        if synonyms.intersection(guide_words):
            overlap.add(q)
        else:
            # Substring / stem check
            for gw in guide_words:
                if len(q) >= 4 and len(gw) >= 4 and (q in gw or gw in q):
                    overlap.add(q)
                    break

    score = len(overlap) / len(query_words)
    return round(min(1.0, score * 1.2), 4)



def compute_appeal_score(guide: str) -> Dict[str, float]:
    """Quantitative & qualitative assessment of guide appeal, formatting, and actionability."""
    if not guide or not guide.strip():
        return {
            "structure_score": 0.0,
            "actionability_score": 0.0,
            "conciseness_score": 0.0,
            "overall_appeal": 0.0,
        }

    guide_lower = guide.lower()

    # 1. Structural Formatting Score (Checks for markdown headers, bullet points, key sections)
    structure_hits = 0
    if "###" in guide or "mini-guide" in guide_lower:
        structure_hits += 1
    if "-" in guide or "*" in guide:
        structure_hits += 1
    if "must-visit" in guide_lower or "visit" in guide_lower:
        structure_hits += 1
    if "secret" in guide_lower or "tip" in guide_lower or "spot" in guide_lower:
        structure_hits += 1
    if "best time" in guide_lower or "am" in guide_lower or "pm" in guide_lower or "morning" in guide_lower or "sunset" in guide_lower:
        structure_hits += 1
    structure_score = round(structure_hits / 5.0, 4)

    # 2. Actionability & Specificity Score (Concrete nouns, timing, recommendations)
    time_mentions = len(re.findall(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", guide_lower))
    named_entities = len(re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", guide))
    actionability_score = round(min(1.0, (structure_score * 0.4) + (0.2 * min(2, time_mentions)) + (0.05 * min(8, named_entities))), 4)

    # 3. Conciseness Score (Optimal mini-guide: 35 - 180 words without runaway loops)
    word_count = len(guide.split())
    if word_count < 15:
        conciseness_score = 0.3
    elif 30 <= word_count <= 180:
        conciseness_score = 1.0
    elif 180 < word_count <= 250:
        conciseness_score = 0.8
    else:
        conciseness_score = 0.5

    # Overall appeal (Scale 1 to 5)
    composite = (structure_score * 0.4) + (actionability_score * 0.4) + (conciseness_score * 0.2)
    overall_appeal = round(1.0 + (composite * 4.0), 2)

    return {
        "structure_score": structure_score,
        "actionability_score": actionability_score,
        "conciseness_score": conciseness_score,
        "overall_appeal": overall_appeal,
    }


def evaluate_guide(guide: str, destination: str, interests: str) -> Dict[str, Any]:
    """Evaluate a single generated guide across all defined criteria."""
    rel = compute_relevance_score(guide, destination, interests)
    appeal_dict = compute_appeal_score(guide)
    return {
        "relevance_score": rel,
        "structure_score": appeal_dict["structure_score"],
        "actionability_score": appeal_dict["actionability_score"],
        "conciseness_score": appeal_dict["conciseness_score"],
        "overall_appeal": appeal_dict["overall_appeal"],
    }


def run_comparative_evaluation(
    test_cases: Optional[List[Dict[str, str]]] = None,
    num_samples: int = 4,
    max_new_tokens: int = 90,
    device: Optional[str] = None,
    adapter_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run evaluation benchmark comparing Base GPT-2 vs. LoRA fine-tuned model."""
    dataset = load_travel_dataset()
    cases = test_cases or dataset[:num_samples]

    dev = get_device(device)
    base_model, base_tok = load_generator_model("base", device=dev)
    lora_model, lora_tok = load_generator_model("lora", adapter_dir=adapter_dir, device=dev)

    base_results: List[Dict[str, Any]] = []
    lora_results: List[Dict[str, Any]] = []

    for item in cases:
        dest = item["destination"]
        inte = item["interest"]

        # Base generation
        b_res = generate_travel_guide(
            destination=dest,
            interests=inte,
            model_mode="base",
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            model=base_model,
            tokenizer=base_tok,
            device=dev,
        )
        b_eval = evaluate_guide(b_res["guide"], dest, inte)
        base_results.append({**b_res, **b_eval})

        # LoRA generation
        l_res = generate_travel_guide(
            destination=dest,
            interests=inte,
            model_mode="lora",
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            model=lora_model,
            tokenizer=lora_tok,
            device=dev,
        )
        l_eval = evaluate_guide(l_res["guide"], dest, inte)
        lora_results.append({**l_res, **l_eval})

    def avg(lst: List[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    base_summary = {
        "avg_relevance": avg([r["relevance_score"] for r in base_results]),
        "avg_structure": avg([r["structure_score"] for r in base_results]),
        "avg_actionability": avg([r["actionability_score"] for r in base_results]),
        "avg_overall_appeal": avg([r["overall_appeal"] for r in base_results]),
        "avg_latency_ms": avg([r["latency_ms"] for r in base_results]),
    }

    lora_summary = {
        "avg_relevance": avg([r["relevance_score"] for r in lora_results]),
        "avg_structure": avg([r["structure_score"] for r in lora_results]),
        "avg_actionability": avg([r["actionability_score"] for r in lora_results]),
        "avg_overall_appeal": avg([r["overall_appeal"] for r in lora_results]),
        "avg_latency_ms": avg([r["latency_ms"] for r in lora_results]),
    }

    return {
        "status": "completed",
        "num_samples": len(cases),
        "base_model": base_summary,
        "lora_model": lora_summary,
        "sample_comparisons": [
            {
                "destination": cases[i]["destination"],
                "interest": cases[i]["interest"],
                "base_guide": base_results[i]["guide"],
                "base_appeal": base_results[i]["overall_appeal"],
                "lora_guide": lora_results[i]["guide"],
                "lora_appeal": lora_results[i]["overall_appeal"],
            }
            for i in range(len(cases))
        ],
    }


def milestone_4(
    num_samples: int = 2,
    max_new_tokens: int = 80,
    device: Optional[str] = None,
    adapter_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Milestone 4: Evaluate relevance and appeal of generated recommendations."""
    return run_comparative_evaluation(
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        device=device,
        adapter_dir=adapter_dir,
    )


# =====================================================================
# Milestone 5: Adapter Merging for Inference
# =====================================================================

def merge_adapter_weights(
    base_model_name_or_path: str = DEFAULT_MODEL_NAME,
    adapter_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Dict[str, Any]]:
    """Merge trained LoRA adapter weights directly into base model weights for zero-overhead inference."""
    a_dir = Path(adapter_path or DEFAULT_ADAPTER_DIR)
    o_dir = Path(output_path or DEFAULT_MERGED_DIR)
    o_dir.mkdir(parents=True, exist_ok=True)
    dev = get_device(device)

    tok_path = str(a_dir) if (a_dir / "tokenizer_config.json").exists() else base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
    peft_model = PeftModel.from_pretrained(base_model, str(a_dir))

    # Merge weights into base architecture and unload PEFT wrappers
    merged_model = peft_model.merge_and_unload()
    merged_model.to(dev)
    merged_model.eval()

    # Save standalone merged model
    merged_model.save_pretrained(str(o_dir))
    tokenizer.save_pretrained(str(o_dir))

    stats = {
        "status": "completed",
        "output_dir": str(o_dir),
        "base_model": base_model_name_or_path,
        "is_peft_model": isinstance(merged_model, PeftModel),
        "num_parameters": sum(p.numel() for p in merged_model.parameters()),
    }
    return merged_model, tokenizer, stats


def benchmark_inference_efficiency(
    prompt_text: Optional[str] = None,
    num_runs: int = 3,
    max_new_tokens: int = 50,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Benchmark generation latency between merged model and dynamic LoRA adapter model."""
    dev = get_device(device)
    prompt = prompt_text or format_travel_prompt("Port Luminara", "Artisan Coffee & Bookstores")

    # Load adapter model
    lora_model, lora_tok = load_generator_model("lora", device=dev)
    # Load merged model
    merged_model, merged_tok = load_generator_model("merged", device=dev)

    lora_latencies = []
    for _ in range(num_runs):
        res = generate_travel_guide(
            destination="Port Luminara",
            interests="Artisan Coffee & Bookstores",
            model_mode="lora",
            max_new_tokens=max_new_tokens,
            model=lora_model,
            tokenizer=lora_tok,
            device=dev,
        )
        lora_latencies.append(res["latency_ms"])

    merged_latencies = []
    for _ in range(num_runs):
        res = generate_travel_guide(
            destination="Port Luminara",
            interests="Artisan Coffee & Bookstores",
            model_mode="merged",
            max_new_tokens=max_new_tokens,
            model=merged_model,
            tokenizer=merged_tok,
            device=dev,
        )
        merged_latencies.append(res["latency_ms"])

    avg_lora = round(sum(lora_latencies) / len(lora_latencies), 2)
    avg_merged = round(sum(merged_latencies) / len(merged_latencies), 2)
    speedup_pct = round(((avg_lora - avg_merged) / avg_lora) * 100, 2) if avg_lora > 0 else 0.0

    return {
        "num_runs": num_runs,
        "avg_lora_latency_ms": avg_lora,
        "avg_merged_latency_ms": avg_merged,
        "speedup_percent": speedup_pct,
    }


def milestone_5(
    adapter_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Milestone 5: Merge trained LoRA adapter weights and verify standalone inference."""
    merged_model, tokenizer, stats = merge_adapter_weights(
        adapter_path=adapter_path,
        output_path=output_path,
        device=device,
    )
    # Perform test generation with merged model
    sample_gen = generate_travel_guide(
        destination="Port Luminara",
        interests="Starlight Canals & Jazz",
        model_mode="merged",
        max_new_tokens=50,
        model=merged_model,
        tokenizer=tokenizer,
        device=device,
    )
    return {
        "status": "completed",
        "stats": stats,
        "sample_generation": sample_gen,
    }

