"""Unit and integration test suite for Local Travel Guide Generator (Milestones 1-5)."""
import pytest

# Milestone 1: Prepare a small dataset of travel interests and corresponding local recommendations for a fictional city.
def test_milestone_1():
    from app.core import milestone_1, load_travel_dataset, format_travel_prompt

    res = milestone_1()
    assert res["status"] == "completed"
    assert res["total_samples"] >= 15
    assert "Port Luminara" in res["destinations"]
    assert "sample" in res
    assert "prompt" in res["sample"]
    assert "completion" in res["sample"]

    # Verify formatting helper
    prompt = format_travel_prompt("Port Luminara", "Artisan Coffee")
    assert "Destination: Port Luminara" in prompt
    assert "Interests: Artisan Coffee" in prompt

    # Verify full dataset content
    dataset = load_travel_dataset()
    assert len(dataset) == res["total_samples"]
    for row in dataset:
        assert row["destination"] == "Port Luminara"
        assert len(row["interest"]) > 3
        assert len(row["completion"]) > 20
        assert "Guide:" in row["prompt"]



# Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).
def test_milestone_2(tmp_path):
    from app.core import inject_lora_adapter, train_lora_adapter, milestone_2

    model, tokenizer, stats = inject_lora_adapter("gpt2", r=4, lora_alpha=8)
    assert stats["base_model"] == "gpt2"
    assert stats["r"] == 4
    assert stats["trainable_params"] > 0
    # LoRA trains only a small fraction of all weights (< 1%)
    assert stats["percent_trainable"] < 1.0
    assert "c_attn" in stats["target_modules"]

    # Test short training run
    out_dir = tmp_path / "test_lora_adapter"
    res = milestone_2(epochs=1, max_steps=2, output_dir=str(out_dir), device="cpu")
    assert res["status"] == "completed"
    assert (out_dir / "adapter_config.json").exists()
    assert res["training"]["steps"] == 2
    assert res["training"]["final_loss"] is not None



# Milestone 3: Develop a Streamlit UI to input interests and destination, generating a mini-guide.
def test_milestone_3():
    import py_compile
    from app.core import milestone_3, clean_generated_guide, format_travel_prompt

    # Test py_compile on app/main.py
    py_compile.compile("app/main.py", doraise=True)

    # Test clean_generated_guide helper
    dummy_prompt = format_travel_prompt("Port Luminara", "Artisan Coffee")
    raw = f"{dummy_prompt}\n### Mini-Guide: Top Coffee\n- Visit Roast & Parchment\nDestination: next"
    cleaned = clean_generated_guide(raw, dummy_prompt)
    assert "Mini-Guide: Top Coffee" in cleaned
    assert "Destination: next" not in cleaned

    # Test milestone 3 generation logic
    res = milestone_3(
        destination="Port Luminara",
        interests="Artisan Coffee & Bookstores",
        model_mode="lora",
        max_new_tokens=40,
        device="cpu",
    )
    assert res["status"] == "completed"
    assert res["destination"] == "Port Luminara"
    assert "guide" in res
    assert len(res["guide"]) > 0
    assert "latency_ms" in res



# Milestone 4: Evaluate the relevance and appeal of the generated recommendations.
def test_milestone_4():
    from app.core import (
        milestone_4,
        compute_relevance_score,
        compute_appeal_score,
        evaluate_guide,
    )

    # Test scoring functions directly
    sample_guide = (
        "### Mini-Guide: Cozy Haven\n"
        "- **Must-Visit**: Roast & Parchment Cafe on Belltower Hill\n"
        "- **Secret Tip**: Ask for the poet's balcony\n"
        "- **Best Time**: 8:30 AM to 11:00 AM"
    )
    rel = compute_relevance_score(sample_guide, "Port Luminara", "Artisan Coffee & Bookstores")
    assert rel > 0.0

    appeal = compute_appeal_score(sample_guide)
    assert appeal["structure_score"] >= 0.8
    assert appeal["overall_appeal"] >= 3.5

    # Test evaluation on empty string handles gracefully
    empty_eval = evaluate_guide("", "Port Luminara", "Artisan Coffee")
    assert empty_eval["relevance_score"] == 0.0
    assert empty_eval["overall_appeal"] == 0.0

    # Run milestone 4 evaluation
    res = milestone_4(num_samples=2, max_new_tokens=60, device="cpu")
    assert res["status"] == "completed"
    assert "base_model" in res
    assert "lora_model" in res
    assert res["num_samples"] == 2
    assert "sample_comparisons" in res


# Milestone 5: Implement adapter merging for inference.
def test_milestone_5(tmp_path):
    from app.core import (
        milestone_5,
        merge_adapter_weights,
        benchmark_inference_efficiency,
        load_generator_model,
        generate_travel_guide,
    )
    from peft import PeftModel

    merged_out = tmp_path / "test_merged_model"
    merged_m, tokenizer, stats = merge_adapter_weights(
        output_path=str(merged_out),
        device="cpu",
    )
    assert stats["status"] == "completed"
    assert not stats["is_peft_model"]
    assert not isinstance(merged_m, PeftModel)
    assert (merged_out / "config.json").exists()

    # Test milestone 5 function
    res = milestone_5(output_path=str(merged_out), device="cpu")
    assert res["status"] == "completed"
    assert "sample_generation" in res
    assert res["sample_generation"]["status"] == "completed"
    assert len(res["sample_generation"]["guide"]) > 0


