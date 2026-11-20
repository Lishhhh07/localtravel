# 🧭 Local Travel Guide Generator

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-LoRA-orange)
![Model](https://img.shields.io/badge/Model-GPT--2-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
[![Tests](https://github.com/Lishhhh07/localtravel/actions/workflows/ci.yml/badge.svg)](https://github.com/Lishhhh07/localtravel/actions)

An AI-powered local travel guide generator that uses **Low-Rank Adaptation (LoRA)** to fine-tune GPT-2 for generating structured, personalized travel recommendations.

The project explores how parameter-efficient fine-tuning can adapt a relatively small language model to a specialized domain without updating all of its original parameters.

It includes a synthetic travel dataset, a LoRA training pipeline, a Streamlit application, model evaluation, baseline comparisons, and merged-model inference.

---

## 1. Project Overview

Travel recommendations are often generic and fail to reflect individual interests. This project explores a lightweight approach to generating more structured and personalized recommendations.

The system accepts a destination and traveler interests, then generates a concise travel mini-guide containing:

- Recommended places and landmarks
- Interest-specific activities
- Local tips and hidden attractions
- Suggested visiting times

The model is adapted using LoRA rather than full-model fine-tuning, reducing the number of trainable parameters and the size of the trained adapter.

**Primary objective:** Demonstrate the complete workflow of preparing a domain-specific dataset, fine-tuning a pretrained language model, evaluating its outputs, and deploying it through an interactive application.

## 2. Key Features

- **Parameter-efficient fine-tuning:** Uses PEFT LoRA to adapt GPT-2.
- **Custom dataset:** 20 structured travel recommendation examples for Port Luminara.
- **Interactive application:** Streamlit interface for entering travel preferences and generating guides.
- **Model comparison:** Compare outputs from base GPT-2, LoRA fine-tuned GPT-2, and the merged model.
- **Evaluation:** Measures keyword relevance, formatting adherence, actionability, appeal, and generation latency.
- **Adapter merging:** Combines learned LoRA weights with the original model for standalone inference.
- **Local persistence:** Saves generated guides in a local catalog.

## 3. Project Milestones

| Milestone | Implementation | Status |
|---|---|---|
| 1. Dataset preparation | Created 20 structured travel examples | Complete |
| 2. LoRA training | Implemented adapter injection and GPT-2 fine-tuning | Complete |
| 3. Streamlit interface | Built interactive travel guide generation dashboard | Complete |
| 4. Model evaluation | Implemented baseline comparisons and evaluation metrics | Complete |
| 5. Adapter merging | Implemented merged-model inference using PEFT | Complete |

### Milestone 1: Dataset Preparation

Created a synthetic dataset containing 20 input-output pairs for a fictional coastal city called **Port Luminara**.

The dataset covers different traveler interests, including:

- Artisan coffee and bookstores
- Nightlife and canals
- Local food and night markets
- Historical architecture
- Gardens and conservatories
- Glassblowing and local crafts
- Maritime attractions
- Wellness and thermal baths
- Coastal hiking and vineyards

Dataset location: `data/travel_guides.json`

### Milestone 2: LoRA Fine-Tuning

The project uses GPT-2 (124M parameters) as its pretrained language model.

LoRA introduces trainable low-rank matrices into selected attention projections instead of updating all model parameters.

**Configuration:**

| Parameter | Value |
|---|---|
| Base model | GPT-2 |
| Fine-tuning method | PEFT LoRA |
| Target module | `c_attn` |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Optimizer | AdamW |
| Learning rate | 0.0005 |
| Epochs | 8 |
| Batch size | 2 |
| Random seed | 42 |
| Trainable parameters | 294,912 |
| Trainable parameter percentage | Approximately 0.236% |

Only a small fraction of the original model parameters are trained.

### Milestone 3: Streamlit Application

The application provides an interface for entering destination and traveler interests.

Its main functionality includes:

- Travel preference inputs
- Personalized guide generation
- Base-model and fine-tuned-model comparison
- Generation latency display
- Evaluation metrics
- Saved guide catalog
- Downloadable Markdown guides

Launch the application using:

```bash
streamlit run app/main.py
```

### Milestone 4: Model Evaluation

The project evaluates generated guides against the original GPT-2 baseline.

The implemented evaluation dimensions include:

1. Keyword relevance: Measures overlap between expected and generated travel concepts.
2. Structure adherence: Checks whether generated content follows the expected guide format.
3. Actionability: Evaluates whether recommendations provide useful details.
4. Overall appeal: A rating-based assessment of the generated guide.
5. Latency: Measures the time required to generate a response.

### Milestone 5: Adapter Merging

The trained LoRA adapter can be merged into the original GPT-2 model using:

```python
merged_model = model.merge_and_unload()
```

Conceptually, the merged weights are:

\[
W_{\text{merged}} = W_0 + \frac{\alpha}{r}BA
\]

Where:

- \(W_0\) represents the original model weights.
- \(A\) and \(B\) are the learned low-rank matrices.
- \(r\) is the LoRA rank.
- \(\alpha\) is the scaling factor.

This removes the need to maintain a separate adapter during inference.

---

## 4. System Architecture

```text
              USER INPUT
       Destination + Interests
                   |
                   v
           PROMPT GENERATION
                   |
                   v
        +----------------------+
        |   Model Selection    |
        +----------------------+
           |        |        |
           v        v        v
         Base     LoRA     Merged
         GPT-2   Adapter   GPT-2
           |        |        |
           v        v        v
        Generated Travel Recommendations
                   |
                   v
          EVALUATION ENGINE
                   |
          +--------+--------+
          |        |        |
       Relevance Structure Latency
          |        |        |
          +--------+--------+
                   |
                   v
          STREAMLIT DASHBOARD
                   |
          +--------+--------+
          |                 |
      Save Guide       Download Guide
```

### Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Base language model | GPT-2 |
| Deep learning | PyTorch |
| Fine-tuning | Hugging Face PEFT |
| Model loading | Transformers |
| Tokenization | GPT-2 Tokenizer |
| Interface | Streamlit |
| Data processing | JSON |
| Testing | Pytest |
| Containerization | Docker |

---

## 5. Dataset Provenance

The project uses an original synthetic dataset stored in:

`data/travel_guides.json`

It contains 20 structured prompt-completion examples centered around the fictional city of Port Luminara.

The city and its attractions were created specifically for this experiment. The recommendations are fictional and should not be interpreted as verified real-world travel information.

Each record follows this structure:

```json
{
  "id": "pl-01",
  "destination": "Port Luminara",
  "interest": "Artisan Coffee & Hidden Bookstores",
  "prompt": "Destination: Port Luminara\nInterests: Artisan Coffee & Hidden Bookstores\nGenerate a concise, personalized travel mini-guide with top recommendations:\nGuide:",
  "completion": "### Mini-Guide: Cozy Literary Haven\n- Must-Visit: The Whispering Archive..."
}
```

The dataset is intentionally small and is primarily intended to demonstrate domain adaptation rather than establish a general-purpose travel recommendation benchmark.

---

## 6. Experimental Results

The following are the project's recorded experimental results. They should be interpreted in the context of the small synthetic dataset and evaluation set.

### Training Loss

Recorded training loss over eight epochs:

| Epoch | Training Loss |
|---|---:|
| Initial | 4.7349 |
| 1 | 4.6022 |
| 2 | 4.0419 |
| 3 | 3.6899 |
| 4 | 3.3210 |
| 5 | 2.9144 |
| 6 | 2.5802 |
| 7 | 2.2415 |
| 8 | 2.0139 |

The recorded training loss decreased over the training process, indicating that the model adapted to the training examples.

### Baseline Comparison

Average recorded results across five test archetypes:

| Metric | Base GPT-2 | LoRA | Merged |
|---|---:|---:|---:|
| Keyword relevance | 20.2% | 78.6% | 78.6% |
| Structure adherence | 8% | 92% | 92% |
| Actionability | 20% | 84% | 84% |
| Overall appeal | 1.66/5 | 4.60/5 | 4.60/5 |
| Latency | 172.2 ms | 194.8 ms | 172.2 ms |

**Observations:**

- LoRA fine-tuning improved the recorded keyword relevance and formatting scores.
- The fine-tuned model generated more structured travel guides than the unadapted baseline.
- Dynamic adapter inference introduced additional latency in the recorded experiment.
- The merged model retained the fine-tuned output scores while recovering approximately the baseline latency.

### Adapter Size

The recorded LoRA adapter size is approximately 2.36 MB.

This is substantially smaller than the full GPT-2 model weights.

The adapter can therefore be distributed separately from the pretrained model, provided the base model is downloaded when needed.

---

## 7. Example Output

**Input:**

```text
Destination: Port Luminara
Interests: Artisan Coffee & Hidden Bookstores
```

**Base GPT-2:**

The recorded baseline output demonstrates generic completion behavior, including unrelated real-world references and limited adherence to the requested guide structure.

**LoRA Fine-Tuned GPT-2:**

```markdown
### Mini-Guide: Cozy Literary Haven

- Must-Visit: The Whispering Archive on Belltower Hill — a 300-year-old sanctuary filled with floor-to-ceiling cedar shelves and rare nautical maps.

- Coffee Spot: Roast & Parchment Cafe, famed for roasted chicory mocha and quiet window nooks overlooking the harbor.

- Local Secret: Inquire with the back-counter barista for the key to the second-floor poet's balcony.

- Best Time: 8:30 AM to 11:00 AM while morning fog wraps the cobblestones.
```

The fine-tuned output follows the requested format and uses the fictional locations represented in the training dataset.

---

## 8. Installation and Setup

### Prerequisites

- Python 3.12
- Git
- Internet connection for downloading the pretrained GPT-2 model
- Sufficient RAM and disk space for PyTorch and Transformers

### Clone the Repository

```bash
git clone https://github.com/Lishhhh07/localtravel.git

cd localtravel
```

### Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv

.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python3.12 -m venv .venv

source .venv/bin/activate
```

### Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app/main.py
```

Open the following address in your browser:

`http://localhost:8501`

### Run Tests

```bash
pytest -v
```

The repository also includes a GitHub Actions workflow that runs the test suite on pushes and pull requests.

---

## 9. Reproducibility

The following configuration describes the recorded training experiment:

| Setting | Value |
|---|---|
| Base model | GPT-2 |
| Dataset | `data/travel_guides.json` |
| Fine-tuning | LoRA |
| Rank | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Learning rate | 0.0005 |
| Optimizer | AdamW |
| Epochs | 8 |
| Batch size | 2 |
| Random seed | 42 |

The trained adapter is stored in:

`models/lora_adapter/`

Training from scratch requires downloading the original GPT-2 model and running the training pipeline.

The exact experimental results may vary depending on the hardware, library versions, and generation settings.

---

## 10. Project Structure

```text
local-travel-guide-generator/
│
├── app/
│   ├── __init__.py
│   ├── core.py
│   ├── llm.py
│   ├── main.py
│   └── storage.py
│
├── data/
│   └── travel_guides.json
│
├── models/
│   └── lora_adapter/
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── training_meta.json
│       └── README.md
│
├── tests/
│   ├── test_core.py
│   └── test_smoke.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .env.example
├── .gitignore
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 11. Limitations

This project is an experimental demonstration of parameter-efficient fine-tuning, not a production-ready travel recommendation service.

Current limitations include:

1. **Small dataset:** Only 20 examples are used for training, limiting generalization.
2. **Fictional destination:** Port Luminara is not a real city, and its recommendations are synthetic.
3. **Model capacity:** GPT-2 is a relatively small language model and can generate repetitive or inaccurate responses.
4. **Limited evaluation:** The evaluation covers five test archetypes and does not establish performance across a broad population of destinations.
5. **Out-of-distribution queries:** The model may not generate reliable recommendations for destinations or interests outside its training domain.
6. **Hardware dependence:** Generation latency and training performance vary across machines.

The results demonstrate improvements on the project's specific evaluation examples, not general superiority over larger language models.

---

## 12. Future Improvements

Potential next steps include:

- Expand the dataset with more diverse travel scenarios.
- Introduce real-world destination datasets with verified sources.
- Evaluate on a larger held-out test set.
- Add human evaluation and more robust semantic relevance metrics.
- Experiment with larger pretrained language models.
- Add multilingual travel guide generation.
- Introduce retrieval-augmented generation for current destination information.
- Deploy the application as a hosted service.

---

## 13. License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 14. Acknowledgements

This project uses open-source tools and pretrained model infrastructure, including:

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [PyTorch](https://pytorch.org/)
- [Streamlit](https://streamlit.io/)

The project uses GPT-2 as its pretrained base model. Refer to the original model documentation for its licensing and usage terms.