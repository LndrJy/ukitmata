# UkitMata

Local, self-hostable intelligent form digitization. Reads handwritten and printed
**government documents**, extracts structured field/value data, and improves over time
from human corrections — all running on your own hardware, no cloud calls.

## Engines

| Engine | Model | Role |
|--------|-------|------|
| Vision | `Imgscope-OCR-2B` (Qwen2-VL) | Reads the form (printed + handwriting) |
| Logic  | `Llama 3.1 8B` (via Ollama)   | Cleans & structures the raw OCR text |
| HTR *(optional)* | `microsoft/trocr-base-handwritten` | Second-opinion on low-confidence handwriting 
## Architecture

```
Photo ─▶ preprocess ─▶ detect form type ─▶ VLM extract ─▶ LLM structure ─▶ store
                                                                            │
                          low-confidence fields ─▶ human review/correct ────┤
                                                                            ▼
                          corrected (image, fields) pairs ─▶ periodic LoRA fine-tune
```

One core (`ukitmata.pipeline`), two front-ends:
- **Workstation:** `ukitmata-ui`   (Gradio)
- **Server/API:**  `ukitmata-api`  (FastAPI)

## Adding a new form

Drop a YAML file in [`ukitmata/schemas/forms/`](ukitmata/schemas/forms/). No code change.
See [`bir_2306.yaml`](ukitmata/schemas/forms/bir_2306.yaml) for the template.

## Quick start

```bash
# 1. Install Ollama and pull the logic model
#    https://ollama.com/download
ollama pull llama3.1

# 2. Install UkitMata
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -e ".[dev]"

# 3. Run the workstation UI
ukitmata-ui
#    ...or the API
ukitmata-api
```

First run downloads the Qwen2-VL weights from Hugging Face (~5 GB).

## Hardware

Single NVIDIA GPU, **24 GB VRAM recommended** (16 GB minimum). See `docs` / project notes.

## Layout

```
ukitmata/
  config.py        central settings (env-overridable)
  models/          vlm.py · llm.py · htr.py     (engine wrappers)
  pipeline/        preprocess.py · detect.py · extract.py
  schemas/         registry.py + forms/*.yaml   (one YAML per form)
  store/           db.py · files.py             (documents, extractions, corrections)
  learning/        dataset.py · finetune.py     (human-in-the-loop retraining)
  api/main.py      FastAPI service
  ui/app.py        Gradio workstation app
```
