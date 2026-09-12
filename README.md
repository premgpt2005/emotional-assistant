# Emotion-Aware AI Assistant

A conversational assistant that detects the emotion behind a user's message
using a fine-tuned T5 model, then generates a reply tailored to that emotion.

## Stack
- **HuggingFace Transformers** — model pipelines
- `mrm8488/t5-base-finetuned-emotion` — T5 fine-tuned for emotion classification
  (sadness, joy, love, anger, fear, surprise)
- `google/flan-t5-base` — generates the emotion-tailored reply
- **Streamlit** — optional chat UI

## Setup

```bash
pip install -r requirements.txt
```

No API key needed — both models run locally via HuggingFace `transformers`
(they'll download automatically on first run, ~1GB total).

## Run

CLI:
```bash
python main.py
```

Streamlit UI:
```bash
streamlit run app.py
```

## How it works
1. `emotion_detector.py` prefixes the input with `"emotion: "` and feeds it
   to the fine-tuned T5 model, which outputs one of six emotion labels.
2. `response_generator.py` maps that emotion to a response strategy
   (e.g. "respond with gentle encouragement" for sadness) and prompts
   `flan-t5-base` to phrase a short, natural reply. If generation produces
   something too short/degenerate, it falls back to a curated template.

## Notes
- First run will be slow while models download and load into memory.
- Runs on CPU; a GPU (if available) will speed up inference automatically via PyTorch.
