# EmotionWare — Emotion-Aware AI Assistant

A simple Python notebook that detects the emotional tone of a user's message and generates a context-appropriate reply using Hugging Face Transformers.

## What it does

1. Takes a line of text input from the user.
2. Runs sentiment analysis on it to classify the emotion as **POSITIVE**, **NEGATIVE**, or **NEUTRAL**.
3. Builds a different prompt depending on the detected sentiment (empathetic, celebratory, or acknowledging).
4. Feeds that prompt to a text-generation model to produce a short, natural-language reply.
5. Prints the generated reply back to the user.

## How it works

The notebook is built around two Hugging Face `pipeline`s:

| Stage | Task | Model |
|---|---|---|
| Sentiment detection | `sentiment-analysis` | `distilbert-base-uncased-finetuned-sst-2-english` (default) |
| Reply generation | `text2text-generation` | `google/flan-t5-large` |

The flow:

```
user input → classifier (sentiment) → if/else prompt builder → generator (reply) → printed output
```

Based on the sentiment label, one of three prompt templates is chosen:

- **NEGATIVE** → asks the generator for a short, kind, empathetic reply that acknowledges the feeling and offers support.
- **POSITIVE** → asks the generator for a short, enthusiastic reply that celebrates the good mood.
- **NEUTRAL** → asks the generator for a short reply acknowledging the statement.

## Requirements

```bash
pip install transformers==4.46.3 torch
```

A GPU is used automatically if available (`torch.cuda.is_available()`); otherwise it falls back to CPU (slower, especially for `flan-t5-large`).

## Usage

Run the notebook cells in order. The final cell is interactive:

```
--Emotion Aware AI assistant---
Hello! I am an AI assistant. How are u feeling today?
> I am not getting a job
(Debug: NEGATIVE sentiment with 99.98% confidence)
Generating reply......
you can always ask for help
```

## Notes / possible improvements

- The sentiment pipeline currently loads without an explicitly pinned model name; pinning one avoids the "no model specified" warning and guarantees reproducible results.
- `flan-t5-large` on CPU is slow — consider a smaller model (e.g. `google/flan-t5-base`) for faster iteration.
- The POSITIVE prompt template currently says "user is feeling bad" — this looks like a copy-paste leftover and should say "user is feeling good" instead.
- The assistant only handles a single turn; wrapping the last cell in a loop would let it hold a short conversation.
