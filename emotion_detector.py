"""
emotion_detector.py
Detects emotion from text using a T5 model fine-tuned for emotion
classification, via HuggingFace's text2text-generation pipeline.
"""
from functools import lru_cache
from transformers import pipeline

# mrm8488/t5-base-finetuned-emotion is a T5 model fine-tuned on the
# dair-ai/emotion dataset (labels: sadness, joy, love, anger, fear, surprise)
MODEL_NAME = "mrm8488/t5-base-finetuned-emotion"

VALID_EMOTIONS = {"sadness", "joy", "love", "anger", "fear", "surprise"}


@lru_cache(maxsize=1)
def _get_pipeline():
    return pipeline("text2text-generation", model=MODEL_NAME)


def detect_emotion(text: str) -> str:
    """Returns one of: sadness, joy, love, anger, fear, surprise."""
    pipe = _get_pipeline()
    result = pipe(f"emotion: {text}", max_length=8)[0]["generated_text"].strip().lower()
    return result if result in VALID_EMOTIONS else "neutral"
