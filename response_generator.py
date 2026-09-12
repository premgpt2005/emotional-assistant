"""
response_generator.py
Generates an emotion-aware response by combining the detected emotion
with a lightweight conversational strategy per emotion, then using a
small HuggingFace text-generation pipeline to phrase the reply naturally.
"""
import random
from functools import lru_cache
from transformers import pipeline

GENERATOR_MODEL = "google/flan-t5-base"

# Response strategy per detected emotion: sets the tone/intent of the reply.
EMOTION_STRATEGIES = {
    "sadness": "Respond with warmth and gentle encouragement. Acknowledge the feeling without being dismissive.",
    "joy": "Respond with matching enthusiasm and share in the excitement.",
    "love": "Respond warmly and affirmatively, reflecting the positive sentiment.",
    "anger": "Respond calmly, acknowledge the frustration, and avoid escalating.",
    "fear": "Respond reassuringly and offer grounded, practical next steps.",
    "surprise": "Respond with curiosity and openness, inviting more detail.",
    "neutral": "Respond helpfully and naturally.",
}

FALLBACK_TEMPLATES = {
    "sadness": [
        "I'm sorry you're going through that. I'm here if you want to talk more about it.",
        "That sounds tough. Take your time — I'm listening.",
    ],
    "joy": [
        "That's wonderful to hear! Tell me more about it.",
        "Love that energy! What's got you feeling this good?",
    ],
    "love": [
        "That's really sweet. Sounds like it means a lot to you.",
        "That's a lovely feeling to hold onto.",
    ],
    "anger": [
        "That sounds frustrating. Want to walk through what happened?",
        "I hear you — that's a valid thing to be upset about.",
    ],
    "fear": [
        "That sounds stressful. Let's figure out a next step together.",
        "It's okay to feel uneasy about that. What's worrying you most?",
    ],
    "surprise": [
        "Whoa, that's unexpected! What happened?",
        "Didn't see that coming — tell me more!",
    ],
    "neutral": [
        "Got it — how can I help with that?",
        "Sure, let's dig into that.",
    ],
}


@lru_cache(maxsize=1)
def _get_generator():
    return pipeline("text2text-generation", model=GENERATOR_MODEL)


def generate_response(user_text: str, emotion: str) -> str:
    """
    Generates a reply using flan-t5's instruction-following ability,
    guided by the emotion-specific strategy. Falls back to a templated
    reply if the model output is empty or degenerate.
    """
    strategy = EMOTION_STRATEGIES.get(emotion, EMOTION_STRATEGIES["neutral"])
    prompt = (
        f"You are an empathetic assistant. The user seems to feel '{emotion}'. "
        f"{strategy} User said: \"{user_text}\". Write a short, natural reply (1-2 sentences)."
    )

    try:
        pipe = _get_generator()
        output = pipe(prompt, max_length=60, do_sample=True, temperature=0.7)[0]["generated_text"].strip()
        if len(output) > 5:
            return output
    except Exception:
        pass

    return random.choice(FALLBACK_TEMPLATES.get(emotion, FALLBACK_TEMPLATES["neutral"]))
