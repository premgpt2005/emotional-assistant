"""
app.py
Streamlit chat UI for the Emotion-Aware AI Assistant.
"""
import streamlit as st

from emotion_detector import detect_emotion
from response_generator import generate_response

st.set_page_config(page_title="Emotion-Aware Assistant", page_icon="🎭", layout="centered")
st.title("🎭 Emotion-Aware AI Assistant")
st.caption("Detects emotion from your message using a fine-tuned T5 model, then replies accordingly.")

EMOJI = {
    "sadness": "😢", "joy": "😄", "love": "❤️",
    "anger": "😠", "fear": "😨", "surprise": "😲", "neutral": "🙂",
}

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

if user_input := st.chat_input("How are you feeling? Tell me anything..."):
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Reading the emotion..."):
            emotion = detect_emotion(user_input)
            reply = generate_response(user_input, emotion)
        st.markdown(f"**{EMOJI.get(emotion, '🙂')} detected: `{emotion}`**")
        st.write(reply)
        st.session_state.history.append({"role": "assistant", "content": reply})
