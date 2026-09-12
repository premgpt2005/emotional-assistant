"""
main.py
CLI chat loop for the Emotion-Aware AI Assistant.

Usage:
    python main.py
"""
from emotion_detector import detect_emotion
from response_generator import generate_response


def main():
    print("Emotion-Aware AI Assistant (type 'quit' to exit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Assistant: Take care!")
            break
        if not user_input:
            continue

        emotion = detect_emotion(user_input)
        reply = generate_response(user_input, emotion)

        print(f"[detected emotion: {emotion}]")
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()
