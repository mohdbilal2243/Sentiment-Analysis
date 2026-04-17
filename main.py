import os
from Generate_Transcription import extract_vocals, transcribe_audio_whisperx
from sentiment import analyze_sentiment_groq


def process_audio(input_audio_path):

    # Extract Vocals
    vocals_path, base_name  = extract_vocals(input_audio_path)

    # transcription
    transcript_path = transcribe_audio_whisperx(vocals_path, base_name)

    # Read transcript file
    with open(transcript_path, "r", encoding = "utf-8") as f:
        full_text = f.read()

    # Sentiment Analysis
    sentiment = analyze_sentiment_groq(full_text, base_name)

    return {
        "Vocals" : vocals_path,
        "Transcription" : transcript_path,
        "Sentiment" : sentiment
    }

if __name__ == "__main__":
    input_audio_path = "data/Aud_3_English.WAV"

    result = process_audio(input_audio_path)

    print("\nFINAL OUTPUT:")
    print(result)
