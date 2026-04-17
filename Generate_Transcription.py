import os
from typing import Any

os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"
import subprocess
import whisperx
import torch

def create_folder(path):
    os.makedirs(path, exist_ok=True)

# Extract vocals using Demucs
def extract_vocals(input_audio_path: object, base_output_dir: object = "result") -> tuple[str, Any]:

    # Extract base filename (without extension)
    filename = os.path.splitext(os.path.basename(input_audio_path))[0]

    # Create structured folders
    audio_root = os.path.join(base_output_dir, filename)
    vocals_dir = os.path.join(audio_root, "vocals")

    create_folder(vocals_dir)

    vocals_path = os.path.join(vocals_dir, f"{filename}_vocals.mp3")


    # Check if already processed
    if not os.path.exists(vocals_path):
        print("Extracting vocals using Demucs...")

        command = [
            "python",
            "-m",
            "demucs",
            "--two-stems=vocals",
            "-n", "mdx_extra_q",
            "--float32",
            "--mp3",
            "-o", "temp_separated",
            input_audio_path
        ]

        subprocess.run(command, check=True)

        demucs_output = os.path.join(
            "temp_separated",
            "mdx_extra_q",
            filename,
            "vocals.mp3"
        )

        # Move to structured folder
        import shutil
        shutil.move(demucs_output, vocals_path)

        shutil.rmtree("temp_separated", ignore_errors=True)

        print(f"Saved Vocals: {vocals_path}")

    else:
        print("Vocals Already exist...")


    return vocals_path, filename


def transcribe_audio_whisperx(audio_path, filename, base_output_dir="result"):

    # Create output folder
    output_dir = os.path.join(base_output_dir, filename, "transcripts")
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"{filename}_transcript.txt")

    if not os.path.exists(file_path):

        print("Loading WhisperX model...")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load WhisperX model
        model = whisperx.load_model("base", device)

        print("📝 Transcribing audio...")
        result = model.transcribe(audio_path)

        # Align for better timestamps
        print("⏱️ Aligning timestamps...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=device
        )

        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio_path,
            device
        )

        # Save formatted transcript
        with open(file_path, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                start = round(segment["start"], 2)
                end = round(segment["end"], 2)
                text = segment["text"]

                line = f"[{start}s - {end}s] {text}"
                f.write(line + "\n")

        print(f"✅ Transcript saved at: {file_path}")

    else:
        print("Transcript Already exist...")

    return file_path

# if __name__ == "__main__":
#     vocals_path, base_name = extract_vocals("data/Aud_6_English.WAV")
#
#     transcript_path = transcribe_audio_whisperx(vocals_path, base_name)
#
