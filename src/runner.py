import torch
from audio.generate_audio import AudioGenerator

AUDIO_MODEL_NAME = "facebook/mms-tts-por"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    audio_generator = AudioGenerator(
        model_name=AUDIO_MODEL_NAME,
        device=DEVICE,
        output_path="audio.wav"
    )

    text = {
        "intro": "Olá mundo!"
    }

    audio_generator.generate(text=text)