import torch
from audio.generate_audio import AudioGenerator
from soundtrack.generate_soundtrack import SoundTrackGenerator

AUDIO_MODEL_NAME = "facebook/mms-tts-por"
SOUNDTRACK_MODEL_NAME = "facebook/musicgen-small"

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

    sound_generator = SoundTrackGenerator(
        model_name=SOUNDTRACK_MODEL_NAME,
        device=DEVICE,
        output_path="soundtrack.wav"
    )

    audio_generator.generate(text=text)
    sound_generator.generate(text="Trilha sonora eletrônica, atmosfera épica, 120 BPM, pad suave, baixo pulsante")
