import torch
from audio.generate_audio import AudioGenerator
from soundtrack.generate_soundtrack import SoundTrackGenerator
from brown_noise.generate_brown_noise import BrownNoiseGenerator
from utils.utils import audio_to_spectrogram_video

AUDIO_MODEL_NAME = "facebook/mms-tts-por"
SOUNDTRACK_MODEL_NAME = "facebook/musicgen-small"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    print("Generating text to speech - audio...")
    audio_generator = AudioGenerator(
        model_name=AUDIO_MODEL_NAME,
        device=DEVICE,
        output_path="audio.wav"
    )
    # audio_output = audio_generator.generate(text="Olá mundo!")
    
    print("Generating sound track...")
    sound_generator = SoundTrackGenerator(
        model_name=SOUNDTRACK_MODEL_NAME,
        device=DEVICE,
        output_path="brown_noise.wav",
        number_tokens=512
    )
    # noise_audio_output = sound_generator.generate(text="A deep continuous brown noise, low-frequency ambient sound, steady and relaxing, without melody or instruments, pure noise texture.")
    # combining noise audios (1 hour) 
    # print("Combining results for noise audio") 
    # sound_generator.combine([noise_audio_output.cpu().numpy().squeeze()] * 720) 

    print("Generating brown noise")
    brown = BrownNoiseGenerator(
        duration=60, # 60 seconds
        sample_rate=44100, # 44.1 khz 
        volume=0.95, # factor between [0.0 and 1.]
        output_path="naive_brown_noise.wav"
    )

    noise = brown.generate()
    print("Combing noise - 1 hour")
    brown.combine(media_list=[noise] * 60)

    print("Creating video from audio")
    audio_to_spectrogram_video("combined_naive_brown_noise.wav", "brown_noise_spectrogram.mp4")

