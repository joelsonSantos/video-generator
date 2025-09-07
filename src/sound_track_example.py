import torch
from soundtrack.generate_soundtrack import SoundTrackGenerator

SOUNDTRACK_MODEL_NAME = "facebook/musicgen-small"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":    
    print("Generating sound track...")
    sound_generator = SoundTrackGenerator(
        model_name=SOUNDTRACK_MODEL_NAME,
        device=DEVICE,
        output_path="brown_noise.wav",
        number_tokens=512
    )
    noise_audio_output = sound_generator.generate(text="A deep continuous brown noise, low-frequency ambient sound, steady and relaxing, without melody or instruments, pure noise texture.")
    # combining noise audios (1 hour) 
    print("Combining results for noise audio") 
    sound_generator.combine([noise_audio_output.cpu().numpy().squeeze()] * 720) 
