from brown_noise.generate_brown_noise import BrownNoiseGenerator
from utils.utils import merge_image_audio_to_video

if __name__ == "__main__":
    print("Generating brown noise")
    brown = BrownNoiseGenerator(
        duration=60, # 60 seconds
        sample_rate=44100, # 44.1 khz 
        volume=0.95, # factor between [0.0 and 1.]
        output_path="naive_brown_noise.wav"
    )

    noise = brown.generate()
    print("Combing noise - 1 hour")
    brown.combine(media_list=[noise] * 14)

    print("Creating video from audio")
    # audio_to_spectrogram_video("combined_naive_brown_noise.wav", "brown_noise_spectrogram.mp4")
    merge_image_audio_to_video(image_file="noise.jpg", audio_file="combined_naive_brown_noise.wav", output_file="output_15min.mp4")
