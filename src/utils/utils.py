import numpy as np
import matplotlib.pyplot as plt
from moviepy import AudioFileClip, VideoClip
from scipy.io import wavfile
from scipy.signal import spectrogram

def audio_to_spectrogram_video(audio_path: str, output_path: str, fps: int =30) -> str:
    """
    Create a video with spectrogram from an audio file.
    
    Args:
        audio_path (str): audio path file (.wav).
        output_path (str): output path video (.mp4).
        fps (int): frames by second.
    """
    
    # loading audio
    samplerate, data = wavfile.read(audio_path)
    if data.ndim > 1:
        data = data[:, 0]

    # normalizing
    data = data / np.max(np.abs(data))

    # create audio clip
    audio_clip = AudioFileClip(audio_path)

    # function to generate frame (t -> time in seconds)
    def make_frame(t):
        start = int((t - 0.5) * samplerate)
        end = int((t + 0.5) * samplerate)
        start = max(start, 0)
        end = min(end, len(data))

        segment = data[start:end]

        # Calculate spectrogram slice
        f, _, Sxx = spectrogram(segment, samplerate, nperseg=256, noverlap=128)
        Sxx = 10 * np.log10(Sxx + 1e-8)

        # plot spectrogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        ax.imshow(Sxx, aspect="auto", origin="lower", cmap="magma")
        plt.tight_layout(pad=0)
        
        # convert to numpy
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        # Convert to RGB
        frame = buf[:, :, :3].copy()
        
        plt.close(fig)
        return frame

    # create video from frames
    video = VideoClip(make_frame, duration=audio_clip.duration)
    video = video.with_audio(audio_clip)

    # export
    video.write_videofile(output_path, fps=fps)
    return output_path
