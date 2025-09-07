import numpy as np
import soundfile as sf
import torch

from tqdm import tqdm

from typing import Any

from commons.base_model import BaseModel

class BrownNoiseGenerator(BaseModel):

    def __init__(self, duration: float, sample_rate: int, volume: float, output_path: str):
        self.duration = duration
        self.sample_rate = sample_rate
        self.output_path = output_path
        self.volume = volume

    def load_model():
        pass

    def generate(self) -> np.ndarray:
        """
        Generate Brownian (Brown) noise using NumPy.
        
        Returns:
            np.ndarray: array of Brown noise values in range [-1, 1]
        """
        n_samples = int(self.duration * self.sample_rate)

        # White noise
        white = np.random.randn(n_samples)

        # Integração cumulativa -> Brown noise
        brown = np.cumsum(white)

        # Normalizar para [-1, 1]
        brown = brown / np.max(np.abs(brown))

        # Ajustar volume
        brown = brown * self.volume
        
        self.save(brown)
        return brown
    
    def save(self, media: Any) -> str:
        # Save if requested
        sf.write(self.output_path, media, self.sample_rate)

    def combine(self, media_list):
        # Combining audios with pause
        final_audio = media_list[0]
        for second_audio in tqdm(media_list[1:], desc="combining audios"):
            final_audio = torch.cat((torch.from_numpy(final_audio), torch.from_numpy(second_audio)), dim=0)
            final_audio = final_audio.numpy()

        # --- 5. Salving final final
        sf.write("combined_" + self.output_path, final_audio, self.sample_rate)
        return "combined_" + self.output_path

