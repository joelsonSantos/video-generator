import soundfile as sf
import scipy
import torch

from typing import Any
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from commons.base_model import BaseModel

class SoundTrackGenerator(BaseModel):

    def __init__(self, model_name: str, device: str, output_path: str, number_tokens: int = 1503):
        self.model_name = model_name
        self.device = device
        self.output_path = output_path
        self.number_tokens = number_tokens
        self.model = None
        self.processor = None

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(self.model_name)
        
    def generate(self, text: str) -> Any:
        self.load_model()

        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )

        audio_values = self.model.generate(**inputs, max_new_tokens=self.number_tokens)

        # saving audio
        self.save(audio_values)

        return audio_values

    def combine(self, media_list):
        sampling_rate = self.model.config.sampling_rate

        # Combining audios with pause
        final_audio = media_list[0]
        for second_audio in media_list[1:]:
            final_audio = torch.cat((torch.from_numpy(final_audio), torch.from_numpy(second_audio)), dim=0)
            final_audio = final_audio.numpy()

        # --- 5. Salving final final
        sf.write("combined_" + self.output_path, final_audio, sampling_rate)
        return "combined_" + self.output_path

    def save(self, media) -> str:
        sampling_rate = self.model.config.audio_encoder.sampling_rate
        scipy.io.wavfile.write(self.output_path, rate=sampling_rate, data=media[0, 0].numpy())
        return self.output_path
