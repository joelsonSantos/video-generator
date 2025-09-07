import torch
from transformers import VitsModel, AutoTokenizer
import soundfile as sf
import json

from commons.base_model import BaseModel

class AudioGenerator(BaseModel):
    """Concrete implementation of BaseModel for audio generation"""

    def __init__(self, model_name: str, device: str, output_path: str, silence: float = 0.3):
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.device = device
        self.output_path = output_path
        self.silence = silence

    def load_model(self):
        """Load the model and tokenizer."""
        self.model = VitsModel.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def generate(self, text: dict) -> str:
        """
        Generate audio from text using the specified model.
        Args:
            text (dict): Input text data.
        Returns:
            str: Path to the generated audio file.
        """
        
        if self.model is None or self.tokenizer is None:
            self.load_model()

        generated_audios = []
        for section, token in text.items():
            print(f"Processing section: '{section}'")

            inputs = self.tokenizer(token, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model(**inputs).waveform

            audio_data = output.squeeze().cpu().numpy()
            generated_audios.append(audio_data)

        sampling_rate = self.model.config.sampling_rate
        silence = torch.zeros(int(sampling_rate * self.silence))

        # Combining audios with pause
        final_audio = generated_audios[0]
        for second_audio in generated_audios[1:]:
            final_audio = torch.cat((torch.from_numpy(final_audio), silence, torch.from_numpy(second_audio)), dim=0)
            final_audio = final_audio.numpy()

        # --- 5. Salving final final
        sf.write(self.output_path, final_audio, sampling_rate)
        return self.output_path



