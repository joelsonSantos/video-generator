from transformers import AutoProcessor, MusicgenForConditionalGeneration
from commons.base_model import BaseModel
import soundfile as sf
import scipy
        
class SoundTrackGenerator(BaseModel):

    def __init__(self, model_name: str, device: str, output_path: str, duration: int = 30):
        self.model_name = model_name
        self.device = device
        self.output_path = output_path
        self.model = None
        self.duration = duration
        self.processor = None

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(self.model_name)
        

    def generate(self, text: str) -> str:

        self.load_model()

        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )

        audio_values = self.model.generate(**inputs, max_new_tokens=256)
        sampling_rate = self.model.config.audio_encoder.sampling_rate
        scipy.io.wavfile.write(self.output_path, rate=sampling_rate, data=audio_values[0, 0].numpy())
        return self.output_path
