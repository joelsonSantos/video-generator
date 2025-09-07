from typing import Protocol, Any

class BaseModel(Protocol):
    """Base interface for models."""

    def load_model():
        """
        Load the model and tokenizer.
        Args:
            model_name (str): Name of the model to load.
            device (str): Device to load the model on.
        """
        ... 

    def generate(self, text: str) -> str:
        """
        Generate output from the model.
        Args:
            text (str): Input text data.
        Returns:
            str: Generated output path.
        """
        ...

    def combine(self, media_list: list) -> Any:
        """
        Combine multiple media into a single file
        Args:
            media_list (list): sound_track, audio, images
        Returns:
            Any: sound_track, audio, images.
        """
        ...

    def save(self, media: Any) -> str:
        """
        Saving in a file the media
        Args:
            media (Any): audio, image, sound_track
        Returns:
            str: Generated output path
        """