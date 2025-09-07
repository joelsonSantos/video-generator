from typing import Protocol

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

    def generate(self, text: dict|str) -> str:
        """
        Generate output from the model.
        Args:
            text (dict): Input text data.
        Returns:
            str: Generated output path.
        """
        ...