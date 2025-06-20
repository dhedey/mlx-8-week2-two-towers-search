from dataclasses import dataclass
import torch.nn as nn
import torch
import os
from typing import Optional, Self

class PersistableData:
    def to_dict(self):
        output = vars(self)
        for key, value in output.items():
            if isinstance(value, PersistableData):
                output[key] = value.to_dict()

        return output
    
    @classmethod
    def from_dict(cls, d):
        """You might need to override this method in subclasses if they have nested Persistable objects."""
        return cls(**d)

def select_device():
    DEVICE_IF_MPS_SUPPORT = 'cpu' # or 'mps' - but it doesn't work well with EmbeddingBag
    device = torch.device('cuda' if torch.cuda.is_available() else DEVICE_IF_MPS_SUPPORT if torch.backends.mps.is_available() else 'cpu')
    
    print(f'Selected device: {device}')
    return device

@dataclass
class TrainingState(PersistableData):
    epoch: int
    optimizer_state: dict
    total_training_time_seconds: Optional[float] = None
    latest_training_loss: Optional[float] = None
    latest_validation_loss: Optional[float] = None

class PersistableModel(nn.Module):
    registered_types: dict[str, type] = {}

    def __init__(self):
        super(PersistableModel, self).__init__()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ in PersistableModel.registered_types:
            raise ValueError(f"Persistable Model {cls.__name__} is a duplicate classname. Use a new class name.")
        PersistableModel.registered_types[cls.__name__] = cls

    def get_device(self):
        return next(self.parameters()).device

    @classmethod
    def build_creation_state(cls) -> dict:
        """This method should return a dictionary with the state needed to recreate the model."""
        raise NotImplementedError("This class method should be implemented by subclasses.")

    @classmethod
    def create(cls, creation_state: dict, for_evaluation_only: bool) -> Self:
        """This method should return a new model from the creation state."""
        raise NotImplementedError("This class method should be implemented by subclasses.")
    
    @staticmethod
    def _model_path(model_name: str) -> str:
        model_folder = os.path.join(os.path.dirname(__file__), "data")
        return os.path.join(model_folder, f"{model_name}.pt")

    def save_model_data(self, model_name: str, training_state: TrainingState):
        model_path = PersistableModel._model_path(model_name)
        torch.save({
            "model": {
                "class_name": type(self).__name__,
                "weights": self.state_dict(),
                "creation_state": self.build_creation_state()
            },
            "training": training_state.to_dict(),
        }, model_path)
        print(f"Model saved to {model_path}")
    
    @classmethod
    def load(cls, model_name: str, for_evaluation_only: bool, override_class_name = None, device: Optional[str] = None) -> tuple[Self, TrainingState]:
        model_path = PersistableModel._model_path(model_name)
        if device is None:
            device = select_device()

        loaded_model_data = torch.load(model_path, map_location=device)
        print(f"Model data read from {model_path}")
        loaded_class_name = loaded_model_data["model"]["class_name"]
        actual_class_name = override_class_name if override_class_name is not None else loaded_class_name

        registered_types = PersistableModel.registered_types
        if actual_class_name not in registered_types:
            raise ValueError(f"Model class {actual_class_name} is not a known PersistableModel. Available classes: {list(registered_types.keys())}")
        actual_class: type[PersistableModel] = registered_types[actual_class_name]

        if not issubclass(actual_class, cls):
            raise ValueError(f"The model {model_name} was attempted to be loaded with {cls.__name__}.load(\"{model_name}\") (loaded class name = {loaded_class_name}, override class name = {override_class_name}), but {actual_class} is not a subclass of {cls}.")

        training_state = TrainingState.from_dict(loaded_model_data["training"])
        model = actual_class.create(
            creation_state=loaded_model_data["model"]["creation_state"],
            for_evaluation_only=for_evaluation_only
        )
        model.load_state_dict(loaded_model_data["model"]["weights"])
        model.to(device)
        return model, training_state

    @classmethod
    def load_for_evaluation(cls, model_name: str, device: Optional[str] = None) -> Self:
        model, _ = cls.load(model_name=model_name, device=device, for_evaluation_only=True)
        model.eval()
        return model
    
    @classmethod
    def load_for_training(cls, model_name: str, device: Optional[str] = None) -> tuple[Self, TrainingState]:
        model, training_state = cls.load(model_name=model_name, device=device, for_evaluation_only=False)
        model.train()
        return (model, training_state)