
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "backend" / "artifacts" / "best_resnet18_finetuned.pt"

DEVICE = torch.device("cpu")

CLASS_NAMES = {
    0: "normal",
    1: "abnormal",
}

MODEL_NAME = "Fine-tuned ResNet18"
MODEL_VERSION = "1.0"


def load_model():
    """Load the trained ResNet18 model."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Load the model once when the service starts.
MODEL = load_model()


def predict_image(image: Image.Image) -> dict:
    """Predict normal or abnormal and return model scores."""
    image = image.convert("RGB")
    tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = MODEL(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(probabilities.argmax().item())

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "class_scores": {
            CLASS_NAMES[index]: round(
                float(probabilities[index].item()), 6
            )
            for index in CLASS_NAMES
        },
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "disclaimer": (
            "Research and educational use only. This prediction "
            "is not a medical diagnosis."
        ),
    }
