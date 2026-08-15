import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import os
from huggingface_hub import hf_hub_download

DISPOSAL_INSTRUCTIONS = {
    "cardboard": {
        "emoji": "📦",
        "bin": "Blue Recycling Bin",
        "instructions": "Flatten cardboard boxes before recycling. Remove any food residue or grease. Place in the blue recycling bin.",
        "nea_tip": "Clean and dry cardboard can be recycled. Soiled or wet cardboard should go to the general waste bin."
    },
    "glass": {
        "emoji": "🍶",
        "bin": "Blue Recycling Bin",
        "instructions": "Rinse glass bottles and jars. Remove lids (recycle separately if metal). Place in the blue recycling bin.",
        "nea_tip": "Broken glass should be wrapped in newspaper before disposal in the general waste bin for safety."
    },
    "metal": {
        "emoji": "🥫",
        "bin": "Blue Recycling Bin",
        "instructions": "Rinse metal cans and tins. Crush if possible to save space. Place in the blue recycling bin.",
        "nea_tip": "Aluminium and steel cans are recyclable. Large metal items should go to a recycling centre."
    },
    "paper": {
        "emoji": "📄",
        "bin": "Blue Recycling Bin",
        "instructions": "Bundle newspapers and magazines. Place clean paper in the blue recycling bin.",
        "nea_tip": "Avoid recycling paper that is soiled with food or liquids. Remove plastic windows from envelopes."
    },
    "plastic": {
        "emoji": "🧴",
        "bin": "Blue Recycling Bin",
        "instructions": "Rinse plastic bottles and containers. Check for recycling symbol. Place in the blue recycling bin.",
        "nea_tip": "Singapore recycles plastics #1 (PET) and #2 (HDPE) most efficiently. Plastic bags go to dedicated bag recycling points."
    },
    "trash": {
        "emoji": "🗑️",
        "bin": "General Waste Bin",
        "instructions": "This item cannot be recycled. Dispose of in the general waste bin.",
        "nea_tip": "When in doubt, throw it out. Contaminated recyclables can spoil entire batches of recycling."
    }
}

HF_REPO_ID = "YOUR_HF_USERNAME/waste-classifier-mobilenetv2"

def load_model_and_classes():
    """Download model from HuggingFace Hub and load it."""
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="waste_classifier.pth")
    classes_path = hf_hub_download(repo_id=HF_REPO_ID, filename="classes.json")

    with open(classes_path, 'r') as f:
        classes = json.load(f)

    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, len(classes))
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    return model, classes


def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def predict(image: Image.Image, model, classes: list) -> dict:
    """Run inference on a PIL image."""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = probabilities.max(0)

    predicted_class = classes[predicted_idx.item()]
    confidence_pct = confidence.item() * 100

    top3 = sorted(
        [(classes[i], probabilities[i].item() * 100) for i in range(len(classes))],
        key=lambda x: x[1],
        reverse=True
    )[:3]

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence_pct, 2),
        "top3": top3,
        "disposal": DISPOSAL_INSTRUCTIONS.get(predicted_class, {})
    }


_model = None
_classes = None

def get_model_and_classes():
    global _model, _classes
    if _model is None:
        _model, _classes = load_model_and_classes()
    return _model, _classes