from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import sys
import os

sys.path.append(os.path.dirname(__file__))
from predict import predict, get_model_and_classes

app = FastAPI(
    title="Singapore Waste Classifier API",
    description="Upload an image to classify waste type and get NEA-aligned disposal instructions.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.on_event("startup")
async def load_model():
    get_model_and_classes()
    print("Model loaded and ready.")


@app.get("/")
def root():
    return {"message": "Singapore Waste Classifier API is running."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/classes")
def get_classes():
    _, classes = get_model_and_classes()
    return {"classes": classes}


@app.post("/predict")
async def classify_waste(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    model, classes = get_model_and_classes()
    result = predict(image, model, classes)

    return {
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "top3_predictions": [
            {"class": cls, "confidence": round(conf, 2)}
            for cls, conf in result["top3"]
        ],
        "disposal_instructions": result["disposal"]
    }