import os
import io
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

def build_model(num_classes=4):
    """Reconstruct EfficientNet-B0 with custom classifier head."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )
    return model

def load_model():
    """Load the trained model. Falls back to demo mode if no model found."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=4)

    # Search for model file in common locations
    model_paths = [
        "brain_tumor_classifier.pth",
        "model/brain_tumor_classifier.pth",
        "./model/brain_tumor_classifier.pth",
    ]

    for path in model_paths:
        if os.path.exists(path):
            model.load_state_dict(
                torch.load(path, map_location=device, weights_only=True)
            )
            model.eval()
            model.to(device)
            print(f"[OK] Model loaded from: {path}")
            return model, device, True

    # Demo mode — model not trained yet
    print("[WARN] No trained model found. Running in DEMO mode.")
    model.eval()
    model.to(device)
    return model, device, False

MODEL, DEVICE, MODEL_LOADED = load_model()

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

app = FastAPI(title="NeuroScan AI Backend")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    try:
        with open("fronted.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="fronted.html not found.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    probs = probabilities.cpu().numpy()
    pred_idx = np.argmax(probs)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx] * 100)
    
    prob_dict = {CLASS_NAMES[i]: float(probs[i] * 100) for i in range(len(CLASS_NAMES))}
    
    return JSONResponse(content={
        "prediction": pred_class,
        "confidence": confidence,
        "probabilities": prob_dict,
        "demo_mode": not MODEL_LOADED
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
