import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# api.py
import os
import shutil
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from predict import predict, load_model

# ─────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────
app = FastAPI(
    title="PetVision AI API",
    description="Upload a pet image and get skin condition analysis",
    version="1.0.0"
)

# ─────────────────────────────────────────
# CORS — allows mobile app to call this API
# Without this your app will get blocked
# ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allows any app to call your API
    allow_methods=["*"],        # allows GET, POST etc
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# Load model ONCE when server starts
# Not on every request — that would be slow
# ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on {device}...")

model = load_model("best_model.pth", device)
print("Model loaded successfully!")

# ─────────────────────────────────────────
# Your 6 classes from training
# ─────────────────────────────────────────
CLASSES = ["Dermatitis", "Fungal_infections", "Healthy",
           "Hypersensitivity", "demodicosis", "ringworm"]

SEVERITY_MAP = {
    "Healthy":           "normal",
    "Hypersensitivity":  "moderate",
    "Dermatitis":        "moderate",
    "Fungal_infections": "mild",
    "demodicosis":       "critical",
    "ringworm":          "mild"
}

RECOMMENDATIONS = {
    "Healthy":           "Your pet looks healthy! Keep up regular checkups.",
    "Hypersensitivity":  "Possible allergic reaction. Avoid triggers and consult a vet this week.",
    "Dermatitis":        "Skin inflammation detected. Vet visit recommended within 2-3 days.",
    "Fungal_infections": "Possible fungal infection. Antifungal treatment needed. See vet soon.",
    "demodicosis":       "Demodex mites detected. Requires immediate veterinary treatment.",
    "ringworm":          "Ringworm detected. Contagious — isolate pet and see vet within 24 hours."
}

TIMEFRAME = {
    "Healthy":           "No action needed",
    "Hypersensitivity":  "Within 1 week",
    "Dermatitis":        "Within 2-3 days",
    "Fungal_infections": "Within 1 week",
    "demodicosis":       "Immediate",
    "ringworm":          "Within 24 hours"
}

# ─────────────────────────────────────────
# Route 1 — Health Check
# Visit this in browser to confirm API works
# GET: https://yourapi.com/
# ─────────────────────────────────────────
@app.get("/")
def home():
    return {
        "status":  "PetVision AI is running",
        "model":   "ResNet50 — Skin Condition Detector",
        "classes": CLASSES,
        "usage":   "POST /analyze with an image file"
    }

# ─────────────────────────────────────────
# Route 2 — Main Analysis Endpoint
# This is what your mobile app will call
# POST: https://yourapi.com/analyze
# ─────────────────────────────────────────
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    # Step 1 — Validate file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are accepted. Please upload JPG or PNG."
        )

    # Step 2 — Save uploaded image temporarily
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Step 3 — Run ML model
        result = predict(temp_path, model, device)

        detected = result["detected_issue"]

        # Step 4 — Build full response
        return {
            "success":        True,
            "detected_issue": detected,
            "confidence":     result["confidence"],
            "severity":       SEVERITY_MAP[detected],
            "recommendation": RECOMMENDATIONS[detected],
            "timeframe":      TIMEFRAME[detected],
            "all_scores":     result["all_scores"],
            "pet_is_healthy": detected == "Healthy"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

    finally:
        # Step 5 — Always delete temp file even if error occurs
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ─────────────────────────────────────────
# Route 3 — Test without uploading image
# GET: https://yourapi.com/test
# Useful to confirm model predicts correctly
# ─────────────────────────────────────────
@app.get("/test")
def test():
    return {
        "message":    "API is working correctly",
        "device":     str(device),
        "classes":    CLASSES,
        "model_ready": True
    }
