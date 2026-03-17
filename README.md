# 🐾 PetVision — Pet Skin Condition Detector

PetVision is a deep learning-based pet skin condition classifier that detects 6 common skin conditions in dogs from images using a fine-tuned ResNet50 model.
---
## 🔍 Conditions Detected

| Condition | Description |
|---|---|
| Dermatitis | Skin inflammation |
| Fungal Infection | Fungal skin disease |
| Healthy | No condition detected |
| Hypersensitivity | Allergic skin reactions |
| Demodicosis | Mite-caused skin condition |
| Ringworm | Fungal ring-shaped infection |
---
## 🧠 Model

- Architecture: **ResNet50** (Transfer Learning)
- Framework: **PyTorch**
- Input: Pet skin image (JPG/PNG)
- Output: Predicted condition + confidence score

> 📥 Download model weights (`best_model.pth`): https://drive.google.com/file/d/1n03LKTnHiaq8oc2_4XB2pRNOoSYa_RDA/view?usp=drivesdk
---

## 📁 Project Structure
```
petvision/
├── api.py          # FastAPI server
├── model.py        # ResNet50 model definition
├── predict.py      # Prediction engine
├── train.py        # Training script
├── dataset.py      # Dataset loader
├── requirments.txt # Dependencies
└── best_model.pth  # Model weights (download separately)
```
## 👩‍💻 Built By

**Arpita** — [GitHub](https://github.com/ARPITA27032024)
