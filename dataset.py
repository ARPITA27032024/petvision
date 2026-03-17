import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import build_model

# ✅ Must match training class_to_idx EXACTLY
CLASSES = ["Dermatitis", "Fungal_infections", "Healthy",
           "Hypersensitivity", "demodicosis", "ringworm"]

# Base severity — starting point before confidence adjustment
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
# Dynamic Severity Function
# Adjusts severity based on confidence score
# ─────────────────────────────────────────
def get_severity(detected, confidence):

    # Healthy is always normal regardless of confidence
    if detected == "Healthy":
        return "normal"

    # Get base severity from manual table
    base = SEVERITY_MAP[detected]

    # ── HIGH CONFIDENCE (above 90%) ──
    # Model is very sure — upgrade severity
    if confidence > 90:
        if base == "mild":
            return "moderate"    # mild → moderate
        if base == "moderate":
            return "critical"    # moderate → critical
        if base == "critical":
            return "critical"    # already max, stays critical

    # ── MEDIUM CONFIDENCE (50% to 90%) ──
    # Model is reasonably sure — keep base severity
    if 50 <= confidence <= 90:
        return base

    # ── LOW CONFIDENCE (below 50%) ──
    # Model is not sure — downgrade severity
    if confidence < 50:
        if base == "critical":
            return "moderate"    # critical → moderate
        if base == "moderate":
            return "mild"        # moderate → mild
        if base == "mild":
            return "mild"        # already lowest, stays mild

    return base


# ─────────────────────────────────────────
# Dynamic Recommendation Function
# Changes advice based on final severity
# ─────────────────────────────────────────
def get_recommendation(detected, final_severity, confidence):

    # Healthy — always same message
    if detected == "Healthy":
        return "Your pet looks healthy! Keep up regular checkups."

    # Low confidence — always add uncertainty warning
    if confidence < 50:
        base_rec = RECOMMENDATIONS[detected]
        return f"Uncertain detection ({confidence}% confidence). {base_rec} Consider getting a clearer photo for better results."

    # Critical severity — urgent message
    if final_severity == "critical":
        return f"URGENT: {RECOMMENDATIONS[detected]} Do not delay — seek veterinary care immediately."

    # Moderate severity — standard message
    if final_severity == "moderate":
        return f"{RECOMMENDATIONS[detected]} Monitor closely and book a vet appointment soon."

    # Mild severity — relaxed message
    if final_severity == "mild":
        return f"{RECOMMENDATIONS[detected]} Keep an eye on symptoms and consult a vet if it worsens."

    return RECOMMENDATIONS[detected]


# ─────────────────────────────────────────
# Dynamic Timeframe Function
# Adjusts urgency based on final severity
# ─────────────────────────────────────────
def get_timeframe(detected, final_severity, confidence):

    if detected == "Healthy":
        return "No action needed"

    if confidence < 50:
        return "Get a clearer photo first"

    if final_severity == "critical":
        return "Immediate"

    if final_severity == "moderate":
        return "Within 24-48 hours"

    if final_severity == "mild":
        return "Within 1 week"

    return TIMEFRAME[detected]


def load_model(weights_path, device):
    model = build_model(num_classes=len(CLASSES), device=device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def predict(image_path, model, device):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image  = Image.open(r"C:\Users\arpit\OneDrive\Desktop\AI TRAINING\vision\dog4.png").convert("RGB")  
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits     = model(tensor)
        probs      = F.softmax(logits, dim=1)[0]
        confidence = probs.max().item()
        class_idx  = probs.argmax().item()
        detected   = CLASSES[class_idx]

    # Convert to percentage
    confidence_pct = round(confidence * 100, 1)

    # ── Dynamic calculations ──
    final_severity      = get_severity(detected, confidence_pct)
    final_recommendation = get_recommendation(detected, final_severity, confidence_pct)
    final_timeframe     = get_timeframe(detected, final_severity, confidence_pct)

    # Debug — print all scores visually
    print("\n--- All Class Probabilities ---")
    for i, cls in enumerate(CLASSES):
        bar = "█" * int(probs[i].item() * 30)
        print(f"{cls:<20} {probs[i].item()*100:5.1f}%  {bar}")
    print("------------------------------")

    # Debug — print severity logic
    
    print(f"Confidence       : {confidence_pct}%")
    print(f"Final Severity   : {final_severity}  ← adjusted by confidence")

    return {
        "detected_issue":   detected,
        "confidence":       confidence_pct,
        
        "severity":         final_severity,          # ← dynamic
        "recommendation":   final_recommendation,    # ← dynamic
        "timeframe":        final_timeframe,         # ← dynamic
        "all_scores": {
            cls: round(probs[i].item() * 100, 1)
            for i, cls in enumerate(CLASSES)
        }
    }


if __name__ == "__main__":
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model("best_model.pth", device)

    image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    result     = predict(image_path, model, device)

    print(f"\nDetected         : {result['detected_issue']}")
    print(f"Confidence       : {result['confidence']}%")
    
    print(f"Final Severity   : {result['severity']}")
    print(f"Timeframe        : {result['timeframe']}")
    print(f"Recommendation   : {result['recommendation']}")
