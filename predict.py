"""
╔══════════════════════════════════════════════════════════╗
║           PetVision AI — Prediction Engine               ║
║           Pet Skin Condition Detector v2.0               ║
╚══════════════════════════════════════════════════════════╝
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path 

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from model import build_model


# ════════════════════════════════════════════════════════════
#  TERMINAL COLORS  (works on Windows + Mac + Linux)
# ════════════════════════════════════════════════════════════
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

    BLACK   = "\033[30m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"

    BG_RED    = "\033[41m"
    BG_GREEN  = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE   = "\033[44m"

    @staticmethod
    def red(t):     return f"{C.RED}{t}{C.RESET}"
    @staticmethod
    def green(t):   return f"{C.GREEN}{t}{C.RESET}"
    @staticmethod
    def yellow(t):  return f"{C.YELLOW}{t}{C.RESET}"
    @staticmethod
    def cyan(t):    return f"{C.CYAN}{t}{C.RESET}"
    @staticmethod
    def bold(t):    return f"{C.BOLD}{t}{C.RESET}"
    @staticmethod
    def dim(t):     return f"{C.DIM}{t}{C.RESET}"
    @staticmethod
    def magenta(t): return f"{C.MAGENTA}{t}{C.RESET}"


# ════════════════════════════════════════════════════════════
#  LOGGING SETUP
#  Saves all predictions to petvision.log automatically
# ════════════════════════════════════════════════════════════
logging.basicConfig(
    filename="petvision.log",
    filemode="a",
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger("PetVisionAI")


# ════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════

# ✅ Must match training class_to_idx EXACTLY
CLASSES = [
    "Dermatitis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity",
    "demodicosis",
    "ringworm"
]

# Severity icons for terminal display
SEVERITY_ICONS = {
    "normal":   "✅",
    "mild":     "💛",
    "moderate": "⚠️ ",
    "critical": "🚨",
}

# Severity colors for terminal display
SEVERITY_COLORS = {
    "normal":   C.green,
    "mild":     C.yellow,
    "moderate": C.yellow,
    "critical": C.red,
}

# Base severity — starting point before confidence adjustment
SEVERITY_MAP = {
    "Healthy":           "normal",
    "Hypersensitivity":  "moderate",
    "Dermatitis":        "moderate",
    "Fungal_infections": "mild",
    "demodicosis":       "critical",
    "ringworm":          "mild",
}

RECOMMENDATIONS = {
    "Healthy":           "Your pet looks healthy! Maintain regular vet checkups and a balanced diet.",
    "Hypersensitivity":  "Possible allergic reaction detected. Identify and remove triggers. Consult a vet this week.",
    "Dermatitis":        "Skin inflammation detected. Avoid irritants and schedule a vet visit within 2-3 days.",
    "Fungal_infections": "Possible fungal infection. Antifungal treatment needed. See a vet soon.",
    "demodicosis":       "Demodex mite infestation detected. Requires immediate veterinary treatment.",
    "ringworm":          "Ringworm detected. Highly contagious — isolate your pet and see a vet within 24 hours.",
}

TIMEFRAME = {
    "Healthy":           "No action needed",
    "Hypersensitivity":  "Within 1 week",
    "Dermatitis":        "Within 2-3 days",
    "Fungal_infections": "Within 1 week",
    "demodicosis":       "Immediate",
    "ringworm":          "Within 24 hours",
}

VET_SPECIALTY = {
    "Healthy":           "General Practitioner",
    "Hypersensitivity":  "Veterinary Dermatologist",
    "Dermatitis":        "Veterinary Dermatologist",
    "Fungal_infections": "General Practitioner",
    "demodicosis":       "Veterinary Dermatologist",
    "ringworm":          "General Practitioner",
}

# Minimum confidence threshold — below this we flag as uncertain
CONFIDENCE_THRESHOLD = 40.0

# ════════════════════════════════════════════════════════════
#  DYNAMIC SEVERITY
# ════════════════════════════════════════════════════════════
def get_severity(detected: str, confidence: float) -> str:
    """
    Dynamically adjusts severity based on model confidence.

    HIGH   (> 90%) → upgrade one level
    MEDIUM (50-90%) → keep base level
    LOW    (< 50%) → downgrade one level
    """
    if detected == "Healthy":
        return "normal"

    base = SEVERITY_MAP[detected]

    if confidence > 90:
        upgrade = {"mild": "moderate", "moderate": "critical", "critical": "critical"}
        return upgrade.get(base, base)

    if 50 <= confidence <= 90:
        return base

    if confidence < 50:
        downgrade = {"critical": "moderate", "moderate": "mild", "mild": "mild"}
        return downgrade.get(base, base)

    return base


# ════════════════════════════════════════════════════════════
#  DYNAMIC RECOMMENDATION
# ════════════════════════════════════════════════════════════
def get_recommendation(detected: str, final_severity: str, confidence: float) -> str:
    """
    Returns a recommendation message that adapts to:
    - Whether the pet is healthy
    - How severe the condition is
    - How confident the model is
    """
    if detected == "Healthy":
        return RECOMMENDATIONS["Healthy"]

    base_rec = RECOMMENDATIONS[detected]

    if confidence < CONFIDENCE_THRESHOLD:
        return (
            f"Low confidence detection ({confidence:.1f}%). "
            f"{base_rec} "
            f"Please retake the photo in better lighting for a more accurate result."
        )

    prefix = {
        "critical": "🚨 URGENT: ",
        "moderate": "⚠️  ATTENTION: ",
        "mild":     "💛 ADVISORY: ",
    }.get(final_severity, "")

    suffix = {
        "critical": " Do NOT delay — seek emergency veterinary care immediately.",
        "moderate": " Monitor closely and book a vet appointment as soon as possible.",
        "mild":     " Watch for worsening symptoms and consult a vet if concerned.",
    }.get(final_severity, "")

    return f"{prefix}{base_rec}{suffix}"


# ════════════════════════════════════════════════════════════
#  DYNAMIC TIMEFRAME
# ════════════════════════════════════════════════════════════
def get_timeframe(detected: str, final_severity: str, confidence: float) -> str:
    """Returns urgency timeframe based on final severity and confidence."""
    if detected == "Healthy":
        return "No action needed"

    if confidence < CONFIDENCE_THRESHOLD:
        return "Retake photo for accurate result"

    return {
        "critical": "⚡ Immediate — go now",
        "moderate": "🕐 Within 24-48 hours",
        "mild":     "📅 Within 1 week",
        "normal":   "✅ No action needed",
    }.get(final_severity, TIMEFRAME[detected])


# ════════════════════════════════════════════════════════════
#  CONFIDENCE LABEL
# ════════════════════════════════════════════════════════════
def get_confidence_label(confidence: float) -> str:
    """Returns a human readable confidence label."""
    if confidence >= 90: return "Very High"
    if confidence >= 75: return "High"
    if confidence >= 50: return "Moderate"
    if confidence >= 30: return "Low"
    return "Very Low"


# ════════════════════════════════════════════════════════════
#  LOAD MODEL
# ════════════════════════════════════════════════════════════
def load_model(weights_path: str, device: torch.device):
    """
    Loads trained ResNet50 model from .pth file.
    Raises clear errors if file is missing or corrupted.
    """
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"\n{C.red('✗ Model file not found:')} {weights_path}\n"
            f"{C.dim('  Run train.py first to generate best_model.pth')}"
        )

    print(f"{C.dim('  Loading model weights from')} {C.cyan(str(weights_path))}...")

    try:
        model = build_model(num_classes=len(CLASSES), device=device)
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        logger.info(f"Model loaded successfully from {weights_path}")
        print(f"  {C.green('✓ Model loaded successfully')}\n")
        return model

    except RuntimeError as e:
        raise RuntimeError(
            f"\n{C.red('✗ Model loading failed.')}\n"
            f"{C.dim('  This usually means the model was trained with different settings.')}\n"
            f"{C.dim(f'  Error: {e}')}"
        )


# ════════════════════════════════════════════════════════════
#  PREDICT
# ════════════════════════════════════════════════════════════
def predict(image_path: str, model, device: torch.device) -> dict:
    """
    Runs full prediction pipeline on a single image.

    Args:
        image_path : path to pet image (JPG, PNG, HEIC)
        model      : loaded PyTorch model
        device     : cpu or cuda

    Returns:
        dict with detected_issue, confidence, severity,
             recommendation, timeframe, all_scores
    """
    image_path = Path(image_path)

    # ── Validate image file ──
    if not image_path.exists():
        raise FileNotFoundError(
            f"{C.red('✗ Image not found:')} {image_path}"
        )

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}
    if image_path.suffix.lower() not in valid_extensions:
        raise ValueError(
            f"{C.red('✗ Unsupported file type:')} {image_path.suffix}\n"
            f"{C.dim(f'  Supported types: {valid_extensions}')}"
        )

    # ── Load and preprocess image ──
    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError(
            f"{C.red('✗ Cannot read image file:')} {image_path}\n"
            f"{C.dim('  File may be corrupted or not a valid image.')}"
        )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    tensor = transform(image).unsqueeze(0).to(device)

    # ── Run inference ──
    with torch.no_grad():
        logits     = model(tensor)
        probs      = F.softmax(logits, dim=1)[0]
        confidence = probs.max().item()
        class_idx  = probs.argmax().item()
        detected   = CLASSES[class_idx]

    # ── Convert to percentage ──
    confidence_pct = round(confidence * 100, 1)
    all_scores     = {
        cls: round(probs[i].item() * 100, 1)
        for i, cls in enumerate(CLASSES)
    }

    # ── Dynamic calculations ──
    final_severity       = get_severity(detected, confidence_pct)
    final_recommendation = get_recommendation(detected, final_severity, confidence_pct)
    final_timeframe      = get_timeframe(detected, final_severity, confidence_pct)
    confidence_label     = get_confidence_label(confidence_pct)
    vet_specialty        = VET_SPECIALTY.get(detected, "General Practitioner")
    is_uncertain         = confidence_pct < CONFIDENCE_THRESHOLD

    # ── Log result ──
    logger.info(
        f"PREDICTION | image={image_path.name} | "
        f"detected={detected} | confidence={confidence_pct}% | "
        f"severity={final_severity}"
    )

    return {
        "detected_issue":   detected,
        "confidence":       confidence_pct,
        "confidence_label": confidence_label,
        "is_uncertain":     is_uncertain,
        "base_severity":    SEVERITY_MAP.get(detected, "normal"),
        "severity":         final_severity,
        "recommendation":   final_recommendation,
        "timeframe":        final_timeframe,
        "vet_specialty":    vet_specialty,
        "all_scores":       all_scores,
        "image_analyzed":   image_path.name,
        "analyzed_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ════════════════════════════════════════════════════════════
#  BEAUTIFUL TERMINAL PRINTER
# ════════════════════════════════════════════════════════════
def print_result(result: dict) -> None:
    """Prints prediction result in a beautiful, readable terminal format."""

    sev        = result["severity"]
    sev_color  = SEVERITY_COLORS.get(sev, C.cyan)   # ✅ fixed
    sev_icon   = SEVERITY_ICONS.get(sev, "•")
    conf       = result["confidence"]

    # ── Confidence bar ──
    filled     = int(conf / 100 * 30)
    empty      = 30 - filled
    bar_color  = C.green if conf >= 75 else C.yellow if conf >= 50 else C.red
    bar        = bar_color("█" * filled) + C.dim("░" * empty)

    # ── Scores bar for each class ──
    def score_bar(pct):
        filled = int(pct / 100 * 20)
        color  = C.green if pct >= 60 else C.yellow if pct >= 30 else C.dim
        return color("▓" * filled) + C.dim("░" * (20 - filled))

    w = 58  # box width

    print()
    print(C.bold(C.cyan("╔" + "═" * w + "╗")))
    print(C.bold(C.cyan("║")) + C.bold(f"  🐾  PetVision AI — Analysis Report".center(w)) + C.bold(C.cyan("║")))
    print(C.bold(C.cyan("║")) + C.dim(f"  {result['analyzed_at']}  ·  {result['image_analyzed']}".center(w)) + C.bold(C.cyan("║")))
    print(C.bold(C.cyan("╠" + "═" * w + "╣")))

    # Detected condition
    print(C.bold(C.cyan("║")) + f"  {C.dim('DETECTED')}".ljust(w + 9) + C.bold(C.cyan("║")))
    print(C.bold(C.cyan("║")) + f"  {C.bold(result['detected_issue'])}".ljust(w + 9) + C.bold(C.cyan("║")))

    print(C.bold(C.cyan("╠" + "─" * w + "╣")))

    # Confidence
    print(C.bold(C.cyan("║")) + f"  {C.dim('CONFIDENCE')}  {bar}  {bar_color(C.bold(f'{conf}%'))}  {C.dim(result['confidence_label'])}".ljust(w + 56) + C.bold(C.cyan("║")))

    # Uncertainty warning
    if result["is_uncertain"]:
        print(C.bold(C.cyan("║")) + f"  {C.yellow('⚠  Low confidence — retake photo in better lighting')}".ljust(w + 9) + C.bold(C.cyan("║")))

    print(C.bold(C.cyan("╠" + "─" * w + "╣")))

    # Severity
    base_sev = result["base_severity"]
    fin_sev  = result["severity"]
    adjusted = C.dim(f"  (adjusted from {base_sev})") if base_sev != fin_sev else ""
    print(C.bold(C.cyan("║")) + f"  {C.dim('SEVERITY')}  {sev_icon}  {sev_color(C.bold(fin_sev.upper()))}{adjusted}".ljust(w + 36) + C.bold(C.cyan("║")))

    print(C.bold(C.cyan("╠" + "─" * w + "╣")))

    # Timeframe and vet
    print(C.bold(C.cyan("║")) + f"  {C.dim('TIMEFRAME')}   {result['timeframe']}".ljust(w + 9) + C.bold(C.cyan("║")))
    print(C.bold(C.cyan("║")) + f"  {C.dim('SEE')}         {result['vet_specialty']}".ljust(w + 9) + C.bold(C.cyan("║")))

    print(C.bold(C.cyan("╠" + "─" * w + "╣")))

    # Recommendation
    rec_lines = [result["recommendation"][i:i+52] for i in range(0, len(result["recommendation"]), 52)]
    print(C.bold(C.cyan("║")) + f"  {C.dim('RECOMMENDATION')}".ljust(w + 9) + C.bold(C.cyan("║")))
    for line in rec_lines:
        print(C.bold(C.cyan("║")) + f"  {line}".ljust(w) + C.bold(C.cyan("║")))

    print(C.bold(C.cyan("╠" + "─" * w + "╣")))

    # All class scores
    print(C.bold(C.cyan("║")) + f"  {C.dim('ALL CLASS PROBABILITIES')}".ljust(w + 9) + C.bold(C.cyan("║")))
    for cls, score in sorted(result["all_scores"].items(), key=lambda x: -x[1]):
        marker = " ◄" if cls == result["detected_issue"] else "  "
        row    = f"  {cls:<22} {score_bar(score)}  {score:5.1f}%{marker}"
        print(C.bold(C.cyan("║")) + row.ljust(w + 27) + C.bold(C.cyan("║")))

    print(C.bold(C.cyan("╚" + "═" * w + "╝")))
    print(C.dim("  ⚕  This is a screening tool only. Always consult a licensed veterinarian.\n"))


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Header ──
    print()
    print(C.bold(C.green("  ██████╗ ███████╗████████╗██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗")))
    print(C.bold(C.cyan("  PetVision AI  ·  Skin Condition Detector  ·  v2.0")))
    print(C.dim("  ─────────────────────────────────────────────────────"))
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  {C.dim('Device :')} {C.cyan(str(device).upper())}")
    print(f"  {C.dim('Classes:')} {C.cyan(str(len(CLASSES)))} conditions")
    print()

    # ── Load model ──
    try:
        model = load_model("best_model.pth", device)
    except (FileNotFoundError, RuntimeError) as e:
        print(e)
        sys.exit(1)

    # ── Get image path ──
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print(C.dim("  Usage: python predict.py <image_path>"))
        print(C.dim("  Example: python predict.py my_dog.jpg"))
        print()
        image_path = input(C.cyan("  Enter image path: ")).strip().strip('"')

    print(f"\n  {C.dim('Analyzing')} {C.cyan(image_path)} ...")

    # ── Run prediction ──
    try:
        result = predict(image_path, model, device)
        print_result(result)

    except FileNotFoundError as e:
        print(f"\n  {e}")
        logger.error(f"File not found: {image_path}")
        sys.exit(1)

    except ValueError as e:
        print(f"\n  {e}")
        logger.error(f"Invalid image: {image_path} — {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\n  {C.red('✗ Unexpected error:')} {e}")
        logger.exception(f"Unexpected error during prediction: {e}")
        sys.exit(1)