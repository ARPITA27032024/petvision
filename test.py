# test.py — test multiple images at once
import requests

images = [
    r"C:\Users\arpit\OneDrive\Desktop\AI TRAINING\vision\test_dog.jpg.png",
    r"C:\Users\arpit\OneDrive\Desktop\AI TRAINING\vision\test_dog2.jpg.png",
    r"C:\Users\arpit\OneDrive\Desktop\AI TRAINING\vision\goggy_image3.png",
]

for path in images:
    with open(path, "rb") as f:
        response = requests.post(
            "http://localhost:8000/analyze",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )
    result = response.json()
    print(f"\nImage : {path.split(chr(92))[-1]}")
    print(f"Result: {result['detected_issue']} ({result['confidence']}%)")
    print(f"Scores: {result['all_scores']}")