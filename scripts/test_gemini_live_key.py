import os
import requests
import json
import base64
from PIL import Image, ImageDraw
import io

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

key = os.environ.get("GEMINI_API_KEY", "").strip()

log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "gemini_api_debug.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)

with open(log_file, "w", encoding="utf-8") as log:
    log.write(f"Key loaded: {key}\n")

    # Create dummy synthetic image with text "Cognova MultiModal Test"
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 80), "Cognova MultiModal Test", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    model_candidates = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-pro"]

    for m_name in model_candidates:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": "Extract all text from this image."},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": b64_str
                            }
                        }
                    ]
                }
            ]
        }
        try:
            r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=15)
            log.write(f"Model '{m_name}' HTTP Status: {r.status_code}\n")
            log.write(f"Response Body: {r.text[:300]}\n\n")
        except Exception as e:
            log.write(f"Model '{m_name}' Exception: {e}\n\n")

print("Debug finished. Check reports/gemini_api_debug.log")
