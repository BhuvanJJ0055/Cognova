import os
import requests
import json
import base64
from PIL import Image, ImageDraw
import io

key = "AIzaSyDv_-9r9mB5LC1VNqBTEKm9FvZrGzdE87k"

log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "gemini_direct_response.txt")
os.makedirs(os.path.dirname(log_file), exist_ok=True)

with open(log_file, "w", encoding="utf-8") as f:
    f.write(f"Key being tested: {key}\n")

    # 1. Test Text prompt
    url_text = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
    payload_text = {"contents": [{"parts": [{"text": "Hello, explain what you can do."}]}]}
    try:
        r = requests.post(url_text, json=payload_text, headers={"Content-Type": "application/json"}, timeout=15)
        f.write(f"Text Call HTTP Status: {r.status_code}\n")
        f.write(f"Text Response: {r.text[:500]}\n\n")
    except Exception as e:
        f.write(f"Text Exception: {e}\n\n")

    # 2. Test Image Vision prompt
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 80), "Cognova Vision AI Test OCR", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    models = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
    for m in models:
        url_vision = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}"
        payload_vision = {
            "contents": [
                {
                    "parts": [
                        {"text": "Extract all text from this image and explain what is in the image."},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64_str
                            }
                        }
                    ]
                }
            ]
        }
        try:
            r = requests.post(url_vision, json=payload_vision, headers={"Content-Type": "application/json"}, timeout=15)
            f.write(f"Vision Model '{m}' HTTP Status: {r.status_code}\n")
            f.write(f"Vision Response: {r.text[:500]}\n\n")
        except Exception as e:
            f.write(f"Vision Exception '{m}': {e}\n\n")

print("Finished direct test.")
