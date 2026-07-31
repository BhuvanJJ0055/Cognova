import os
import io
import base64
import requests
from PIL import Image, ImageDraw

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

key = os.environ.get("GEMINI_API_KEY", "")
print(f"Testing Key: {key[:15]}...")

# Create synthetic image with text "VoxCRM CONNECT. MANAGE. GROW."
img = Image.new("RGB", (600, 300), color=(255, 255, 255))
draw = ImageDraw.Draw(img)
draw.text((50, 100), "VoxCRM CONNECT. MANAGE. GROW.", fill=(0, 0, 0))

buf = io.BytesIO()
img.save(buf, format="JPEG")
b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

for m_name in ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": "Extract all text from this image."},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64_str}}
                ]
            }
        ]
    }
    r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=15)
    print(f"Model '{m_name}' HTTP Status: {r.status_code}")
    if r.status_code == 200:
        res_json = r.json()
        candidates = res_json.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                print(f"🎉 REST Response Text: {parts[0].get('text', '').strip()}")
                break
    else:
        print(f"  Error Body: {r.text[:150]}")
