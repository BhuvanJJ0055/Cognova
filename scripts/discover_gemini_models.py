"""
Discover available Gemini models for the given API key and test vision capability.
"""
import os
import requests
import json
import base64
import io
from PIL import Image, ImageDraw

KEY = os.environ.get("GEMINI_API_KEY", "").strip()
if not KEY:
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("GEMINI_API_KEY="):
                    KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "gemini_models_list.txt")
os.makedirs(os.path.dirname(LOG), exist_ok=True)

with open(LOG, "w", encoding="utf-8") as f:
    # 1. List all available models
    for api_ver in ["v1", "v1beta"]:
        list_url = f"https://generativelanguage.googleapis.com/{api_ver}/models?key={KEY}&pageSize=100"
        try:
            r = requests.get(list_url, timeout=10)
            f.write(f"\n=== API Version: {api_ver} — HTTP {r.status_code} ===\n")
            if r.status_code == 200:
                models = r.json().get("models", [])
                vision_models = []
                for m in models:
                    name = m.get("name", "")
                    methods = m.get("supportedGenerationMethods", [])
                    if "generateContent" in methods:
                        f.write(f"  ✅ {name} | {', '.join(methods)}\n")
                        if any(x in name for x in ["flash", "pro", "vision"]):
                            vision_models.append(name)
            else:
                f.write(f"  Error: {r.text[:200]}\n")
        except Exception as e:
            f.write(f"  Exception: {e}\n")

    # 2. Try vision request with models from list
    img = Image.new("RGB", (400, 200), (255, 255, 255))
    ImageDraw.Draw(img).text((20, 60), "Cognova Vision AI Test 2026", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    test_models = [
        "models/gemini-2.5-flash",
        "models/gemini-2.5-flash-lite",
        "models/gemini-2.5-pro",
        "models/gemini-2.0-flash-lite",
        "models/gemini-1.5-flash-latest",
        "models/gemini-1.5-flash-002",
    ]

    f.write("\n=== Vision API Tests ===\n")
    for model_path in test_models:
        model_id = model_path.replace("models/", "")
        for ver in ["v1", "v1beta"]:
            url = f"https://generativelanguage.googleapis.com/{ver}/models/{model_id}:generateContent?key={KEY}"
            payload = {
                "contents": [{"parts": [
                    {"text": "Extract all text from this image."},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
                ]}]
            }
            try:
                r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=15)
                f.write(f"  {ver}/{model_id}: HTTP {r.status_code}")
                if r.status_code == 200:
                    cands = r.json().get("candidates", [])
                    text = cands[0].get("content", {}).get("parts", [{}])[0].get("text", "") if cands else ""
                    f.write(f" ✅ Text: {text[:80]}\n")
                    break
                else:
                    body = r.json().get("error", {}).get("message", r.text[:100])
                    f.write(f" ❌ {body[:120]}\n")
            except Exception as e:
                f.write(f" ⚠️ {e}\n")

print("Done. Check reports/gemini_models_list.txt")
