import os
from PIL import Image

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

key = os.environ.get("GEMINI_API_KEY", "")
print(f"Loaded Key: {key[:15]}...")

try:
    import importlib
    genai = importlib.import_module("google.generativeai")
    genai.configure(api_key=key)
    for model_name in ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]:
        try:
            m = genai.GenerativeModel(model_name)
            res = m.generate_content("Say hello in 5 words.")
            print(f"✅ Success with model '{model_name}': {res.text.strip()}")
            break
        except Exception as e:
            print(f"Model '{model_name}' error: {e}")
except Exception as main_e:
    print(f"Gemini API Error: {main_e}")
