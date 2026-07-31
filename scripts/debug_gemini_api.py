import os
import requests
import json

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

key = os.environ.get("GEMINI_API_KEY", "")
print(f"Debug: Testing Key = {key}")

# 1. List available models
list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
try:
    res = requests.get(list_url, timeout=10)
    print(f"List Models Status: {res.status_code}")
    if res.status_code == 200:
        models = res.json().get("models", [])
        print("Available models:")
        for m in models:
            if "generateContent" in m.get("supportedGenerationMethods", []):
                print(f"  - {m.get('name')}")
    else:
        print(f"List models error: {res.text}")
except Exception as e:
    print(f"Error listing models: {e}")

# 2. Test generateContent text
test_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
payload = {"contents": [{"parts": [{"text": "Hello world"}]}]}
try:
    res = requests.post(test_url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
    print(f"\nTest Text Status: {res.status_code}")
    if res.status_code == 200:
        print(f"Text Response: {res.json()}")
    else:
        print(f"Text error: {res.text}")
except Exception as e:
    print(f"Text Exception: {e}")
