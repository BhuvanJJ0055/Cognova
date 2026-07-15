import os
import requests
import socket

def check_dns():
    print("[Diagnose] Checking DNS resolution for generativelanguage.googleapis.com...")
    try:
        ip = socket.gethostbyname("generativelanguage.googleapis.com")
        print(f"  Passed: Resolved to IP {ip}")
        return True
    except Exception as e:
        print(f"  ❌ Failed: DNS lookup failed. Error: {e}")
        return False

def check_general_internet():
    print("[Diagnose] Checking general internet connection (httpbin.org)...")
    try:
        res = requests.get("https://httpbin.org/delay/0", timeout=5)
        if res.status_code == 200:
            print("  Passed: Internet connection is active.")
            return True
        else:
            print(f"  ❌ Failed: HTTP Status {res.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Failed: Cannot connect to general internet. Error: {e}")
        return False

def check_google_endpoint():
    print("[Diagnose] Checking direct connection to Google Generative API host...")
    try:
        res = requests.get("https://generativelanguage.googleapis.com/", timeout=5)
        print(f"  Passed: Connected successfully! Host responded with status {res.status_code}")
        return True
    except Exception as e:
        print(f"  ❌ Failed: Host did not respond (timed out). Error: {e}")
        return False

def test_gemini():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key and os.path.exists(".env"):
        print("Reading key from .env file...")
        with open(".env", "r") as f:
            for line in f:
                if "=" in line:
                    key, val = line.strip().split("=", 1)
                    if key.strip() == "GEMINI_API_KEY":
                        api_key = val.strip().strip("'").strip('"')
                        break

    # Run Diagnostics First
    print("=== RUNNING NETWORK DIAGNOSTICS ===")
    dns_ok = check_dns()
    internet_ok = check_general_internet()
    google_ok = check_google_endpoint()
    
    # Check proxy settings
    proxies = {
        "http": os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"),
        "https": os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    }
    if proxies["http"] or proxies["https"]:
        print(f"[Diagnose] System proxies detected: {proxies}")
    else:
        print("[Diagnose] No system proxies detected in environment variables.")
    print("===================================\n")

    if not api_key:
        api_key = input("No GEMINI_API_KEY found. Please paste your key here: ").strip()

    if not api_key:
        print("Error: No API key provided.")
        return

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    # 1. Query available models
    print("Fetching list of available models for your API key...")
    list_models_url = "https://generativelanguage.googleapis.com/v1beta/models"
    try:
        res = requests.get(list_models_url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            models = data.get("models", [])
            print("\nAvailable models on your account supporting generateContent:")
            for m in models:
                methods = m.get('supportedGenerationMethods', [])
                if 'generateContent' in methods:
                    print(f"  - {m.get('name')}")
        else:
            print(f"\n❌ Failed to list models (HTTP {res.status_code}):")
            print(res.text)
    except Exception as e:
        print(f"\n❌ Error listing models: {e}")

    # 2. Try querying all available models that support generateContent
    print("\nTesting model connectivity dynamically from the fetched model list:")
    try:
        res = requests.get(list_models_url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            models = data.get("models", [])
            generate_models = [m.get('name') for m in models if 'generateContent' in m.get('supportedGenerationMethods', [])]
            
            if not generate_models:
                print("No models supporting generateContent found on this account.")
                return
                
            for full_model_name in generate_models:
                # full_model_name is like 'models/gemini-2.0-flash'
                # Extract the short name for logging
                short_name = full_model_name.replace("models/", "")
                print(f"\nTrying to query {full_model_name}...")
                v1beta_url = f"https://generativelanguage.googleapis.com/v1beta/{full_model_name}:generateContent"
                payload = {
                    "contents": [{
                        "parts": [{"text": f"Say 'Hello!' in 2 words."}]
                    }]
                }
                try:
                    response = requests.post(v1beta_url, json=payload, headers=headers, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        text = data.get("candidates", [])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                        print(f"✅ SUCCESS on {short_name}! Response: \"{text}\"")
                    else:
                        print(f"❌ {short_name} API Error (HTTP {response.status_code}):")
                        try:
                            err_json = response.json()
                            print(f"  Message: {err_json.get('error', {}).get('message')}")
                            print(f"  Status: {err_json.get('error', {}).get('status')}")
                        except Exception:
                            print(f"  Raw: {response.text}")
                except Exception as e:
                    print(f"❌ {short_name} Connection Exception: {e}")
        else:
            print("Could not retrieve models list for dynamic testing.")
    except Exception as e:
        print(f"Error during dynamic testing: {e}")

if __name__ == "__main__":
    test_gemini()
