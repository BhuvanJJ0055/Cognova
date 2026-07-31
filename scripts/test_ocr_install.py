"""
Test OCR installation - verifies Tesseract is working correctly.
Run after: install_tesseract.bat
"""
import os
import sys
import io
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

# ─── Create a test image with known text ──────────────────────────────────────
def create_test_image():
    img = Image.new("RGB", (800, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Try to use a system font for clearer text
    try:
        font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 36)
    except Exception:
        font = ImageFont.load_default()
    draw.text((30, 60), "VoxCRM CONNECT. MANAGE. GROW.", fill=(0, 0, 0), font=font)
    draw.text((30, 120), "Task 5: Multimodal Vision Test 2026", fill=(20, 20, 20), font=font)
    return img

def test_pytesseract(img):
    print("\n[1] Testing pytesseract...")
    import shutil
    import importlib
    try:
        pytesseract = importlib.import_module("pytesseract")
        # Find tesseract binary
        tess_paths = [
            shutil.which("tesseract"),
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        tess_bin = next((p for p in tess_paths if p and os.path.exists(p)), None)
        if tess_bin:
            pytesseract.pytesseract.tesseract_cmd = tess_bin
            print(f"   Tesseract binary: {tess_bin}")
            enhanced = ImageEnhance.Contrast(img.convert("L")).enhance(2.0)
            text = pytesseract.image_to_string(enhanced).strip()
            if text:
                print(f"   ✅ OCR SUCCESS!\n   Extracted text:\n   ---\n   {text}\n   ---")
                return True
            else:
                print("   ⚠️ Tesseract ran but returned empty text.")
        else:
            print("   ❌ Tesseract binary not found on system.")
    except ImportError:
        print("   ❌ pytesseract not installed. Run: pip install pytesseract")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    return False

def test_subprocess_tesseract(img):
    print("\n[2] Testing subprocess Tesseract CLI...")
    import subprocess
    import shutil
    tess_paths = [
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    ]
    tess_bin = next((p for p in tess_paths if p and os.path.exists(p)), None)
    if not tess_bin:
        print("   ❌ Tesseract binary not found.")
        return False
    try:
        tmp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "ocr_test.png")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        img.save(tmp)
        res = subprocess.run([tess_bin, tmp, "stdout"], capture_output=True, text=True, timeout=10)
        os.remove(tmp)
        if res.stdout.strip():
            print(f"   ✅ CLI OCR SUCCESS!\n   Extracted:\n   ---\n   {res.stdout.strip()}\n   ---")
            return True
        else:
            print(f"   ⚠️ Tesseract CLI ran but no output. stderr: {res.stderr[:100]}")
    except Exception as e:
        print(f"   ❌ CLI Error: {e}")
    return False

def test_easyocr(img):
    print("\n[3] Testing EasyOCR...")
    import importlib
    try:
        easyocr = importlib.import_module("easyocr")
        print("   Loading EasyOCR reader (first run may download model ~40MB)...")
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        results = reader.readtext(buf.getvalue())
        text = "\n".join([item[1] for item in results])
        if text.strip():
            print(f"   ✅ EasyOCR SUCCESS!\n   Extracted:\n   ---\n   {text}\n   ---")
            return True
        else:
            print("   ⚠️ EasyOCR ran but returned empty text.")
    except ImportError:
        print("   ❌ easyocr not installed. Run: pip install easyocr")
    except Exception as e:
        print(f"   ❌ EasyOCR error: {e}")
    return False

if __name__ == "__main__":
    print("=" * 60)
    print("  Cognova OCR Engine Installation Test")
    print("=" * 60)
    img = create_test_image()
    # Save test image for inspection
    report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    os.makedirs(report_dir, exist_ok=True)
    img.save(os.path.join(report_dir, "ocr_test_image.png"))
    print(f"Test image saved to: reports/ocr_test_image.png")

    ok1 = test_pytesseract(img)
    ok2 = test_subprocess_tesseract(img) if not ok1 else True
    ok3 = test_easyocr(img) if not ok1 and not ok2 else None

    print("\n" + "=" * 60)
    if ok1 or ok2:
        print("  ✅ OCR is WORKING! Text extraction will succeed in Streamlit.")
        print("  Restart Streamlit: streamlit run app.py")
    elif ok3:
        print("  ✅ EasyOCR is WORKING! Text extraction will succeed in Streamlit.")
    else:
        print("  ❌ No OCR engine found. Please:")
        print("  1. Run install_tesseract.bat (installs Tesseract + pytesseract)")
        print("  2. OR: pip install easyocr")
    print("=" * 60)
