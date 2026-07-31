@echo off
title Tesseract OCR + pytesseract Installer
cd /d c:\Users\bhuva\Cognova
echo.
echo ============================================================
echo  STEP 1: Install pytesseract Python package
echo ============================================================
pip install pytesseract pillow --quiet
if %errorlevel% neq 0 (
    pip install pytesseract pillow
)
echo pytesseract installed OK.

echo.
echo ============================================================
echo  STEP 2: Check if Tesseract binary is already installed
echo ============================================================
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    echo Tesseract already installed at C:\Program Files\Tesseract-OCR\
    goto :test
)
if exist "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe" (
    echo Tesseract already installed at C:\Program Files (x86)\Tesseract-OCR\
    goto :test
)

echo Tesseract NOT found. Downloading from GitHub...
echo Download size: ~50 MB (Tesseract v5.4.0 official Windows installer)
echo.

echo ============================================================
echo  STEP 3: Download Tesseract Installer (GitHub Releases)
echo ============================================================
set "TESS_URL=https://github.com/UB-Mannheim/tesseract/releases/download/v5.4.0.20240606/tesseract-ocr-w64-setup-5.4.0.20240606.exe"
set "TESS_DEST=%TEMP%\tesseract-setup.exe"

echo Downloading from: %TESS_URL%
powershell -Command "Invoke-WebRequest -Uri '%TESS_URL%' -OutFile '%TESS_DEST%' -UseBasicParsing"
if %errorlevel% neq 0 (
    echo PowerShell download failed. Trying curl...
    curl -L -o "%TESS_DEST%" "%TESS_URL%"
)

if not exist "%TESS_DEST%" (
    echo.
    echo ERROR: Download failed. Please manually download from:
    echo   https://github.com/UB-Mannheim/tesseract/releases/download/v5.4.0.20240606/tesseract-ocr-w64-setup-5.4.0.20240606.exe
    echo Then run it and install to: C:\Program Files\Tesseract-OCR\
    pause
    exit /b 1
)

echo Download complete!
echo.
echo ============================================================
echo  STEP 4: Running Tesseract Installer
echo ============================================================
echo NOTE: A UAC prompt may appear. Click YES to allow installation.
"%TESS_DEST%" /SILENT /NORESTART /DIR="C:\Program Files\Tesseract-OCR"
echo Installer finished.

:test
echo.
echo ============================================================
echo  STEP 5: Verifying Tesseract
echo ============================================================
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    echo [OK] Found at: C:\Program Files\Tesseract-OCR\tesseract.exe
    "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
) else (
    echo [WARN] Not found at expected path. Check Program Files.
)

echo.
echo ============================================================
echo  STEP 6: Running OCR test
echo ============================================================
python scripts\test_ocr_install.py

echo.
echo ============================================================
echo  Done! Restart Streamlit: streamlit run app.py
echo ============================================================
pause
