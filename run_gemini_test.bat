@echo off
cd /d c:\Users\bhuva\Cognova
python scripts\run_gemini_test_directly.py
echo.
echo ====== OUTPUT FROM LOG ======
type reports\gemini_direct_response.txt
pause
