@echo off
cd /d c:\Users\bhuva\Cognova
echo Running model discovery (listing what your API key can access)...
python scripts\discover_gemini_models.py
echo.
echo ====== DISCOVERED MODELS ======
type reports\gemini_models_list.txt
echo.
echo ====== Done. Now restart Streamlit to apply the fix: ======
echo streamlit run app.py
pause
