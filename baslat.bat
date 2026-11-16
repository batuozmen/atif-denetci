@echo off
title TUBITAK Akademik Makale ve Semantik Atif Denetcisi
echo ============================================================================
echo   TUBITAK AKADEMIK MAKALE, KAYNAKCA VE SEMANTIK ATIF DOGRULAMA SISTEMI
echo ============================================================================
echo.
echo [1/3] Sanal ortam (.venv) kontrol ediliyor...
cd /d "%~dp0"
set "STREAMLIT_EXE=.venv\Scripts\streamlit.exe"
if not exist "%STREAMLIT_EXE%" set "STREAMLIT_EXE=%USERPROFILE%\Desktop\zelisimv1\.venv\Scripts\streamlit.exe"
if not exist "%STREAMLIT_EXE%" (
    echo       Bu bilgisayarda sanal ortam (.venv) henuz kurulmamis.
    echo       Otomatik kurulum sihirbazi [kurulum_yap.bat] baslatiliyor...
    call kurulum_yap.bat
    set "STREAMLIT_EXE=.venv\Scripts\streamlit.exe"
)
if not exist "%STREAMLIT_EXE%" (
    echo HATA: streamlit.exe bulunamadi! Lutfen once kurulum_yap.bat dosyasini calistirin.
    pause
    exit /b 1
)
echo       Kullanilan ortam: %STREAMLIT_EXE%
echo [2/3] Yerel Yapay Zeka Hakemi (Ollama) servisi kontrol ediliyor...
ollama --version >nul 2>&1
if not errorlevel 1 (
    start "" /min ollama serve >nul 2>&1
)
echo [3/3] Web arayuzu baslatiliyor (Tarayicinizda otomatik acilacak)...
echo       Kapatmak icin bu pencereyi kapatabilirsiniz.
echo.
"%STREAMLIT_EXE%" run app.py
pause
