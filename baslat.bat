@echo off
title TUBITAK Akademik Makale ve Semantik Atif Denetcisi
echo ============================================================================
echo   TUBITAK AKADEMIK MAKALE, KAYNAKCA VE SEMANTIK ATIF DOGRULAMA SISTEMI
echo ============================================================================
echo.
echo [1/2] Sanal ortam (.venv) kontrol ediliyor...
cd /d "%~dp0"
set "STREAMLIT_EXE=.venv\Scripts\streamlit.exe"
if not exist "%STREAMLIT_EXE%" set "STREAMLIT_EXE=%USERPROFILE%\Desktop\zelisimv1\.venv\Scripts\streamlit.exe"
if not exist "%STREAMLIT_EXE%" (
    echo HATA: streamlit.exe bulunamadi! Lutfen kurulumun tamamlandigindan emin olun.
    pause
    exit /b 1
)
echo       Kullanilan ortam: %STREAMLIT_EXE%
echo [2/2] Web arayuzu baslatiliyor (Tarayicinizda otomatik acilacak)...
echo       Kapatmak icin bu pencereyi kapatabilirsiniz.
echo.
"%STREAMLIT_EXE%" run app.py
pause
