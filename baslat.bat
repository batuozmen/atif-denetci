@echo off
chcp 65001 > nul
title TUBITAK Akademik Makale ve Semantik Atif Denetcisi
echo ============================================================================
echo   TUBITAK AKADEMIK MAKALE, KAYNAKCA VE SEMANTIK ATIF DOGRULAMA SISTEMI
echo ============================================================================
echo.
echo [1/2] Sanal ortam (.venv) kontrol ediliyor...
cd /d "%~dp0"

set "STREAMLIT_EXE="
if exist ".venv\Scripts\streamlit.exe" (
    set "STREAMLIT_EXE=.venv\Scripts\streamlit.exe"
) else if exist "%USERPROFILE%\Desktop\zelisimv1\.venv\Scripts\streamlit.exe" (
    set "STREAMLIT_EXE=%USERPROFILE%\Desktop\zelisimv1\.venv\Scripts\streamlit.exe"
) else if exist "%USERPROFILE%\OneDrive\Desktop\zelisimv1\.venv\Scripts\streamlit.exe" (
    set "STREAMLIT_EXE=%USERPROFILE%\OneDrive\Desktop\zelisimv1\.venv\Scripts\streamlit.exe"
) else if exist "%USERPROFILE%\OneDrive\Masaüstü\zelisimv1\.venv\Scripts\streamlit.exe" (
    set "STREAMLIT_EXE=%USERPROFILE%\OneDrive\Masaüstü\zelisimv1\.venv\Scripts\streamlit.exe"
)

if "%STREAMLIT_EXE%"=="" (
    echo HATA: streamlit.exe bulunamadi! Lutfen .venv kurulumunun tamamlandigindan emin olun.
    pause
    exit /b 1
)

echo       Kullanilan ortam: %STREAMLIT_EXE%
echo [2/2] Web arayuzu baslatiliyor (Tarayicinizda otomatik acilacak)...
echo       Kapatmak icin bu pencereyi kapatabilirsiniz.
echo.
"%STREAMLIT_EXE%" run app.py
pause
