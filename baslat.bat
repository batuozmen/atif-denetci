@echo off
chcp 65001 > nul
title TUBITAK Akademik Makale ve Semantik Atif Denetcisi
echo ============================================================================
echo   TUBITAK AKADEMIK MAKALE, KAYNAKCA VE SEMANTIK ATIF DOGRULAMA SISTEMI
echo ============================================================================
echo.
echo [1/2] Sanal ortam (.venv) kontrol ediliyor...
cd /d "%~dp0"
if not exist ".venv\Scripts\streamlit.exe" (
    echo HATA: .venv bulunamadi! Lutfen kurulumun tamamlandigindan emin olun.
    pause
    exit /b 1
)
echo [2/2] Web arayuzu baslatiliyor (Tarayicinizda otomatik acilacak)...
echo       Kapatmak icin bu pencereyi kapatabilirsiniz.
echo.
".venv\Scripts\streamlit.exe" run app.py
pause
