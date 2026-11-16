@echo off
title TUBITAK Atif Denetcisi - Yeni Bilgisayar Tam Kurulum Sihirbazi
echo ============================================================================
echo   TUBITAK AKADEMIK MAKALE, KAYNAKCA VE SEMANTIK ATIF DOGRULAMA SISTEMI
echo   YENI BILGISAYAR TAM FONKSIYONLU OTOMATIK KURULUM SIHIRBAZI
echo ============================================================================
echo.
cd /d "%~dp0"

echo [1/4] Python kurulumu kontrol ediliyor...
python --version >nul 2>&1
if errorlevel 1 goto :PYTHON_KUR
goto :PYTHON_TAMAM

:PYTHON_KUR
echo       Python bulunamadi! Windows Winget ile Python 3.12 otomatik kuruluyor...
winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
python --version >nul 2>&1
if errorlevel 1 goto :PYTHON_HATA

:PYTHON_TAMAM
echo       [OK] Python hazir.
echo.

echo [2/4] Sanal ortam ve gerekli yapay zeka kutuphaneleri kuruluyor...
if not exist ".venv\Scripts\python.exe" python -m venv .venv
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
echo       [OK] Tum Python ve Vektor Arama kutuphaneleri hazir.
echo.

echo [3/4] Yerel Yapay Zeka Hakemi Ollama kontrol ediliyor...
ollama --version >nul 2>&1
if errorlevel 1 winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements
echo       [OK] Ollama kontrolu tamamlandi.
echo.

echo [4/4] Yapay Zeka Hakem Modeli qwen2.5:3b indiriliyor ve dogrulaniyor...
ollama pull qwen2.5:3b
echo.
echo ============================================================================
echo   TEBRIKLER! KURULUM TAMAMLANDI.
echo   Artik baslat.bat dosyasina cift tiklayarak sistemi tam kapasiteyle
echo   calistirabilirsiniz!
echo ============================================================================
pause
exit /b 0

:PYTHON_HATA
echo HATA: Python otomatik kurulamadi. Lutfen https://www.python.org/downloads/ adresinden
echo Python 3.12 indirip kurarken Add Python to PATH kutucugunu isaretleyin.
pause
exit /b 1
