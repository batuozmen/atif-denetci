@echo off
title TUBITAK Atif Denetcisi - Yeni Bilgisayar Tam Kurulum Sihirbazi
echo ============================================================================
echo   TUBITAK AKADEMIK MAKALE, KAYNAKCA VE SEMANTIK ATIF DOGRULAMA SISTEMI
echo   YENI BILGISAYAR %100 TAM FONKSIYONLU OTOMATIK KURULUM SIHIRBAZI
echo ============================================================================
echo.
cd /d "%~dp0"

echo [1/4] Python kurulumu kontrol ediliyor...
python --version >nul 2>&1
if errorlevel 1 (
    echo       Python bulunamadi! Windows Winget ile Python 3.12 otomatik kuruluyor...
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo HATA: Python otomatik kurulamadi. Lutfen https://www.python.org/downloads/ adresinden
        echo Python 3.12 indirip kurarken "Add Python to PATH" kutucugunu isaretleyin.
        pause
        exit /b 1
    )
)
echo       [OK] Python hazir.
echo.

echo [2/4] Sanal ortam (.venv) ve gerekli yapay zeka kutuphaneleri kuruluyor...
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo       Bazilari guncel surumle tekrar deneniyor...
    ".venv\Scripts\python.exe" -m pip install streamlit pandas pymupdf python-docx requests beautifulsoup4 lxml sentence-transformers torch ollama scikit-learn
)
echo       [OK] Tum Python ve Vektor Arama kutuphaneleri hazir.
echo.

echo [3/4] Yerel Yapay Zeka Hakemi (Ollama) kontrol ediliyor...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo       Ollama bulunamadi! Windows Winget ile Ollama otomatik kuruluyor...
    winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements
)
echo       [OK] Ollama kontrolu tamamlandi.
echo.

echo [4/4] Yapay Zeka Hakem Modeli (qwen2.5:3b) indiriliyor / dogrulaniyor...
start "" /min ollama serve >nul 2>&1
timeout /t 3 /nobreak >nul
ollama pull qwen2.5:3b
echo.
echo ============================================================================
echo   TEBRIKLER! KURULUM %100 TAMAMLANDI.
echo   Artik [baslat.bat] dosyasina cift tiklayarak sistemi tam kapasiteyle
echo   (8 Kanalli Internet Avcisi + Cok Dilli RAG + Yerel Yapay Zeka Hakemi)
echo   calistirabilirsiniz!
echo ============================================================================
pause
