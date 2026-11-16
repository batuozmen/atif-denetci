@echo off
title TUBITAK Akademik Makale ve Semantik Atif Denetcisi
echo ============================================================================
echo   TUBITAK AKADEMIK MAKALE, KAYNAKCA VE SEMANTIK ATIF DOGRULAMA SISTEMI
echo ============================================================================
echo.
echo [1/2] Sanal ortam kontrol ediliyor...
cd /d "%~dp0"
set "STREAMLIT_EXE=.venv\Scripts\streamlit.exe"
if not exist "%STREAMLIT_EXE%" set "STREAMLIT_EXE=%USERPROFILE%\Desktop\zelisimv1\.venv\Scripts\streamlit.exe"
if not exist "%STREAMLIT_EXE%" goto :KURULUM_GEREKLI
goto :BASLAT

:KURULUM_GEREKLI
echo       Bu bilgisayarda sanal ortam henuz kurulmamis.
echo       Otomatik kurulum sihirbazi kurulum_yap.bat baslatiliyor...
if exist "kurulum_yap.bat" call kurulum_yap.bat
set "STREAMLIT_EXE=.venv\Scripts\streamlit.exe"
if not exist "%STREAMLIT_EXE%" goto :HATA

:BASLAT
echo       Kullanilan ortam: %STREAMLIT_EXE%
echo [2/2] Web arayuzu baslatiliyor...
echo       Kapatmak icin bu pencereyi kapatabilirsiniz.
echo.
"%STREAMLIT_EXE%" run app.py
pause
exit /b 0

:HATA
echo HATA: streamlit.exe bulunamadi! Lutfen once kurulum_yap.bat dosyasini calistirin.
pause
exit /b 1
