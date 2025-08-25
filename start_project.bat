@echo off
echo ========================================
echo TEKNOFEST 2025 - Eğitim Teknolojileri
echo ========================================
echo.
echo Başlatma seçenekleri:
echo.
echo [1] Frontend (Next.js)
echo [2] Backend API Server
echo [3] Her İkisi
echo [4] Çıkış
echo.
set /p choice="Seçiminiz (1-4): "

if "%choice%"=="1" goto frontend
if "%choice%"=="2" goto backend
if "%choice%"=="3" goto both
if "%choice%"=="4" goto end

:frontend
echo.
echo Frontend başlatılıyor...
cd frontend
start cmd /k "npm run dev"
echo Frontend http://localhost:3000 adresinde başlatıldı
pause
goto end

:backend
echo.
echo Backend başlatılıyor...
start cmd /k "python -m src.mcp_server.production_server"
echo Backend http://localhost:8000 adresinde başlatıldı
echo API Dokümantasyonu: http://localhost:8000/docs
pause
goto end

:both
echo.
echo Frontend ve Backend başlatılıyor...
cd frontend
start cmd /k "npm run dev"
cd ..
start cmd /k "python -m src.mcp_server.production_server"
echo.
echo ========================================
echo Servisler başlatıldı:
echo Frontend: http://localhost:3000
echo Backend: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo ========================================
pause
goto end

:end
echo.
echo Program kapatılıyor...
exit