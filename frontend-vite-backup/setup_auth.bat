@echo off
cls
echo ================================================================
echo       TEKNOFEST 2025 - Authentication System Setup
echo ================================================================
echo.

cd /d C:\Users\husey\teknofest-2025-egitim-eylemci\frontend

echo [STEP 1] Installing lucide-react for icons...
echo.
call npm install lucide-react

echo.
echo [STEP 2] Starting development server...
echo.

echo ================================================================
echo      Authentication System Ready!
echo ================================================================
echo.
echo Login sayfasına erişmek için:
echo 1. Tarayıcıda http://localhost:5173 adresine gidin
echo 2. Demo modda herhangi bir email/şifre ile giriş yapabilirsiniz
echo.
echo Özellikler:
echo - Öğrenci/Öğretmen girişi
echo - Form validation
echo - Şifre güç göstergesi
echo - JWT token yönetimi
echo - Auto-refresh token
echo - Responsive tasarım
echo.

call npm run dev
