import React, { useState, useEffect } from 'react';
import { 
  Mail, 
  Lock, 
  Eye, 
  EyeOff, 
  LogIn, 
  AlertCircle,
  Loader2,
  Shield,
  Check,
  User,
  BookOpen,
  Trophy
} from 'lucide-react';

interface LoginFormData {
  email: string;
  password: string;
  rememberMe: boolean;
}

interface ValidationErrors {
  email?: string;
  password?: string;
  general?: string;
}

const LoginPage: React.FC = () => {
  const [formData, setFormData] = useState<LoginFormData>({
    email: '',
    password: '',
    rememberMe: false
  });

  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<ValidationErrors>({});
  const [showTwoFactor, setShowTwoFactor] = useState(false);
  const [twoFactorCode, setTwoFactorCode] = useState('');
  const [loginAttempts, setLoginAttempts] = useState(0);
  const [isBlocked, setIsBlocked] = useState(false);
  const [blockTimer, setBlockTimer] = useState(0);

  useEffect(() => {
    // Check for saved credentials
    const savedEmail = localStorage.getItem('rememberedEmail');
    if (savedEmail) {
      setFormData(prev => ({ ...prev, email: savedEmail, rememberMe: true }));
    }

    // Check if user is blocked
    const blockEndTime = localStorage.getItem('loginBlockEndTime');
    if (blockEndTime) {
      const remainingTime = parseInt(blockEndTime) - Date.now();
      if (remainingTime > 0) {
        setIsBlocked(true);
        setBlockTimer(Math.ceil(remainingTime / 1000));
      } else {
        localStorage.removeItem('loginBlockEndTime');
      }
    }
  }, []);

  useEffect(() => {
    // Countdown timer for blocked state
    if (blockTimer > 0) {
      const timer = setTimeout(() => {
        setBlockTimer(blockTimer - 1);
      }, 1000);

      return () => clearTimeout(timer);
    } else if (isBlocked && blockTimer === 0) {
      setIsBlocked(false);
      setLoginAttempts(0);
      localStorage.removeItem('loginBlockEndTime');
    }
  }, [blockTimer, isBlocked]);

  const validateEmail = (email: string): boolean => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const validateForm = (): boolean => {
    const newErrors: ValidationErrors = {};

    if (!formData.email) {
      newErrors.email = 'E-posta adresi gereklidir';
    } else if (!validateEmail(formData.email)) {
      newErrors.email = 'Geçerli bir e-posta adresi giriniz';
    }

    if (!formData.password) {
      newErrors.password = 'Şifre gereklidir';
    } else if (formData.password.length < 6) {
      newErrors.password = 'Şifre en az 6 karakter olmalıdır';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));

    // Clear error for this field
    if (errors[name as keyof ValidationErrors]) {
      setErrors(prev => ({ ...prev, [name]: undefined }));
    }
  };

  const handleLogin = async () => {
    if (isBlocked) {
      setErrors({ general: `Çok fazla başarısız deneme. ${blockTimer} saniye bekleyiniz.` });
      return;
    }

    if (!validateForm()) {
      return;
    }

    setIsLoading(true);
    setErrors({});

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Simulate login logic
      const isValidCredentials = formData.email === 'demo@teknofest.com' && formData.password === 'Demo123!';

      if (!isValidCredentials) {
        const newAttempts = loginAttempts + 1;
        setLoginAttempts(newAttempts);

        if (newAttempts >= 5) {
          const blockEndTime = Date.now() + 300000; // 5 minutes
          localStorage.setItem('loginBlockEndTime', blockEndTime.toString());
          setIsBlocked(true);
          setBlockTimer(300);
          setErrors({ general: 'Çok fazla başarısız deneme. 5 dakika bekleyiniz.' });
        } else {
          setErrors({ 
            general: `Geçersiz e-posta veya şifre. ${5 - newAttempts} deneme hakkınız kaldı.` 
          });
        }
        return;
      }

      // Check if 2FA is enabled (simulate)
      const has2FA = formData.email === 'demo@teknofest.com';
      
      if (has2FA) {
        setShowTwoFactor(true);
      } else {
        // Successful login
        if (formData.rememberMe) {
          localStorage.setItem('rememberedEmail', formData.email);
        } else {
          localStorage.removeItem('rememberedEmail');
        }

        // Store JWT token
        const mockToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...';
        localStorage.setItem('authToken', mockToken);
        localStorage.setItem('refreshToken', 'refresh_token_here');
        
        // Reset login attempts
        setLoginAttempts(0);
        
        // Redirect to dashboard
        console.log('Login successful! Redirecting to dashboard...');
        window.location.href = '/dashboard';
      }
    } catch (error) {
      setErrors({ general: 'Bir hata oluştu. Lütfen tekrar deneyiniz.' });
    } finally {
      setIsLoading(false);
    }
  };

  const handle2FASubmit = async () => {
    if (twoFactorCode.length !== 6) {
      setErrors({ general: 'Lütfen 6 haneli kodu giriniz' });
      return;
    }

    setIsLoading(true);

    try {
      // Simulate 2FA verification
      await new Promise(resolve => setTimeout(resolve, 1500));

      if (twoFactorCode === '123456') {
        // Successful 2FA
        if (formData.rememberMe) {
          localStorage.setItem('rememberedEmail', formData.email);
        }

        const mockToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...';
        localStorage.setItem('authToken', mockToken);
        
        console.log('2FA successful! Redirecting...');
        window.location.href = '/dashboard';
      } else {
        setErrors({ general: 'Geçersiz doğrulama kodu' });
      }
    } catch (error) {
      setErrors({ general: 'Doğrulama başarısız' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSocialLogin = async (provider: string) => {
    setIsLoading(true);
    try {
      // Simulate OAuth flow
      await new Promise(resolve => setTimeout(resolve, 1500));
      console.log(`Logging in with ${provider}...`);
      window.location.href = `/auth/${provider}`;
    } catch (error) {
      setErrors({ general: `${provider} ile giriş başarısız` });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isLoading && !isBlocked) {
      handleLogin();
    }
  };

  // Two-Factor Authentication Screen
  if (showTwoFactor) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-8">
          <div className="text-center mb-8">
            <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <Shield className="w-10 h-10 text-white" />
            </div>
            <h2 className="text-3xl font-bold text-gray-800 mb-2">İki Faktörlü Doğrulama</h2>
            <p className="text-gray-600">
              Telefonunuzdaki authenticator uygulamasından 6 haneli kodu giriniz
            </p>
          </div>

          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Doğrulama Kodu
              </label>
              <input
                type="text"
                value={twoFactorCode}
                onChange={(e) => {
                  const value = e.target.value.replace(/\D/g, '');
                  if (value.length <= 6) {
                    setTwoFactorCode(value);
                  }
                }}
                placeholder="000000"
                maxLength={6}
                className="w-full px-4 py-3 text-center text-2xl font-mono tracking-widest border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                autoFocus
              />
            </div>

            {errors.general && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-center gap-2">
                <AlertCircle className="w-5 h-5" />
                <span className="text-sm">{errors.general}</span>
              </div>
            )}

            <button
              onClick={handle2FASubmit}
              disabled={isLoading || twoFactorCode.length !== 6}
              className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white py-3 rounded-lg font-medium hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Doğrulanıyor...
                </>
              ) : (
                <>
                  <Check className="w-5 h-5" />
                  Doğrula ve Giriş Yap
                </>
              )}
            </button>

            <button
              onClick={() => {
                setShowTwoFactor(false);
                setTwoFactorCode('');
              }}
              className="w-full text-gray-600 hover:text-gray-800 font-medium"
            >
              Geri Dön
            </button>

            <div className="text-center">
              <a href="#" className="text-sm text-blue-600 hover:text-blue-700">
                Kodunuzu mu kaybettiniz?
              </a>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Main Login Screen
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 flex">
      {/* Left Side - Login Form */}
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-md">
          {/* Logo */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center gap-2 mb-6">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                <span className="text-white text-2xl font-bold">T</span>
              </div>
              <h1 className="text-3xl font-bold text-gray-800">TEKNOFEST 2025</h1>
            </div>
            <h2 className="text-2xl font-semibold text-gray-700">Hoş Geldiniz!</h2>
            <p className="text-gray-600 mt-2">Eğitim platformuna giriş yapın</p>
          </div>

          {/* Login Form */}
          <div className="space-y-6" onKeyPress={handleKeyPress}>
            {/* Email Input */}
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-2">
                E-posta Adresi
              </label>
              <div className="relative">
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={formData.email}
                  onChange={handleInputChange}
                  placeholder="ornek@email.com"
                  className={`w-full pl-12 pr-4 py-3 border-2 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all ${
                    errors.email ? 'border-red-300' : 'border-gray-300'
                  }`}
                  disabled={isBlocked}
                />
                <Mail className="absolute left-4 top-3.5 w-5 h-5 text-gray-400" />
              </div>
              {errors.email && (
                <p className="mt-1 text-sm text-red-600 flex items-center gap-1">
                  <AlertCircle className="w-4 h-4" />
                  {errors.email}
                </p>
              )}
            </div>

            {/* Password Input */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-2">
                Şifre
              </label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  id="password"
                  name="password"
                  value={formData.password}
                  onChange={handleInputChange}
                  placeholder="••••••••"
                  className={`w-full pl-12 pr-12 py-3 border-2 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all ${
                    errors.password ? 'border-red-300' : 'border-gray-300'
                  }`}
                  disabled={isBlocked}
                />
                <Lock className="absolute left-4 top-3.5 w-5 h-5 text-gray-400" />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-4 top-3.5 text-gray-400 hover:text-gray-600"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
              {errors.password && (
                <p className="mt-1 text-sm text-red-600 flex items-center gap-1">
                  <AlertCircle className="w-4 h-4" />
                  {errors.password}
                </p>
              )}
            </div>

            {/* Remember Me & Forgot Password */}
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="rememberMe"
                  name="rememberMe"
                  checked={formData.rememberMe}
                  onChange={handleInputChange}
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <label htmlFor="rememberMe" className="ml-2 text-sm text-gray-700">
                  Beni hatırla
                </label>
              </div>
              <a href="/forgot-password" className="text-sm text-blue-600 hover:text-blue-700">
                Şifremi unuttum
              </a>
            </div>

            {/* Error Message */}
            {errors.general && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-center gap-2">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                <span className="text-sm">{errors.general}</span>
              </div>
            )}

            {/* Block Timer */}
            {isBlocked && (
              <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 px-4 py-3 rounded-lg">
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-5 h-5" />
                  <span className="text-sm font-medium">Hesap geçici olarak kilitlendi</span>
                </div>
                <p className="text-sm mt-1">
                  Lütfen {Math.floor(blockTimer / 60)}:{(blockTimer % 60).toString().padStart(2, '0')} dakika bekleyiniz
                </p>
              </div>
            )}

            {/* Submit Button */}
            <button
              onClick={handleLogin}
              disabled={isLoading || isBlocked}
              className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white py-3 rounded-lg font-medium hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Giriş yapılıyor...
                </>
              ) : (
                <>
                  <LogIn className="w-5 h-5" />
                  Giriş Yap
                </>
              )}
            </button>
          </div>

          {/* Divider */}
          <div className="relative my-8">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-300"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-4 bg-white text-gray-500">veya</span>
            </div>
          </div>

          {/* Social Login */}
          <div className="space-y-3">
            <button
              onClick={() => handleSocialLogin('google')}
              disabled={isLoading || isBlocked}
              className="w-full flex items-center justify-center gap-3 px-4 py-3 border-2 border-gray-300 rounded-lg hover:bg-gray-50 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
              <span className="text-gray-700 font-medium">Google ile Giriş Yap</span>
            </button>

            <button
              onClick={() => handleSocialLogin('microsoft')}
              disabled={isLoading || isBlocked}
              className="w-full flex items-center justify-center gap-3 px-4 py-3 border-2 border-gray-300 rounded-lg hover:bg-gray-50 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24">
                <path fill="#F25022" d="M11.4 11.4H0V0h11.4v11.4z"/>
                <path fill="#00A4EF" d="M24 11.4H12.6V0H24v11.4z"/>
                <path fill="#7FBA00" d="M11.4 24H0V12.6h11.4V24z"/>
                <path fill="#FFB900" d="M24 24H12.6V12.6H24V24z"/>
              </svg>
              <span className="text-gray-700 font-medium">Microsoft ile Giriş Yap</span>
            </button>
          </div>

          {/* Sign Up Link */}
          <p className="text-center mt-8 text-gray-600">
            Hesabınız yok mu?{' '}
            <a href="/register" className="text-blue-600 hover:text-blue-700 font-medium">
              Kayıt Olun
            </a>
          </p>

          {/* Demo Credentials */}
          <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
            <p className="text-sm text-blue-800 font-medium mb-1">Demo Hesap:</p>
            <p className="text-xs text-blue-700">E-posta: demo@teknofest.com</p>
            <p className="text-xs text-blue-700">Şifre: Demo123!</p>
            <p className="text-xs text-blue-700 mt-1">2FA Kodu: 123456</p>
          </div>
        </div>
      </div>

      {/* Right Side - Info Panel */}
      <div className="hidden lg:flex flex-1 bg-gradient-to-br from-blue-600 to-purple-700 p-12 items-center justify-center">
        <div className="max-w-lg text-white">
          <h2 className="text-4xl font-bold mb-6">
            Geleceğin Eğitimi, Bugün Başlıyor!
          </h2>
          <p className="text-lg mb-8 text-blue-100">
            TEKNOFEST 2025 Eğitim Teknolojileri platformuna hoş geldiniz. 
            Yapay zeka destekli öğrenme deneyimi ile eğitimde yeni bir çağ başlıyor.
          </p>
          
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-white/20 rounded-lg flex items-center justify-center">
                <User className="w-6 h-6" />
              </div>
              <div>
                <h3 className="font-semibold">10,000+ Öğrenci</h3>
                <p className="text-sm text-blue-100">Aktif kullanıcı</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-white/20 rounded-lg flex items-center justify-center">
                <BookOpen className="w-6 h-6" />
              </div>
              <div>
                <h3 className="font-semibold">500+ Ders</h3>
                <p className="text-sm text-blue-100">İnteraktif içerik</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-white/20 rounded-lg flex items-center justify-center">
                <Trophy className="w-6 h-6" />
              </div>
              <div>
                <h3 className="font-semibold">%95 Başarı</h3>
                <p className="text-sm text-blue-100">Öğrenci memnuniyeti</p>
              </div>
            </div>
          </div>

          <div className="mt-12 p-6 bg-white/10 backdrop-blur-sm rounded-xl border border-white/20">
            <h4 className="font-semibold mb-3">Güvenlik Özellikleri</h4>
            <ul className="space-y-2 text-sm text-blue-100">
              <li className="flex items-center gap-2">
                <Check className="w-4 h-4" />
                İki faktörlü doğrulama (2FA)
              </li>
              <li className="flex items-center gap-2">
                <Check className="w-4 h-4" />
                256-bit SSL şifreleme
              </li>
              <li className="flex items-center gap-2">
                <Check className="w-4 h-4" />
                KVKK uyumlu veri güvenliği
              </li>
              <li className="flex items-center gap-2">
                <Check className="w-4 h-4" />
                Otomatik oturum sonlandırma
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;