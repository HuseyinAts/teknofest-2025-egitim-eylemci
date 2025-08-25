import React, { useState } from 'react';
import { 
  User, 
  Lock, 
  Mail, 
  Eye, 
  EyeOff, 
  GraduationCap,
  BookOpen,
  AlertCircle,
  CheckCircle,
  Loader2,
  School,
  Hash,
  Users,
  Sparkles,
  ChevronRight,
  Info
} from 'lucide-react';
import { useAuth } from '../../contexts/AuthContext';

const AuthPage = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [role, setRole] = useState<'student' | 'teacher'>('student');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [successMessage, setSuccessMessage] = useState('');
  
  // Form states
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    name: '',
    grade: '',
    school: '',
    teacherCode: ''
  });

  const [passwordStrength, setPasswordStrength] = useState({
    score: 0,
    label: 'Zayıf',
    color: 'bg-red-500'
  });

  const { login, register } = useAuth();

  // Password strength calculator
  const calculatePasswordStrength = (password: string) => {
    let score = 0;
    
    if (password.length >= 8) score++;
    if (password.length >= 12) score++;
    if (/[A-Z]/.test(password)) score++;
    if (/[a-z]/.test(password)) score++;
    if (/[0-9]/.test(password)) score++;
    if (/[!@#$%^&*]/.test(password)) score++;
    
    const strength = 
      score <= 2 ? { label: 'Zayıf', color: 'bg-red-500' } :
      score <= 4 ? { label: 'Orta', color: 'bg-yellow-500' } :
      { label: 'Güçlü', color: 'bg-green-500' };
    
    setPasswordStrength({
      score: Math.min((score / 6) * 100, 100),
      ...strength
    });
  };

  // Handle input change
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    
    // Clear error for this field
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: '' }));
    }
    
    // Calculate password strength
    if (name === 'password') {
      calculatePasswordStrength(value);
    }
  };

  // Validate form
  const validateForm = () => {
    const newErrors: Record<string, string> = {};
    
    // Email validation
    if (!formData.email) {
      newErrors.email = 'Email adresi gereklidir';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      newErrors.email = 'Geçerli bir email adresi giriniz';
    }
    
    // Password validation
    if (!formData.password) {
      newErrors.password = 'Şifre gereklidir';
    } else if (formData.password.length < 6) {
      newErrors.password = 'Şifre en az 6 karakter olmalıdır';
    }
    
    // Registration specific validations
    if (!isLogin) {
      if (!formData.name) {
        newErrors.name = 'Ad Soyad gereklidir';
      }
      
      if (!formData.confirmPassword) {
        newErrors.confirmPassword = 'Şifre tekrarı gereklidir';
      } else if (formData.password !== formData.confirmPassword) {
        newErrors.confirmPassword = 'Şifreler eşleşmiyor';
      }
      
      if (role === 'student' && !formData.grade) {
        newErrors.grade = 'Sınıf seçimi gereklidir';
      }
      
      if (role === 'teacher' && !formData.teacherCode) {
        newErrors.teacherCode = 'Öğretmen kodu gereklidir';
      }
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Handle submit
  const handleSubmit = async () => {
    if (!validateForm()) return;
    
    setIsLoading(true);
    setSuccessMessage('');
    
    try {
      if (isLogin) {
        await login({ email: formData.email, password: formData.password, role });
        setSuccessMessage('Giriş başarılı! Yönlendiriliyorsunuz...');
      } else {
        await register({ 
          email: formData.email,
          password: formData.password,
          confirmPassword: formData.confirmPassword,
          name: formData.name,
          role,
          grade: formData.grade ? parseInt(formData.grade) : undefined,
          school: formData.school,
          teacherCode: formData.teacherCode
        });
        setSuccessMessage('Kayıt başarılı! Yönlendiriliyorsunuz...');
      }
    } catch (error: any) {
      setErrors({ general: error.message || 'İşlem başarısız. Lütfen tekrar deneyin.' });
    } finally {
      setIsLoading(false);
    }
  };

  // Handle enter key press
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSubmit();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center p-4">
      {/* Background decoration */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-300 rounded-full opacity-20 blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-300 rounded-full opacity-20 blur-3xl" />
      </div>

      <div className="relative w-full max-w-md">
        {/* Logo and Title */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full mb-4">
            <Sparkles className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            TEKNOFEST 2025
          </h1>
          <p className="text-gray-600 mt-2">Eğitim Teknolojileri Platformu</p>
        </div>

        {/* Auth Card */}
        <div className="bg-white rounded-2xl shadow-xl p-8">
          {/* Tab Switcher */}
          <div className="flex space-x-1 bg-gray-100 rounded-lg p-1 mb-6">
            <button
              onClick={() => setIsLogin(true)}
              className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${
                isLogin 
                  ? 'bg-white text-blue-600 shadow-sm' 
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              Giriş Yap
            </button>
            <button
              onClick={() => setIsLogin(false)}
              className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${
                !isLogin 
                  ? 'bg-white text-blue-600 shadow-sm' 
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              Kayıt Ol
            </button>
          </div>

          {/* Role Selector */}
          <div className="flex space-x-4 mb-6">
            <button
              onClick={() => setRole('student')}
              className={`flex-1 p-3 rounded-lg border-2 transition-all ${
                role === 'student'
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <GraduationCap className={`w-6 h-6 mx-auto mb-1 ${
                role === 'student' ? 'text-blue-500' : 'text-gray-400'
              }`} />
              <p className={`text-sm font-medium ${
                role === 'student' ? 'text-blue-700' : 'text-gray-600'
              }`}>Öğrenci</p>
            </button>
            <button
              onClick={() => setRole('teacher')}
              className={`flex-1 p-3 rounded-lg border-2 transition-all ${
                role === 'teacher'
                  ? 'border-purple-500 bg-purple-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <BookOpen className={`w-6 h-6 mx-auto mb-1 ${
                role === 'teacher' ? 'text-purple-500' : 'text-gray-400'
              }`} />
              <p className={`text-sm font-medium ${
                role === 'teacher' ? 'text-purple-700' : 'text-gray-600'
              }`}>Öğretmen</p>
            </button>
          </div>

          {/* Error Message */}
          {errors.general && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-red-500" />
              <p className="text-sm text-red-700">{errors.general}</p>
            </div>
          )}

          {/* Success Message */}
          {successMessage && (
            <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-500" />
              <p className="text-sm text-green-700">{successMessage}</p>
            </div>
          )}

          {/* Form Fields */}
          <div className="space-y-4" onKeyPress={handleKeyPress}>
            {/* Name Field (Registration only) */}
            {!isLogin && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Ad Soyad
                </label>
                <div className="relative">
                  <User className="absolute left-3 top-3 w-5 h-5 text-gray-400" />
                  <input
                    type="text"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    className={`w-full pl-10 pr-3 py-2.5 border rounded-lg focus:outline-none focus:ring-2 transition-all ${
                      errors.name 
                        ? 'border-red-300 focus:ring-red-200' 
                        : 'border-gray-300 focus:ring-blue-200'
                    }`}
                    placeholder="Adınız Soyadınız"
                  />
                </div>
                {errors.name && (
                  <p className="mt-1 text-sm text-red-600">{errors.name}</p>
                )}
              </div>
            )}

            {/* Email Field */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Email Adresi
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-3 w-5 h-5 text-gray-400" />
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  className={`w-full pl-10 pr-3 py-2.5 border rounded-lg focus:outline-none focus:ring-2 transition-all ${
                    errors.email 
                      ? 'border-red-300 focus:ring-red-200' 
                      : 'border-gray-300 focus:ring-blue-200'
                  }`}
                  placeholder="ornek@email.com"
                />
              </div>
              {errors.email && (
                <p className="mt-1 text-sm text-red-600">{errors.email}</p>
              )}
            </div>

            {/* Password Field */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Şifre
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-3 w-5 h-5 text-gray-400" />
                <input
                  type={showPassword ? 'text' : 'password'}
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  className={`w-full pl-10 pr-10 py-2.5 border rounded-lg focus:outline-none focus:ring-2 transition-all ${
                    errors.password 
                      ? 'border-red-300 focus:ring-red-200' 
                      : 'border-gray-300 focus:ring-blue-200'
                  }`}
                  placeholder="••••••••"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-3 text-gray-400 hover:text-gray-600"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
              {errors.password && (
                <p className="mt-1 text-sm text-red-600">{errors.password}</p>
              )}
              
              {/* Password Strength Indicator */}
              {!isLogin && formData.password && (
                <div className="mt-2">
                  <div className="flex items-center gap-2">
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full transition-all ${passwordStrength.color}`}
                        style={{ width: `${passwordStrength.score}%` }}
                      />
                    </div>
                    <span className="text-xs font-medium text-gray-600">
                      {passwordStrength.label}
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Confirm Password (Registration only) */}
            {!isLogin && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Şifre Tekrarı
                </label>
                <div className="relative">
                  <Lock className="absolute left-3 top-3 w-5 h-5 text-gray-400" />
                  <input
                    type={showConfirmPassword ? 'text' : 'password'}
                    name="confirmPassword"
                    value={formData.confirmPassword}
                    onChange={handleChange}
                    className={`w-full pl-10 pr-10 py-2.5 border rounded-lg focus:outline-none focus:ring-2 transition-all ${
                      errors.confirmPassword 
                        ? 'border-red-300 focus:ring-red-200' 
                        : 'border-gray-300 focus:ring-blue-200'
                    }`}
                    placeholder="••••••••"
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute right-3 top-3 text-gray-400 hover:text-gray-600"
                  >
                    {showConfirmPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
                {errors.confirmPassword && (
                  <p className="mt-1 text-sm text-red-600">{errors.confirmPassword}</p>
                )}
              </div>
            )}

            {/* Role-specific fields */}
            {!isLogin && role === 'student' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Sınıf
                </label>
                <div className="relative">
                  <Hash className="absolute left-3 top-3 w-5 h-5 text-gray-400" />
                  <select
                    name="grade"
                    value={formData.grade}
                    onChange={handleChange}
                    className={`w-full pl-10 pr-3 py-2.5 border rounded-lg focus:outline-none focus:ring-2 transition-all appearance-none ${
                      errors.grade 
                        ? 'border-red-300 focus:ring-red-200' 
                        : 'border-gray-300 focus:ring-blue-200'
                    }`}
                  >
                    <option value="">Sınıf Seçiniz</option>
                    {[...Array(12)].map((_, i) => (
                      <option key={i + 1} value={i + 1}>
                        {i + 1}. Sınıf
                      </option>
                    ))}
                  </select>
                </div>
                {errors.grade && (
                  <p className="mt-1 text-sm text-red-600">{errors.grade}</p>
                )}
              </div>
            )}

            {!isLogin && role === 'teacher' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Öğretmen Kodu
                </label>
                <div className="relative">
                  <Users className="absolute left-3 top-3 w-5 h-5 text-gray-400" />
                  <input
                    type="text"
                    name="teacherCode"
                    value={formData.teacherCode}
                    onChange={handleChange}
                    className={`w-full pl-10 pr-3 py-2.5 border rounded-lg focus:outline-none focus:ring-2 transition-all ${
                      errors.teacherCode 
                        ? 'border-red-300 focus:ring-red-200' 
                        : 'border-gray-300 focus:ring-blue-200'
                    }`}
                    placeholder="TEKNOFEST-2025"
                  />
                </div>
                {errors.teacherCode && (
                  <p className="mt-1 text-sm text-red-600">{errors.teacherCode}</p>
                )}
                <p className="mt-1 text-xs text-gray-500 flex items-center gap-1">
                  <Info className="w-3 h-3" />
                  Öğretmen kodunu okulunuzdan alabilirsiniz
                </p>
              </div>
            )}

            {/* School Field (Registration only) */}
            {!isLogin && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Okul (Opsiyonel)
                </label>
                <div className="relative">
                  <School className="absolute left-3 top-3 w-5 h-5 text-gray-400" />
                  <input
                    type="text"
                    name="school"
                    value={formData.school}
                    onChange={handleChange}
                    className="w-full pl-10 pr-3 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-200 transition-all"
                    placeholder="Okul adı"
                  />
                </div>
              </div>
            )}

            {/* Remember me & Forgot password (Login only) */}
            {isLogin && (
              <div className="flex items-center justify-between">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                  />
                  <span className="ml-2 text-sm text-gray-600">Beni hatırla</span>
                </label>
                <button
                  type="button"
                  className="text-sm text-blue-600 hover:text-blue-700 font-medium"
                >
                  Şifremi unuttum
                </button>
              </div>
            )}

            {/* Submit Button */}
            <button
              onClick={handleSubmit}
              disabled={isLoading}
              className={`w-full py-3 px-4 rounded-lg font-medium text-white transition-all flex items-center justify-center gap-2 ${
                isLoading
                  ? 'bg-gray-400 cursor-not-allowed'
                  : role === 'student'
                  ? 'bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 shadow-lg'
                  : 'bg-gradient-to-r from-purple-500 to-purple-600 hover:from-purple-600 hover:to-purple-700 shadow-lg'
              }`}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  İşleniyor...
                </>
              ) : (
                <>
                  {isLogin ? 'Giriş Yap' : 'Kayıt Ol'}
                  <ChevronRight className="w-5 h-5" />
                </>
              )}
            </button>
          </div>

          {/* Alternative Actions */}
          <div className="mt-6 text-center">
            <p className="text-sm text-gray-600">
              {isLogin ? "Hesabınız yok mu?" : "Zaten hesabınız var mı?"}
              <button
                type="button"
                onClick={() => setIsLogin(!isLogin)}
                className="ml-1 font-medium text-blue-600 hover:text-blue-700"
              >
                {isLogin ? 'Kayıt Ol' : 'Giriş Yap'}
              </button>
            </p>
          </div>

          {/* Demo Info */}
          <div className="mt-4 p-3 bg-blue-50 rounded-lg">
            <p className="text-xs text-blue-700 text-center">
              <Info className="w-3 h-3 inline mr-1" />
              Demo modunda çalışıyor. Herhangi bir email/şifre ile giriş yapabilirsiniz.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-sm text-gray-500">
          <p>© 2025 TEKNOFEST Eğitim Teknolojileri</p>
          <p className="mt-1">Tüm hakları saklıdır.</p>
        </div>
      </div>
    </div>
  );
};

export default AuthPage;
