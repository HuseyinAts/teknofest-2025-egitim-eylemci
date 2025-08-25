import React, { useState } from 'react';
import { 
  User, 
  Lock, 
  Mail, 
  Eye, 
  EyeOff, 
  GraduationCap,
  Users,
  ArrowRight,
  Phone,
  Calendar,
  School,
  CheckCircle,
  Brain
} from 'lucide-react';

const RegisterPage = () => {
  const [step, setStep] = useState(1);
  const [userType, setUserType] = useState('student');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [formData, setFormData] = useState({
    // Step 1
    firstName: '',
    lastName: '',
    email: '',
    phone: '',
    // Step 2
    password: '',
    confirmPassword: '',
    // Step 3 - Student specific
    school: '',
    grade: '',
    birthDate: '',
    // Step 3 - Teacher specific
    subject: '',
    experience: '',
    // Terms
    acceptTerms: false,
    acceptPrivacy: false
  });
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<any>({});

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const target = e.target;
    const name = target.name;
    const value = target.type === 'checkbox' 
      ? (target as HTMLInputElement).checked 
      : target.value;
    
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors((prev: any) => ({ ...prev, [name]: '' }));
    }
  };

  const validateStep = (stepNumber: number) => {
    const newErrors: any = {};
    
    if (stepNumber === 1) {
      if (!formData.firstName) newErrors.firstName = 'Ad gereklidir';
      if (!formData.lastName) newErrors.lastName = 'Soyad gereklidir';
      if (!formData.email) {
        newErrors.email = 'E-posta adresi gereklidir';
      } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
        newErrors.email = 'Geçerli bir e-posta adresi giriniz';
      }
      if (!formData.phone) newErrors.phone = 'Telefon numarası gereklidir';
    }
    
    if (stepNumber === 2) {
      if (!formData.password) {
        newErrors.password = 'Şifre gereklidir';
      } else if (formData.password.length < 8) {
        newErrors.password = 'Şifre en az 8 karakter olmalıdır';
      }
      if (!formData.confirmPassword) {
        newErrors.confirmPassword = 'Şifre tekrarı gereklidir';
      } else if (formData.password !== formData.confirmPassword) {
        newErrors.confirmPassword = 'Şifreler eşleşmiyor';
      }
    }
    
    if (stepNumber === 3) {
      if (userType === 'student') {
        if (!formData.school) newErrors.school = 'Okul adı gereklidir';
        if (!formData.grade) newErrors.grade = 'Sınıf seviyesi gereklidir';
        if (!formData.birthDate) newErrors.birthDate = 'Doğum tarihi gereklidir';
      } else {
        if (!formData.subject) newErrors.subject = 'Branş gereklidir';
        if (!formData.experience) newErrors.experience = 'Deneyim süresi gereklidir';
      }
      if (!formData.acceptTerms) newErrors.acceptTerms = 'Kullanım şartlarını kabul etmelisiniz';
      if (!formData.acceptPrivacy) newErrors.acceptPrivacy = 'Gizlilik politikasını kabul etmelisiniz';
    }
    
    return newErrors;
  };

  const handleNextStep = () => {
    const newErrors = validateStep(step);
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }
    
    if (step < 3) {
      setStep(step + 1);
    } else {
      handleSubmit();
    }
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    
    try {
      const response = await fetch('http://localhost:8000/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...formData,
          userType: userType
        })
      });

      if (response.ok) {
        const data = await response.json();
        // Store token and redirect
        localStorage.setItem('token', data.token);
        localStorage.setItem('userType', userType);
        window.location.href = '/dashboard';
      } else {
        const errorData = await response.json();
        setErrors({ general: errorData.message || 'Kayıt başarısız. Lütfen bilgilerinizi kontrol edin.' });
      }
    } catch (error) {
      console.error('Registration error:', error);
      setErrors({ general: 'Bir hata oluştu. Lütfen daha sonra tekrar deneyin.' });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 flex items-center justify-center p-4">
      {/* Background decorations */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-300 rounded-full opacity-20 blur-3xl"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-300 rounded-full opacity-20 blur-3xl"></div>
      </div>

      <div className="relative w-full max-w-lg">
        {/* Logo and Title */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl shadow-lg mb-4">
            <Brain className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            TEKNOFEST 2025
          </h1>
          <p className="text-gray-600">Eğitim Teknolojileri Platformu - Kayıt</p>
        </div>

        {/* Progress Steps */}
        <div className="flex items-center justify-center mb-8">
          {[1, 2, 3].map((s) => (
            <React.Fragment key={s}>
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold transition-all ${
                  step >= s
                    ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white'
                    : 'bg-gray-200 text-gray-500'
                }`}
              >
                {step > s ? <CheckCircle className="w-5 h-5" /> : s}
              </div>
              {s < 3 && (
                <div
                  className={`w-20 h-1 transition-all ${
                    step > s ? 'bg-gradient-to-r from-blue-500 to-purple-600' : 'bg-gray-200'
                  }`}
                />
              )}
            </React.Fragment>
          ))}
        </div>

        {/* Main Card */}
        <div className="bg-white rounded-2xl shadow-xl p-8">
          {/* Error Message */}
          {errors.general && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-600 rounded-lg text-sm">
              {errors.general}
            </div>
          )}

          {/* User Type Selector (Only on Step 1) */}
          {step === 1 && (
            <div className="grid grid-cols-2 gap-3 mb-6">
              <button
                onClick={() => setUserType('student')}
                className={`flex items-center justify-center gap-2 py-3 px-4 rounded-lg font-medium transition-all ${
                  userType === 'student'
                    ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                <GraduationCap className="w-5 h-5" />
                Öğrenci
              </button>
              <button
                onClick={() => setUserType('teacher')}
                className={`flex items-center justify-center gap-2 py-3 px-4 rounded-lg font-medium transition-all ${
                  userType === 'teacher'
                    ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                <Users className="w-5 h-5" />
                Öğretmen
              </button>
            </div>
          )}

          {/* Step 1: Personal Information */}
          {step === 1 && (
            <div className="space-y-4">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">Kişisel Bilgiler</h2>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Ad</label>
                  <input
                    type="text"
                    name="firstName"
                    value={formData.firstName}
                    onChange={handleInputChange}
                    className={`w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 transition-all ${
                      errors.firstName 
                        ? 'border-red-300 focus:ring-red-500' 
                        : 'border-gray-300 focus:ring-purple-500'
                    }`}
                    placeholder="Adınız"
                  />
                  {errors.firstName && (
                    <p className="mt-1 text-sm text-red-600">{errors.firstName}</p>
                  )}
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Soyad</label>
                  <input
                    type="text"
                    name="lastName"
                    value={formData.lastName}
                    onChange={handleInputChange}
                    className={`w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 transition-all ${
                      errors.lastName 
                        ? 'border-red-300 focus:ring-red-500' 
                        : 'border-gray-300 focus:ring-purple-500'
                    }`}
                    placeholder="Soyadınız"
                  />
                  {errors.lastName && (
                    <p className="mt-1 text-sm text-red-600">{errors.lastName}</p>
                  )}
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">E-posta</label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type="email"
                    name="email"
                    value={formData.email}
                    onChange={handleInputChange}
                    className={`w-full pl-10 pr-4 py-3 border rounded-lg focus:outline-none focus:ring-2 transition-all ${
                      errors.email 
                        ? 'border-red-300 focus:ring-red-500' 
                        : 'border-gray-300 focus:ring-purple-500'
                    }`}
                    placeholder="ornek@email.com"
                  />
                </div>
                {errors.email && (
                  <p className="mt-1 text-sm text-red-600">{errors.email}</p>
                )}
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Telefon</label>
                <div className="relative">
                  <Phone className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type="tel"
                    name="phone"
                    value={formData.phone}
                    onChange={handleInputChange}
                    className={`w-full pl-10 pr-4 py-3 border rounded-lg focus:outline-none focus:ring-2 transition-all ${
                      errors.phone 
                        ? 'border-red-300 focus:ring-red-500' 
                        : 'border-gray-300 focus:ring-purple-500'
                    }`}
                    placeholder="5XX XXX XX XX"
                  />
                </div>
                {errors.phone && (
                  <p className="mt-1 text-sm text-red-600">{errors.phone}</p>
                )}
              </div>
            </div>
          )}

          {/* Step 2: Security */}
          {step === 2 && (
            <div className="space-y-4">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">Güvenlik Bilgileri</h2>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Şifre</label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type={showPassword ? 'text' : 'password'}
                    name="password"
                    value={formData.password}
                    onChange={handleInputChange}
                    className={`w-full pl-10 pr-12 py-3 border rounded-lg focus:outline-none focus:ring-2 transition-all ${
                      errors.password 
                        ? 'border-red-300 focus:ring-red-500' 
                        : 'border-gray-300 focus:ring-purple-500'
                    }`}
                    placeholder="En az 8 karakter"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
                  >
                    {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
                {errors.password && (
                  <p className="mt-1 text-sm text-red-600">{errors.password}</p>
                )}
                <div className="mt-2 space-y-1">
                  <div className="flex items-center gap-2">
                    <div className={`h-1 flex-1 rounded ${formData.password.length >= 8 ? 'bg-green-500' : 'bg-gray-200'}`} />
                    <span className="text-xs text-gray-500">8+ karakter</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className={`h-1 flex-1 rounded ${/[A-Z]/.test(formData.password) ? 'bg-green-500' : 'bg-gray-200'}`} />
                    <span className="text-xs text-gray-500">Büyük harf</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className={`h-1 flex-1 rounded ${/[0-9]/.test(formData.password) ? 'bg-green-500' : 'bg-gray-200'}`} />
                    <span className="text-xs text-gray-500">Rakam</span>
                  </div>
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Şifre Tekrar</label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type={showConfirmPassword ? 'text' : 'password'}
                    name="confirmPassword"
                    value={formData.confirmPassword}
                    onChange={handleInputChange}
                    className={`w-full pl-10 pr-12 py-3 border rounded-lg focus:outline-none focus:ring-2 transition-all ${
                      errors.confirmPassword 
                        ? 'border-red-300 focus:ring-red-500' 
                        : 'border-gray-300 focus:ring-purple-500'
                    }`}
                    placeholder="Şifrenizi tekrar giriniz"
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
                  >
                    {showConfirmPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
                {errors.confirmPassword && (
                  <p className="mt-1 text-sm text-red-600">{errors.confirmPassword}</p>
                )}
              </div>
            </div>
          )}

          {/* Step 3: Role-specific Information */}
          {step === 3 && (
            <div className="space-y-4">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">
                {userType === 'student' ? 'Öğrenci Bilgileri' : 'Öğretmen Bilgileri'}
              </h2>
              
              {userType === 'student' ? (
                <>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Okul</label>
                    <div className="relative">
                      <School className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                      <input
                        type="text"
                        name="school"
                        value={formData.school}
                        onChange={handleInputChange}
                        className={`w-full pl-10 pr-4 py-3 border rounded-lg focus:outline-none focus:ring-2 transition-all ${
                          errors.school 
                            ? 'border-red-300 focus:ring-red-500' 
                            : 'border-gray-300 focus:ring-purple-500'
                        }`}
                        placeholder="Okulunuzun adı"
                      />
                    </div>
                    {errors.school && (
                      <p className="mt-1 text-sm text-red-600">{errors.school}</p>
                    )}
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Sınıf</label>
                      <select
                        name="grade"
                        value={formData.grade}
                        onChange={handleInputChange}
                        className={`w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 transition-all ${
                          errors.grade 
                            ? 'border-red-300 focus:ring-red-500' 
                            : 'border-gray-300 focus:ring-purple-500'
                        }`}
                      >
                        <option value="">Seçiniz</option>
                        {[9, 10, 11, 12].map(g => (
                          <option key={g} value={g}>{g}. Sınıf</option>
                        ))}
                      </select>
                      {errors.grade && (
                        <p className="mt-1 text-sm text-red-600">{errors.grade}</p>
                      )}
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Doğum Tarihi</label>
                      <input
                        type="date"
                        name="birthDate"
                        value={formData.birthDate}
                        onChange={handleInputChange}
                        className={`w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 transition-all ${
                          errors.birthDate 
                            ? 'border-red-300 focus:ring-red-500' 
                            : 'border-gray-300 focus:ring-purple-500'
                        }`}
                      />
                      {errors.birthDate && (
                        <p className="mt-1 text-sm text-red-600">{errors.birthDate}</p>
                      )}
                    </div>
                  </div>
                </>
              ) : (
                <>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Branş</label>
                    <select
                      name="subject"
                      value={formData.subject}
                      onChange={handleInputChange}
                      className={`w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 transition-all ${
                        errors.subject 
                          ? 'border-red-300 focus:ring-red-500' 
                          : 'border-gray-300 focus:ring-purple-500'
                      }`}
                    >
                      <option value="">Seçiniz</option>
                      <option value="math">Matematik</option>
                      <option value="physics">Fizik</option>
                      <option value="chemistry">Kimya</option>
                      <option value="biology">Biyoloji</option>
                      <option value="turkish">Türkçe</option>
                      <option value="english">İngilizce</option>
                      <option value="history">Tarih</option>
                      <option value="geography">Coğrafya</option>
                    </select>
                    {errors.subject && (
                      <p className="mt-1 text-sm text-red-600">{errors.subject}</p>
                    )}
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Deneyim</label>
                    <select
                      name="experience"
                      value={formData.experience}
                      onChange={handleInputChange}
                      className={`w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 transition-all ${
                        errors.experience 
                          ? 'border-red-300 focus:ring-red-500' 
                          : 'border-gray-300 focus:ring-purple-500'
                      }`}
                    >
                      <option value="">Seçiniz</option>
                      <option value="0-2">0-2 yıl</option>
                      <option value="3-5">3-5 yıl</option>
                      <option value="6-10">6-10 yıl</option>
                      <option value="10+">10+ yıl</option>
                    </select>
                    {errors.experience && (
                      <p className="mt-1 text-sm text-red-600">{errors.experience}</p>
                    )}
                  </div>
                </>
              )}
              
              {/* Terms and Conditions */}
              <div className="space-y-3 mt-6">
                <label className="flex items-start gap-3">
                  <input
                    type="checkbox"
                    name="acceptTerms"
                    checked={formData.acceptTerms}
                    onChange={handleInputChange}
                    className="w-4 h-4 mt-1 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
                  />
                  <span className="text-sm text-gray-600">
                    <a href="#" className="text-purple-600 hover:underline">Kullanım şartlarını</a> okudum ve kabul ediyorum
                  </span>
                </label>
                {errors.acceptTerms && (
                  <p className="text-sm text-red-600">{errors.acceptTerms}</p>
                )}
                
                <label className="flex items-start gap-3">
                  <input
                    type="checkbox"
                    name="acceptPrivacy"
                    checked={formData.acceptPrivacy}
                    onChange={handleInputChange}
                    className="w-4 h-4 mt-1 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
                  />
                  <span className="text-sm text-gray-600">
                    <a href="#" className="text-purple-600 hover:underline">Gizlilik politikasını</a> okudum ve kabul ediyorum
                  </span>
                </label>
                {errors.acceptPrivacy && (
                  <p className="text-sm text-red-600">{errors.acceptPrivacy}</p>
                )}
              </div>
            </div>
          )}

          {/* Navigation Buttons */}
          <div className="flex gap-3 mt-6">
            {step > 1 && (
              <button
                onClick={() => setStep(step - 1)}
                className="flex-1 py-3 px-4 border border-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-50 transition-all"
              >
                Geri
              </button>
            )}
            
            <button
              onClick={handleNextStep}
              disabled={isLoading}
              className="flex-1 bg-gradient-to-r from-blue-500 to-purple-600 text-white py-3 px-4 rounded-lg font-medium hover:from-blue-600 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  Kayıt yapılıyor...
                </>
              ) : (
                <>
                  {step === 3 ? 'Kayıt Ol' : 'İleri'}
                  <ArrowRight className="w-5 h-5" />
                </>
              )}
            </button>
          </div>

          {/* Sign In Link */}
          <p className="text-center mt-6 text-sm text-gray-600">
            Zaten hesabınız var mı?{' '}
            <a href="/login" className="text-purple-600 hover:text-purple-700 font-medium">
              Giriş yapın
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default RegisterPage;