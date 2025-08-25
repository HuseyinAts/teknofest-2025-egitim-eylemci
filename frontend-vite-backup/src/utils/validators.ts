// Form Validation Utilities
import { ValidationError } from '../types/auth.types';

// Email validation
export const validateEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

// Password strength validation
export const validatePassword = (password: string): {
  isValid: boolean;
  errors: string[];
} => {
  const errors: string[] = [];
  
  if (password.length < 8) {
    errors.push('Şifre en az 8 karakter olmalıdır');
  }
  if (!/[A-Z]/.test(password)) {
    errors.push('En az bir büyük harf içermelidir');
  }
  if (!/[a-z]/.test(password)) {
    errors.push('En az bir küçük harf içermelidir');
  }
  if (!/[0-9]/.test(password)) {
    errors.push('En az bir rakam içermelidir');
  }
  if (!/[!@#$%^&*]/.test(password)) {
    errors.push('En az bir özel karakter içermelidir (!@#$%^&*)');
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
};

// Password match validation
export const validatePasswordMatch = (password: string, confirmPassword: string): boolean => {
  return password === confirmPassword && password.length > 0;
};

// Name validation
export const validateName = (name: string): boolean => {
  return name.trim().length >= 2 && name.trim().length <= 50;
};

// Grade validation
export const validateGrade = (grade: number): boolean => {
  return grade >= 1 && grade <= 12;
};

// Turkish phone number validation
export const validatePhone = (phone: string): boolean => {
  const phoneRegex = /^(\+90|0)?5[0-9]{9}$/;
  return phoneRegex.test(phone.replace(/\s/g, ''));
};

// Form validation for login
export const validateLoginForm = (email: string, password: string): ValidationError[] => {
  const errors: ValidationError[] = [];
  
  if (!email) {
    errors.push({ field: 'email', message: 'Email adresi gereklidir' });
  } else if (!validateEmail(email)) {
    errors.push({ field: 'email', message: 'Geçerli bir email adresi giriniz' });
  }
  
  if (!password) {
    errors.push({ field: 'password', message: 'Şifre gereklidir' });
  } else if (password.length < 6) {
    errors.push({ field: 'password', message: 'Şifre en az 6 karakter olmalıdır' });
  }
  
  return errors;
};

// Form validation for registration
export const validateRegisterForm = (
  email: string,
  password: string,
  confirmPassword: string,
  name: string,
  role: string,
  grade?: number
): ValidationError[] => {
  const errors: ValidationError[] = [];
  
  // Email validation
  if (!email) {
    errors.push({ field: 'email', message: 'Email adresi gereklidir' });
  } else if (!validateEmail(email)) {
    errors.push({ field: 'email', message: 'Geçerli bir email adresi giriniz' });
  }
  
  // Password validation
  if (!password) {
    errors.push({ field: 'password', message: 'Şifre gereklidir' });
  } else {
    const passwordValidation = validatePassword(password);
    if (!passwordValidation.isValid) {
      errors.push({ field: 'password', message: passwordValidation.errors[0] });
    }
  }
  
  // Confirm password validation
  if (!confirmPassword) {
    errors.push({ field: 'confirmPassword', message: 'Şifre tekrarı gereklidir' });
  } else if (!validatePasswordMatch(password, confirmPassword)) {
    errors.push({ field: 'confirmPassword', message: 'Şifreler eşleşmiyor' });
  }
  
  // Name validation
  if (!name) {
    errors.push({ field: 'name', message: 'Ad Soyad gereklidir' });
  } else if (!validateName(name)) {
    errors.push({ field: 'name', message: 'Ad Soyad 2-50 karakter arasında olmalıdır' });
  }
  
  // Role validation
  if (!role || !['student', 'teacher'].includes(role)) {
    errors.push({ field: 'role', message: 'Geçerli bir rol seçiniz' });
  }
  
  // Grade validation for students
  if (role === 'student' && grade) {
    if (!validateGrade(grade)) {
      errors.push({ field: 'grade', message: 'Sınıf 1-12 arasında olmalıdır' });
    }
  }
  
  return errors;
};

// Password strength calculator
export const getPasswordStrength = (password: string): {
  score: number;
  label: string;
  color: string;
} => {
  let score = 0;
  
  if (password.length >= 8) score++;
  if (password.length >= 12) score++;
  if (/[A-Z]/.test(password)) score++;
  if (/[a-z]/.test(password)) score++;
  if (/[0-9]/.test(password)) score++;
  if (/[!@#$%^&*]/.test(password)) score++;
  
  const strengthLevels = [
    { min: 0, max: 2, label: 'Zayıf', color: 'red' },
    { min: 3, max: 4, label: 'Orta', color: 'orange' },
    { min: 5, max: 6, label: 'Güçlü', color: 'green' }
  ];
  
  const level = strengthLevels.find(l => score >= l.min && score <= l.max) || strengthLevels[0];
  
  return {
    score: Math.min((score / 6) * 100, 100),
    label: level.label,
    color: level.color
  };
};
