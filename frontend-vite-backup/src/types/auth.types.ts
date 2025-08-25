// Authentication Types
export interface User {
  id: string;
  email: string;
  name: string;
  role: 'student' | 'teacher' | 'admin';
  grade?: number;
  school?: string;
  avatar?: string;
  createdAt: Date;
  lastLogin: Date;
}

export interface LoginCredentials {
  email: string;
  password: string;
  role: 'student' | 'teacher';
}

export interface RegisterData {
  email: string;
  password: string;
  confirmPassword: string;
  name: string;
  role: 'student' | 'teacher';
  grade?: number;
  school?: string;
  teacherCode?: string; // Öğretmen kayıt kodu
}

export interface AuthResponse {
  success: boolean;
  message: string;
  user?: User;
  token?: string;
  refreshToken?: string;
}

export interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (credentials: LoginCredentials) => Promise<void>;
  register: (data: RegisterData) => Promise<void>;
  logout: () => void;
  refreshToken: () => Promise<void>;
}

export interface ValidationError {
  field: string;
  message: string;
}
