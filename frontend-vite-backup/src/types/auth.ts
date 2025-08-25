export interface User {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  role: UserRole;
  avatar?: string;
  phoneNumber?: string;
  birthDate?: string;
  grade?: number;
  school?: string;
  learningStyle?: LearningStyle;
  createdAt: string;
  updatedAt: string;
  emailVerified: boolean;
  isActive: boolean;
}

export enum UserRole {
  STUDENT = 'student',
  TEACHER = 'teacher',
  PARENT = 'parent',
  ADMIN = 'admin',
}

export enum LearningStyle {
  VISUAL = 'visual',
  AUDITORY = 'auditory',
  READING_WRITING = 'reading_writing',
  KINESTHETIC = 'kinesthetic',
  MIXED = 'mixed',
}

export interface LoginRequest {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface RegisterRequest {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  role: UserRole;
  grade?: number;
  school?: string;
  phoneNumber?: string;
}

export interface AuthResponse {
  user: User;
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface RefreshTokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}