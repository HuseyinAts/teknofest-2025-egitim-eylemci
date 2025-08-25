import apiService from './api';
import { LoginRequest, RegisterRequest, AuthResponse, User, RefreshTokenResponse } from '../types/auth';

class AuthService {
  async login(credentials: LoginRequest): Promise<AuthResponse> {
    const response = await apiService.post<AuthResponse>('/api/auth/login', credentials);
    return response.data;
  }

  async register(userData: RegisterRequest): Promise<AuthResponse> {
    const response = await apiService.post<AuthResponse>('/api/auth/register', userData);
    return response.data;
  }

  async logout(): Promise<void> {
    try {
      await apiService.post('/api/auth/logout');
    } catch (error) {
      console.error('Logout error:', error);
    }
  }

  async refreshToken(refreshToken: string): Promise<RefreshTokenResponse> {
    const response = await apiService.post<RefreshTokenResponse>('/api/auth/refresh', {
      refresh_token: refreshToken,
    });
    return response.data;
  }

  async getCurrentUser(): Promise<User> {
    const response = await apiService.get<User>('/api/auth/me');
    return response.data;
  }

  async updateProfile(data: Partial<User>): Promise<User> {
    const response = await apiService.patch<User>('/api/auth/profile', data);
    return response.data;
  }

  async changePassword(oldPassword: string, newPassword: string): Promise<void> {
    await apiService.post('/api/auth/change-password', {
      old_password: oldPassword,
      new_password: newPassword,
    });
  }

  async requestPasswordReset(email: string): Promise<void> {
    await apiService.post('/api/auth/forgot-password', { email });
  }

  async resetPassword(token: string, newPassword: string): Promise<void> {
    await apiService.post('/api/auth/reset-password', {
      token,
      new_password: newPassword,
    });
  }

  async verifyEmail(token: string): Promise<void> {
    await apiService.post('/api/auth/verify-email', { token });
  }

  async resendVerificationEmail(): Promise<void> {
    await apiService.post('/api/auth/resend-verification');
  }
}

export const authService = new AuthService();
export default authService;