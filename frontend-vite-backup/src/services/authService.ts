// Authentication Service
import { LoginCredentials, RegisterData, AuthResponse, User } from '../types/auth.types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';
const TOKEN_KEY = 'teknofest_auth_token';
const REFRESH_TOKEN_KEY = 'teknofest_refresh_token';
const USER_KEY = 'teknofest_user';

class AuthService {
  // Token management
  private getToken(): string | null {
    return localStorage.getItem(TOKEN_KEY);
  }

  private setToken(token: string): void {
    localStorage.setItem(TOKEN_KEY, token);
  }

  private removeToken(): void {
    localStorage.removeItem(TOKEN_KEY);
  }

  private getRefreshToken(): string | null {
    return localStorage.getItem(REFRESH_TOKEN_KEY);
  }

  private setRefreshToken(token: string): void {
    localStorage.setItem(REFRESH_TOKEN_KEY, token);
  }

  private removeRefreshToken(): void {
    localStorage.removeItem(REFRESH_TOKEN_KEY);
  }

  // User management
  private getStoredUser(): User | null {
    const userStr = localStorage.getItem(USER_KEY);
    if (userStr) {
      try {
        return JSON.parse(userStr);
      } catch {
        return null;
      }
    }
    return null;
  }

  private setStoredUser(user: User): void {
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  }

  private removeStoredUser(): void {
    localStorage.removeItem(USER_KEY);
  }

  // API Headers
  private getHeaders(includeAuth: boolean = false): HeadersInit {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };

    if (includeAuth) {
      const token = this.getToken();
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }
    }

    return headers;
  }

  // Login
  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    try {
      // Simulated API call - replace with actual endpoint
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify(credentials),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Giriş başarısız');
      }

      const data: AuthResponse = await response.json();

      if (data.success && data.token && data.user) {
        this.setToken(data.token);
        if (data.refreshToken) {
          this.setRefreshToken(data.refreshToken);
        }
        this.setStoredUser(data.user);
      }

      return data;
    } catch (error) {
      // For development - simulate successful login
      if (import.meta.env.DEV) {
        const mockUser: User = {
          id: '1',
          email: credentials.email,
          name: credentials.role === 'student' ? 'Öğrenci Test' : 'Öğretmen Test',
          role: credentials.role,
          grade: credentials.role === 'student' ? 10 : undefined,
          school: 'TEKNOFEST Lisesi',
          avatar: `https://ui-avatars.com/api/?name=${credentials.email}&background=random`,
          createdAt: new Date(),
          lastLogin: new Date(),
        };

        const mockResponse: AuthResponse = {
          success: true,
          message: 'Giriş başarılı',
          user: mockUser,
          token: '',  // Token must be provided by backend
          refreshToken: '',  // Token must be provided by backend
        };

        this.setToken(mockResponse.token!);
        this.setRefreshToken(mockResponse.refreshToken!);
        this.setStoredUser(mockResponse.user!);

        return mockResponse;
      }

      throw error;
    }
  }

  // Register
  async register(data: RegisterData): Promise<AuthResponse> {
    try {
      // Simulated API call - replace with actual endpoint
      const response = await fetch(`${API_BASE_URL}/auth/register`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Kayıt başarısız');
      }

      const responseData: AuthResponse = await response.json();

      if (responseData.success && responseData.token && responseData.user) {
        this.setToken(responseData.token);
        if (responseData.refreshToken) {
          this.setRefreshToken(responseData.refreshToken);
        }
        this.setStoredUser(responseData.user);
      }

      return responseData;
    } catch (error) {
      // For development - simulate successful registration
      if (import.meta.env.DEV) {
        const mockUser: User = {
          id: Date.now().toString(),
          email: data.email,
          name: data.name,
          role: data.role,
          grade: data.grade,
          school: data.school || 'TEKNOFEST Lisesi',
          avatar: `https://ui-avatars.com/api/?name=${data.name}&background=random`,
          createdAt: new Date(),
          lastLogin: new Date(),
        };

        const mockResponse: AuthResponse = {
          success: true,
          message: 'Kayıt başarılı',
          user: mockUser,
          token: '',  // Token must be provided by backend
          refreshToken: '',  // Token must be provided by backend
        };

        this.setToken(mockResponse.token!);
        this.setRefreshToken(mockResponse.refreshToken!);
        this.setStoredUser(mockResponse.user!);

        return mockResponse;
      }

      throw error;
    }
  }

  // Logout
  async logout(): Promise<void> {
    try {
      const token = this.getToken();
      if (token) {
        // Call logout endpoint if needed
        await fetch(`${API_BASE_URL}/auth/logout`, {
          method: 'POST',
          headers: this.getHeaders(true),
        });
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear local storage
      this.removeToken();
      this.removeRefreshToken();
      this.removeStoredUser();
    }
  }

  // Refresh token
  async refreshToken(): Promise<AuthResponse> {
    const refreshToken = this.getRefreshToken();
    if (!refreshToken) {
      throw new Error('No refresh token available');
    }

    try {
      const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify({ refreshToken }),
      });

      if (!response.ok) {
        throw new Error('Token refresh failed');
      }

      const data: AuthResponse = await response.json();

      if (data.success && data.token) {
        this.setToken(data.token);
        if (data.refreshToken) {
          this.setRefreshToken(data.refreshToken);
        }
      }

      return data;
    } catch (error) {
      // For development - simulate token refresh
      if (import.meta.env.DEV) {
        const user = this.getStoredUser();
        if (user) {
          const mockResponse: AuthResponse = {
            success: true,
            message: 'Token yenilendi',
            user,
            token: '',  // Token must be provided by backend
            refreshToken: 'mock-refresh-token-refreshed-' + Date.now(),
          };

          this.setToken(mockResponse.token!);
          this.setRefreshToken(mockResponse.refreshToken!);

          return mockResponse;
        }
      }

      this.logout();
      throw error;
    }
  }

  // Verify token
  async verifyToken(): Promise<boolean> {
    const token = this.getToken();
    if (!token) {
      return false;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/auth/verify`, {
        method: 'GET',
        headers: this.getHeaders(true),
      });

      return response.ok;
    } catch (error) {
      // For development - always return true if token exists
      if (import.meta.env.DEV) {
        return true;
      }
      return false;
    }
  }

  // Get current user
  getCurrentUser(): User | null {
    return this.getStoredUser();
  }

  // Check if authenticated
  isAuthenticated(): boolean {
    return !!this.getToken() && !!this.getStoredUser();
  }

  // Update user profile
  async updateProfile(userData: Partial<User>): Promise<User> {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/profile`, {
        method: 'PATCH',
        headers: this.getHeaders(true),
        body: JSON.stringify(userData),
      });

      if (!response.ok) {
        throw new Error('Profile update failed');
      }

      const updatedUser: User = await response.json();
      this.setStoredUser(updatedUser);

      return updatedUser;
    } catch (error) {
      // For development
      if (import.meta.env.DEV) {
        const currentUser = this.getStoredUser();
        if (currentUser) {
          const updatedUser = { ...currentUser, ...userData };
          this.setStoredUser(updatedUser);
          return updatedUser;
        }
      }
      throw error;
    }
  }

  // Password reset request
  async requestPasswordReset(email: string): Promise<void> {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/reset-password`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify({ email }),
      });

      if (!response.ok) {
        throw new Error('Password reset request failed');
      }
    } catch (error) {
      // For development
      if (import.meta.env.DEV) {
        console.log('Password reset email would be sent to:', email);
        return;
      }
      throw error;
    }
  }
}

export default new AuthService();
