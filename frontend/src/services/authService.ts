/**
 * Secure Authentication Service with HttpOnly Cookies
 * TEKNOFEST 2025 - Production Ready Authentication
 */

import axios, { AxiosInstance } from 'axios';
import { jwtDecode } from 'jwt-decode';

interface LoginCredentials {
  email: string;
  password: string;
}

interface RegisterData extends LoginCredentials {
  name: string;
}

interface AuthResponse {
  user: User;
  message?: string;
  csrfToken?: string;
}

interface User {
  id: string;
  email: string;
  name: string;
  role: string;
  createdAt: string;
}

interface TokenPayload {
  sub: string;
  email: string;
  exp: number;
  iat: number;
}

class AuthService {
  private api: AxiosInstance;
  private csrfToken: string | null = null;
  private refreshPromise: Promise<void> | null = null;

  constructor() {
    const baseURL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    
    this.api = axios.create({
      baseURL: `${baseURL}/api`,
      timeout: 10000,
      withCredentials: true, // Important: send cookies with requests
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor to add CSRF token
    this.api.interceptors.request.use(
      (config) => {
        if (this.csrfToken) {
          config.headers['X-CSRF-Token'] = this.csrfToken;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor to handle token refresh
    this.api.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config;

        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;

          // Prevent multiple refresh calls
          if (!this.refreshPromise) {
            this.refreshPromise = this.refreshToken();
          }

          try {
            await this.refreshPromise;
            this.refreshPromise = null;
            return this.api(originalRequest);
          } catch (refreshError) {
            this.refreshPromise = null;
            // Redirect to login
            window.location.href = '/login';
            return Promise.reject(refreshError);
          }
        }

        return Promise.reject(error);
      }
    );
  }

  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    try {
      const response = await this.api.post<AuthResponse>('/auth/login', credentials);
      
      // Store CSRF token if provided
      if (response.data.csrfToken) {
        this.csrfToken = response.data.csrfToken;
        // Store in session storage as backup (not for auth, just for CSRF)
        sessionStorage.setItem('csrf_token', response.data.csrfToken);
      }

      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async register(data: RegisterData): Promise<AuthResponse> {
    try {
      const response = await this.api.post<AuthResponse>('/auth/register', data);
      
      if (response.data.csrfToken) {
        this.csrfToken = response.data.csrfToken;
        sessionStorage.setItem('csrf_token', response.data.csrfToken);
      }

      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async logout(): Promise<void> {
    try {
      await this.api.post('/auth/logout');
      this.csrfToken = null;
      sessionStorage.removeItem('csrf_token');
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Always redirect to login
      window.location.href = '/login';
    }
  }

  async getCurrentUser(): Promise<User> {
    try {
      const response = await this.api.get<{ user: User }>('/auth/me');
      return response.data.user;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async refreshToken(): Promise<void> {
    try {
      const response = await this.api.post<{ csrfToken?: string }>('/auth/refresh');
      
      if (response.data.csrfToken) {
        this.csrfToken = response.data.csrfToken;
        sessionStorage.setItem('csrf_token', response.data.csrfToken);
      }
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async verifySession(): Promise<boolean> {
    try {
      const response = await this.api.get('/auth/verify');
      return response.data.valid === true;
    } catch (error) {
      return false;
    }
  }

  async changePassword(oldPassword: string, newPassword: string): Promise<void> {
    try {
      await this.api.post('/auth/change-password', {
        old_password: oldPassword,
        new_password: newPassword,
      });
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async requestPasswordReset(email: string): Promise<void> {
    try {
      await this.api.post('/auth/forgot-password', { email });
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async resetPassword(token: string, newPassword: string): Promise<void> {
    try {
      await this.api.post('/auth/reset-password', {
        token,
        new_password: newPassword,
      });
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async enable2FA(): Promise<{ qr_code: string; secret: string }> {
    try {
      const response = await this.api.post<{ qr_code: string; secret: string }>('/auth/2fa/enable');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async verify2FA(code: string): Promise<void> {
    try {
      await this.api.post('/auth/2fa/verify', { code });
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async disable2FA(code: string): Promise<void> {
    try {
      await this.api.post('/auth/2fa/disable', { code });
    } catch (error) {
      throw this.handleError(error);
    }
  }

  private handleError(error: any): Error {
    if (axios.isAxiosError(error)) {
      const message = error.response?.data?.detail || error.response?.data?.message || error.message;
      return new Error(message);
    }
    return error;
  }

  // Session management helpers
  async checkSessionExpiry(): Promise<boolean> {
    try {
      const response = await this.api.get('/auth/session/check');
      const expiresAt = response.data.expires_at;
      
      if (expiresAt) {
        const expiryTime = new Date(expiresAt).getTime();
        const now = Date.now();
        const timeUntilExpiry = expiryTime - now;
        
        // Refresh if less than 5 minutes remaining
        if (timeUntilExpiry < 5 * 60 * 1000 && timeUntilExpiry > 0) {
          await this.refreshToken();
        }
        
        return timeUntilExpiry > 0;
      }
      
      return false;
    } catch (error) {
      return false;
    }
  }

  // Initialize CSRF token on app start
  async initializeCSRF(): Promise<void> {
    try {
      const response = await this.api.get<{ csrfToken: string }>('/auth/csrf');
      this.csrfToken = response.data.csrfToken;
      sessionStorage.setItem('csrf_token', response.data.csrfToken);
    } catch (error) {
      console.error('Failed to initialize CSRF token:', error);
    }
  }

  // Get stored CSRF token
  getCSRFToken(): string | null {
    if (!this.csrfToken) {
      this.csrfToken = sessionStorage.getItem('csrf_token');
    }
    return this.csrfToken;
  }
}

// Create singleton instance
const authService = new AuthService();

// Initialize CSRF on load
if (typeof window !== 'undefined') {
  authService.initializeCSRF();
}

export default authService;
export { authService, AuthService };
export type { User, LoginCredentials, RegisterData, AuthResponse };