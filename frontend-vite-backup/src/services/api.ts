import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';
import { toast } from 'react-hot-toast';
import { store } from '../store';
import { logout } from '../store/slices/authSlice';

interface ApiError {
  message: string;
  code?: string;
  details?: any;
}

class ApiService {
  private instance: AxiosInstance;
  private isRefreshing = false;
  private refreshSubscribers: ((token: string) => void)[] = [];

  constructor() {
    this.instance = axios.create({
      baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
      timeout: parseInt(import.meta.env.VITE_API_TIMEOUT || '30000'),
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    this.instance.interceptors.request.use(
      (config) => {
        const token = this.getAccessToken();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    this.instance.interceptors.response.use(
      (response) => response,
      async (error: AxiosError<ApiError>) => {
        const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean };

        if (error.response?.status === 401 && !originalRequest._retry) {
          if (!this.isRefreshing) {
            this.isRefreshing = true;
            originalRequest._retry = true;

            try {
              const newToken = await this.refreshToken();
              this.onRefreshed(newToken);
              this.refreshSubscribers = [];
              return this.instance(originalRequest);
            } catch (refreshError) {
              this.handleAuthError();
              return Promise.reject(refreshError);
            } finally {
              this.isRefreshing = false;
            }
          }

          return new Promise((resolve) => {
            this.subscribeTokenRefresh((token: string) => {
              if (originalRequest.headers) {
                originalRequest.headers.Authorization = `Bearer ${token}`;
              }
              resolve(this.instance(originalRequest));
            });
          });
        }

        this.handleApiError(error);
        return Promise.reject(error);
      }
    );
  }

  private getAccessToken(): string | null {
    const state = store.getState();
    return state.auth.accessToken;
  }

  private async refreshToken(): Promise<string> {
    const state = store.getState();
    const refreshToken = state.auth.refreshToken;

    if (!refreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await this.instance.post('/api/auth/refresh', {
      refresh_token: refreshToken,
    });

    const { access_token } = response.data;
    return access_token;
  }

  private onRefreshed(token: string) {
    this.refreshSubscribers.forEach((callback) => callback(token));
  }

  private subscribeTokenRefresh(callback: (token: string) => void) {
    this.refreshSubscribers.push(callback);
  }

  private handleAuthError() {
    store.dispatch(logout());
    window.location.href = '/login';
    toast.error('Oturumunuz sona erdi. Lütfen tekrar giriş yapın.');
  }

  private handleApiError(error: AxiosError<ApiError>) {
    if (error.response) {
      const { status, data } = error.response;
      const message = data?.message || 'Bir hata oluştu';

      switch (status) {
        case 400:
          toast.error(`Geçersiz istek: ${message}`);
          break;
        case 403:
          toast.error('Bu işlem için yetkiniz bulunmuyor');
          break;
        case 404:
          toast.error('İstenen kaynak bulunamadı');
          break;
        case 429:
          toast.error('Çok fazla istek gönderildi. Lütfen bekleyin.');
          break;
        case 500:
          toast.error('Sunucu hatası. Lütfen daha sonra tekrar deneyin.');
          break;
        default:
          if (status >= 400 && status < 500) {
            toast.error(message);
          } else if (status >= 500) {
            toast.error('Sunucu hatası oluştu');
          }
      }
    } else if (error.request) {
      toast.error('Sunucuya ulaşılamıyor. İnternet bağlantınızı kontrol edin.');
    } else {
      toast.error('Beklenmeyen bir hata oluştu');
    }
  }

  async get<T = any>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.instance.get<T>(url, config);
  }

  async post<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.instance.post<T>(url, data, config);
  }

  async put<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.instance.put<T>(url, data, config);
  }

  async patch<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.instance.patch<T>(url, data, config);
  }

  async delete<T = any>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.instance.delete<T>(url, config);
  }

  async upload<T = any>(url: string, formData: FormData, onProgress?: (progress: number) => void): Promise<AxiosResponse<T>> {
    return this.instance.post<T>(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    });
  }
}

export const apiService = new ApiService();

export default apiService;