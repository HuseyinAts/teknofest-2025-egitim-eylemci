/**
 * API Service Configuration
 * Centralized API client setup with interceptors and error handling
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosError } from 'axios';
import { env, getEnv } from '@/config/environment';

// Types
interface ApiError {
  message: string;
  code?: string;
  status?: number;
  details?: any;
}

interface RetryConfig {
  retries: number;
  retryDelay: number;
  retryCondition?: (error: AxiosError) => boolean;
}

/**
 * Create axios instance with configuration
 */
const createApiClient = (): AxiosInstance => {
  const instance = axios.create({
    baseURL: env.apiUrl,
    timeout: env.apiTimeout,
    headers: {
      'Content-Type': 'application/json',
      'X-App-Version': env.appVersion,
      'X-Environment': env.environment,
    },
    withCredentials: true, // Include cookies for authentication
  });
  
  // Request interceptor
  instance.interceptors.request.use(
    (config) => {
      // Add authentication token if available
      const token = getAuthToken();
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      
      // Add CSRF token if enabled
      if (env.security.csrfEnabled) {
        const csrfToken = getCsrfToken();
        if (csrfToken) {
          config.headers['X-CSRF-Token'] = csrfToken;
        }
      }
      
      // Add request ID for tracing
      config.headers['X-Request-ID'] = generateRequestId();
      
      // Log request in development
      if (env.isDevelopment) {
        console.log(`[API Request] ${config.method?.toUpperCase()} ${config.url}`, config.data);
      }
      
      return config;
    },
    (error) => {
      console.error('[API Request Error]', error);
      return Promise.reject(error);
    }
  );
  
  // Response interceptor
  instance.interceptors.response.use(
    (response) => {
      // Log response in development
      if (env.isDevelopment) {
        console.log(`[API Response] ${response.config.url}`, response.data);
      }
      
      // Handle pagination metadata
      if (response.headers['x-total-count']) {
        response.data._metadata = {
          totalCount: parseInt(response.headers['x-total-count'], 10),
          page: parseInt(response.headers['x-page'] || '1', 10),
          pageSize: parseInt(response.headers['x-page-size'] || '10', 10),
        };
      }
      
      return response;
    },
    async (error: AxiosError<ApiError>) => {
      // Log error
      console.error('[API Response Error]', error.response?.data || error.message);
      
      // Handle specific error cases
      if (error.response) {
        switch (error.response.status) {
          case 401:
            // Unauthorized - clear auth and redirect to login
            handleUnauthorized();
            break;
          
          case 403:
            // Forbidden - show permission error
            handleForbidden();
            break;
          
          case 429:
            // Rate limit - retry with exponential backoff
            return handleRateLimit(error);
          
          case 500:
          case 502:
          case 503:
          case 504:
            // Server error - retry if configured
            return handleServerError(error);
          
          default:
            // Other errors
            break;
        }
      } else if (error.request) {
        // Network error
        handleNetworkError(error);
      }
      
      // Transform error for consistent handling
      const apiError: ApiError = {
        message: error.response?.data?.message || error.message || 'An error occurred',
        code: error.response?.data?.code || error.code,
        status: error.response?.status,
        details: error.response?.data?.details,
      };
      
      return Promise.reject(apiError);
    }
  );
  
  return instance;
};

/**
 * Get authentication token from storage
 */
const getAuthToken = (): string | null => {
  if (typeof window === 'undefined') return null;
  
  // Try to get from cookie first (more secure)
  const cookies = document.cookie.split(';');
  const authCookie = cookies.find(c => c.trim().startsWith('auth_token='));
  if (authCookie) {
    return authCookie.split('=')[1];
  }
  
  // Fallback to localStorage
  return localStorage.getItem('auth_token');
};

/**
 * Get CSRF token
 */
const getCsrfToken = (): string | null => {
  if (typeof window === 'undefined') return null;
  
  // Try to get from meta tag
  const metaTag = document.querySelector('meta[name="csrf-token"]');
  if (metaTag) {
    return metaTag.getAttribute('content');
  }
  
  // Try to get from cookie
  const cookies = document.cookie.split(';');
  const csrfCookie = cookies.find(c => c.trim().startsWith('csrf_token='));
  if (csrfCookie) {
    return csrfCookie.split('=')[1];
  }
  
  return null;
};

/**
 * Generate unique request ID for tracing
 */
const generateRequestId = (): string => {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

/**
 * Handle unauthorized error
 */
const handleUnauthorized = (): void => {
  // Clear auth data
  if (typeof window !== 'undefined') {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user');
    
    // Redirect to login
    window.location.href = '/login?expired=true';
  }
};

/**
 * Handle forbidden error
 */
const handleForbidden = (): void => {
  // Show permission error message
  if (typeof window !== 'undefined') {
    // Dispatch event for global error handler
    window.dispatchEvent(new CustomEvent('api:forbidden', {
      detail: { message: 'You do not have permission to access this resource' }
    }));
  }
};

/**
 * Handle rate limit error with retry
 */
const handleRateLimit = async (error: AxiosError): Promise<any> => {
  const retryAfter = error.response?.headers['retry-after'];
  const delay = retryAfter ? parseInt(retryAfter, 10) * 1000 : 5000;
  
  console.log(`Rate limited. Retrying after ${delay}ms...`);
  
  // Wait and retry
  await new Promise(resolve => setTimeout(resolve, delay));
  return axios.request(error.config!);
};

/**
 * Handle server error with retry
 */
const handleServerError = async (error: AxiosError): Promise<any> => {
  const config = error.config as AxiosRequestConfig & { _retry?: number };
  
  // Initialize retry count
  if (!config._retry) {
    config._retry = 0;
  }
  
  // Max 3 retries
  if (config._retry >= 3) {
    return Promise.reject(error);
  }
  
  config._retry++;
  
  // Exponential backoff
  const delay = Math.min(1000 * Math.pow(2, config._retry), 10000);
  console.log(`Server error. Retry ${config._retry}/3 after ${delay}ms...`);
  
  await new Promise(resolve => setTimeout(resolve, delay));
  return axios.request(config);
};

/**
 * Handle network error
 */
const handleNetworkError = (error: AxiosError): void => {
  console.error('Network error:', error.message);
  
  if (typeof window !== 'undefined') {
    // Check if offline
    if (!navigator.onLine) {
      window.dispatchEvent(new CustomEvent('api:offline', {
        detail: { message: 'You are offline. Please check your internet connection.' }
      }));
    } else {
      window.dispatchEvent(new CustomEvent('api:network-error', {
        detail: { message: 'Network error. Please try again later.' }
      }));
    }
  }
};

/**
 * Create API client with retry capability
 */
const createApiClientWithRetry = (retryConfig?: RetryConfig): AxiosInstance => {
  const client = createApiClient();
  
  if (retryConfig) {
    client.interceptors.response.use(
      undefined,
      async (error: AxiosError) => {
        const config = error.config as AxiosRequestConfig & { _retryCount?: number };
        
        if (!config || !retryConfig.retryCondition || !retryConfig.retryCondition(error)) {
          return Promise.reject(error);
        }
        
        config._retryCount = config._retryCount || 0;
        
        if (config._retryCount >= retryConfig.retries) {
          return Promise.reject(error);
        }
        
        config._retryCount++;
        
        await new Promise(resolve => setTimeout(resolve, retryConfig.retryDelay));
        return client.request(config);
      }
    );
  }
  
  return client;
};

// Create default API client
const apiClient = createApiClient();

// Create specialized clients
const authClient = createApiClientWithRetry({
  retries: 3,
  retryDelay: 1000,
  retryCondition: (error) => !error.response || error.response.status >= 500,
});

const uploadClient = createApiClient();
uploadClient.defaults.headers['Content-Type'] = 'multipart/form-data';
uploadClient.defaults.timeout = 300000; // 5 minutes for uploads

// API service methods
export const api = {
  // Generic methods
  get: <T = any>(url: string, config?: AxiosRequestConfig) => 
    apiClient.get<T>(url, config),
  
  post: <T = any>(url: string, data?: any, config?: AxiosRequestConfig) => 
    apiClient.post<T>(url, data, config),
  
  put: <T = any>(url: string, data?: any, config?: AxiosRequestConfig) => 
    apiClient.put<T>(url, data, config),
  
  patch: <T = any>(url: string, data?: any, config?: AxiosRequestConfig) => 
    apiClient.patch<T>(url, data, config),
  
  delete: <T = any>(url: string, config?: AxiosRequestConfig) => 
    apiClient.delete<T>(url, config),
  
  // Specialized methods
  auth: {
    login: (credentials: any) => authClient.post('/api/auth/login', credentials),
    logout: () => authClient.post('/api/auth/logout'),
    refresh: () => authClient.post('/api/auth/refresh'),
    register: (data: any) => authClient.post('/api/auth/register', data),
  },
  
  upload: {
    file: (url: string, file: File, onProgress?: (progress: number) => void) => {
      const formData = new FormData();
      formData.append('file', file);
      
      return uploadClient.post(url, formData, {
        onUploadProgress: (progressEvent) => {
          if (onProgress && progressEvent.total) {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            onProgress(progress);
          }
        },
      });
    },
  },
};

// Export clients for direct use if needed
export { apiClient, authClient, uploadClient };
export type { ApiError };
export default api;