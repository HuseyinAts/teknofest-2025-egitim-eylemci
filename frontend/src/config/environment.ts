/**
 * Environment Configuration for Frontend
 * Centralizes all environment variables and provides type-safe access
 */

interface EnvironmentConfig {
  // API Configuration
  apiUrl: string;
  wsUrl: string;
  apiTimeout: number;
  
  // Application Settings
  appName: string;
  appVersion: string;
  environment: 'development' | 'staging' | 'production';
  isProduction: boolean;
  isDevelopment: boolean;
  
  // Feature Flags
  features: {
    registration: boolean;
    aiChat: boolean;
    analytics: boolean;
    maintenanceMode: boolean;
    webSocket: boolean;
    pushNotifications: boolean;
    offlineMode: boolean;
  };
  
  // External Services
  sentry: {
    enabled: boolean;
    dsn: string;
    environment: string;
    tracesSampleRate: number;
  };
  
  analytics: {
    enabled: boolean;
    googleAnalyticsId: string;
    mixpanelToken: string;
  };
  
  // Security
  security: {
    encryptLocalStorage: boolean;
    csrfEnabled: boolean;
    sessionTimeout: number; // in minutes
  };
  
  // CDN Configuration
  cdn: {
    enabled: boolean;
    baseUrl: string;
    imageOptimization: boolean;
  };
}

/**
 * Get environment variable with fallback
 */
const getEnvVar = (key: string, defaultValue: string = ''): string => {
  // In Next.js, we need to use process.env directly or NEXT_PUBLIC_ prefix
  const value = process.env[key] || process.env[`NEXT_PUBLIC_${key}`] || defaultValue;
  return value;
};

/**
 * Parse boolean environment variable
 */
const getEnvBool = (key: string, defaultValue: boolean = false): boolean => {
  const value = getEnvVar(key, String(defaultValue));
  return value === 'true' || value === '1';
};

/**
 * Parse number environment variable
 */
const getEnvNumber = (key: string, defaultValue: number = 0): number => {
  const value = getEnvVar(key, String(defaultValue));
  return parseInt(value, 10) || defaultValue;
};

/**
 * Determine current environment
 */
const getCurrentEnvironment = (): 'development' | 'staging' | 'production' => {
  const env = getEnvVar('NODE_ENV', 'development').toLowerCase();
  if (env === 'production') return 'production';
  if (env === 'staging') return 'staging';
  return 'development';
};

/**
 * Build API URL based on environment
 */
const buildApiUrl = (): string => {
  // Check for explicit API URL first
  const explicitUrl = getEnvVar('API_URL', '');
  if (explicitUrl) return explicitUrl;
  
  // Build based on environment
  const protocol = getEnvVar('API_PROTOCOL', 'http');
  const host = getEnvVar('API_HOST', 'localhost');
  const port = getEnvVar('API_PORT', '8003');
  
  // In production, don't include port if using standard ports
  if (getCurrentEnvironment() === 'production') {
    if ((protocol === 'https' && port === '443') || (protocol === 'http' && port === '80')) {
      return `${protocol}://${host}`;
    }
  }
  
  return `${protocol}://${host}:${port}`;
};

/**
 * Build WebSocket URL based on API URL
 */
const buildWsUrl = (): string => {
  const apiUrl = buildApiUrl();
  const wsProtocol = apiUrl.startsWith('https') ? 'wss' : 'ws';
  return apiUrl.replace(/^https?/, wsProtocol) + '/ws';
};

/**
 * Environment configuration singleton
 */
class Environment {
  private static instance: Environment;
  private config: EnvironmentConfig;
  
  private constructor() {
    const currentEnv = getCurrentEnvironment();
    
    this.config = {
      // API Configuration
      apiUrl: buildApiUrl(),
      wsUrl: buildWsUrl(),
      apiTimeout: getEnvNumber('API_TIMEOUT', 30000),
      
      // Application Settings
      appName: getEnvVar('APP_NAME', 'TEKNOFEST 2025'),
      appVersion: getEnvVar('APP_VERSION', '1.0.0'),
      environment: currentEnv,
      isProduction: currentEnv === 'production',
      isDevelopment: currentEnv === 'development',
      
      // Feature Flags
      features: {
        registration: getEnvBool('FEATURE_REGISTRATION_ENABLED', true),
        aiChat: getEnvBool('FEATURE_AI_CHAT', true),
        analytics: getEnvBool('FEATURE_ANALYTICS', true),
        maintenanceMode: getEnvBool('FEATURE_MAINTENANCE_MODE', false),
        webSocket: getEnvBool('FEATURE_WEBSOCKET', true),
        pushNotifications: getEnvBool('FEATURE_PUSH_NOTIFICATIONS', false),
        offlineMode: getEnvBool('FEATURE_OFFLINE_MODE', true),
      },
      
      // External Services
      sentry: {
        enabled: getEnvBool('SENTRY_ENABLED', currentEnv === 'production'),
        dsn: getEnvVar('SENTRY_DSN', ''),
        environment: getEnvVar('SENTRY_ENVIRONMENT', currentEnv),
        tracesSampleRate: parseFloat(getEnvVar('SENTRY_TRACES_SAMPLE_RATE', '0.1')),
      },
      
      analytics: {
        enabled: getEnvBool('ANALYTICS_ENABLED', currentEnv === 'production'),
        googleAnalyticsId: getEnvVar('GA_TRACKING_ID', ''),
        mixpanelToken: getEnvVar('MIXPANEL_TOKEN', ''),
      },
      
      // Security
      security: {
        encryptLocalStorage: getEnvBool('ENCRYPT_LOCAL_STORAGE', currentEnv === 'production'),
        csrfEnabled: getEnvBool('CSRF_ENABLED', true),
        sessionTimeout: getEnvNumber('SESSION_TIMEOUT_MINUTES', 30),
      },
      
      // CDN Configuration
      cdn: {
        enabled: getEnvBool('CDN_ENABLED', false),
        baseUrl: getEnvVar('CDN_URL', ''),
        imageOptimization: getEnvBool('CDN_IMAGE_OPTIMIZATION', true),
      },
    };
    
    // Validate configuration in production
    if (this.config.isProduction) {
      this.validateProductionConfig();
    }
  }
  
  /**
   * Get environment configuration instance
   */
  public static getInstance(): Environment {
    if (!Environment.instance) {
      Environment.instance = new Environment();
    }
    return Environment.instance;
  }
  
  /**
   * Get configuration
   */
  public getConfig(): EnvironmentConfig {
    return this.config;
  }
  
  /**
   * Get specific configuration value
   */
  public get<K extends keyof EnvironmentConfig>(key: K): EnvironmentConfig[K] {
    return this.config[key];
  }
  
  /**
   * Check if feature is enabled
   */
  public isFeatureEnabled(feature: keyof EnvironmentConfig['features']): boolean {
    return this.config.features[feature];
  }
  
  /**
   * Get API endpoint URL
   */
  public getApiEndpoint(path: string): string {
    const baseUrl = this.config.apiUrl;
    const cleanPath = path.startsWith('/') ? path : `/${path}`;
    return `${baseUrl}/api${cleanPath}`;
  }
  
  /**
   * Get CDN URL for assets
   */
  public getCdnUrl(path: string): string {
    if (!this.config.cdn.enabled || !this.config.cdn.baseUrl) {
      return path;
    }
    
    const cleanPath = path.startsWith('/') ? path : `/${path}`;
    return `${this.config.cdn.baseUrl}${cleanPath}`;
  }
  
  /**
   * Validate production configuration
   */
  private validateProductionConfig(): void {
    const errors: string[] = [];
    
    // Check required production settings
    if (!this.config.apiUrl || this.config.apiUrl.includes('localhost')) {
      errors.push('Production API URL must not use localhost');
    }
    
    if (!this.config.sentry.dsn && this.config.sentry.enabled) {
      errors.push('Sentry DSN is required when Sentry is enabled');
    }
    
    if (!this.config.security.encryptLocalStorage) {
      console.warn('Local storage encryption is disabled in production');
    }
    
    if (errors.length > 0) {
      console.error('Production configuration errors:', errors);
      // In production, we might want to prevent startup
      if (this.config.isProduction) {
        throw new Error(`Configuration errors: ${errors.join(', ')}`);
      }
    }
  }
  
  /**
   * Log configuration (excluding sensitive data)
   */
  public logConfig(): void {
    const safeConfig = {
      ...this.config,
      sentry: {
        ...this.config.sentry,
        dsn: this.config.sentry.dsn ? '***REDACTED***' : '',
      },
      analytics: {
        ...this.config.analytics,
        mixpanelToken: this.config.analytics.mixpanelToken ? '***REDACTED***' : '',
      },
    };
    
    console.log('Environment Configuration:', safeConfig);
  }
}

// Export singleton instance
export const env = Environment.getInstance().getConfig();
export const getEnv = () => Environment.getInstance();
export default env;

// Export type
export type { EnvironmentConfig };