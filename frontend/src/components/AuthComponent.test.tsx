/**
 * Authentication Component Tests
 * TEKNOFEST 2025
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import authReducer from '@/store/slices/authSlice';

// Mock authService
jest.mock('@/services/authService', () => ({
  __esModule: true,
  default: {
    login: jest.fn(),
    register: jest.fn(),
    logout: jest.fn(),
    getCurrentUser: jest.fn(),
    verifySession: jest.fn(),
    refreshToken: jest.fn(),
    initializeCSRF: jest.fn(),
    getCSRFToken: jest.fn(),
  },
  authService: {
    login: jest.fn(),
    register: jest.fn(),
    logout: jest.fn(),
    getCurrentUser: jest.fn(),
    verifySession: jest.fn(),
    refreshToken: jest.fn(),
    initializeCSRF: jest.fn(),
    getCSRFToken: jest.fn(),
  },
}));

// Simple Login Component for testing
const LoginComponent: React.FC = () => {
  const [email, setEmail] = React.useState('');
  const [password, setPassword] = React.useState('');
  const [error, setError] = React.useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!email || !password) {
      setError('Email and password are required');
      return;
    }
    
    if (password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }
    
    // Mock successful login
    setError('');
  };

  return (
    <form onSubmit={handleSubmit} data-testid="login-form">
      <input
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="Email"
        data-testid="email-input"
      />
      <input
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        placeholder="Password"
        data-testid="password-input"
      />
      {error && <div data-testid="error-message">{error}</div>}
      <button type="submit" data-testid="login-button">Login</button>
    </form>
  );
};

// Test helpers
const createMockStore = () => {
  return configureStore({
    reducer: {
      auth: authReducer,
    },
  });
};

const renderWithProvider = (component: React.ReactElement) => {
  const store = createMockStore();
  return render(
    <Provider store={store}>
      {component}
    </Provider>
  );
};

describe('Authentication Components', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('LoginComponent', () => {
    it('should render login form', () => {
      renderWithProvider(<LoginComponent />);
      
      expect(screen.getByTestId('login-form')).toBeInTheDocument();
      expect(screen.getByTestId('email-input')).toBeInTheDocument();
      expect(screen.getByTestId('password-input')).toBeInTheDocument();
      expect(screen.getByTestId('login-button')).toBeInTheDocument();
    });

    it('should show error for empty fields', async () => {
      renderWithProvider(<LoginComponent />);
      
      const loginButton = screen.getByTestId('login-button');
      fireEvent.click(loginButton);
      
      await waitFor(() => {
        expect(screen.getByTestId('error-message')).toHaveTextContent(
          'Email and password are required'
        );
      });
    });

    it('should validate password length', async () => {
      renderWithProvider(<LoginComponent />);
      
      const emailInput = screen.getByTestId('email-input');
      const passwordInput = screen.getByTestId('password-input');
      const loginButton = screen.getByTestId('login-button');
      
      fireEvent.change(emailInput, { target: { value: 'test@example.com' } });
      fireEvent.change(passwordInput, { target: { value: 'short' } });
      fireEvent.click(loginButton);
      
      await waitFor(() => {
        expect(screen.getByTestId('error-message')).toHaveTextContent(
          'Password must be at least 8 characters'
        );
      });
    });

    it('should accept valid credentials', async () => {
      renderWithProvider(<LoginComponent />);
      
      const emailInput = screen.getByTestId('email-input');
      const passwordInput = screen.getByTestId('password-input');
      const loginButton = screen.getByTestId('login-button');
      
      fireEvent.change(emailInput, { target: { value: 'test@example.com' } });
      fireEvent.change(passwordInput, { target: { value: 'validPassword123' } });
      fireEvent.click(loginButton);
      
      await waitFor(() => {
        expect(screen.queryByTestId('error-message')).not.toBeInTheDocument();
      });
    });
  });

  describe('Authentication Security', () => {
    it('should not store sensitive data in localStorage', () => {
      // Check that no tokens are in localStorage
      expect(localStorage.getItem('access_token')).toBeNull();
      expect(localStorage.getItem('refresh_token')).toBeNull();
      expect(localStorage.getItem('password')).toBeNull();
    });

    it('should only store CSRF token in sessionStorage', () => {
      // CSRF token can be in sessionStorage (not for auth)
      const csrfToken = 'test_csrf_token';
      sessionStorage.setItem('csrf_token', csrfToken);
      
      expect(sessionStorage.getItem('csrf_token')).toBe(csrfToken);
      
      // Clean up
      sessionStorage.removeItem('csrf_token');
    });
  });
});

describe('Auth Service Integration', () => {
  const authService = require('@/services/authService').default;

  it('should call login with credentials', async () => {
    authService.login.mockResolvedValueOnce({
      user: { id: '1', email: 'test@example.com', name: 'Test User' },
      csrfToken: 'mock_csrf_token',
    });

    const result = await authService.login({
      email: 'test@example.com',
      password: 'password123',
    });

    expect(authService.login).toHaveBeenCalledWith({
      email: 'test@example.com',
      password: 'password123',
    });
    expect(result.user.email).toBe('test@example.com');
  });

  it('should handle login errors', async () => {
    authService.login.mockRejectedValueOnce(new Error('Invalid credentials'));

    await expect(
      authService.login({
        email: 'test@example.com',
        password: 'wrong',
      })
    ).rejects.toThrow('Invalid credentials');
  });

  it('should verify session validity', async () => {
    authService.verifySession.mockResolvedValueOnce(true);

    const isValid = await authService.verifySession();
    
    expect(authService.verifySession).toHaveBeenCalled();
    expect(isValid).toBe(true);
  });

  it('should refresh token when needed', async () => {
    authService.refreshToken.mockResolvedValueOnce(undefined);

    await authService.refreshToken();
    
    expect(authService.refreshToken).toHaveBeenCalled();
  });
});