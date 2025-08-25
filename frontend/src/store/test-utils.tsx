import React, { PropsWithChildren } from 'react';
import { render as rtlRender } from '@testing-library/react';
import { configureStore } from '@reduxjs/toolkit';
import { Provider } from 'react-redux';
import authReducer from './slices/authSlice';
import userReducer from './slices/userSlice';
import learningReducer from './slices/learningSlice';
import assessmentReducer from './slices/assessmentSlice';
import uiReducer from './slices/uiSlice';
import { apiSlice } from './slices/apiSlice';
import type { RootState, AppStore } from './index';

interface ExtendedRenderOptions {
  preloadedState?: Partial<RootState>;
  store?: AppStore;
}

export function renderWithProviders(
  ui: React.ReactElement,
  {
    preloadedState = {},
    store = configureStore({
      reducer: {
        auth: authReducer,
        user: userReducer,
        learning: learningReducer,
        assessment: assessmentReducer,
        ui: uiReducer,
        [apiSlice.reducerPath]: apiSlice.reducer,
      },
      preloadedState,
      middleware: (getDefaultMiddleware) =>
        getDefaultMiddleware({
          serializableCheck: false,
        }).concat(apiSlice.middleware),
    }),
    ...renderOptions
  }: ExtendedRenderOptions = {}
) {
  function Wrapper({ children }: PropsWithChildren<{}>): JSX.Element {
    return <Provider store={store}>{children}</Provider>;
  }

  return { store, ...rtlRender(ui, { wrapper: Wrapper, ...renderOptions }) };
}

export const createMockStore = (preloadedState?: Partial<RootState>) => {
  return configureStore({
    reducer: {
      auth: authReducer,
      user: userReducer,
      learning: learningReducer,
      assessment: assessmentReducer,
      ui: uiReducer,
      [apiSlice.reducerPath]: apiSlice.reducer,
    },
    preloadedState,
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware({
        serializableCheck: false,
      }).concat(apiSlice.middleware),
  });
};

export const mockAuthState = {
  user: {
    id: 'test-user-id',
    email: 'test@example.com',
    name: 'Test User',
    role: 'student',
  },
  token: 'mock-jwt-token',
  refreshToken: 'mock-refresh-token',
  isAuthenticated: true,
  loading: false,
  error: null,
};

export const mockUserState = {
  profile: {
    id: 'test-user-id',
    email: 'test@example.com',
    name: 'Test User',
    role: 'student' as const,
    preferences: {
      language: 'tr',
      theme: 'light' as const,
      notifications: {
        email: true,
        push: true,
        sms: false,
      },
      accessibility: {
        fontSize: 'medium' as const,
        highContrast: false,
        screenReader: false,
      },
    },
    stats: {
      coursesCompleted: 5,
      totalLearningHours: 120,
      currentStreak: 7,
      points: 1500,
      level: 3,
      achievements: ['first-course', 'week-streak'],
    },
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  loading: false,
  error: null,
  updateStatus: 'idle' as const,
};

export const mockLearningState = {
  paths: [],
  currentPath: null,
  sessions: [],
  currentSession: null,
  assessments: [],
  loading: false,
  error: null,
};

export const mockAssessmentState = {
  assessments: [],
  currentAssessment: null,
  currentQuestion: 0,
  answers: {},
  results: [],
  history: [],
  loading: false,
  submitting: false,
  error: null,
  timer: {
    isActive: false,
    remainingTime: 0,
  },
};

export * from '@testing-library/react';