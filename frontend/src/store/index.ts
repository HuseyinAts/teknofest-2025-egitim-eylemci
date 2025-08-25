import { configureStore, ThunkAction, Action } from '@reduxjs/toolkit';
import { setupListeners } from '@reduxjs/toolkit/query';
import authReducer from './slices/authSlice';
import learningReducer from './slices/learningSlice';
import uiReducer from './slices/uiSlice';
import userReducer from './slices/userSlice';
import assessmentReducer from './slices/assessmentSlice';
import { apiSlice } from './slices/apiSlice';
import { errorMiddleware } from './middleware/errorMiddleware';
import { loggerMiddleware } from './middleware/loggerMiddleware';

const isDevelopment = process.env.NODE_ENV === 'development';

export const makeStore = () => {
  const store = configureStore({
    reducer: {
      auth: authReducer,
      user: userReducer,
      learning: learningReducer,
      assessment: assessmentReducer,
      ui: uiReducer,
      [apiSlice.reducerPath]: apiSlice.reducer,
    },
    middleware: (getDefaultMiddleware) => {
      let middleware = getDefaultMiddleware({
        serializableCheck: {
          ignoredActions: [
            'auth/setUser',
            'auth/setTokens',
            'api/executeMutation/pending',
            'api/executeQuery/pending',
          ],
          ignoredActionPaths: ['meta.arg', 'payload.timestamp'],
          ignoredPaths: [
            'auth.user.createdAt',
            'auth.user.updatedAt',
            'learning.currentPath.startedAt',
            'assessment.history',
          ],
        },
        immutableCheck: {
          warnAfter: 128,
        },
        thunk: {
          extraArgument: {
            api: apiSlice,
          },
        },
      })
        .concat(apiSlice.middleware)
        .concat(errorMiddleware);

      if (isDevelopment) {
        middleware = middleware.concat(loggerMiddleware);
      }

      return middleware;
    },
    devTools: isDevelopment && {
      name: 'Teknofest Education Platform',
      trace: true,
      traceLimit: 25,
      features: {
        pause: true,
        lock: true,
        persist: true,
        export: true,
        import: 'custom',
        jump: true,
        skip: false,
        reorder: true,
        dispatch: true,
        test: true,
      },
    },
    preloadedState: undefined,
  });

  setupListeners(store.dispatch);

  return store;
};

export const store = makeStore();

export type AppStore = ReturnType<typeof makeStore>;
export type RootState = ReturnType<AppStore['getState']>;
export type AppDispatch = AppStore['dispatch'];
export type AppThunk<ReturnType = void> = ThunkAction<
  ReturnType,
  RootState,
  unknown,
  Action<string>
>;