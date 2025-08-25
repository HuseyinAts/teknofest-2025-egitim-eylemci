import { persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage';
import { encryptTransform } from 'redux-persist-transform-encrypt';

const encryptor = encryptTransform({
  secretKey: process.env.NEXT_PUBLIC_PERSIST_SECRET_KEY || 'teknofest-2025-secret-key',
  onError: function (error) {
    console.error('Persist Encryption Error:', error);
  },
});

export const authPersistConfig = {
  key: 'auth',
  storage,
  whitelist: ['token', 'refreshToken', 'user'],
  transforms: [encryptor],
};

export const userPersistConfig = {
  key: 'user',
  storage,
  whitelist: ['profile'],
  transforms: [encryptor],
};

export const learningPersistConfig = {
  key: 'learning',
  storage,
  whitelist: ['currentPath', 'currentSession'],
};

export const assessmentPersistConfig = {
  key: 'assessment',
  storage,
  whitelist: ['answers', 'currentQuestion'],
  blacklist: ['timer'],
};

export const uiPersistConfig = {
  key: 'ui',
  storage,
  whitelist: ['theme', 'language', 'sidebarOpen'],
};