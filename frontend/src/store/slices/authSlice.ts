import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import authService, { User, LoginCredentials, RegisterData } from '@/services/authService';

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  loading: boolean;
  error: string | null;
  sessionValid: boolean;
  csrfToken: string | null;
}

const initialState: AuthState = {
  user: null,
  isAuthenticated: false,
  loading: false,
  error: null,
  sessionValid: false,
  csrfToken: null,
};

export const login = createAsyncThunk(
  'auth/login',
  async (credentials: LoginCredentials) => {
    const response = await authService.login(credentials);
    return response;
  }
);

export const register = createAsyncThunk(
  'auth/register',
  async (data: RegisterData) => {
    const response = await authService.register(data);
    return response;
  }
);

export const fetchCurrentUser = createAsyncThunk(
  'auth/fetchCurrentUser',
  async () => {
    const user = await authService.getCurrentUser();
    return user;
  }
);

export const logout = createAsyncThunk('auth/logout', async () => {
  await authService.logout();
});

export const verifySession = createAsyncThunk(
  'auth/verifySession',
  async () => {
    const isValid = await authService.verifySession();
    if (isValid) {
      const user = await authService.getCurrentUser();
      return { isValid, user };
    }
    return { isValid, user: null };
  }
);

export const refreshSession = createAsyncThunk(
  'auth/refreshSession',
  async () => {
    await authService.refreshToken();
    const user = await authService.getCurrentUser();
    return user;
  }
);

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    setUser: (state, action: PayloadAction<User | null>) => {
      state.user = action.payload;
      state.isAuthenticated = !!action.payload;
    },
    clearError: (state) => {
      state.error = null;
    },
    setCSRFToken: (state, action: PayloadAction<string>) => {
      state.csrfToken = action.payload;
    },
    resetAuth: (state) => {
      state.user = null;
      state.isAuthenticated = false;
      state.sessionValid = false;
      state.csrfToken = null;
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Login
      .addCase(login.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(login.fulfilled, (state, action) => {
        state.loading = false;
        state.user = action.payload.user;
        state.isAuthenticated = true;
        state.sessionValid = true;
        if (action.payload.csrfToken) {
          state.csrfToken = action.payload.csrfToken;
        }
      })
      .addCase(login.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Giriş başarısız';
        state.isAuthenticated = false;
      })
      // Register
      .addCase(register.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(register.fulfilled, (state, action) => {
        state.loading = false;
        state.user = action.payload.user;
        state.isAuthenticated = true;
        state.sessionValid = true;
        if (action.payload.csrfToken) {
          state.csrfToken = action.payload.csrfToken;
        }
      })
      .addCase(register.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Kayıt başarısız';
      })
      // Fetch Current User
      .addCase(fetchCurrentUser.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchCurrentUser.fulfilled, (state, action) => {
        state.loading = false;
        state.user = action.payload;
        state.isAuthenticated = true;
        state.sessionValid = true;
      })
      .addCase(fetchCurrentUser.rejected, (state) => {
        state.loading = false;
        state.isAuthenticated = false;
        state.user = null;
      })
      // Logout
      .addCase(logout.fulfilled, (state) => {
        state.user = null;
        state.isAuthenticated = false;
        state.sessionValid = false;
        state.csrfToken = null;
      })
      // Verify Session
      .addCase(verifySession.fulfilled, (state, action) => {
        state.sessionValid = action.payload.isValid;
        state.isAuthenticated = action.payload.isValid;
        state.user = action.payload.user;
      })
      .addCase(verifySession.rejected, (state) => {
        state.sessionValid = false;
        state.isAuthenticated = false;
        state.user = null;
      })
      // Refresh Session
      .addCase(refreshSession.fulfilled, (state, action) => {
        state.user = action.payload;
        state.sessionValid = true;
      })
      .addCase(refreshSession.rejected, (state) => {
        state.sessionValid = false;
        state.isAuthenticated = false;
        state.user = null;
      });
  },
});

export const { setUser, clearError, setCSRFToken, resetAuth } = authSlice.actions;
export default authSlice.reducer;