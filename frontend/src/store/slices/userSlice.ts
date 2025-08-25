import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import axios from 'axios';

export interface UserProfile {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  role: 'student' | 'teacher' | 'admin';
  preferences: {
    language: string;
    theme: 'light' | 'dark' | 'system';
    notifications: {
      email: boolean;
      push: boolean;
      sms: boolean;
    };
    accessibility: {
      fontSize: 'small' | 'medium' | 'large';
      highContrast: boolean;
      screenReader: boolean;
    };
  };
  stats: {
    coursesCompleted: number;
    totalLearningHours: number;
    currentStreak: number;
    points: number;
    level: number;
    achievements: string[];
  };
  createdAt: string;
  updatedAt: string;
}

export interface UserState {
  profile: UserProfile | null;
  loading: boolean;
  error: string | null;
  updateStatus: 'idle' | 'loading' | 'succeeded' | 'failed';
}

const initialState: UserState = {
  profile: null,
  loading: false,
  error: null,
  updateStatus: 'idle',
};

export const fetchUserProfile = createAsyncThunk(
  'user/fetchProfile',
  async (userId: string, { rejectWithValue }) => {
    try {
      const response = await axios.get(`/api/users/${userId}`);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data || 'Failed to fetch user profile');
    }
  }
);

export const updateUserProfile = createAsyncThunk(
  'user/updateProfile',
  async ({ userId, updates }: { userId: string; updates: Partial<UserProfile> }, { rejectWithValue }) => {
    try {
      const response = await axios.patch(`/api/users/${userId}`, updates);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data || 'Failed to update user profile');
    }
  }
);

export const updateUserPreferences = createAsyncThunk(
  'user/updatePreferences',
  async ({ userId, preferences }: { userId: string; preferences: Partial<UserProfile['preferences']> }, { rejectWithValue }) => {
    try {
      const response = await axios.patch(`/api/users/${userId}/preferences`, preferences);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data || 'Failed to update preferences');
    }
  }
);

export const uploadAvatar = createAsyncThunk(
  'user/uploadAvatar',
  async ({ userId, file }: { userId: string; file: File }, { rejectWithValue }) => {
    try {
      const formData = new FormData();
      formData.append('avatar', file);
      
      const response = await axios.post(`/api/users/${userId}/avatar`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data || 'Failed to upload avatar');
    }
  }
);

const userSlice = createSlice({
  name: 'user',
  initialState,
  reducers: {
    setUser: (state, action: PayloadAction<UserProfile>) => {
      state.profile = action.payload;
      state.error = null;
    },
    clearUser: (state) => {
      state.profile = null;
      state.error = null;
      state.updateStatus = 'idle';
    },
    updateUserStats: (state, action: PayloadAction<Partial<UserProfile['stats']>>) => {
      if (state.profile) {
        state.profile.stats = { ...state.profile.stats, ...action.payload };
      }
    },
    addAchievement: (state, action: PayloadAction<string>) => {
      if (state.profile && !state.profile.stats.achievements.includes(action.payload)) {
        state.profile.stats.achievements.push(action.payload);
      }
    },
    incrementStreak: (state) => {
      if (state.profile) {
        state.profile.stats.currentStreak += 1;
      }
    },
    resetStreak: (state) => {
      if (state.profile) {
        state.profile.stats.currentStreak = 0;
      }
    },
    setTheme: (state, action: PayloadAction<'light' | 'dark' | 'system'>) => {
      if (state.profile) {
        state.profile.preferences.theme = action.payload;
      }
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchUserProfile.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchUserProfile.fulfilled, (state, action) => {
        state.loading = false;
        state.profile = action.payload;
        state.error = null;
      })
      .addCase(fetchUserProfile.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      })
      .addCase(updateUserProfile.pending, (state) => {
        state.updateStatus = 'loading';
      })
      .addCase(updateUserProfile.fulfilled, (state, action) => {
        state.updateStatus = 'succeeded';
        state.profile = action.payload;
      })
      .addCase(updateUserProfile.rejected, (state, action) => {
        state.updateStatus = 'failed';
        state.error = action.payload as string;
      })
      .addCase(updateUserPreferences.fulfilled, (state, action) => {
        if (state.profile) {
          state.profile.preferences = action.payload.preferences;
        }
      })
      .addCase(uploadAvatar.fulfilled, (state, action) => {
        if (state.profile) {
          state.profile.avatar = action.payload.avatarUrl;
        }
      });
  },
});

export const {
  setUser,
  clearUser,
  updateUserStats,
  addAchievement,
  incrementStreak,
  resetStreak,
  setTheme,
} = userSlice.actions;

export default userSlice.reducer;