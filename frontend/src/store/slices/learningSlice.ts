import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import api, { LearningPath, StudySession, Assessment } from '@/lib/api';

interface LearningState {
  paths: LearningPath[];
  currentPath: LearningPath | null;
  sessions: StudySession[];
  currentSession: StudySession | null;
  assessments: Assessment[];
  loading: boolean;
  error: string | null;
}

const initialState: LearningState = {
  paths: [],
  currentPath: null,
  sessions: [],
  currentSession: null,
  assessments: [],
  loading: false,
  error: null,
};

export const fetchLearningPaths = createAsyncThunk(
  'learning/fetchPaths',
  async () => {
    const response = await api.learningPaths.getAll();
    return response;
  }
);

export const fetchLearningPath = createAsyncThunk(
  'learning/fetchPath',
  async (id: string) => {
    const response = await api.learningPaths.getById(id);
    return response;
  }
);

export const createStudySession = createAsyncThunk(
  'learning/createSession',
  async (data: Partial<StudySession>) => {
    const response = await api.studySessions.create(data);
    return response;
  }
);

export const endStudySession = createAsyncThunk(
  'learning/endSession',
  async ({ id, notes }: { id: string; notes?: string }) => {
    const response = await api.studySessions.end(id, notes);
    return response;
  }
);

export const fetchAssessments = createAsyncThunk(
  'learning/fetchAssessments',
  async () => {
    const response = await api.assessments.getAll();
    return response;
  }
);

const learningSlice = createSlice({
  name: 'learning',
  initialState,
  reducers: {
    setCurrentPath: (state, action: PayloadAction<LearningPath | null>) => {
      state.currentPath = action.payload;
    },
    setCurrentSession: (state, action: PayloadAction<StudySession | null>) => {
      state.currentSession = action.payload;
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchLearningPaths.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchLearningPaths.fulfilled, (state, action) => {
        state.loading = false;
        state.paths = action.payload;
      })
      .addCase(fetchLearningPaths.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Öğrenme yolları yüklenemedi';
      })
      .addCase(fetchLearningPath.fulfilled, (state, action) => {
        state.currentPath = action.payload;
      })
      .addCase(createStudySession.fulfilled, (state, action) => {
        state.currentSession = action.payload;
        state.sessions.push(action.payload);
      })
      .addCase(endStudySession.fulfilled, (state, action) => {
        state.currentSession = null;
        const index = state.sessions.findIndex(s => s.id === action.payload.id);
        if (index !== -1) {
          state.sessions[index] = action.payload;
        }
      })
      .addCase(fetchAssessments.fulfilled, (state, action) => {
        state.assessments = action.payload;
      });
  },
});

export const { setCurrentPath, setCurrentSession, clearError } = learningSlice.actions;
export default learningSlice.reducer;