import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import axios from 'axios';

export interface Question {
  id: string;
  type: 'multiple-choice' | 'true-false' | 'short-answer' | 'essay' | 'code';
  question: string;
  options?: string[];
  correctAnswer?: string | string[];
  points: number;
  difficulty: 'easy' | 'medium' | 'hard';
  topic: string;
  explanation?: string;
  hints?: string[];
  timeLimit?: number;
}

export interface Assessment {
  id: string;
  title: string;
  description: string;
  type: 'quiz' | 'exam' | 'practice' | 'diagnostic';
  questions: Question[];
  totalPoints: number;
  passingScore: number;
  timeLimit?: number;
  attempts: number;
  maxAttempts?: number;
  status: 'not-started' | 'in-progress' | 'completed' | 'expired';
  createdAt: string;
  updatedAt: string;
}

export interface AssessmentResult {
  id: string;
  assessmentId: string;
  userId: string;
  score: number;
  percentage: number;
  passed: boolean;
  answers: {
    questionId: string;
    answer: string | string[];
    isCorrect: boolean;
    pointsEarned: number;
    timeSpent: number;
  }[];
  startedAt: string;
  completedAt: string;
  feedback?: string;
}

export interface AssessmentState {
  assessments: Assessment[];
  currentAssessment: Assessment | null;
  currentQuestion: number;
  answers: Record<string, string | string[]>;
  results: AssessmentResult[];
  history: AssessmentResult[];
  loading: boolean;
  submitting: boolean;
  error: string | null;
  timer: {
    isActive: boolean;
    remainingTime: number;
  };
}

const initialState: AssessmentState = {
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

export const fetchAssessments = createAsyncThunk(
  'assessment/fetchAll',
  async (filters?: { type?: string; status?: string }, { rejectWithValue }) => {
    try {
      const params = new URLSearchParams(filters as any);
      const response = await axios.get(`/api/assessments?${params}`);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data || 'Failed to fetch assessments');
    }
  }
);

export const fetchAssessment = createAsyncThunk(
  'assessment/fetchOne',
  async (id: string, { rejectWithValue }) => {
    try {
      const response = await axios.get(`/api/assessments/${id}`);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data || 'Failed to fetch assessment');
    }
  }
);

export const startAssessment = createAsyncThunk(
  'assessment/start',
  async (assessmentId: string, { rejectWithValue }) => {
    try {
      const response = await axios.post(`/api/assessments/${assessmentId}/start`);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data || 'Failed to start assessment');
    }
  }
);

export const submitAssessment = createAsyncThunk(
  'assessment/submit',
  async ({ assessmentId, answers }: { assessmentId: string; answers: Record<string, any> }, { rejectWithValue }) => {
    try {
      const response = await axios.post(`/api/assessments/${assessmentId}/submit`, { answers });
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data || 'Failed to submit assessment');
    }
  }
);

export const fetchAssessmentHistory = createAsyncThunk(
  'assessment/fetchHistory',
  async (userId: string, { rejectWithValue }) => {
    try {
      const response = await axios.get(`/api/users/${userId}/assessment-history`);
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data || 'Failed to fetch assessment history');
    }
  }
);

export const generatePracticeQuestions = createAsyncThunk(
  'assessment/generatePractice',
  async ({ topic, difficulty, count }: { topic: string; difficulty: string; count: number }, { rejectWithValue }) => {
    try {
      const response = await axios.post('/api/assessments/generate-practice', {
        topic,
        difficulty,
        count,
      });
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data || 'Failed to generate practice questions');
    }
  }
);

const assessmentSlice = createSlice({
  name: 'assessment',
  initialState,
  reducers: {
    setCurrentAssessment: (state, action: PayloadAction<Assessment | null>) => {
      state.currentAssessment = action.payload;
      state.currentQuestion = 0;
      state.answers = {};
      if (action.payload && action.payload.timeLimit) {
        state.timer.remainingTime = action.payload.timeLimit * 60;
        state.timer.isActive = false;
      }
    },
    setAnswer: (state, action: PayloadAction<{ questionId: string; answer: string | string[] }>) => {
      state.answers[action.payload.questionId] = action.payload.answer;
    },
    nextQuestion: (state) => {
      if (state.currentAssessment && state.currentQuestion < state.currentAssessment.questions.length - 1) {
        state.currentQuestion += 1;
      }
    },
    previousQuestion: (state) => {
      if (state.currentQuestion > 0) {
        state.currentQuestion -= 1;
      }
    },
    goToQuestion: (state, action: PayloadAction<number>) => {
      if (state.currentAssessment && action.payload >= 0 && action.payload < state.currentAssessment.questions.length) {
        state.currentQuestion = action.payload;
      }
    },
    startTimer: (state) => {
      state.timer.isActive = true;
    },
    stopTimer: (state) => {
      state.timer.isActive = false;
    },
    updateTimer: (state, action: PayloadAction<number>) => {
      state.timer.remainingTime = action.payload;
      if (action.payload <= 0) {
        state.timer.isActive = false;
      }
    },
    clearAssessmentError: (state) => {
      state.error = null;
    },
    resetAssessment: (state) => {
      state.currentAssessment = null;
      state.currentQuestion = 0;
      state.answers = {};
      state.timer = { isActive: false, remainingTime: 0 };
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchAssessments.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchAssessments.fulfilled, (state, action) => {
        state.loading = false;
        state.assessments = action.payload;
      })
      .addCase(fetchAssessments.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      })
      .addCase(fetchAssessment.fulfilled, (state, action) => {
        state.currentAssessment = action.payload;
        if (action.payload.timeLimit) {
          state.timer.remainingTime = action.payload.timeLimit * 60;
        }
      })
      .addCase(startAssessment.fulfilled, (state, action) => {
        state.currentAssessment = action.payload;
        state.currentQuestion = 0;
        state.answers = {};
        state.timer.isActive = true;
      })
      .addCase(submitAssessment.pending, (state) => {
        state.submitting = true;
      })
      .addCase(submitAssessment.fulfilled, (state, action) => {
        state.submitting = false;
        state.results.push(action.payload);
        state.currentAssessment = null;
        state.answers = {};
        state.timer = { isActive: false, remainingTime: 0 };
      })
      .addCase(submitAssessment.rejected, (state, action) => {
        state.submitting = false;
        state.error = action.payload as string;
      })
      .addCase(fetchAssessmentHistory.fulfilled, (state, action) => {
        state.history = action.payload;
      })
      .addCase(generatePracticeQuestions.fulfilled, (state, action) => {
        state.currentAssessment = action.payload;
        state.currentQuestion = 0;
        state.answers = {};
      });
  },
});

export const {
  setCurrentAssessment,
  setAnswer,
  nextQuestion,
  previousQuestion,
  goToQuestion,
  startTimer,
  stopTimer,
  updateTimer,
  clearAssessmentError,
  resetAssessment,
} = assessmentSlice.actions;

export default assessmentSlice.reducer;