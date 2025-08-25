import { createSelector } from '@reduxjs/toolkit';
import { RootState } from './index';

export const selectAuth = (state: RootState) => state.auth;
export const selectUser = (state: RootState) => state.user;
export const selectLearning = (state: RootState) => state.learning;
export const selectAssessment = (state: RootState) => state.assessment;
export const selectUI = (state: RootState) => state.ui;

export const selectIsAuthenticated = createSelector(
  [selectAuth],
  (auth) => !!auth?.token && !!auth?.user
);

export const selectUserProfile = createSelector(
  [selectUser],
  (user) => user?.profile
);

export const selectUserStats = createSelector(
  [selectUserProfile],
  (profile) => profile?.stats || {
    coursesCompleted: 0,
    totalLearningHours: 0,
    currentStreak: 0,
    points: 0,
    level: 0,
    achievements: [],
  }
);

export const selectUserLevel = createSelector(
  [selectUserStats],
  (stats) => {
    const points = stats.points;
    if (points < 100) return 1;
    if (points < 500) return 2;
    if (points < 1000) return 3;
    if (points < 2500) return 4;
    if (points < 5000) return 5;
    if (points < 10000) return 6;
    if (points < 20000) return 7;
    if (points < 40000) return 8;
    if (points < 80000) return 9;
    return 10;
  }
);

export const selectActiveLearningPaths = createSelector(
  [selectLearning],
  (learning) => learning?.paths?.filter(path => path.status === 'active') || []
);

export const selectCompletedLearningPaths = createSelector(
  [selectLearning],
  (learning) => learning?.paths?.filter(path => path.status === 'completed') || []
);

export const selectLearningProgress = createSelector(
  [selectLearning],
  (learning) => {
    if (!learning?.currentPath) return 0;
    const { completedModules = 0, totalModules = 1 } = learning.currentPath;
    return Math.round((completedModules / totalModules) * 100);
  }
);

export const selectUpcomingAssessments = createSelector(
  [selectAssessment],
  (assessment) => assessment?.assessments?.filter(a => 
    a.status === 'not-started' && 
    new Date(a.scheduledAt) > new Date()
  ) || []
);

export const selectAssessmentHistory = createSelector(
  [selectAssessment],
  (assessment) => assessment?.history || []
);

export const selectAverageScore = createSelector(
  [selectAssessmentHistory],
  (history) => {
    if (!history.length) return 0;
    const total = history.reduce((sum, result) => sum + result.percentage, 0);
    return Math.round(total / history.length);
  }
);

export const selectPassRate = createSelector(
  [selectAssessmentHistory],
  (history) => {
    if (!history.length) return 0;
    const passed = history.filter(result => result.passed).length;
    return Math.round((passed / history.length) * 100);
  }
);

export const selectCurrentQuestionProgress = createSelector(
  [selectAssessment],
  (assessment) => {
    if (!assessment?.currentAssessment) return { current: 0, total: 0 };
    return {
      current: assessment.currentQuestion + 1,
      total: assessment.currentAssessment.questions.length,
    };
  }
);

export const selectAnsweredQuestions = createSelector(
  [selectAssessment],
  (assessment) => Object.keys(assessment?.answers || {}).length
);

export const selectUnansweredQuestions = createSelector(
  [selectAssessment, selectCurrentQuestionProgress],
  (assessment, progress) => {
    if (!assessment?.currentAssessment) return [];
    const answered = Object.keys(assessment.answers || {});
    return assessment.currentAssessment.questions
      .map((q, index) => ({ ...q, index }))
      .filter(q => !answered.includes(q.id));
  }
);

export const selectStudyTime = createSelector(
  [selectLearning],
  (learning) => {
    if (!learning?.currentSession) return 0;
    const start = new Date(learning.currentSession.startedAt).getTime();
    const now = Date.now();
    return Math.floor((now - start) / 1000);
  }
);

export const selectTotalStudyTime = createSelector(
  [selectLearning],
  (learning) => {
    if (!learning?.sessions) return 0;
    return learning.sessions.reduce((total, session) => {
      const duration = session.duration || 0;
      return total + duration;
    }, 0);
  }
);

export const selectRecentAchievements = createSelector(
  [selectUserStats],
  (stats) => {
    const achievements = stats.achievements || [];
    return achievements.slice(-5).reverse();
  }
);

export const selectNotificationCount = createSelector(
  [(state: RootState) => state],
  (state) => {
    return 0;
  }
);

export const selectTheme = createSelector(
  [selectUserProfile, selectUI],
  (profile, ui) => {
    if (profile?.preferences?.theme === 'system') {
      return typeof window !== 'undefined' && 
             window.matchMedia('(prefers-color-scheme: dark)').matches 
             ? 'dark' 
             : 'light';
    }
    return profile?.preferences?.theme || ui?.theme || 'light';
  }
);