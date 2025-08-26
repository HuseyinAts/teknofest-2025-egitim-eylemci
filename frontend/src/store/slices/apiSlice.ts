import { createApi, fetchBaseQuery, retry } from '@reduxjs/toolkit/query/react';
import type { RootState } from '../index';

const baseQuery = fetchBaseQuery({
  baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api',
  credentials: 'include',
  prepareHeaders: (headers, { getState }) => {
    const token = (getState() as RootState).auth?.token;
    if (token) {
      headers.set('authorization', `Bearer ${token}`);
    }
    headers.set('Content-Type', 'application/json');
    return headers;
  },
  timeout: 30000,
});

const baseQueryWithRetry = retry(baseQuery, { maxRetries: 3 });

export const apiSlice = createApi({
  reducerPath: 'api',
  baseQuery: baseQueryWithRetry,
  tagTypes: [
    'User',
    'LearningPath',
    'Assessment',
    'StudySession',
    'Course',
    'Progress',
    'Notification',
    'Achievement',
  ],
  endpoints: (builder) => ({
    getUser: builder.query({
      query: (id) => `/users/${id}`,
      providesTags: ['User'],
      transformResponse: (response: any) => response.data,
      keepUnusedDataFor: 300,
    }),
    
    updateUser: builder.mutation({
      query: ({ id, ...updates }) => ({
        url: `/users/${id}`,
        method: 'PATCH',
        body: updates,
      }),
      invalidatesTags: ['User'],
      transformResponse: (response: any) => response.data,
    }),
    
    getLearningPaths: builder.query({
      query: (params) => ({
        url: '/learning-paths',
        params,
      }),
      providesTags: ['LearningPath'],
      transformResponse: (response: any) => response.data,
      keepUnusedDataFor: 600,
    }),
    
    getLearningPath: builder.query({
      query: (id) => `/learning-paths/${id}`,
      providesTags: (result, error, id) => [{ type: 'LearningPath', id }],
      transformResponse: (response: any) => response.data,
    }),
    
    createLearningPath: builder.mutation({
      query: (data) => ({
        url: '/learning-paths',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: ['LearningPath'],
      transformResponse: (response: any) => response.data,
    }),
    
    updateLearningPath: builder.mutation({
      query: ({ id, ...updates }) => ({
        url: `/learning-paths/${id}`,
        method: 'PATCH',
        body: updates,
      }),
      invalidatesTags: (result, error, { id }) => [
        { type: 'LearningPath', id },
        'LearningPath',
      ],
      transformResponse: (response: any) => response.data,
    }),
    
    getAssessments: builder.query({
      query: (params) => ({
        url: '/assessments',
        params,
      }),
      providesTags: ['Assessment'],
      transformResponse: (response: any) => response.data,
      keepUnusedDataFor: 300,
    }),
    
    getAssessment: builder.query({
      query: (id) => `/assessments/${id}`,
      providesTags: (result, error, id) => [{ type: 'Assessment', id }],
      transformResponse: (response: any) => response.data,
    }),
    
    startAssessment: builder.mutation({
      query: (id) => ({
        url: `/assessments/${id}/start`,
        method: 'POST',
      }),
      invalidatesTags: (result, error, id) => [{ type: 'Assessment', id }],
      transformResponse: (response: any) => response.data,
    }),
    
    submitAssessment: builder.mutation({
      query: ({ id, answers }) => ({
        url: `/assessments/${id}/submit`,
        method: 'POST',
        body: { answers },
      }),
      invalidatesTags: ['Assessment', 'Progress'],
      transformResponse: (response: any) => response.data,
    }),
    
    getStudySessions: builder.query({
      query: (params) => ({
        url: '/study-sessions',
        params,
      }),
      providesTags: ['StudySession'],
      transformResponse: (response: any) => response.data,
    }),
    
    createStudySession: builder.mutation({
      query: (data) => ({
        url: '/study-sessions',
        method: 'POST',
        body: data,
      }),
      invalidatesTags: ['StudySession', 'Progress'],
      transformResponse: (response: any) => response.data,
    }),
    
    endStudySession: builder.mutation({
      query: ({ id, notes }) => ({
        url: `/study-sessions/${id}/end`,
        method: 'POST',
        body: { notes },
      }),
      invalidatesTags: ['StudySession', 'Progress'],
      transformResponse: (response: any) => response.data,
    }),
    
    getCourses: builder.query({
      query: (params) => ({
        url: '/courses',
        params,
      }),
      providesTags: ['Course'],
      transformResponse: (response: any) => response.data,
      keepUnusedDataFor: 1800,
    }),
    
    getCourse: builder.query({
      query: (id) => `/courses/${id}`,
      providesTags: (result, error, id) => [{ type: 'Course', id }],
      transformResponse: (response: any) => response.data,
    }),
    
    enrollCourse: builder.mutation({
      query: (courseId) => ({
        url: `/courses/${courseId}/enroll`,
        method: 'POST',
      }),
      invalidatesTags: ['Course', 'Progress'],
      transformResponse: (response: any) => response.data,
    }),
    
    getUserProgress: builder.query({
      query: (userId) => `/users/${userId}/progress`,
      providesTags: ['Progress'],
      transformResponse: (response: any) => response.data,
    }),
    
    updateProgress: builder.mutation({
      query: ({ userId, courseId, progress }) => ({
        url: `/users/${userId}/progress`,
        method: 'POST',
        body: { courseId, progress },
      }),
      invalidatesTags: ['Progress'],
      transformResponse: (response: any) => response.data,
    }),
    
    getNotifications: builder.query({
      query: (params) => ({
        url: '/notifications',
        params,
      }),
      providesTags: ['Notification'],
      transformResponse: (response: any) => response.data,
    }),
    
    markNotificationRead: builder.mutation({
      query: (id) => ({
        url: `/notifications/${id}/read`,
        method: 'PATCH',
      }),
      invalidatesTags: ['Notification'],
      transformResponse: (response: any) => response.data,
      async onQueryStarted(id, { dispatch, queryFulfilled }) {
        const patchResult = dispatch(
          apiSlice.util.updateQueryData('getNotifications', undefined, (draft) => {
            const notification = draft.find((n: any) => n.id === id);
            if (notification) {
              notification.read = true;
            }
          })
        );
        try {
          await queryFulfilled;
        } catch {
          patchResult.undo();
        }
      },
    }),
    
    getAchievements: builder.query({
      query: (userId) => `/users/${userId}/achievements`,
      providesTags: ['Achievement'],
      transformResponse: (response: any) => response.data,
      keepUnusedDataFor: 3600,
    }),
    
    searchContent: builder.query({
      query: (query) => ({
        url: '/search',
        params: { q: query },
      }),
      transformResponse: (response: any) => response.data,
      keepUnusedDataFor: 60,
    }),
    
    getRecommendations: builder.query({
      query: (userId) => `/users/${userId}/recommendations`,
      transformResponse: (response: any) => response.data,
      keepUnusedDataFor: 1800,
    }),
    
    generateContent: builder.mutation({
      query: (data) => ({
        url: '/ai/generate',
        method: 'POST',
        body: data,
      }),
      transformResponse: (response: any) => response.data,
    }),
    
    analyzePerformance: builder.query({
      query: ({ userId, dateRange }) => ({
        url: `/users/${userId}/analytics`,
        params: dateRange,
      }),
      transformResponse: (response: any) => response.data,
      keepUnusedDataFor: 300,
    }),
  }),
});

export const {
  useGetUserQuery,
  useUpdateUserMutation,
  useGetLearningPathsQuery,
  useGetLearningPathQuery,
  useCreateLearningPathMutation,
  useUpdateLearningPathMutation,
  useGetAssessmentsQuery,
  useGetAssessmentQuery,
  useStartAssessmentMutation,
  useSubmitAssessmentMutation,
  useGetStudySessionsQuery,
  useCreateStudySessionMutation,
  useEndStudySessionMutation,
  useGetCoursesQuery,
  useGetCourseQuery,
  useEnrollCourseMutation,
  useGetUserProgressQuery,
  useUpdateProgressMutation,
  useGetNotificationsQuery,
  useMarkNotificationReadMutation,
  useGetAchievementsQuery,
  useSearchContentQuery,
  useGetRecommendationsQuery,
  useGenerateContentMutation,
  useAnalyzePerformanceQuery,
} = apiSlice;