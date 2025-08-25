import { Middleware, isRejectedWithValue } from '@reduxjs/toolkit';
import { toast } from 'react-hot-toast';

interface ErrorResponse {
  message?: string;
  status?: number;
  data?: {
    message?: string;
    error?: string;
  };
}

export const errorMiddleware: Middleware = () => (next) => (action) => {
  if (isRejectedWithValue(action)) {
    const error = action.payload as ErrorResponse;
    
    const errorMessage = 
      error?.data?.message || 
      error?.message || 
      error?.data?.error ||
      'An unexpected error occurred';

    if (error?.status === 401) {
      toast.error('Session expired. Please login again.');
      window.location.href = '/login';
    } else if (error?.status === 403) {
      toast.error('You do not have permission to perform this action.');
    } else if (error?.status === 404) {
      toast.error('Resource not found.');
    } else if (error?.status === 429) {
      toast.error('Too many requests. Please try again later.');
    } else if (error?.status && error.status >= 500) {
      toast.error('Server error. Please try again later.');
      console.error('Server Error:', error);
    } else {
      toast.error(errorMessage);
    }

    if (process.env.NODE_ENV === 'development') {
      console.error('API Error:', {
        action: action.type,
        payload: action.payload,
        meta: action.meta,
      });
    }
  }

  return next(action);
};