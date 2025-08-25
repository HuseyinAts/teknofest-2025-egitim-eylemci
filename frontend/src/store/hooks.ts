import { useDispatch, useSelector, TypedUseSelectorHook } from 'react-redux';
import type { RootState, AppDispatch } from './index';
import { useMemo } from 'react';
import { bindActionCreators } from '@reduxjs/toolkit';

export const useAppDispatch = () => useDispatch<AppDispatch>();
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;

export function useActionCreators<T extends Record<string, (...args: any[]) => any>>(
  actions: T
): T {
  const dispatch = useAppDispatch();
  return useMemo(
    () => bindActionCreators(actions, dispatch) as T,
    [actions, dispatch]
  );
}

export const useAuth = () => useAppSelector((state) => state.auth);
export const useUser = () => useAppSelector((state) => state.user);
export const useLearning = () => useAppSelector((state) => state.learning);
export const useAssessment = () => useAppSelector((state) => state.assessment);
export const useUI = () => useAppSelector((state) => state.ui);

export const useIsAuthenticated = () => {
  const auth = useAuth();
  return !!auth?.token && !!auth?.user;
};

export const useUserRole = () => {
  const user = useUser();
  return user?.profile?.role || null;
};

export const useCurrentLearningPath = () => {
  const learning = useLearning();
  return learning?.currentPath || null;
};

export const useCurrentAssessment = () => {
  const assessment = useAssessment();
  return assessment?.currentAssessment || null;
};

export const useLoadingStates = () => {
  const user = useUser();
  const learning = useLearning();
  const assessment = useAssessment();
  
  return {
    userLoading: user?.loading || false,
    learningLoading: learning?.loading || false,
    assessmentLoading: assessment?.loading || false,
    anyLoading: user?.loading || learning?.loading || assessment?.loading || false,
  };
};

export const useErrors = () => {
  const user = useUser();
  const learning = useLearning();
  const assessment = useAssessment();
  
  return {
    userError: user?.error,
    learningError: learning?.error,
    assessmentError: assessment?.error,
    hasError: !!(user?.error || learning?.error || assessment?.error),
  };
};