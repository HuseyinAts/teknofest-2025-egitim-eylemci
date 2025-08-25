'use client';

import React from 'react';
import {
  Box,
  CircularProgress,
  Typography,
  Skeleton,
  Fade,
  LinearProgress,
} from '@mui/material';

interface LoadingScreenProps {
  message?: string;
  variant?: 'circular' | 'linear' | 'skeleton';
  fullScreen?: boolean;
}

export const LoadingScreen: React.FC<LoadingScreenProps> = ({
  message = 'Yükleniyor...',
  variant = 'circular',
  fullScreen = false,
}) => {
  const content = (
    <>
      {variant === 'circular' && (
        <CircularProgress
          size={48}
          thickness={4}
          sx={{ color: 'primary.main' }}
          aria-label="Yükleniyor"
        />
      )}
      
      {variant === 'linear' && (
        <Box sx={{ width: '100%', maxWidth: 400 }}>
          <LinearProgress />
        </Box>
      )}
      
      {message && variant !== 'skeleton' && (
        <Typography
          variant="body1"
          color="text.secondary"
          sx={{ mt: 2 }}
        >
          {message}
        </Typography>
      )}
    </>
  );

  if (variant === 'skeleton') {
    return (
      <Box sx={{ p: 3 }}>
        <Skeleton variant="text" width="60%" height={40} />
        <Skeleton variant="text" width="40%" height={30} />
        <Box sx={{ mt: 2 }}>
          <Skeleton variant="rectangular" height={200} />
        </Box>
        <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
          <Skeleton variant="rectangular" width={100} height={40} />
          <Skeleton variant="rectangular" width={100} height={40} />
        </Box>
      </Box>
    );
  }

  if (fullScreen) {
    return (
      <Fade in timeout={300}>
        <Box
          sx={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: 'rgba(255, 255, 255, 0.95)',
            zIndex: 9999,
          }}
        >
          {content}
        </Box>
      </Fade>
    );
  }

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: 200,
        p: 3,
      }}
    >
      {content}
    </Box>
  );
};

interface ContentLoaderProps {
  loading: boolean;
  error?: Error | null;
  children: React.ReactNode;
  loadingComponent?: React.ReactNode;
  errorComponent?: React.ReactNode;
  retry?: () => void;
}

export const ContentLoader: React.FC<ContentLoaderProps> = ({
  loading,
  error,
  children,
  loadingComponent,
  errorComponent,
  retry,
}) => {
  if (loading) {
    return <>{loadingComponent || <LoadingScreen />}</>;
  }

  if (error) {
    if (errorComponent) {
      return <>{errorComponent}</>;
    }

    return (
      <Box
        sx={{
          textAlign: 'center',
          p: 3,
          minHeight: 200,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Typography variant="h6" color="error" gutterBottom>
          Bir hata oluştu
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          {error.message || 'Beklenmeyen bir hata oluştu'}
        </Typography>
        {retry && (
          <Box sx={{ mt: 2 }}>
            <button
              onClick={retry}
              className="px-4 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 transition"
              aria-label="Tekrar dene"
            >
              Tekrar Dene
            </button>
          </Box>
        )}
      </Box>
    );
  }

  return <>{children}</>;
};

export default LoadingScreen;