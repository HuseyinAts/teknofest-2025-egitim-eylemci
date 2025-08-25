/**
 * Offline Indicator Component
 * TEKNOFEST 2025 - Production Ready
 */

import React, { useState, useEffect } from 'react';
import {
  Snackbar,
  Alert,
  Badge,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  List,
  ListItem,
  ListItemText,
  LinearProgress,
  Typography,
  Box,
  Chip
} from '@mui/material';
import {
  CloudOff,
  Cloud,
  Sync,
  SyncProblem,
  CheckCircle,
  Error,
  Info,
  CloudQueue
} from '@mui/icons-material';
import { useOffline } from '../hooks/useOffline';
import { formatDistanceToNow } from 'date-fns';

const OfflineIndicator: React.FC = () => {
  const {
    isOnline,
    queueSize,
    sync,
    syncing,
    syncResults,
    lastSyncTime,
    cacheStats,
    clearCache,
    clearing,
    updateAvailable,
    skipWaiting
  } = useOffline();

  const [showSnackbar, setShowSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState<'success' | 'error' | 'warning' | 'info'>('info');
  const [showSyncDialog, setShowSyncDialog] = useState(false);
  const [previousOnlineStatus, setPreviousOnlineStatus] = useState(isOnline);

  // Handle online/offline transitions
  useEffect(() => {
    if (previousOnlineStatus !== isOnline) {
      setPreviousOnlineStatus(isOnline);
      
      if (isOnline) {
        setSnackbarMessage('Connection restored. Syncing offline data...');
        setSnackbarSeverity('success');
        setShowSnackbar(true);
        
        // Auto-sync when coming back online
        if (queueSize > 0) {
          sync();
        }
      } else {
        setSnackbarMessage('You are currently offline. Your data will be saved locally.');
        setSnackbarSeverity('warning');
        setShowSnackbar(true);
      }
    }
  }, [isOnline, previousOnlineStatus, queueSize, sync]);

  // Handle service worker updates
  useEffect(() => {
    if (updateAvailable) {
      setSnackbarMessage('A new version is available. Click to update.');
      setSnackbarSeverity('info');
      setShowSnackbar(true);
    }
  }, [updateAvailable]);

  const handleManualSync = async () => {
    const results = await sync();
    const successCount = results.filter(r => r.success).length;
    const failureCount = results.filter(r => !r.success).length;
    
    if (failureCount === 0) {
      setSnackbarMessage(`Successfully synced ${successCount} requests`);
      setSnackbarSeverity('success');
    } else {
      setSnackbarMessage(`Synced ${successCount} requests, ${failureCount} failed`);
      setSnackbarSeverity('warning');
    }
    setShowSnackbar(true);
  };

  const formatCacheSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <>
      {/* Status Icon */}
      <Box sx={{ position: 'fixed', bottom: 20, right: 20, zIndex: 1000 }}>
        <Tooltip title={isOnline ? 'Online' : 'Offline'}>
          <Badge badgeContent={queueSize} color="error" invisible={queueSize === 0}>
            <IconButton
              color={isOnline ? 'primary' : 'default'}
              onClick={() => setShowSyncDialog(true)}
              sx={{
                backgroundColor: 'background.paper',
                boxShadow: 2,
                '&:hover': {
                  backgroundColor: 'background.paper',
                  transform: 'scale(1.1)'
                }
              }}
            >
              {isOnline ? (
                queueSize > 0 ? <CloudQueue /> : <Cloud />
              ) : (
                <CloudOff />
              )}
            </IconButton>
          </Badge>
        </Tooltip>
      </Box>

      {/* Sync Dialog */}
      <Dialog
        open={showSyncDialog}
        onClose={() => setShowSyncDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Typography variant="h6">Offline Status</Typography>
            <Chip
              label={isOnline ? 'Online' : 'Offline'}
              color={isOnline ? 'success' : 'default'}
              icon={isOnline ? <Cloud /> : <CloudOff />}
            />
          </Box>
        </DialogTitle>
        
        <DialogContent>
          {/* Connection Status */}
          <Box mb={2}>
            <Typography variant="subtitle2" color="textSecondary">
              Connection Status
            </Typography>
            <Typography variant="body1">
              {isOnline
                ? 'You are connected to the internet'
                : 'You are working offline. Changes will be synced when connection is restored.'}
            </Typography>
          </Box>

          {/* Queue Status */}
          {queueSize > 0 && (
            <Box mb={2}>
              <Typography variant="subtitle2" color="textSecondary">
                Pending Requests
              </Typography>
              <Typography variant="body1">
                {queueSize} request{queueSize > 1 ? 's' : ''} waiting to be synced
              </Typography>
            </Box>
          )}

          {/* Last Sync Time */}
          {lastSyncTime && (
            <Box mb={2}>
              <Typography variant="subtitle2" color="textSecondary">
                Last Sync
              </Typography>
              <Typography variant="body1">
                {formatDistanceToNow(lastSyncTime, { addSuffix: true })}
              </Typography>
            </Box>
          )}

          {/* Cache Stats */}
          {cacheStats && (
            <Box mb={2}>
              <Typography variant="subtitle2" color="textSecondary">
                Cache Statistics
              </Typography>
              <Typography variant="body2">
                Entries: {cacheStats.cacheEntries}
              </Typography>
              <Typography variant="body2">
                Size: {formatCacheSize(cacheStats.cacheSize || 0)}
              </Typography>
            </Box>
          )}

          {/* Sync Progress */}
          {syncing && (
            <Box mb={2}>
              <Typography variant="subtitle2" color="textSecondary">
                Syncing...
              </Typography>
              <LinearProgress />
            </Box>
          )}

          {/* Sync Results */}
          {syncResults.length > 0 && !syncing && (
            <Box mb={2}>
              <Typography variant="subtitle2" color="textSecondary">
                Recent Sync Results
              </Typography>
              <List dense>
                {syncResults.slice(0, 5).map((result, index) => (
                  <ListItem key={index}>
                    <ListItemText
                      primary={
                        <Box display="flex" alignItems="center" gap={1}>
                          {result.success ? (
                            <CheckCircle color="success" fontSize="small" />
                          ) : (
                            <Error color="error" fontSize="small" />
                          )}
                          <Typography variant="body2">
                            {result.request.method} {result.request.url}
                          </Typography>
                        </Box>
                      }
                      secondary={result.error}
                    />
                  </ListItem>
                ))}
              </List>
            </Box>
          )}
        </DialogContent>

        <DialogActions>
          <Button
            onClick={clearCache}
            disabled={clearing}
            color="secondary"
          >
            Clear Cache
          </Button>
          <Button
            onClick={handleManualSync}
            disabled={!isOnline || syncing || queueSize === 0}
            startIcon={<Sync />}
            variant="contained"
          >
            Sync Now
          </Button>
          <Button onClick={() => setShowSyncDialog(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar Notifications */}
      <Snackbar
        open={showSnackbar}
        autoHideDuration={6000}
        onClose={() => setShowSnackbar(false)}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setShowSnackbar(false)}
          severity={snackbarSeverity}
          action={
            updateAvailable ? (
              <Button color="inherit" size="small" onClick={skipWaiting}>
                Update
              </Button>
            ) : undefined
          }
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </>
  );
};

export default OfflineIndicator;