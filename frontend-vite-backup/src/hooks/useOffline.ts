/**
 * React Hooks for Offline Functionality
 * TEKNOFEST 2025 - Production Ready
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import offlineManager, { CacheStrategy, SyncResult } from '../services/offlineManager';

// Hook for offline status
export function useOfflineStatus() {
  const [isOnline, setIsOnline] = useState(offlineManager.getOnlineStatus());
  const [queueSize, setQueueSize] = useState(offlineManager.getOfflineQueueSize());

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
    };

    const handleOffline = () => {
      setIsOnline(false);
    };

    const handleQueueUpdate = () => {
      setQueueSize(offlineManager.getOfflineQueueSize());
    };

    offlineManager.on('online', handleOnline);
    offlineManager.on('offline', handleOffline);
    offlineManager.on('request-queued', handleQueueUpdate);
    offlineManager.on('sync-complete', handleQueueUpdate);

    return () => {
      offlineManager.off('online', handleOnline);
      offlineManager.off('offline', handleOffline);
      offlineManager.off('request-queued', handleQueueUpdate);
      offlineManager.off('sync-complete', handleQueueUpdate);
    };
  }, []);

  return { isOnline, queueSize };
}

// Hook for offline-aware fetch
export function useOfflineFetch() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const fetch = useCallback(async (
    url: string,
    options?: RequestInit & { cacheStrategy?: CacheStrategy; cacheTTL?: number }
  ) => {
    setLoading(true);
    setError(null);

    // Cancel previous request if exists
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // Create new abort controller
    abortControllerRef.current = new AbortController();

    try {
      const { cacheStrategy, cacheTTL, ...fetchOptions } = options || {};
      
      const response = await offlineManager.request(
        url,
        {
          ...fetchOptions,
          signal: abortControllerRef.current.signal
        },
        {
          strategy: cacheStrategy,
          ttl: cacheTTL
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setLoading(false);
      return data;
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        setError(err as Error);
      }
      setLoading(false);
      throw err;
    }
  }, []);

  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }, []);

  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  return { fetch, loading, error, cancel };
}

// Hook for offline sync
export function useOfflineSync() {
  const [syncing, setSyncing] = useState(false);
  const [syncResults, setSyncResults] = useState<SyncResult[]>([]);
  const [lastSyncTime, setLastSyncTime] = useState<Date | null>(null);

  const sync = useCallback(async () => {
    setSyncing(true);
    try {
      const results = await offlineManager.syncOfflineData();
      setSyncResults(results);
      setLastSyncTime(new Date());
      return results;
    } finally {
      setSyncing(false);
    }
  }, []);

  useEffect(() => {
    const handleSyncComplete = (results: SyncResult[]) => {
      setSyncResults(results);
      setLastSyncTime(new Date());
      setSyncing(false);
    };

    const handleSyncStart = () => {
      setSyncing(true);
    };

    offlineManager.on('sync-start', handleSyncStart);
    offlineManager.on('sync-complete', handleSyncComplete);

    return () => {
      offlineManager.off('sync-start', handleSyncStart);
      offlineManager.off('sync-complete', handleSyncComplete);
    };
  }, []);

  return { sync, syncing, syncResults, lastSyncTime };
}

// Hook for cache management
export function useOfflineCache() {
  const [cacheStats, setCacheStats] = useState<any>(null);
  const [clearing, setClearing] = useState(false);

  const getCacheStats = useCallback(async () => {
    const stats = await offlineManager.getCacheStats();
    setCacheStats(stats);
    return stats;
  }, []);

  const clearCache = useCallback(async () => {
    setClearing(true);
    try {
      await offlineManager.clearCache();
      await getCacheStats();
    } finally {
      setClearing(false);
    }
  }, [getCacheStats]);

  useEffect(() => {
    getCacheStats();

    const handleCacheUpdate = () => {
      getCacheStats();
    };

    offlineManager.on('cache-updated', handleCacheUpdate);
    offlineManager.on('cache-cleared', handleCacheUpdate);

    return () => {
      offlineManager.off('cache-updated', handleCacheUpdate);
      offlineManager.off('cache-cleared', handleCacheUpdate);
    };
  }, [getCacheStats]);

  return { cacheStats, clearCache, clearing, refreshStats: getCacheStats };
}

// Hook for service worker updates
export function useServiceWorkerUpdate() {
  const [updateAvailable, setUpdateAvailable] = useState(false);

  const skipWaiting = useCallback(() => {
    offlineManager.skipWaiting();
    window.location.reload();
  }, []);

  useEffect(() => {
    const handleUpdateAvailable = () => {
      setUpdateAvailable(true);
    };

    offlineManager.on('update-available', handleUpdateAvailable);

    return () => {
      offlineManager.off('update-available', handleUpdateAvailable);
    };
  }, []);

  return { updateAvailable, skipWaiting };
}

// Compound hook for all offline features
export function useOffline() {
  const status = useOfflineStatus();
  const fetch = useOfflineFetch();
  const sync = useOfflineSync();
  const cache = useOfflineCache();
  const update = useServiceWorkerUpdate();

  return {
    ...status,
    fetch: fetch.fetch,
    loading: fetch.loading,
    error: fetch.error,
    sync: sync.sync,
    syncing: sync.syncing,
    syncResults: sync.syncResults,
    lastSyncTime: sync.lastSyncTime,
    cacheStats: cache.cacheStats,
    clearCache: cache.clearCache,
    clearing: cache.clearing,
    updateAvailable: update.updateAvailable,
    skipWaiting: update.skipWaiting
  };
}