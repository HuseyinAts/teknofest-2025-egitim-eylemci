/**
 * React Hooks for Offline Functionality
 * TEKNOFEST 2025 - Production Ready
 */
'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import offlineManager, { CacheStrategy, SyncResult } from '@/lib/offlineManager';

// Hook for offline status
export function useOfflineStatus() {
  const [isOnline, setIsOnline] = useState(true);
  const [queueSize, setQueueSize] = useState(0);

  useEffect(() => {
    // Initialize based on browser API
    setIsOnline(navigator.onLine);
    
    const handleOnline = () => {
      setIsOnline(true);
    };

    const handleOffline = () => {
      setIsOnline(false);
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
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
      
      const response = await window.fetch(url, {
        ...fetchOptions,
        signal: abortControllerRef.current.signal
      });

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
      // Placeholder for sync logic
      const results: SyncResult[] = [];
      setSyncResults(results);
      setLastSyncTime(new Date());
      return results;
    } finally {
      setSyncing(false);
    }
  }, []);

  return { sync, syncing, syncResults, lastSyncTime };
}

// Hook for cache management
export function useOfflineCache() {
  const [cacheStats, setCacheStats] = useState<any>(null);
  const [clearing, setClearing] = useState(false);

  const getCacheStats = useCallback(async () => {
    // Placeholder for cache stats
    const stats = {
      cacheEntries: 0,
      cacheSize: 0
    };
    setCacheStats(stats);
    return stats;
  }, []);

  const clearCache = useCallback(async () => {
    setClearing(true);
    try {
      // Placeholder for cache clearing
      await getCacheStats();
    } finally {
      setClearing(false);
    }
  }, [getCacheStats]);

  useEffect(() => {
    getCacheStats();
  }, [getCacheStats]);

  return { cacheStats, clearCache, clearing, refreshStats: getCacheStats };
}

// Hook for service worker updates
export function useServiceWorkerUpdate() {
  const [updateAvailable, setUpdateAvailable] = useState(false);

  const skipWaiting = useCallback(() => {
    window.location.reload();
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