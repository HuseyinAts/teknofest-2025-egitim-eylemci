/**
 * Offline Manager for React Application
 * TEKNOFEST 2025 - Production Ready
 */

import { EventEmitter } from 'events';

export enum CacheStrategy {
  CACHE_FIRST = 'cache-first',
  NETWORK_FIRST = 'network-first',
  CACHE_ONLY = 'cache-only',
  NETWORK_ONLY = 'network-only',
  STALE_WHILE_REVALIDATE = 'stale-while-revalidate'
}

export interface OfflineRequest {
  id: string;
  url: string;
  method: string;
  headers?: HeadersInit;
  body?: any;
  timestamp: number;
  retryCount: number;
  maxRetries: number;
}

export interface CacheConfig {
  strategy: CacheStrategy;
  ttl: number;
  maxSize: number;
  enableCompression: boolean;
}

export interface SyncResult {
  success: boolean;
  request: OfflineRequest;
  error?: string;
}

class OfflineManager extends EventEmitter {
  private static instance: OfflineManager;
  private isOnline: boolean = navigator.onLine;
  private serviceWorkerRegistration: ServiceWorkerRegistration | null = null;
  private offlineQueue: OfflineRequest[] = [];
  private syncInProgress: boolean = false;
  private db: IDBDatabase | null = null;
  private readonly DB_NAME = 'teknofest_offline';
  private readonly DB_VERSION = 1;

  private constructor() {
    super();
    this.initialize();
  }

  public static getInstance(): OfflineManager {
    if (!OfflineManager.instance) {
      OfflineManager.instance = new OfflineManager();
    }
    return OfflineManager.instance;
  }

  private async initialize(): Promise<void> {
    // Register service worker
    await this.registerServiceWorker();
    
    // Initialize IndexedDB
    await this.initializeDatabase();
    
    // Setup event listeners
    this.setupEventListeners();
    
    // Load offline queue
    await this.loadOfflineQueue();
    
    // Check initial online status
    this.checkOnlineStatus();
  }

  private async registerServiceWorker(): Promise<void> {
    if ('serviceWorker' in navigator) {
      try {
        this.serviceWorkerRegistration = await navigator.serviceWorker.register(
          '/service-worker.js',
          { scope: '/' }
        );
        
        console.log('Service Worker registered successfully');
        
        // Listen for service worker updates
        this.serviceWorkerRegistration.addEventListener('updatefound', () => {
          this.handleServiceWorkerUpdate();
        });
        
        // Listen for messages from service worker
        navigator.serviceWorker.addEventListener('message', (event) => {
          this.handleServiceWorkerMessage(event);
        });
        
      } catch (error) {
        console.error('Service Worker registration failed:', error);
      }
    }
  }

  private async initializeDatabase(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.DB_NAME, this.DB_VERSION);
      
      request.onerror = () => {
        console.error('Failed to open IndexedDB');
        reject(request.error);
      };
      
      request.onsuccess = () => {
        this.db = request.result;
        console.log('IndexedDB initialized');
        resolve();
      };
      
      request.onupgradeneeded = (event: IDBVersionChangeEvent) => {
        const db = (event.target as IDBOpenDBRequest).result;
        
        // Create offline queue store
        if (!db.objectStoreNames.contains('offline_queue')) {
          const store = db.createObjectStore('offline_queue', {
            keyPath: 'id',
            autoIncrement: true
          });
          store.createIndex('timestamp', 'timestamp', { unique: false });
        }
        
        // Create cache store
        if (!db.objectStoreNames.contains('cache')) {
          const cacheStore = db.createObjectStore('cache', {
            keyPath: 'key'
          });
          cacheStore.createIndex('expires', 'expires', { unique: false });
        }
      };
    });
  }

  private setupEventListeners(): void {
    // Online/offline events
    window.addEventListener('online', () => this.handleOnline());
    window.addEventListener('offline', () => this.handleOffline());
    
    // Visibility change
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden && this.isOnline) {
        this.syncOfflineData();
      }
    });
    
    // Page unload - save state
    window.addEventListener('beforeunload', () => {
      this.saveOfflineQueue();
    });
  }

  private handleOnline(): void {
    console.log('Network: Online');
    this.isOnline = true;
    this.emit('online');
    
    // Start syncing offline data
    this.syncOfflineData();
  }

  private handleOffline(): void {
    console.log('Network: Offline');
    this.isOnline = false;
    this.emit('offline');
  }

  private checkOnlineStatus(): void {
    // Use multiple methods to check online status
    const checks = [
      this.checkNavigatorOnline(),
      this.checkFetch(),
      this.checkWebSocket()
    ];
    
    Promise.race(checks).then((online) => {
      this.isOnline = online;
      this.emit(online ? 'online' : 'offline');
    });
  }

  private async checkNavigatorOnline(): Promise<boolean> {
    return navigator.onLine;
  }

  private async checkFetch(): Promise<boolean> {
    try {
      const response = await fetch('/api/health', {
        method: 'HEAD',
        cache: 'no-cache'
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  private async checkWebSocket(): Promise<boolean> {
    return new Promise((resolve) => {
      const ws = new WebSocket(`${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`);
      const timeout = setTimeout(() => {
        ws.close();
        resolve(false);
      }, 3000);
      
      ws.onopen = () => {
        clearTimeout(timeout);
        ws.close();
        resolve(true);
      };
      
      ws.onerror = () => {
        clearTimeout(timeout);
        resolve(false);
      };
    });
  }

  public async request(
    url: string,
    options: RequestInit = {},
    cacheConfig?: Partial<CacheConfig>
  ): Promise<Response> {
    const config: CacheConfig = {
      strategy: CacheStrategy.NETWORK_FIRST,
      ttl: 3600000, // 1 hour
      maxSize: 50 * 1024 * 1024, // 50MB
      enableCompression: true,
      ...cacheConfig
    };
    
    // Add offline status header
    const headers = new Headers(options.headers);
    headers.set('X-Network-Status', this.isOnline ? 'online' : 'offline');
    
    const requestOptions: RequestInit = {
      ...options,
      headers
    };
    
    // Handle based on cache strategy
    switch (config.strategy) {
      case CacheStrategy.CACHE_FIRST:
        return this.cacheFirst(url, requestOptions, config);
      
      case CacheStrategy.NETWORK_FIRST:
        return this.networkFirst(url, requestOptions, config);
      
      case CacheStrategy.CACHE_ONLY:
        return this.cacheOnly(url, requestOptions);
      
      case CacheStrategy.NETWORK_ONLY:
        return this.networkOnly(url, requestOptions);
      
      case CacheStrategy.STALE_WHILE_REVALIDATE:
        return this.staleWhileRevalidate(url, requestOptions, config);
      
      default:
        return this.networkFirst(url, requestOptions, config);
    }
  }

  private async cacheFirst(
    url: string,
    options: RequestInit,
    config: CacheConfig
  ): Promise<Response> {
    // Try cache first
    const cached = await this.getFromCache(url);
    if (cached) {
      return new Response(cached.data, {
        status: 200,
        headers: { 'X-Cache': 'HIT' }
      });
    }
    
    // Fallback to network
    try {
      const response = await fetch(url, options);
      if (response.ok) {
        await this.saveToCache(url, await response.clone().text(), config.ttl);
      }
      return response;
    } catch (error) {
      throw new Error(`Request failed and no cache available: ${error}`);
    }
  }

  private async networkFirst(
    url: string,
    options: RequestInit,
    config: CacheConfig
  ): Promise<Response> {
    try {
      const response = await fetch(url, options);
      if (response.ok) {
        await this.saveToCache(url, await response.clone().text(), config.ttl);
      }
      return response;
    } catch (error) {
      // Try cache on network failure
      const cached = await this.getFromCache(url);
      if (cached) {
        return new Response(cached.data, {
          status: 200,
          headers: { 'X-Cache': 'HIT', 'X-Cache-Stale': 'true' }
        });
      }
      
      // Queue for offline sync if POST/PUT/DELETE
      if (options.method && ['POST', 'PUT', 'DELETE'].includes(options.method)) {
        await this.queueOfflineRequest(url, options);
        return new Response(JSON.stringify({
          queued: true,
          message: 'Request queued for synchronization'
        }), {
          status: 202,
          headers: { 'Content-Type': 'application/json' }
        });
      }
      
      throw error;
    }
  }

  private async cacheOnly(
    url: string,
    options: RequestInit
  ): Promise<Response> {
    const cached = await this.getFromCache(url);
    if (cached) {
      return new Response(cached.data, {
        status: 200,
        headers: { 'X-Cache': 'HIT' }
      });
    }
    
    throw new Error('No cache available');
  }

  private async networkOnly(
    url: string,
    options: RequestInit
  ): Promise<Response> {
    return fetch(url, options);
  }

  private async staleWhileRevalidate(
    url: string,
    options: RequestInit,
    config: CacheConfig
  ): Promise<Response> {
    const cached = await this.getFromCache(url);
    
    // Return stale cache immediately
    if (cached) {
      // Revalidate in background
      fetch(url, options).then(async (response) => {
        if (response.ok) {
          await this.saveToCache(url, await response.text(), config.ttl);
        }
      }).catch(console.error);
      
      return new Response(cached.data, {
        status: 200,
        headers: { 'X-Cache': 'HIT', 'X-Cache-Stale': 'true' }
      });
    }
    
    // No cache, fetch from network
    const response = await fetch(url, options);
    if (response.ok) {
      await this.saveToCache(url, await response.clone().text(), config.ttl);
    }
    return response;
  }

  private async getFromCache(key: string): Promise<any | null> {
    if (!this.db) return null;
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['cache'], 'readonly');
      const store = transaction.objectStore('cache');
      const request = store.get(key);
      
      request.onsuccess = () => {
        const result = request.result;
        if (result && result.expires > Date.now()) {
          resolve(result);
        } else {
          resolve(null);
        }
      };
      
      request.onerror = () => reject(request.error);
    });
  }

  private async saveToCache(key: string, data: any, ttl: number): Promise<void> {
    if (!this.db) return;
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['cache'], 'readwrite');
      const store = transaction.objectStore('cache');
      
      const cacheEntry = {
        key,
        data,
        expires: Date.now() + ttl,
        created: Date.now()
      };
      
      const request = store.put(cacheEntry);
      
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  private async queueOfflineRequest(url: string, options: RequestInit): Promise<void> {
    const request: OfflineRequest = {
      id: `${Date.now()}-${Math.random()}`,
      url,
      method: options.method || 'GET',
      headers: options.headers as HeadersInit,
      body: options.body,
      timestamp: Date.now(),
      retryCount: 0,
      maxRetries: 3
    };
    
    this.offlineQueue.push(request);
    await this.saveOfflineQueue();
    
    this.emit('request-queued', request);
  }

  private async loadOfflineQueue(): Promise<void> {
    if (!this.db) return;
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['offline_queue'], 'readonly');
      const store = transaction.objectStore('offline_queue');
      const request = store.getAll();
      
      request.onsuccess = () => {
        this.offlineQueue = request.result || [];
        console.log(`Loaded ${this.offlineQueue.length} offline requests`);
        resolve();
      };
      
      request.onerror = () => reject(request.error);
    });
  }

  private async saveOfflineQueue(): Promise<void> {
    if (!this.db) return;
    
    const transaction = this.db.transaction(['offline_queue'], 'readwrite');
    const store = transaction.objectStore('offline_queue');
    
    // Clear existing
    await new Promise((resolve) => {
      const clearRequest = store.clear();
      clearRequest.onsuccess = () => resolve(undefined);
    });
    
    // Add all queue items
    for (const request of this.offlineQueue) {
      store.add(request);
    }
  }

  public async syncOfflineData(): Promise<SyncResult[]> {
    if (this.syncInProgress || !this.isOnline || this.offlineQueue.length === 0) {
      return [];
    }
    
    this.syncInProgress = true;
    this.emit('sync-start');
    
    const results: SyncResult[] = [];
    const failedRequests: OfflineRequest[] = [];
    
    for (const request of this.offlineQueue) {
      try {
        const response = await fetch(request.url, {
          method: request.method,
          headers: request.headers,
          body: request.body
        });
        
        if (response.ok) {
          results.push({ success: true, request });
        } else {
          request.retryCount++;
          if (request.retryCount < request.maxRetries) {
            failedRequests.push(request);
          }
          results.push({
            success: false,
            request,
            error: `HTTP ${response.status}: ${response.statusText}`
          });
        }
      } catch (error) {
        request.retryCount++;
        if (request.retryCount < request.maxRetries) {
          failedRequests.push(request);
        }
        results.push({
          success: false,
          request,
          error: (error as Error).message
        });
      }
    }
    
    this.offlineQueue = failedRequests;
    await this.saveOfflineQueue();
    
    this.syncInProgress = false;
    this.emit('sync-complete', results);
    
    return results;
  }

  private handleServiceWorkerUpdate(): void {
    const newWorker = this.serviceWorkerRegistration?.installing;
    
    if (newWorker) {
      newWorker.addEventListener('statechange', () => {
        if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
          this.emit('update-available');
        }
      });
    }
  }

  private handleServiceWorkerMessage(event: MessageEvent): void {
    const { type, data } = event.data;
    
    switch (type) {
      case 'sync-complete':
        this.emit('sync-complete', data.results);
        break;
      
      case 'cache-updated':
        this.emit('cache-updated', data);
        break;
      
      default:
        console.log('Unknown service worker message:', type);
    }
  }

  public async clearCache(): Promise<void> {
    // Clear service worker caches
    if ('caches' in window) {
      const cacheNames = await caches.keys();
      await Promise.all(cacheNames.map(name => caches.delete(name)));
    }
    
    // Clear IndexedDB cache
    if (this.db) {
      const transaction = this.db.transaction(['cache'], 'readwrite');
      const store = transaction.objectStore('cache');
      await new Promise((resolve) => {
        const request = store.clear();
        request.onsuccess = () => resolve(undefined);
      });
    }
    
    this.emit('cache-cleared');
  }

  public async getCacheStats(): Promise<any> {
    const stats = {
      isOnline: this.isOnline,
      offlineQueueSize: this.offlineQueue.length,
      cacheSize: 0,
      cacheEntries: 0
    };
    
    // Get cache size from IndexedDB
    if (this.db) {
      const transaction = this.db.transaction(['cache'], 'readonly');
      const store = transaction.objectStore('cache');
      const countRequest = store.count();
      
      await new Promise((resolve) => {
        countRequest.onsuccess = () => {
          stats.cacheEntries = countRequest.result;
          resolve(undefined);
        };
      });
    }
    
    // Get service worker cache stats
    if (navigator.serviceWorker.controller) {
      const channel = new MessageChannel();
      
      const promise = new Promise((resolve) => {
        channel.port1.onmessage = (event) => {
          if (event.data.stats) {
            Object.assign(stats, event.data.stats);
          }
          resolve(undefined);
        };
      });
      
      navigator.serviceWorker.controller.postMessage(
        { type: 'get-cache-stats' },
        [channel.port2]
      );
      
      await promise;
    }
    
    return stats;
  }

  public skipWaiting(): void {
    if (navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({ type: 'skip-waiting' });
    }
  }

  public getOnlineStatus(): boolean {
    return this.isOnline;
  }

  public getOfflineQueueSize(): number {
    return this.offlineQueue.length;
  }
}

export default OfflineManager.getInstance();