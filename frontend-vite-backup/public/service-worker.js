/**
 * Service Worker for Offline Support
 * TEKNOFEST 2025 - Production Ready
 */

const CACHE_NAME = 'teknofest-v1.0.0';
const API_CACHE_NAME = 'teknofest-api-v1.0.0';
const IMAGE_CACHE_NAME = 'teknofest-images-v1.0.0';

// Cache strategies
const CACHE_STRATEGIES = {
  CACHE_FIRST: 'cache-first',
  NETWORK_FIRST: 'network-first',
  CACHE_ONLY: 'cache-only',
  NETWORK_ONLY: 'network-only',
  STALE_WHILE_REVALIDATE: 'stale-while-revalidate'
};

// Static assets to cache
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/offline.html',
  '/assets/css/main.css',
  '/assets/js/app.js'
];

// API endpoints to cache
const CACHEABLE_API_ENDPOINTS = [
  '/api/v1/curriculum',
  '/api/v1/learning-style',
  '/api/v1/data/stats'
];

// Offline queue for failed requests
let offlineQueue = [];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('[Service Worker] Installing...');
  
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[Service Worker] Caching static assets');
      return cache.addAll(STATIC_ASSETS);
    }).then(() => {
      console.log('[Service Worker] Installation complete');
      return self.skipWaiting();
    })
  );
});

// Activate event - clean old caches
self.addEventListener('activate', (event) => {
  console.log('[Service Worker] Activating...');
  
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME && 
              cacheName !== API_CACHE_NAME && 
              cacheName !== IMAGE_CACHE_NAME) {
            console.log('[Service Worker] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      console.log('[Service Worker] Activation complete');
      return self.clients.claim();
    })
  );
});

// Fetch event - handle requests with caching strategies
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Determine cache strategy based on request type
  let strategy = CACHE_STRATEGIES.NETWORK_FIRST;
  
  if (request.method !== 'GET') {
    // Handle non-GET requests (POST, PUT, DELETE)
    event.respondWith(handleNonGetRequest(request));
    return;
  }
  
  // Static assets - cache first
  if (isStaticAsset(url.pathname)) {
    strategy = CACHE_STRATEGIES.CACHE_FIRST;
  }
  
  // API requests - network first with cache fallback
  if (url.pathname.startsWith('/api/')) {
    strategy = CACHE_STRATEGIES.NETWORK_FIRST;
  }
  
  // Images - stale while revalidate
  if (isImageAsset(url.pathname)) {
    strategy = CACHE_STRATEGIES.STALE_WHILE_REVALIDATE;
  }
  
  event.respondWith(handleRequest(request, strategy));
});

// Handle requests based on cache strategy
async function handleRequest(request, strategy) {
  switch (strategy) {
    case CACHE_STRATEGIES.CACHE_FIRST:
      return cacheFirst(request);
    
    case CACHE_STRATEGIES.NETWORK_FIRST:
      return networkFirst(request);
    
    case CACHE_STRATEGIES.CACHE_ONLY:
      return cacheOnly(request);
    
    case CACHE_STRATEGIES.NETWORK_ONLY:
      return networkOnly(request);
    
    case CACHE_STRATEGIES.STALE_WHILE_REVALIDATE:
      return staleWhileRevalidate(request);
    
    default:
      return networkFirst(request);
  }
}

// Cache-first strategy
async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) {
    return cached;
  }
  
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    console.error('[Service Worker] Fetch failed:', error);
    return caches.match('/offline.html');
  }
}

// Network-first strategy
async function networkFirst(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cacheName = request.url.includes('/api/') ? API_CACHE_NAME : CACHE_NAME;
      const cache = await caches.open(cacheName);
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    console.log('[Service Worker] Network request failed, falling back to cache');
    const cached = await caches.match(request);
    if (cached) {
      return cached;
    }
    
    // Return offline page for navigation requests
    if (request.mode === 'navigate') {
      return caches.match('/offline.html');
    }
    
    // Return error response for API requests
    return new Response(JSON.stringify({
      error: 'Offline',
      message: 'No cached data available'
    }), {
      status: 503,
      statusText: 'Service Unavailable',
      headers: new Headers({
        'Content-Type': 'application/json'
      })
    });
  }
}

// Cache-only strategy
async function cacheOnly(request) {
  const cached = await caches.match(request);
  if (cached) {
    return cached;
  }
  
  return new Response('Cache miss', {
    status: 404,
    statusText: 'Not Found'
  });
}

// Network-only strategy
async function networkOnly(request) {
  try {
    return await fetch(request);
  } catch (error) {
    console.error('[Service Worker] Network request failed:', error);
    return new Response('Network error', {
      status: 503,
      statusText: 'Service Unavailable'
    });
  }
}

// Stale-while-revalidate strategy
async function staleWhileRevalidate(request) {
  const cached = await caches.match(request);
  
  const fetchPromise = fetch(request).then((response) => {
    if (response.ok) {
      const cache = caches.open(IMAGE_CACHE_NAME);
      cache.then(c => c.put(request, response.clone()));
    }
    return response;
  }).catch((error) => {
    console.error('[Service Worker] Background fetch failed:', error);
  });
  
  return cached || fetchPromise;
}

// Handle non-GET requests (POST, PUT, DELETE)
async function handleNonGetRequest(request) {
  try {
    const response = await fetch(request);
    return response;
  } catch (error) {
    console.log('[Service Worker] Non-GET request failed, queuing for later');
    
    // Clone request for queuing
    const body = await request.text();
    const queuedRequest = {
      url: request.url,
      method: request.method,
      headers: Object.fromEntries(request.headers.entries()),
      body: body,
      timestamp: Date.now()
    };
    
    // Add to offline queue
    await queueOfflineRequest(queuedRequest);
    
    // Return response indicating request was queued
    return new Response(JSON.stringify({
      queued: true,
      message: 'Request queued for synchronization'
    }), {
      status: 202,
      statusText: 'Accepted',
      headers: new Headers({
        'Content-Type': 'application/json'
      })
    });
  }
}

// Queue offline requests
async function queueOfflineRequest(request) {
  offlineQueue.push(request);
  
  // Store in IndexedDB for persistence
  const db = await openDatabase();
  const tx = db.transaction('offline_queue', 'readwrite');
  const store = tx.objectStore('offline_queue');
  await store.add(request);
  
  // Register sync event
  if ('sync' in self.registration) {
    await self.registration.sync.register('sync-offline-requests');
  }
}

// Sync offline requests when back online
self.addEventListener('sync', async (event) => {
  if (event.tag === 'sync-offline-requests') {
    console.log('[Service Worker] Syncing offline requests');
    event.waitUntil(syncOfflineRequests());
  }
});

// Sync offline requests
async function syncOfflineRequests() {
  const db = await openDatabase();
  const tx = db.transaction('offline_queue', 'readonly');
  const store = tx.objectStore('offline_queue');
  const requests = await store.getAll();
  
  const results = [];
  
  for (const request of requests) {
    try {
      const response = await fetch(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body
      });
      
      if (response.ok) {
        // Remove from queue
        const deleteTx = db.transaction('offline_queue', 'readwrite');
        const deleteStore = deleteTx.objectStore('offline_queue');
        await deleteStore.delete(request.id);
        
        results.push({ success: true, request });
      } else {
        results.push({ success: false, request, error: response.statusText });
      }
    } catch (error) {
      results.push({ success: false, request, error: error.message });
    }
  }
  
  // Notify clients about sync results
  const clients = await self.clients.matchAll();
  clients.forEach(client => {
    client.postMessage({
      type: 'sync-complete',
      results: results
    });
  });
  
  return results;
}

// Open IndexedDB
async function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('teknofest_offline', 1);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      
      if (!db.objectStoreNames.contains('offline_queue')) {
        const store = db.createObjectStore('offline_queue', {
          keyPath: 'id',
          autoIncrement: true
        });
        store.createIndex('timestamp', 'timestamp', { unique: false });
      }
    };
  });
}

// Check if URL is a static asset
function isStaticAsset(pathname) {
  const staticExtensions = ['.html', '.css', '.js', '.json', '.ico'];
  return staticExtensions.some(ext => pathname.endsWith(ext));
}

// Check if URL is an image asset
function isImageAsset(pathname) {
  const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp'];
  return imageExtensions.some(ext => pathname.endsWith(ext));
}

// Message handler for client communication
self.addEventListener('message', (event) => {
  const { type, data } = event.data;
  
  switch (type) {
    case 'skip-waiting':
      self.skipWaiting();
      break;
    
    case 'clear-cache':
      clearAllCaches().then(() => {
        event.ports[0].postMessage({ success: true });
      });
      break;
    
    case 'get-cache-stats':
      getCacheStats().then((stats) => {
        event.ports[0].postMessage({ stats });
      });
      break;
  }
});

// Clear all caches
async function clearAllCaches() {
  const cacheNames = await caches.keys();
  await Promise.all(cacheNames.map(name => caches.delete(name)));
  console.log('[Service Worker] All caches cleared');
}

// Get cache statistics
async function getCacheStats() {
  const stats = {
    caches: {},
    totalSize: 0
  };
  
  const cacheNames = await caches.keys();
  
  for (const name of cacheNames) {
    const cache = await caches.open(name);
    const requests = await cache.keys();
    
    stats.caches[name] = {
      count: requests.length,
      urls: requests.map(r => r.url)
    };
  }
  
  return stats;
}