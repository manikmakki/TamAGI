// TamAGI Service Worker — PWA offline support
const CACHE_NAME = 'tamagi-platform-v11';
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/assets/avatar-vex.js',
  '/assets/avatar-echo.js',
  '/assets/avatar-ember.js?v=8',
  '/assets/avatar-runtime.js?v=8',
  '/assets/app-shell.css?v=18',
  '/assets/app-shell.js?v=18',
  '/assets/home-views.js?v=19',
  '/assets/marked.min.js',
  '/assets/d3.v7.min.js',
  '/assets/panel-bridge.js',
  '/assets/panel-bridge.css',
];

// Install: cache static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

// Activate: clean old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

// Fetch: network-first for API, cache-first for static
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // API calls: always network
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/ws')) {
    event.respondWith(fetch(event.request));
    return;
  }

  if (['/home','/settings','/login'].includes(url.pathname)) { event.respondWith(fetch(event.request)); return; }

  // Refresh the application shell after a release, with an offline fallback.
  if (event.request.mode === 'navigate' || ['/', '/index.html'].includes(url.pathname)) {
    event.respondWith(fetch(event.request).then((response) => {
      if (response.ok && !response.redirected) {
        const copy = response.clone();
        event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copy)));
      }
      return response;
    }).catch(() => caches.match(event.request)));
    return;
  }

  // Static assets: cache-first, fallback to network
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).then((response) => {
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        }
        return response;
      });
    })
  );
});
