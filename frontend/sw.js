// TamAGI Service Worker — PWA offline support
const CACHE_NAME = 'tamagi-v1';
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
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

// ── Push Notifications ────────────────────────────────────────────────────────

self.addEventListener('push', (event) => {
  // Parse payload sent by the backend (JSON: title, body, url, tag)
  let data = { title: 'TamAGI', body: 'New response', url: '/', tag: 'tamagi-response' };
  if (event.data) {
    try { data = { ...data, ...event.data.json() }; } catch (_) { /* use defaults */ }
  }

  event.waitUntil(
    // Check if the app window is already visible and focused.
    // If so, skip the system notification — the user can already see the response.
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clients) => {
      const isVisible = clients.some((c) => c.visibilityState === 'visible' && c.focused);
      if (isVisible) return;

      return self.registration.showNotification(data.title, {
        body:     data.body,
        icon:     '/assets/icon-192.png',
        badge:    '/assets/icon-192.png',   // small monochrome icon (Chrome Android)
        tag:      data.tag,                 // deduplicates — replaces previous same-tag notification
        renotify: false,                    // silent replacement (no re-alert sound)
        data:     { url: data.url },        // passed to notificationclick handler
        vibrate:  [100, 50, 100],           // gentle vibration pattern (mobile)
      });
    })
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const targetUrl = event.notification.data?.url || '/';

  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clients) => {
      // Focus an existing window if one is already open
      for (const client of clients) {
        if (client.url.includes(self.location.origin) && 'focus' in client) {
          client.navigate(targetUrl);
          return client.focus();
        }
      }
      // Otherwise open a new window
      if (self.clients.openWindow) {
        return self.clients.openWindow(targetUrl);
      }
    })
  );
});
