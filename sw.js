// Service worker: keeps every file of the app on the device so it works offline.
// Bump VERSION whenever content changes, so devices download the new files.
const VERSION = "slas-v1";
const FILES = [
 "./",
 "index.html",
 "bank.md",
 "quiz.js",
 "marked.min.js",
 "manifest.webmanifest",
 "icon-192.png",
 "icon-512.png",
 "apple-touch-icon.png",
 "exams/2022-11-07-1.jpg",
 "exams/2022-11-07-2.jpg",
 "exams/2022-11-07-3.jpg",
 "exams/2023-02-06-1.jpg",
 "exams/2023-02-06-2.jpg",
 "exams/2023-02-06-3.jpg",
 "exams/2023-02-06-4.jpg",
 "exams/2023-07-03-1.jpg",
 "exams/2023-07-03-2.jpg",
 "exams/2023-07-03-3.jpg",
 "exams/2023-07-03-4.jpg",
 "exams/2023-09-07-1.jpg",
 "exams/2023-09-07-2.jpg",
 "exams/2023-09-07-3.jpg",
 "exams/2023-09-07-4.jpg",
 "exams/2024-11-12-1.jpg",
 "exams/2024-11-12-2.jpg",
 "exams/2024-11-12-3.jpg",
 "exams/2024-11-12-4.jpg",
 "exams/2024-11-12-5.jpg",
 "exams/2024-11-12-6.jpg",
 "exams/2024-11-12-7.jpg",
 "exams/2024-11-12-8.jpg",
 "exams/2024-11-12-9.jpg",
 "exams/2025-01-09-1.jpg",
 "exams/2025-01-09-2.jpg",
 "exams/2025-01-09-3.jpg",
 "exams/2025-01-09-4.jpg",
 "exams/2025-01-09-5.jpg",
 "exams/2025-01-09-6.jpg",
 "exams/2025-01-09-7.jpg",
 "exams/2025-01-09-8.jpg",
 "exams/2025-02-13-1.jpg",
 "exams/2025-02-13-2.jpg",
 "exams/2025-02-13-3.jpg",
 "exams/2025-02-13-4.jpg",
 "exams/2025-02-13-5.jpg",
 "exams/2025-02-13-6.jpg",
 "exams/2025-02-13-7.jpg",
 "exams/2025-02-13-8.jpg",
 "exams/2026-01-21-1.jpg",
 "exams/2026-01-21-2.jpg",
 "exams/2026-01-21-3.jpg",
 "exams/2026-01-21-4.jpg",
 "exams/2026-01-21-5.jpg",
 "exams/2026-01-21-6.jpg",
 "exams/2026-01-21-7.jpg",
 "exams/2026-01-21-8.jpg",
 "exams/2026-02-11-1.jpg",
 "exams/2026-02-11-2.jpg",
 "exams/2026-02-11-3.jpg",
 "exams/2026-02-11-4.jpg",
 "exams/2026-02-11-5.jpg"
];

self.addEventListener("install", event => {
  event.waitUntil(caches.open(VERSION).then(cache => cache.addAll(FILES)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== VERSION).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

// Text files (page, bank, quiz): try the network first so updates arrive, fall back to the saved copy offline.
// Everything else (images, fonts): saved copy first, network only if missing.
self.addEventListener("fetch", event => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  const isText = url.origin === location.origin && /(\/|\.html|\.md|\.js|\.webmanifest)$/.test(url.pathname);
  if (isText) {
    event.respondWith(
      fetch(req)
        .then(res => {
          if (res.ok) { const copy = res.clone(); caches.open(VERSION).then(c => c.put(req, copy)); }
          return res;
        })
        .catch(() => caches.match(req, { ignoreSearch: true }))
    );
    return;
  }
  event.respondWith(
    caches.match(req).then(hit => hit || fetch(req).then(res => {
      // Also keep Google Fonts and new images for offline use.
      if (res.ok || res.type === "opaque") { const copy = res.clone(); caches.open(VERSION).then(c => c.put(req, copy)); }
      return res;
    }))
  );
});
