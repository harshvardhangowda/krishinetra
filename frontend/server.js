/**
 * KrishiNetra Frontend - Zero-Dependency Local Static Server
 * File: frontend/server.js
 *
 * Serves index.html and assets on http://localhost:5500
 * Compatible with backend CORS configuration.
 */

const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = parseInt(process.env.PORT || "5500", 10);
const HOST = "0.0.0.0";
const FRONTEND_DIR = __dirname;

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".txt": "text/plain; charset=utf-8",
};

const server = http.createServer((req, res) => {
  // Normalize request URL
  let parsedUrl = req.url.split("?")[0];
  if (parsedUrl === "/" || parsedUrl === "") {
    parsedUrl = "/index.html";
  }

  // Prevent directory traversal
  const safePath = path.normalize(decodeURIComponent(parsedUrl)).replace(/^(\.\.[\/\\])+/, "");
  let filePath = path.join(FRONTEND_DIR, safePath);

  // Check if file exists
  fs.stat(filePath, (err, stats) => {
    if (err || !stats.isFile()) {
      // Fallback to index.html for SPA-style routes if not found
      filePath = path.join(FRONTEND_DIR, "index.html");
    }

    const ext = path.extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || "application/octet-stream";

    fs.readFile(filePath, (readErr, content) => {
      if (readErr) {
        res.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
        res.end("500 Internal Server Error: Could not read file.");
        return;
      }

      res.writeHead(200, {
        "Content-Type": contentType,
        "Cache-Control": "no-cache",
        "X-Content-Type-Options": "nosniff",
      });
      res.end(content);
    });
  });
});

server.listen(PORT, HOST, () => {
  console.log(`=======================================================`);
  console.log(`  🌾 KRISHI NETRA Frontend Studio`);
  console.log(`  Live UI: http://localhost:${PORT}`);
  console.log(`  Target Backend API: http://localhost:8000/api/v1/predict`);
  console.log(`=======================================================`);
});
