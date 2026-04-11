cd Release/
./

After a while, you will see the log:
  Connect:
    Local Base URL: http://127.0.0.1:8080/v1
    LAN Base URL:   http://<server-ip>:8080/v1
    API Key:        any non-empty string
    Model:          default

[ov_serve] Endpoints:
  POST /v1/chat/completions  (OpenAI)
  POST /v1/completions       (OpenAI)
  GET  /v1/models            (OpenAI)
  POST /api/chat             (Ollama)
  POST /api/generate         (Ollama)
  GET  /api/tags             (Ollama)
  POST /api/show             (Ollama)
  GET  /health

