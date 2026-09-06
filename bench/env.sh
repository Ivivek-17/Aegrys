# All model caches on D: — C: is full (0 bytes free).
export HF_HOME=D:/Aegrys/.cache/hf
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export OLLAMA_MODELS=D:/Aegrys/.cache/ollama
export PY=D:/Aegrys/.venv/Scripts/python.exe

# S2 needs a D:-backed ollama server (the default one stores models on C:, which is full):
#   OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS=D:/Aegrys/.cache/ollama ollama serve &
