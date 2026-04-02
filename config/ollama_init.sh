#!/bin/bash
# MiniStack Bedrock — Pull default Ollama models
# Run this after starting the ollama container:
#   docker exec ministack-ollama /bin/bash /app/config/ollama_init.sh
#
# Or wait for ollama to be healthy and run:
#   docker exec ministack-ollama ollama pull qwen2.5:3b
#   docker exec ministack-ollama ollama pull nomic-embed-text

set -e

echo "Pulling default models for MiniStack Bedrock..."

# Default inference model (~2GB)
ollama pull qwen2.5:3b

# Embedding model for Knowledge Base (~274MB)
ollama pull nomic-embed-text

echo "All models pulled successfully."
