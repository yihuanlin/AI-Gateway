# AI Gateway

A high-performance AI Gateway built with [Hono](https://hono.dev/) for edge computing environments. This gateway provides unified access to multiple AI providers with intelligent fallback, streaming support, and advanced tools integration.

## 🛠 Quick Start

### Development
```bash
# Install dependencies
bun install

# Start development server
bun dev

# Build for production
bun build
```

## 📡 API Endpoints

### Chat Completions
```
POST /v1/chat/completions
```

### Responses
```
POST /v1/responses
```

### Response Management
```
GET /v1/responses/:response_id     # Get a specific response
GET /v1/responses                  # List all responses
DELETE /v1/responses/:response_id  # Delete a specific response
DELETE /v1/responses/all           # Delete all responses
```

### Models List
```
GET /v1/models
POST /v1/models
```

## 💡 API Usage Examples

### Chat Completions
```bash
# Basic chat completion with password authentication
curl -X POST "$HOSTNAME/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $PASSWORD" \
  -d '{
    "model": "cerebras/gpt-oss-120b",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 1.0,
    "max_tokens": 1000,
    "stream": false
  }'
```

```bash
# Chat with search (enabled by adding tools, even an empty array) and custom provider key
curl -X POST "$HOSTNAME/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer gsk_...,gsk_..." \
  -H "x-tavily-api-key: tvly-dev-..." \
  -d '{
    "model": "groq/moonshotai/kimi-k2-instruct",
    "messages": [
      {"role": "user", "content": "Search for information about CRISPR gene editing"}
    ],
    "tools": []
  }'
```

### Responses API
```bash
curl -X POST "$HOSTNAME/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $PASSWORD" \
  -d '{
    "model": "gemini/gemini-2.5-flash",
    "input": [
      {"role": "user", "content": [{"type": "input_text", "text": "What are the latest developments in AI?"}]}
    ],
    "tools": []
  }'
```

```bash
# Create response with conversation continuation
curl -X POST "$HOSTNAME/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $PASSWORD" \
  -d '{
    "model": "gemini/gemini-2.5-flash",
    "input": "Tell me more about that topic",
    "previous_response_id": "resp_abc123456789",
  }'
```

### Get Models List
```bash
# Get models with specific provider keys
curl -X GET "$HOSTNAME/v1/models" \
  -H "Authorization: Bearer your-vercel-ai-gateway-key" \
  -H "x-chatgpt-api-key: sk-proj-..." \
  -H "x-groq-api-key: gsk_..." \
  -H "x-gemini-api-key: AIzaSy..."
```

```bash
# POST method and password auth also works for models
curl -X POST "$HOSTNAME/v1/models" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $PASSWORD"
```

## 📝 Response Management Examples

### Get a Specific Response
```bash
# Get response by ID
curl -X GET "$HOSTNAME/v1/responses/resp_abc123" \
  -H "Authorization: Bearer $PASSWORD"
```

```bash
# Get response with streaming
curl -X GET "$HOSTNAME/v1/responses/resp_abc123?stream=true" \
  -H "Authorization: Bearer $PASSWORD"
```

### List All Responses
```bash
# List all stored responses
curl -X GET "$HOSTNAME/v1/responses" \
  -H "Authorization: Bearer $PASSWORD"
```

```bash
# List with filters
curl -X GET "$HOSTNAME/v1/responses?prefix=resp_&limit=10" \
  -H "Authorization: Bearer $PASSWORD"
```

```bash
# List with directory structure
curl -X GET "$HOSTNAME/v1/responses?directories=true" \
  -H "Authorization: Bearer $PASSWORD"
```

### Delete a Specific Response
```bash
# Delete response by ID
curl -X DELETE "$HOSTNAME/v1/responses/resp_abc123" \
  -H "Authorization: Bearer $PASSWORD"
```

### Delete All Responses
```bash
# Delete all stored responses
curl -X DELETE "$HOSTNAME/v1/responses/all" \
  -H "Authorization: Bearer $PASSWORD"
```

## 🔧 Environment Variables

```bash
# Required
PASSWORD=your-gateway-password
GATEWAY_API_KEY=your-vercel-ai-gateway-key

# Optional AI Services
TAVILY_API_KEY=tvly-dev-...
PYTHON_API_KEY=your-python-key
PYTHON_URL=https://your-python-executor.com

# Provider-specific keys
CHATGPT_API_KEY=sk-proj-...,sk-proj-...,sk-proj-...
GROQ_API_KEY=gsk_...
CEREBRAS_API_KEY=csk-...
GEMINI_API_KEY=AIzaSy...
CHATGPT_API_KEY=sk-proj-...
DOUBAO_API_KEY=your-volcengine-key
MODELSCOPE_API_KEY=ms-...
GITHUB_API_KEY=github_pat_...
OPENROUTER_API_KEY=sk-or-v1-...
NVIDIA_API_KEY=nvapi-...
MISTRAL_API_KEY=your-mistral-key
COHERE_API_KEY=your-cohere-api-key
MORPH_API_KEY=sk-...
INFINI_API_KEY=sk-...
POIXE_API_KEY=sk-...
NETLIFY_SITE_ID=your-netlify-site-id
NETLIFY_TOKEN=nfp_...
COPILOT_API_KEY=ghu_...
```

## 🔌 Supported Providers

- **Gateway**: Vercel AI Gateway
- **Direct Providers**: Google, OpenAI, Groq, Cerebras, etc.
- **Tools**: Python execution, web search, content extraction

## 📊 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client App    │───▶│   Hono Gateway   │───▶│  AI Providers   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Tools Layer    │
                       │ • Python Exec    │
                       │ • Web Search     │
                       │ • Content Read   │
                       │ • Ensembl API    │
                       │ • Scholar API    │
                       └──────────────────┘
```
