# AI Gateway

A high-performance AI Gateway built with [Hono](https://hono.dev/) for edge computing environments. This gateway provides unified access to multiple AI providers with intelligent fallback, streaming support, advanced tools integration, and multimedia generation capabilities.

## ✨ Features

- **Unified Text API**: Images and videos models accessible through standard chat/responses endpoints  
- **Admin Models**: Special administrative models for system management (admin/magic)
- **Multi-Provider Support**: OpenAI, Google, Groq, Cerebras, Doubao, ModelScope, and more
- **Streaming Support**: Real-time responses with progress indicators
- **Tool Integration**: Python execution, web search, content extraction
- **Response Storage**: Persistent conversation management with Netlify Blobs

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

### Responses (Stored in Netlify Blobs)
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

#### Text Generation
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

#### Image Generation
```bash
# Image editing using Doubao (ByteDance) models
curl -X POST "$HOSTNAME/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $PASSWORD" \
  -d '{
    "model": "image/doubao-vision",
    "messages": [
      {"role": "user", "content": "A beautiful sunset over mountains --size 1280x720 --guidance 7.5"}
    ],
    "stream": true
  }'

# Generate images using ModelScope models
curl -X POST "$HOSTNAME/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $PASSWORD" \
  -d '{
    "model": "image/Qwen/Qwen-Image", # image/ + ModelScope model ID
    "messages": [
      {"role": "user", "content": "Cyberpunk cityscape at night --steps 30 --guidance 3.5"}
    ]
  }'
```

#### Video Generation
```bash
# Generate videos using Doubao Seedance models
curl -X POST "$HOSTNAME/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $PASSWORD" \
  -d '{
    "model": "video/doubao-seedance",
    "messages": [
      {"role": "user", "content": "A cat playing in a garden --ratio 16:9 --duration 5"}
    ],
    "stream": true
  }'
```

#### Advanced Search
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

#### Text Responses
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

#### Image Generation with Responses
```bash
# Generate image with reasoning steps shown
curl -X POST "$HOSTNAME/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $PASSWORD" \
  -d '{
    "model": "image/doubao-vision",
    "input": [
      {"role": "user", "content": [
        {"type": "input_text", "text": "Create a logo for a tech startup"},
        {"type": "input_image", "image_url": "data:image/jpeg;base64,..."}
      ]}
    ],
    "stream": true
  }'
```

#### Administrative Tasks
```bash
# Use admin model for system management
curl -X POST "$HOSTNAME/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $PASSWORD" \
  -d '{
    "model": "admin/magic",
    "input": "deleteall" # Delete all stored responses
  }'
```

#### Conversation Continuation
```bash
# Create response with conversation continuation
curl -X POST "$HOSTNAME/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $PASSWORD" \
  -d '{
    "model": "gemini/gemini-2.5-flash",
    "input": "Tell me more about that topic",
    "previous_response_id": "resp_abc123456789"
  }'
```

### Get Models List
```bash
# Get all available models including text, image, video, and admin models
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

## 🎨 Model Categories

### Text Models
- **LLM Providers**: `provider`/`model` for custom providers, `model` for Vercel AI Gateway, e.g.:
- *For Gemini native image generation:* `openrouter/google/gemini-2.5-flash-image-preview` (OpenRouter), `google/gemini-2.5-flash-image-preview` (Vercel AI Gateway), `gemini/gemini-2.5-flash-image-preview` (Google Generative AI)

### Image Models  
- **Doubao (ByteDance)**: `image/doubao` - i2i and t2i.
- **Hugging Face**: `image/Qwen/Qwen-Image-Edit-vision`, `image/black-forest-labs/FLUX.1-Kontext-dev-vision` etc. (Add a `-vision` suffix to any Hugging Face Inference model, i2i only)
- **ModelScope**: `image/black-forest-labs/FLUX.1-dev`, `image/Qwen/Qwen-Image` etc. (Any ModelScope model, t2i only)
- **Flags**: `--size WxH`, `--ratio A:B`, `--guidance N`, `--steps N`, `--seed N` etc. (Send `/help` for help)

### Video Models
- **Doubao Seedance**: `video/doubao-seedance`, `video/doubao-seedance-pro` (t2v and i2v)
- **Hugging Face**: `video/Wan-AI/Qwen-Wan2.2-I2V-A14B-vision` etc. (Any Hugging Face Inference model, t2v and i2v)
- **Flags**: `--ratio 16:9`, `--duration 3-12`, `--resolution 720p` etc. (Send `/help` for help)

### Admin Models
- **System Management**: `admin/magic` (Send `/help` for help)

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

# Optional S3 bucket to upload images for Doubao i2i and i2v inputs (triggered by /upload in prompt) and Hugging Face i2i and i2v outputs (always enabled if set)
S3_ACCESS_KEY=your-s3-access-key
S3_SECRET_KEY=your-s3-secret-key
S3_PUBLIC_URL=https://your-s3-public-url.com
S3_API=https://your-s3-provider-api.com/your-bucket-name

# Use Netlify Blobs in non-Netlify platforms (Responses endpoint datastore)
NETLIFY_SITE_ID=your-netlify-site-id
NETLIFY_TOKEN=nfp_...

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
COPILOT_API_KEY=ghu_...
POE_API_KEY=your-poe-api-key
HUGGINGFACE_API_KEY=hf_...
```

## 🔌 Supported Providers

### Text Generation
- **Gateway**: Vercel AI Gateway
- **Direct Providers**: Vercel AI Gateway (Gateway), OpenAI (ChatGPT), Google Generative AI (Gemini), Groq, Cerebras, OpenRouter, Poe, Volcengine (Doubao), ModelScope, Infini, Nvidia, Mistral, Poixe, Cohere, Morph, GitHub Models (GitHub), GitHub Copilot (Copilot), etc.

### Multimedia Generation  
- **Doubao (ByteDance)**: i2i, t2i, i2v, and t2v
- **ModelScope**: Community models for t2i
- **Hugging Face**: Community models for i2i, t2v, and i2v

### Tools & Extensions
- **Python Execution**: Code interpreter
- **Web Search**: Tavily API integration  
- **Content Extraction**: Web page reading
- **Research APIs**: Ensembl, Scholar APIs

## 📊 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client App    │───▶│   Hono Gateway   │───▶│  AI Providers   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Modular System  │
                       │ • Text Models    │
                       │ • Image Models   │
                       │ • Video Models   │ 
                       │ • Admin Models   │
                       │ • Tools Layer    │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Data Storage   │
                       │ • Netlify Blobs  │
                       │ • Response Mgmt  │
                       │ • Conversation   │
                       └──────────────────┘
```
