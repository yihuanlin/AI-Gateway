# AI Gateway

A high-performance AI Gateway built with [Hono](https://github.com/honojs/hono) and [AI SDK](https://github.com/vercel/ai) for edge computing environments. This gateway provides unified access to multiple AI providers with intelligent fallback, streaming support, advanced tools integration, and multimedia generation capabilities.

## ✨ Features

- **Unified Text API**: Images and videos models accessible through standard OpenAI chat/responses and Anthropic messages endpoints  
- **Admin Models**: Special administrative models for system management (admin/magic-vision)
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

### Chat Completions (OpenAI)
```
POST /v1/chat/completions
```

### Messages (Anthropic)
```
POST /v1/messages
```

### Responses (OpenAI)
```
POST /v1/responses
```

### Models List (OpenAI)
```
GET /v1/models
```

### Response Management (Stored in Netlify Blobs; OpenAI)
```
GET /v1/responses/:response_id     # Get a specific response
GET /v1/responses                  # List all responses
DELETE /v1/responses/:response_id  # Delete a specific response
DELETE /v1/responses/all           # Delete all responses
POST /v1/chat/completions (model: admin/magic-vision)
POST /v1/responses (model: admin/magic-vision)
```

### Files (Stored in Netlify Blobs)
```
GET /v1/files/:file                # Serve a file from Netlify Blobs
```

## 🔌 Supported Providers

### Text Generation
- **Gateway**: Vercel AI Gateway
- **Direct Providers**: Vercel AI Gateway (Gateway), OpenAI (ChatGPT), Google Generative AI (Gemini), Groq, Cerebras, OpenRouter, Poe, Volcengine (Doubao), ModelScope, Infini, Nvidia, Mistral, Poixe, Cohere, Morph, GitHub Models (GitHub), GitHub Copilot (Copilot), Cloudflare Gateway (Cloudflare), Meituan (LongCat), and any custom OpenAI chat/completions compatible providers.

### Multimedia Generation  
- **Gemini Image (Nano Banana)**: Gemini native image generation: t2i and i2i
- **GPT Image (`image_generation` tool)**: GPT-5 series native image generation: t2i and i2i
- **Black Forest Labs**: FLUX models via Vercel AI Gateway: t2i and i2i
- **Doubao (ByteDance)**: t2i/i2i (Seedream)and t2v/i2v (Seedance)
- **ModelScope**: Community models for t2i and i2i (i2i requires Netlify Blobs)
- **Hugging Face**: Community models for t2i, i2i, t2v, and i2v

### Tools & Extensions
**If required environment variables are set, the following tools are enabled by adding tools in request body (except for when Anthropic format client tools are provided), even an empty array (in [Cherry Studio](https://www.cherry-ai.com), this is triggered by enabling model build-in search):**
- **Code Execution**: [Python Executor API](https://github.com/yihuanlin/python-executor-api) `python_executor` or model build-in (Gateway and Custom Gemini `code_execution`, Gateway Anthropic `code_execution`, Gateway OpenAI `code_interpreter`)
- **Web Search**: [Tavily Search API](https://docs.tavily.com/documentation/api-reference/endpoint/search) `web_search` or model build-in (Gateway and Custom Gemini `google_search`, Gateway OpenAI `web_search_preview`, Gateway Anthropic `web_search`, Gateway Grok `mode = 'on'`, Gateway Perplexity `always on regardless of tools`)
- **Content Extraction**: [Jina Reader API](https://jina.ai/reader/) `fetch` or model build-in (Gateway and Custom `url_context`)

**In OpenAI endpoints, research mode is triggered by detecting keywards `research` and `paper` in conversation. Default search depth and reasoning effort will increase, all tools above (except `python_executor`) and research APIs below will be enabled:**
- **Research APIs**: [Ensembl API](https://rest.ensembl.org) `ensembl_api`, [Semantic Scholar APIs](https://www.semanticscholar.org/product/api) `scholar_search` and `paper_recommendations`

## 🔧 Environment Variables

```bash
# Required
PASSWORD=your-gateway-password
GATEWAY_API_KEY=your-vercel-ai-gateway-key

# Optional tools
TAVILY_API_KEY=tvly-dev-...
PYTHON_API_KEY=your-python-key
PYTHON_URL=https://your-python-executor.com

# Use Netlify Blobs in non-Netlify platforms
NETLIFY_SITE_ID=your-netlify-site-id
NETLIFY_TOKEN=nfp_...
URL=http://localhost:8888 # Optional site URL to upload files

# Optional provider-specific keys
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
LONGCAT_API_KEY=ak_...

# Cloudflare Gateway (GPT-OSS series currently not supported)
CLOUDFLARE_API_KEY=your-cloudflare-api-key
CLOUDFLARE_ACCOUNT_ID=your-cloudflare-account-id
CLOUDFLARE_GATEWAY=your-cloudflare-gateway-name

#Custom OpenAI chat/completions format providers
CUSTOM_API_ENDPOINTS={"internai":{"baseURL":"https://chat.intern-ai.org.cn/api/v1"},"lmstudio":{"baseURL":"http://localhost:1234/v1"}}
INTERNAI_API_KEY=each-custom-provider-must-have-at-least-a-key
LMSTUDIO_API_KEY=each-custom-provider-must-have-at-least-a-key
```

## 🎨 Model Categories

### Text Models
- **LLM Providers**: `provider`/`model` for custom providers, `model` for Vercel AI Gateway, e.g.:
- For **Gemini native image generation**: `google/gemini-3-pro-image` (Vercel AI Gateway), `gemini/gemini-2.5-flash-image` (Google Generative AI)
- For **ChatGPT native image generation**: add `-image` suffix to model ID, e.g. `openai/gpt-5.1-image` (Vercel AI Gateway)

### Image Models  
- **Black Forest Labs**: `image/bfl/flux-2-pro`, `image/bfl/flux-kontext-pro` etc. (`image/bfl/` + BFL model ID via Vercel AI Gateway)
- **Doubao (ByteDance)**: `image/doubao` - i2i and t2i.
- **Hugging Face**: `image/huggingface/black-forest-labs/FLUX.1-Kontext-dev` etc. (`image/huggingface/` + any Hugging Face internal model ID)
- **ModelScope**: `image/modelscope/Qwen/Qwen-Image` etc. (`image/modelscope/` + any ModelScope internal model ID; i2i requires Netlify Blobs)
- **Flags**: `--size WxH`, `--ratio A:B`, `--guidance N`, `--steps N`, `--seed N` etc. (Send `/help` for help)

### Video Models
- **Doubao Seedance**: `video/doubao-seedance`, `video/doubao-seedance-pro` (t2v and i2v)
- **Hugging Face**: `video/Wan-AI/Qwen-Wan2.2-I2V-A14B-vision` etc. (Any Hugging Face Inference model, t2v and i2v)
- **Flags**: `--ratio 16:9`, `--duration 3-12`, `--resolution 720p` etc. (Send `/help` for help)

### Admin Models
- **System Management**: `admin/magic-vision` (Send `/help` for help)

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

### Messages (Anthropic Format)

#### Text Generation with Anthropic Format
```bash
# Basic message completion with Anthropic format
curl -X POST "$HOSTNAME/v1/messages" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $PASSWORD" \
  -d '{
    "model": "anthropic/claude-sonnet-4",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 1000,
    "stream": false
  }'
```

#### Anthropic Format with Tools and Search
```bash
# Message with tools and web search capabilities
curl -X POST "$HOSTNAME/v1/messages" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $PASSWORD" \
  -d '{
    "model": "anthropic/claude-opus-4.5",
    "messages": [
      {"role": "user", "content": "Search for the latest developments in AI research"}
    ],
    "tools": [
      {
        "name": "paper_search",
        "description": "Search the web for papers",
        "input_schema": {
          "type": "object",
          "properties": {
            "query": {"type": "string", "description": "Search query"}
          },
          "required": ["query"]
        }
      }
    ],
    "max_tokens": 2000,
    "stream": true
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
    "model": "admin/magic-vision",
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
  -H "Authorization: Bearer $PASSWORD" \
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
