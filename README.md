# Vercel AI Gateway - Hono Version

A high-performance AI Gateway built with [Hono](https://hono.dev/) for edge computing environments. This gateway provides unified access to multiple AI providers with intelligent fallback, streaming support, and advanced tools integration.

## 🛠 Quick Start

### Development
```bash
# Install dependencies
pnpm install

# Start development server
pnpm dev

# Build for production
pnpm build
```

## 📡 API Endpoints

### Chat Completions
```
POST /v1/chat/completions
```

### Models List
```
GET /v1/models
POST /v1/models
```

## 🔧 Environment Variables

```bash
# Required
PASSWORD=your-gateway-password
GATEWAY_API_KEY=your-gateway-api-key

# Optional AI Services
TAVILY_API_KEY=your-tavily-key
PYTHON_API_KEY=your-python-key
PYTHON_URL=your-python-execution-url

# Provider-specific keys
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GROQ_API_KEY=your-groq-key
# ... etc for other providers
```

## 🔌 Supported Providers

- **Gateway**: Vercel AI Gateway
- **Direct Providers**: Google, Groq, Cerebras, etc.
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
                       └──────────────────┘
```
