import { createGateway } from '@ai-sdk/gateway';
import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'edge';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': '*',
  'Cache-Control': 'public, s-maxage=300, stale-while-revalidate=86400',
  'CDN-Cache-Control': 'public, s-maxage=300, stale-while-revalidate=86400',
  'Vercel-CDN-Cache-Control': 'public, s-maxage=300, stale-while-revalidate=86400',
};

// Supported custom providers configuration
const SUPPORTED_PROVIDERS = {
  cerebras: {
    name: 'cerebras',
    baseURL: 'https://api.cerebras.ai/v1',
  },
  groq: {
    name: 'groq',
    baseURL: 'https://api.groq.com/openai/v1',
  },
  gemini: {
    name: 'gemini',
    baseURL: 'https://generativelanguage.googleapis.com/v1beta/openai',
  },
  doubao: {
    name: 'doubao',
    baseURL: 'https://ark.cn-beijing.volces.com/api/v3',
  },
  modelscope: {
    name: 'modelscope',
    baseURL: 'https://api-inference.modelscope.cn/v1',
  },
  github: {
    name: 'github',
    baseURL: 'https://models.inference.ai.azure.com',
  },
  openrouter: {
    name: 'openrouter',
    baseURL: 'https://openrouter.ai/api/v1',
  },
  nvidia: {
    name: 'nvidia',
    baseURL: 'https://integrate.api.nvidia.com/v1',
  },
  mistral: {
    name: 'mistral',
    baseURL: 'https://api.mistral.ai/v1',
  },
  cohere: {
    name: 'cohere',
    baseURL: 'https://api.cohere.ai/compatibility/v1',
  },
  // morph: {
  //   name: 'morph',
  //   baseURL: 'https://api.morphllm.com/v1',
  // },
  infini: {
    name: 'infini',
    baseURL: 'https://cloud.infini-ai.com/maas/v1',
  },
  poixe: {
    name: 'poixe',
    baseURL: 'https://api.poixe.ai/v1',
  },
};

// Custom model lists for providers that don't support /models endpoint
const CUSTOM_MODEL_LISTS = {
  poixe: [
    { id: 'gpt-5:free', name: 'GPT-5' },
    { id: 'grok-3-mini:free', name: 'Grok 3 Mini' },
    { id: 'grok-4:free', name: 'Grok 4' },
  ],
  doubao: [
    { id: 'doubao-seed-1-6-flash-250715', name: 'Doubao Seed 1.6 Flash' },
    { id: 'doubao-seed-1-6-thinking-250715', name: 'Doubao Seed 1.6 Thinking' },
    { id: 'deepseek-r1-250528', name: 'DeepSeek R1' },
    { id: 'deepseek-v3-250324', name: 'DeepSeek V3' },
    { id: 'kimi-k2-250711', name: 'Kimi K2' },
  ],
  // morph: [
  //   { id: 'morph-v3-fast', name: 'Morph V3 Fast' },
  //   { id: 'morph-v3-large', name: 'Morph V3 Large' },
  //   { id: 'auto', name: 'Morph Automatic Selection' },
  // ],
  cohere: [
    { id: 'command-a-03-2025', name: 'Command A' },
    { id: 'command-a-vision-07-2025', name: 'Cohere A Vision' },
  ],
};

const parseModelName = (model: string) => {

  let baseName = model.split('/').pop() || model;

  if (baseName.endsWith(':free')) {
    baseName = baseName.slice(0, -5);
  }

  const parts = baseName.split('-');
  let displayName;

  displayName = baseName.replace(/-/g, ' ');
  displayName = displayName.split(' ').map(word => {
    const lowerWord = word.toLowerCase();
    if (lowerWord === 'deepseek') {
      return 'DeepSeek';
    } else if (lowerWord === 'ernie') {
      return 'ERNIE';
    } else if (lowerWord === 'mai' || lowerWord === 'ds' || lowerWord === 'r1') {
      return word.toUpperCase();
    } else if (lowerWord === 'gpt') {
      return 'GPT';
    } else if (lowerWord === 'oss') {
      return 'OSS';
    } else if (lowerWord === 'glm') {
      return 'GLM';
    } else if (lowerWord.startsWith('o') && lowerWord.length > 1 && /^\d/.test(lowerWord.slice(1))) {
      // Check if word starts with 'o' followed by a number (OpenAI o models)
      return word.toLowerCase();
    } else if (/^a?\d+[bkmae]$/.test(lowerWord)) {
      return word.toUpperCase();
    } else {
      // Capitalize first letter of each word
      return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
    }
  }).join(' ');

  // Handle special cases that need to keep hyphens
  if (displayName === 'MAI DS R1') {
    displayName = 'MAI-DS-R1';
  } else if (displayName.startsWith('GPT ')) {
    // Replace spaces after GPT with hyphens
    displayName = displayName.replace(/^GPT /, 'GPT-');
  }
  return displayName;
}

// Helper function to fetch models from custom provider
async function fetchProviderModels(providerName: string, apiKey: string) {
  const config = SUPPORTED_PROVIDERS[providerName as keyof typeof SUPPORTED_PROVIDERS];
  if (!config) {
    throw new Error(`Unsupported provider: ${providerName}`);
  }

  const modelsEndpoint = `${config.baseURL}/models`;
  const response = await fetch(modelsEndpoint, {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    next: {
      revalidate: 3600,
    },
  });

  if (!response.ok) {
    throw new Error(`Provider ${providerName} models API failed: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();
  return data;
}

// Helper function to get provider API keys from request headers
function getProviderKeysFromHeaders(req: NextRequest, isPasswordAuth: boolean = false) {
  const providerKeys: Record<string, string[]> = {};

  for (const provider of Object.keys(SUPPORTED_PROVIDERS)) {
    const keyName = `${provider}_api_key`;
    const headerValue = req.headers.get(keyName);
    if (headerValue) {
      const keys = headerValue.split(',').map((k: string) => k.trim());
      providerKeys[provider] = keys;
    }
  }

  // If password auth is enabled, also check for environment variables for all providers
  if (isPasswordAuth) {
    for (const provider of Object.keys(SUPPORTED_PROVIDERS)) {
      const envKeyName = `${provider.toUpperCase()}_API_KEY`;
      const envValue = process.env[envKeyName];
      if (envValue) {
        const keys = envValue.split(',').map(k => k.trim());
        // If provider already has header keys, merge them; otherwise add env keys
        if (providerKeys[provider]) {
          providerKeys[provider].push(...keys);
        } else {
          providerKeys[provider] = keys;
        }
      }
    }
  }

  return providerKeys;
}

// Helper function to filter out unwanted models
function shouldIncludeModel(model: any, providerName?: string) {
  const modelId = model.id.toLowerCase();

  // Common exclusions for all providers
  const commonExclusions = ['gemma', 'rerank', 'distill', 'parse', 'embed', 'bge-', 'tts', 'phi', 'live', 'audio', 'lite', 'qwen2', 'qwen-2', 'qwen1', 'qwq', 'qvq', 'gemini-2.0', 'gemini-1', 'learnlm', 'gemini-exp', 'turbo', 'claude-3', 'voxtral', 'pixtral', 'mixtral', 'ministral', '-24', 'moderation', 'saba', '-ocr-'];
  if (commonExclusions.some(exclusion => modelId.includes(exclusion))) {
    return false;
  }
  if (!modelId.includes('super') && (['nemotron', 'llama'].some(exclusion => modelId.includes(exclusion))) || modelId.includes('nvidia')) {
    return false;
  }

  // Provider-specific exclusions
  if (providerName === 'gemini' && ['veo', 'imagen'].some(exclusion => modelId.includes(exclusion))) {
    return false;
  } else if (providerName === 'openrouter' && !modelId.includes(':free')) {
    return false;
  } else if (providerName !== 'mistral' && modelId.includes('mistral')) {
    return false;
  }

  if (!providerName && ['mistral', 'alibaba', 'cohere', 'deepseek', 'moonshotai', 'morph', 'zai'].some(exclusion => modelId.includes(exclusion))) {
    return false;
  }

  return true;
}

export async function OPTIONS(req: NextRequest) {
  return new NextResponse(null, {
    status: 204,
    headers: corsHeaders,
  });
}

export async function GET(req: NextRequest) {
  const authHeader = req.headers.get('Authorization');
  let apiKey = authHeader?.split(' ')[1];

  // Check for password authentication
  const envPassword = process.env.PASSWORD;
  const isPasswordAuth = !!(envPassword && apiKey && envPassword.trim() === apiKey.trim());

  if (!apiKey) {
    return new NextResponse('Unauthorized', {
      status: 401,
      headers: corsHeaders,
    });
  }

  // Get provider API keys from headers
  const providerKeys = getProviderKeysFromHeaders(req, isPasswordAuth);

  return await getModelsResponse(apiKey, providerKeys, isPasswordAuth);
}

export async function POST(req: NextRequest) {
  const authHeader = req.headers.get('Authorization');
  let apiKey = authHeader?.split(' ')[1];

  // Check for password authentication
  const envPassword = process.env.PASSWORD;
  const isPasswordAuth = !!(envPassword && apiKey && envPassword.trim() === apiKey.trim());

  if (!apiKey) {
    return new NextResponse('Unauthorized', {
      status: 401,
      headers: corsHeaders,
    });
  }

  // Get provider API keys from headers
  const providerKeys = getProviderKeysFromHeaders(req, isPasswordAuth);
  return await getModelsResponse(apiKey, providerKeys, isPasswordAuth);
}

async function getModelsResponse(apiKey: string, providerKeys: Record<string, string[]>, isPasswordAuth: boolean = false) {
  const allModels: any[] = [];
  let gatewayApiKeys: string[] = [];

  if (isPasswordAuth) {
    // For password auth, try to get gateway API keys from environment variables
    const gatewayKey = process.env.GATEWAY_API_KEY;
    if (gatewayKey) {
      gatewayApiKeys = gatewayKey.split(',').map(key => key.trim());
    }
  } else {
    // Use the provided API key for gateway
    gatewayApiKeys = apiKey.split(',').map(key => key.trim());
  }

  let lastError: any;

  // Get models from gateway first (if we have gateway keys)
  if (gatewayApiKeys.length > 0) {
    for (let i = 0; i < gatewayApiKeys.length; i++) {
      const currentApiKey = gatewayApiKeys[i];

      const gateway = createGateway({
        apiKey: currentApiKey,
        baseURL: 'https://ai-gateway.vercel.sh/v1/ai',
      });

      try {
        const availableModels = await gateway.getAvailableModels();

        const now = Math.floor(Date.now() / 1000);
        const gatewayModels = availableModels.models.map(model => ({
          id: model.id,
          name: model.name,
          description: `${model.pricing
            ? ` I: $${(Number(model.pricing.input) * 1000000).toFixed(2)}, O: $${(
              Number(model.pricing.output) * 1000000
            ).toFixed(2)};`
            : ''
            } ${model.description || ''}`,
          object: 'model',
          created: now,
          owned_by: model.name.split('/')[0],
          pricing: model.pricing || {},
          source: 'gateway',
        })).filter(model => shouldIncludeModel(model));

        allModels.push(...gatewayModels);
        break; // Successfully got gateway models
      } catch (error: any) {
        console.error(`Error with gateway API key ${i + 1}/${gatewayApiKeys.length}:`, error);
        lastError = error;

        if (i < gatewayApiKeys.length - 1) {
          continue;
        }
      }
    }
  }

  // Get models from custom providers if provider keys are provided
  for (const [providerName, keys] of Object.entries(providerKeys)) {
    for (let i = 0; i < keys.length; i++) {
      const providerApiKey = keys[i];

      try {
        let formattedModels: any[] = [];
        const now = Math.floor(Date.now() / 1000);

        // Check if this provider has a custom model list (doesn't support /models endpoint)
        if (CUSTOM_MODEL_LISTS[providerName as keyof typeof CUSTOM_MODEL_LISTS]) {
          const customModels = CUSTOM_MODEL_LISTS[providerName as keyof typeof CUSTOM_MODEL_LISTS];
          formattedModels = customModels.map(model => ({
            id: `${providerName}/${model.id}`,
            name: model.name,
            object: 'model',
            created: now,
            owned_by: providerName,
            pricing: {},
            source: providerName,
          })).filter(model => shouldIncludeModel(model, providerName));
        } else {
          // Use regular API call for providers that support /models endpoint
          const providerModels = await fetchProviderModels(providerName, providerApiKey);
          formattedModels = providerModels.data?.map((model: any) => ({
            id: `${providerName}/${model.id.replace('models/', '')}`,
            name: `${model.name?.replace(' (free)', '') || parseModelName(model.id)}`,
            description: model.description || '',
            object: 'model',
            created: model.created || now,
            owned_by: model.owned_by || providerName,
            pricing: model.pricing || {},
            source: providerName,
          })).filter((model: any) => {
            if (!shouldIncludeModel(model, providerName)) {
              return false;
            }
            return true;
          }) || [];
        }

        allModels.push(...formattedModels);
        break; // Successfully got provider models
      } catch (error: any) {
        console.error(`Error with ${providerName} API key ${i + 1}/${keys.length}:`, error);

        if (i < keys.length - 1) {
          continue;
        }
        // Don't fail the entire request if one provider fails
      }
    }
  }

  // If we have models from any source, return them
  if (allModels.length > 0) {
    return NextResponse.json(
      {
        object: 'list',
        data: allModels,
      },
      {
        headers: corsHeaders,
      },
    );
  }

  // If no models were retrieved, return the last error
  console.error('All providers failed. Last error:', lastError);
  const errorMessage = lastError?.message || 'An unknown error occurred';
  const statusCode = lastError?.statusCode || 500;
  return new NextResponse(JSON.stringify({
    error: `All provider(s) failed. Last error: ${errorMessage}`
  }), {
    status: statusCode,
    headers: {
      'Content-Type': 'application/json',
      ...corsHeaders,
    },
  });
}