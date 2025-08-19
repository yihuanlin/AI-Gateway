// ESM-friendly models handler extracted from index.ts, now reusing shared providers
import { createGateway } from '@ai-sdk/gateway';
import { SUPPORTED_PROVIDERS, getProviderKeys, parseModelDisplayName, fetchCopilotToken } from '../shared/providers.mts';

function isSupportedProvider(name: string): name is keyof typeof SUPPORTED_PROVIDERS {
  return Object.prototype.hasOwnProperty.call(SUPPORTED_PROVIDERS, name);
}

function shouldIncludeModel(model: any, providerName?: string) {
  const modelId = String(model.id || '').toLowerCase();
  const commonExclusions = [
    'gemma', 'rerank', 'distill', 'parse', 'embed', 'bge-', 'tts', 'phi', 'live', 'audio', 'lite',
    'qwen2', 'qwen-2', 'qwen1', 'qwq', 'qvq', 'gemini-1', 'learnlm', 'gemini-exp',
    'turbo', 'claude-3', 'voxtral', 'pixtral', 'mixtral', 'ministral', '-24', 'moderation', 'saba', '-ocr-',
    'transcribe', 'dall', 'davinci', 'babbage', 'hailuo', 'kling', 'wan', 'ideogram', 'background'
  ];
  if (commonExclusions.some((e) => modelId.includes(e))) return false;
  if (!modelId.includes('super') && ((['nemotron', 'llama'].some((e) => modelId.includes(e))) || modelId.includes('nvidia'))) return false;
  if (providerName === 'openrouter' && !modelId.includes(':free')) return false;
  if (providerName !== 'mistral' && modelId.includes('mistral')) return false;
  if (providerName === 'chatgpt' && (modelId.split('-').length - 1) > 2) return false;
  if (!providerName && ['mistral', 'alibaba', 'cohere', 'deepseek', 'moonshotai', 'morph', 'zai'].some((e) => modelId.includes(e))) return false;
  return true;
}

// Using shared getProviderKeys

async function fetchProviderModels(providerName: string, apiKey: string) {
  if (!isSupportedProvider(providerName)) {
    throw new Error(`Unsupported provider: ${providerName}`);
  }
  const config = SUPPORTED_PROVIDERS[providerName];
  if (!config) throw new Error(`Unsupported provider: ${providerName}`);
  let modelsEndpoint: string;
  if (providerName === 'github') modelsEndpoint = config.baseURL.replace('inference', 'catalog/models');
  else modelsEndpoint = `${config.baseURL}/models`;

  let response: Response;
  if (providerName === 'gemini') {
    modelsEndpoint = modelsEndpoint + '?key=' + apiKey;
    response = await fetch(modelsEndpoint);
  } else if (providerName === 'copilot') {
    const copilotToken = await fetchCopilotToken(apiKey);
    response = await fetch(modelsEndpoint, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${copilotToken}`,
        'Content-Type': 'application/json',
        "editor-version": "vscode/1.103.1",
        "copilot-vision-request": "true",
        "editor-plugin-version": "copilot-chat/0.30.1",
        "user-agent": "GitHubCopilotChat/0.30.1"
      },
    });
  } else {
    response = await fetch(modelsEndpoint, { method: 'GET', headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' } });
  }
  if (!response.ok) throw new Error(`Provider ${providerName} models API failed: ${response.status} ${response.statusText}`);
  const data = (await response.json()) as any;
  if (providerName === 'gemini') {
    return { data: data.models.map((m: any) => ({ id: m.name, name: m.displayName, description: m.description || '' })) };
  } else if (providerName === 'github') {
    return { data };
  }
  return data;
}

async function getModelsResponse(apiKey: string, providerKeys: Record<string, string[]>, isPasswordAuth: boolean = false) {
  let gatewayApiKeys: string[] = [];
  if (isPasswordAuth) {
    const gatewayKey = process.env.GATEWAY_API_KEY;
    if (gatewayKey) gatewayApiKeys = gatewayKey.split(',').map((k) => k.trim());
  } else {
    gatewayApiKeys = apiKey.split(',').map((k) => k.trim());
  }

  const fetchPromises: Promise<any[]>[] = [];

  if (gatewayApiKeys.length > 0) {
    const randomIndex = Math.floor(Math.random() * gatewayApiKeys.length);
    const currentApiKey = gatewayApiKeys[randomIndex];
    const gatewayPromise = (async () => {
      try {
        if (!currentApiKey) throw new Error('No valid gateway API key found');
        const gateway = createGateway({ apiKey: currentApiKey, baseURL: 'https://ai-gateway.vercel.sh/v1/ai' });
        const availableModels = await gateway.getAvailableModels();
        const now = Math.floor(Date.now() / 1000);
        return availableModels.models
          .map((model: any) => ({
            id: model.id,
            name: model.name,
            description: model.pricing ? ` I: $${(Number(model.pricing.input) * 1000000).toFixed(2)}, O: $${(Number(model.pricing.output) * 1000000).toFixed(2)}; ${model.description || ''}` : (model.description || ''),
            object: 'model',
            created: now,
            owned_by: model.name.split('/')[0],
          }))
          .filter((m: any) => shouldIncludeModel(m));
      } catch (e) {
        return [] as any[];
      }
    })();
    fetchPromises.push(gatewayPromise);
  }

  for (const [providerName, keys] of Object.entries(providerKeys)) {
    if (!keys || keys.length === 0) continue;
    if (!isSupportedProvider(providerName)) continue;
    const randomIndex = Math.floor(Math.random() * keys.length);
    const providerApiKey = keys[randomIndex];
    const providerPromise = (async () => {
      try {
        if (!providerApiKey) throw new Error(`No valid API key found for provider: ${providerName}`);
        let formattedModels: any[] = [];
        const providerModels = await fetchProviderModels(providerName, providerApiKey);
        formattedModels = (providerModels as any).data?.map((model: any) => ({
          id: `${providerName}/${String(model.id).replace('models/', '')}`,
          name: `${model.name?.replace(' (free)', '') || parseModelDisplayName(model.id)}`,
          description: model.description || model.summary || '',
          object: 'model',
          created: model.created || 0,
          owned_by: model.owned_by || providerName,
        }))?.filter((m: any) => shouldIncludeModel(m, providerName)) || [];
        return formattedModels;
      } catch (e) {
        return [] as any[];
      }
    })();
    fetchPromises.push(providerPromise);
  }

  const results = await Promise.allSettled(fetchPromises);
  const allModels: any[] = [];
  results.forEach((r) => { if (r.status === 'fulfilled' && r.value.length > 0) allModels.push(...r.value); });

  // Inject curated image models (always available)
  const curated = [
    { id: 'image/doubao-vision', name: 'Seed Image', object: 'model', created: 0, owned_by: 'doubao' },
    { id: 'image/AI-ModelScope/stable-diffusion-3.5-large-turbo', name: 'Stable Diffusion 3.5 Large', object: 'model', created: 0, owned_by: 'modelscope' },
    { id: 'image/MusePublic/489_ckpt_FLUX_1', name: 'FLUX.1 [dev]', object: 'model', created: 0, owned_by: 'modelscope' },
    { id: 'image/black-forest-labs/FLUX.1-Krea-dev', name: 'FLUX.1 Krea [dev]', object: 'model', created: 0, owned_by: 'modelscope' },
    { id: 'image/MusePublic/FLUX.1-Kontext-Dev', name: 'FLUX.1 Kontext [dev]', object: 'model', created: 0, owned_by: 'modelscope' },
    { id: 'image/MusePublic/flux-high-res', name: 'FLUX.1 [dev] High-Res', object: 'model', created: 0, owned_by: 'modelscope' },
    { id: 'image/Qwen/Qwen-Image', name: 'Qwen-Image', object: 'model', created: 0, owned_by: 'modelscope' },
    { id: 'admin/responses', name: 'Responses Management', object: 'model', created: 0, owned_by: 'admin' },
    { id: 'video/doubao-seedance-pro-vision', name: 'Seedance 1.0 Pro', object: 'model', created: 0, owned_by: 'doubao' },
    { id: 'video/doubao-seedance-lite-vision', name: 'Seedance 1.0 Lite', object: 'model', created: 0, owned_by: 'doubao' }
  ];
  const existingIds = new Set(allModels.map((m) => m.id));
  for (const m of curated) if (!existingIds.has(m.id)) allModels.push(m);

  if (allModels.length > 0) return { object: 'list', data: allModels };
  throw new Error('All provider(s) failed to return models');
}

export async function handleModelsRequest(c: any) {
  const authHeader = c.req.header('Authorization');
  let apiKey = authHeader?.split(' ')[1];
  const envPassword = process.env.PASSWORD;
  const isPasswordAuth = !!(envPassword && apiKey && envPassword.trim() === apiKey.trim());
  if (!apiKey) return c.text('Unauthorized', 401);

  const headers: Record<string, string> = {};
  c.req.raw.headers.forEach((value: string, key: string) => {
    headers[key.toLowerCase().replace(/-/g, '_')] = value;
  });
  const providerKeys = await getProviderKeys(headers as any, authHeader || null, isPasswordAuth);

  try {
    const modelsResponse = await getModelsResponse(apiKey, providerKeys, isPasswordAuth);
    return c.json(modelsResponse);
  } catch (error: any) {
    return c.json({ error: error?.message || 'All provider(s) failed to return models' }, 500);
  }
}
