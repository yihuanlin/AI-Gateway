export const SUPPORTED_PROVIDERS = {
  copilot: { baseURL: 'https://api.githubcopilot.com', tokenURL: 'https://api.github.com/copilot_internal/v2/token' },
  chatgpt: { baseURL: 'https://api.openai.com/v1' },
  doubao: { baseURL: 'https://ark.cn-beijing.volces.com/api/v3' },
  gemini: { baseURL: 'https://generativelanguage.googleapis.com/v1beta' },
  poe: { baseURL: 'https://api.poe.com/v1' },
  cerebras: { baseURL: 'https://api.cerebras.ai/v1' },
  groq: { baseURL: 'https://api.groq.com/openai/v1' },
  modelscope: { baseURL: 'https://api-inference.modelscope.cn/v1' },
  infini: { baseURL: 'https://cloud.infini-ai.com/maas/v1' },
  github: { baseURL: 'https://models.github.ai/inference' },
  openrouter: { baseURL: 'https://openrouter.ai/api/v1' },
  nvidia: { baseURL: 'https://integrate.api.nvidia.com/v1' },
  mistral: { baseURL: 'https://api.mistral.ai/v1' },
  cohere: { baseURL: 'https://api.cohere.ai/compatibility/v1' },
  poixe: { baseURL: 'https://api.poixe.com/v1' },
  huggingface: { baseURL: 'https://router.huggingface.co/v1' },
  cloudflare: { baseURL: `https://gateway.ai.cloudflare.com/v1/${process.env.CLOUDFLARE_GATEWAY}/workers-ai/v1` },
} as const;

export const PROVIDER_KEYS = Object.keys(SUPPORTED_PROVIDERS);

export async function getProviderKeys(headers: any, authHeader: string | null, isPasswordAuth: boolean = false): Promise<Record<string, string[]>> {
  const providerKeys: Record<string, string[]> = {};

  const getHeader = (name: string): string | null => {
    try {
      if (typeof headers?.get === 'function') {
        return headers.get(name) || headers.get(name.toLowerCase()) || null;
      }
    } catch { }
    const underscore = name.replace(/-/g, '_');
    return headers?.[name] || headers?.[name.toLowerCase()] || headers?.[underscore] || headers?.[underscore.toLowerCase()] || null;
  };

  for (const provider of PROVIDER_KEYS) {
    const keyName = `x-${provider}-api-key`;
    const headerValue = getHeader(keyName);
    if (headerValue) {
      providerKeys[provider] = String(headerValue).split(',').map((k: string) => k.trim());
      continue;
    }
    if (isPasswordAuth) {
      const envKeyName = `${provider.toUpperCase()}_API_KEY`;
      const envValue = (process as any).env?.[envKeyName];
      if (envValue) {
        providerKeys[provider] = String(envValue).split(',').map((k: string) => k.trim());
      }
    }
  }

  if (Object.keys(providerKeys).length === 0 && authHeader && !isPasswordAuth) {
    const headerKey = authHeader.split(' ')[1];
    if (headerKey) {
      const keys = headerKey.split(',').map((k: string) => k.trim());
      for (const provider of PROVIDER_KEYS) providerKeys[provider] = keys;
    }
  }
  return providerKeys;
}