import { type WaitResult, lastUserPromptFromMessages, lastUserPromptFromResponsesInput, responsesBase, streamChatSingleText, streamResponsesSingleText, streamChatGenerationElapsed, streamResponsesGenerationElapsed, findLinks, hasImageInMessages, sleep } from './utils.mts';
import { SUPPORTED_PROVIDERS, getProviderKeys } from '../shared/providers.mts';

export type ImageResult = {
  usage: { input_tokens: number; output_tokens: number; total_tokens: number } | null;
  data: any;
};

function toMarkdownImage(url: string): string {
  return `![Generated Image](${url})`;
}

function getHelpForModel(model: string) {
  if (model.startsWith('image/doubao')) {
    return 'Use **Doubao** t2i model *doubao-seedream-3-0-t2i-250415* or i2i *doubao-seededit-3-0-i2i-250628* (if has an input image).\nFlags: `--format url|b64_json`, `--size {WxH}|--ratio {e.g., 16:9}`, `--seed N`, `--guidance F`.\n`/upload` upload input images to storage .';
  }
  if (model.endsWith('-vision') && !model.includes('doubao')) {
    return '**Hugging Face** Image-to-Image models (requires input image).\nFlags: `--guidance F`, `--negative_prompt "text"`, `--steps N (1-100)"`, `--size WxH` or `--ratio A:B`.\n`/upload` upload input images to storage (output images are always uploaded if S3 bucket is configured).\nSpecial prompt trigger for Kontext models:\n`Make a shot in the same scene of...`\n`Remove ...`\n`redepthkontext ...`\n`Place it`\n`Fuse this image into background`\n`Convert this image into pencil drawing art style`\n`Turn this image into the Clay_Toy style.`';
  }
  if (model.startsWith('image/')) {
    return '**ModelScope** Text-to-Image models.\nFlags: `--negative_prompt "text"`, `--steps N (1-100)`, `--guidance F` (or derived from `top_p`/`temperature`), `--size WxH` or `--ratio A:B`, `--seed N`.\nFLUX.1 uses support any ratio. For Qwen models, supported ratios: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3.\n`/upload` upload input images to storage.\nIf prompt contains `miratsu style` or `chibi` with Qwen/Qwen-Image, switches to **MTWLDFC/miratsu_style**.';
  }
  return 'Unknown image model';
}

export async function handleImageForChat(args: {
  model: string;
  messages: any[];
  headers: Headers;
  stream?: boolean;
  temperature?: number;
  top_p?: number;
  authHeader: string | null;
  isPasswordAuth: boolean;
}): Promise<Response> {
  const { model, messages, headers, stream = false, temperature, top_p, authHeader, isPasswordAuth } = args;

  const now = Date.now();
  const last = lastUserPromptFromMessages(messages);
  let prompt = last.text || '';
  const { cleaned, flags } = extractFlags(prompt);
  prompt = cleaned;

  if (prompt.trim() === '/help') {
    const help = getHelpForModel(model);
    if (stream) return streamChatSingleText(model, help);
    const created = Math.floor(now / 1000);
    const payload = { id: `chatcmpl-${now}`, object: 'chat.completion', created, model, choices: [{ index: 0, message: { role: 'assistant', content: help }, finish_reason: 'stop' }], usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } } as any;
    return new Response(JSON.stringify(payload), { headers: { 'Content-Type': 'application/json' } });
  }

  try {
    const waiter = await buildImageGenerationWaiter({
      model,
      prompt,
      flags,
      headers,
      authHeader,
      isPasswordAuth,
      contentParts: last.content || [],
      ...(typeof temperature === 'number' ? { temperature } : {}),
      ...(typeof top_p === 'number' ? { top_p } : {}),
    });
    if (!waiter.ok) {
      return new Response(JSON.stringify({ error: waiter.error }), { status: waiter.status || 400, headers: { 'Content-Type': 'application/json' } });
    }
    if (stream) return streamChatGenerationElapsed(model, waiter.wait, waiter.taskId);
    const res = await waiter.wait(new AbortController().signal);
    if (!res.ok) return new Response(JSON.stringify({ error: res.error }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    const created = Math.floor(now / 1000);
    // Convert usage for Chat endpoint format
    const chatUsage = res.usage ? {
      prompt_tokens: res.usage.input_tokens,
      completion_tokens: res.usage.output_tokens,
      total_tokens: res.usage.total_tokens
    } : { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
    const payload = { id: `chatcmpl-${now}`, object: 'chat.completion', created, model, choices: [{ index: 0, message: { role: 'assistant', content: res.text }, finish_reason: 'stop' }], usage: chatUsage } as any;
    return new Response(JSON.stringify(payload), { headers: { 'Content-Type': 'application/json' } });
  } catch (e: any) {
    return new Response(JSON.stringify({ error: { code: 'network_error', message: e?.message || 'generation failed' } }), { status: 500, headers: { 'Content-Type': 'application/json' } });
  }
}

export async function handleImageForResponses(args: {
  model: string;
  input: any;
  headers: Headers;
  stream?: boolean;
  temperature?: number;
  top_p?: number;
  request_id: string;
  authHeader: string | null;
  isPasswordAuth: boolean;
}): Promise<Response> {
  const { model, input, headers, stream = false, temperature, top_p, request_id, authHeader, isPasswordAuth } = args;
  const now = Date.now();
  const last = lastUserPromptFromResponsesInput(input);
  let prompt = last.text || '';
  const { cleaned, flags } = extractFlags(prompt);
  prompt = cleaned;

  if (prompt.trim() === '/help') {
    const help = getHelpForModel(model);
    const baseObj = responsesBase(now, request_id, model, input, null, false, undefined, undefined, undefined, undefined);
    if (stream) return streamResponsesSingleText(baseObj, help, `msg_${now}`, true);
    const responsePayload = { ...baseObj, status: 'completed', output: [{ type: 'message', id: `msg_${now}`, status: 'completed', role: 'assistant', content: [{ type: 'output_text', text: help }] }], usage: { input_tokens: 0, output_tokens: 0, total_tokens: 0 } } as any;
    return new Response(JSON.stringify(responsePayload), { headers: { 'Content-Type': 'application/json' } });
  }

  const baseObj = responsesBase(now, request_id, model, input, null, false, undefined, undefined, undefined, undefined);

  try {
    const waiter = await buildImageGenerationWaiter({
      model,
      prompt,
      flags,
      headers,
      authHeader,
      isPasswordAuth,
      contentParts: last.content || [],
      ...(typeof temperature === 'number' ? { temperature } : {}),
      ...(typeof top_p === 'number' ? { top_p } : {}),
    });
    if (!waiter.ok) {
      return new Response(JSON.stringify({ error: waiter.error }), { status: waiter.status || 400, headers: { 'Content-Type': 'application/json' } });
    }
    if (stream) return streamResponsesGenerationElapsed({ baseObj, requestId: request_id, waitForResult: waiter.wait, taskId: waiter.taskId, headers });
    const res = await waiter.wait(new AbortController().signal);
    if (!res.ok) return new Response(JSON.stringify({ error: res.error }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    // For Responses endpoint, usage format is already correct: { input_tokens, output_tokens, total_tokens }
    const responsePayload = { ...baseObj, status: 'completed', output: [{ type: 'message', id: 'msg_' + Date.now(), status: 'completed', role: 'assistant', content: [{ type: 'output_text', text: res.text }] }], usage: res.usage ?? { input_tokens: 0, output_tokens: 0, total_tokens: 0 } } as any;
    return new Response(JSON.stringify(responsePayload), { headers: { 'Content-Type': 'application/json' } });
  } catch (e: any) {
    return new Response(JSON.stringify({ error: { code: 'network_error', message: e?.message || 'generation failed' } }), { status: 500, headers: { 'Content-Type': 'application/json' } });
  }
}

function extractFlags(prompt: string) {
  const flags: Record<string, string | number | boolean> = {};
  let cleaned = prompt;
  const flagRegex = /\s--([a-zA-Z_\-]+)(?:\s+([^\s][^\n]*?))?(?=\s--|$)/g;
  cleaned = cleaned.replace(flagRegex, (_m, key, val) => {
    const k = String(key).trim().toLowerCase();
    if (typeof val === 'string' && val.trim().length > 0) {
      const v = val.trim();
      if (/^\d+$/.test(v)) flags[k] = Number(v);
      else if (/^\d+\.\d+$/.test(v)) flags[k] = Number(v);
      else if (v === 'true' || v === 'false') flags[k] = v === 'true';
      else flags[k] = v;
    } else {
      flags[k] = true;
    }
    return '';
  });
  cleaned = cleaned.replace(/\s+/g, ' ').trim();
  return { cleaned, flags };
}

function guidanceFromTopP(topP?: number, temperature?: number): number | undefined {
  if (typeof topP === 'number') {
    const t = typeof temperature === 'number' ? temperature : 1;
    const mapped = 1 + (1 - Math.max(0, Math.min(1, topP))) * 9;
    const adj = Math.max(1.1, Math.min(9, mapped * (t <= 0 ? 1 : 1 / t)));
    return Number(adj.toFixed(2));
  }
  return undefined;
}

function ratioToSize(r: string, model: string): string | null {
  const ratio = String(r).trim();

  // Parse ratio
  const parts = ratio.split(':').map(n => parseFloat(n));
  if (parts.length !== 2 || parts.some(isNaN)) return null;
  const [w, h] = parts;
  if (w === undefined || h === undefined || h === 0) return null;
  const aspectRatio = w / h;

  if (/qwen/i.test(model)) {
    // Use preconfigured values for Qwen
    const qwenMap: Record<string, [number, number]> = {
      '1:1': [1328, 1328], '16:9': [1664, 928], '9:16': [928, 1664],
      '4:3': [1472, 1140], '3:4': [1140, 1472], '3:2': [1584, 1056], '2:3': [1056, 1584],
    };
    const v = qwenMap[ratio];
    return v ? `${v[0]}x${v[1]}` : null;
  } else if (/flux/i.test(model)) {
    // FLUX: base 1440x1440, adapt to ratio
    const base = 1440;
    const area = base * base;
    const width = Math.round(Math.sqrt(area * aspectRatio) / 2) * 2;
    const height = Math.round(area / width / 2) * 2;
    return `${width}x${height}`;
  } else if (/(diffusion|high-res)/i.test(model)) {
    // Diffusion/high-res: base 2048x2048, adapt to ratio
    const base = 2048;
    const area = base * base;
    const width = Math.round(Math.sqrt(area * aspectRatio) / 2) * 2;
    const height = Math.round(area / width / 2) * 2;
    return `${width}x${height}`;
  }

  return null;
}

function getSizeString(flags: Record<string, any>, effectiveModel: string): string | undefined {
  let sizeStr: string | undefined = undefined;
  if (typeof flags['size'] === 'string') {
    sizeStr = flags['size'] as string;
  } else if (typeof flags['ratio'] === 'string') {
    sizeStr = ratioToSize(flags['ratio'] as string, effectiveModel) || undefined;
  } else {
    // Default sizes based on model
    if (/flux/i.test(effectiveModel)) sizeStr = '1440x1440';
    else if (/(diffusion|high-res)/i.test(effectiveModel)) sizeStr = '2048x2048';
    else if (/qwen/i.test(effectiveModel)) sizeStr = '1328x1328';
  }
  return sizeStr;
}

async function buildImageGenerationWaiter(params: {
  model: string;
  prompt: string;
  flags: Record<string, any>;
  headers: Headers;
  authHeader: string | null;
  isPasswordAuth: boolean;
  contentParts: any[];
  temperature?: number;
  top_p?: number;
}): Promise<{ ok: true; wait: (signal: AbortSignal) => Promise<WaitResult>; taskId: string } | { ok: false; error: any; status?: number }> {
  const { model, headers, authHeader, isPasswordAuth, contentParts, flags, temperature, top_p } = params;
  let prompt = params.prompt || '';
  const links = findLinks(prompt);
  const imgs = hasImageInMessages(contentParts || []);
  const hasUploadFlag = prompt.toLowerCase().includes('/upload');
  prompt = prompt.replace(/\/upload/gi, '').trim();

  if (model.startsWith('image/doubao')) {
    let apiKey: string | null = null;
    try {
      const pk = await getProviderKeys(headers as any, authHeader, isPasswordAuth);
      const keys = pk['doubao'] || [];
      if (keys.length > 0) { const idx = Math.floor(Math.random() * keys.length); apiKey = keys[idx] || null; }
    } catch { }
    if (!apiKey) return { ok: false, error: { code: 'no_api_key', message: 'Missing Doubao API key' }, status: 401 };
    const base = SUPPORTED_PROVIDERS.doubao.baseURL;
    const url = `${base}/images/generations`;
    const response_format = (flags['format'] as string) || 'url';
    const watermark = false;
    let payload: any;
    if (imgs.has || links.length > 0) {
      // i2i model (seededit) - ignore size/ratio options but still clean prompt
      const firstUrl = imgs.first || links[0];
      let cleanPrompt = prompt.replace(firstUrl || '', '').trim();
      // Remove size/ratio flags from prompt for i2i model
      cleanPrompt = cleanPrompt.replace(/\s--(?:size|ratio)\s+[^\s]+/g, '').trim();
      const guidance = typeof flags['guidance'] === 'number' ? flags['guidance'] : (typeof top_p === 'number' ? Math.max(1, Math.min(10, (1 - top_p) * 9 + 1)) : 5.5);
      payload = { model: 'doubao-seededit-3-0-i2i-250628', prompt: cleanPrompt, image: firstUrl, response_format, size: 'adaptive', seed: typeof flags['seed'] === 'number' ? flags['seed'] : 21, guidance_scale: guidance, watermark };
    } else {
      const size = (() => {
        if (typeof flags['size'] === 'string') return flags['size'];
        if (typeof flags['ratio'] === 'string') {
          const ratio = flags['ratio'] as string;
          const ratioMap: Record<string, string> = {
            '1:1': '1024x1024',
            '3:4': '864x1152',
            '4:3': '1152x864',
            '16:9': '1280x720',
            '9:16': '720x1280',
            '2:3': '832x1248',
            '3:2': '1248x832',
            '21:9': '1512x648'
          };
          return ratioMap[ratio] || '1280x720';
        }
        return '1280x720';
      })();
      const g = typeof flags['guidance'] === 'number' ? flags['guidance'] : guidanceFromTopP(top_p, temperature) ?? 2.5;
      payload = { model: 'doubao-seedream-3-0-t2i-250415', prompt, response_format, size, seed: typeof flags['seed'] === 'number' ? flags['seed'] : -1, guidance_scale: g, watermark };
    }
    // Generate a synthetic task ID for Doubao since it's synchronous
    const taskId = `doubao_${Date.now()}_${Math.random().toString(36).slice(2)}`;

    const wait = async (_signal: AbortSignal) => {
      try {
        const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` }, body: JSON.stringify(payload) });
        const json: any = await res.json().catch(() => ({} as any));
        if (!res.ok) return { ok: false, error: json?.error || { code: res.status, message: json?.message || res.statusText } } as const;
        const data = json?.data?.[0];
        let urlOrB64 = data?.url || (data?.b64_json ? `data:image/png;base64,${data.b64_json}` : '');

        if (data?.b64_json && hasUploadFlag && process.env.S3_API && process.env.S3_PUBLIC_URL && process.env.S3_ACCESS_KEY && process.env.S3_SECRET_KEY) {
          try {
            const { uploadBase64ToBlob } = await import('../shared/bucket.mts');
            const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
            const blobUrl = await uploadBase64ToBlob(`data:image/png;base64,${data.b64_json}`, timestamp);
            urlOrB64 = blobUrl;
          } catch (blobError) {
            console.warn('Failed to upload to bucket, using base64:', blobError);
          }
        }

        const usage = json?.usage ? {
          input_tokens: 0,
          output_tokens: json.usage.output_tokens || 0,
          total_tokens: json.usage.total_tokens || 0
        } : { input_tokens: 0, output_tokens: 0, total_tokens: 0 };
        return { ok: true, text: toMarkdownImage(urlOrB64), usage, downloadLink: urlOrB64, taskId } as const;
      } catch (e: any) {
        return { ok: false, error: { code: 'network_error', message: e?.message || 'fetch failed' } } as const;
      }
    };
    return { ok: true, wait, taskId };
  }

  if (model.endsWith('-vision')) {
    let apiKey: string | null = null;
    try {
      const pk = await getProviderKeys(headers as any, authHeader, isPasswordAuth);
      const keys = pk['huggingface'] || [];
      if (keys.length > 0) { const idx = Math.floor(Math.random() * keys.length); apiKey = keys[idx] || null; }
    } catch { }
    if (!apiKey) return { ok: false, error: { code: 'no_api_key', message: 'Missing Hugging Face API key' }, status: 401 };

    let modelId = model.replace(/-vision$/, '').replace("image/", '');
    if (/Kontext/i.test(modelId)) {
      if (prompt.toLowerCase().startsWith("remove")) {
        modelId = 'starsfriday/Kontext-Remover-General-LoRA';
      } else if (prompt == "Place it") {
        modelId = 'ilkerzgi/Overlay-Kontext-Dev-LoRA';
      } else if (/Make a shot in the same scene of/i.test(prompt)) {
        modelId = 'peteromallet/Flux-Kontext-InScene';
      } else if (/redepthkontext/i.test(prompt)) {
        modelId = 'thedeoxen/FLUX.1-Kontext-dev-reference-depth-fusion-LORA';
      } else if (prompt == "Fuse this image into background") {
        modelId = 'gokaygokay/Fuse-it-Kontext-Dev-LoRA';
      } else if (prompt == "Convert this image into pencil drawing art style") {
        modelId = 'fal/Pencil-Drawing-Kontext-Dev-LoRA';
      } else if (prompt == "Turn this image into the Clay_Toy style.") {
        modelId = 'Kontext-Style/Clay_Toy_lora';
      }
    }
    const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
    const taskId = `hf_${timestamp}`;

    // Get input image data (must have an image for i2i)
    if (!imgs.has && links.length === 0) {
      return { ok: false, error: { code: 'missing_image', message: 'Image-to-image requires an input image' }, status: 400 };
    }

    const wait = async (_signal: AbortSignal) => {
      try {
        const { InferenceClient } = await import('@huggingface/inference');
        const client = new InferenceClient(apiKey);

        // Get image data
        let imageData: Buffer;
        let inputImageType = 'image/jpeg'; // default
        const imageUrl = imgs.first || links[0] || '';

        if (imageUrl.startsWith('data:')) {
          // Base64 image - extract type from header
          const base64Match = imageUrl.match(/^data:([^;]+);base64,(.+)$/);
          if (!base64Match || !base64Match[2]) {
            return { ok: false, error: { code: 'invalid_image', message: 'Invalid base64 image format' } } as const;
          }
          inputImageType = base64Match[1] || 'image/jpeg';
          imageData = Buffer.from(base64Match[2], 'base64');
        } else {
          // Download from URL
          const response = await fetch(imageUrl);
          if (!response.ok) {
            return { ok: false, error: { code: 'download_failed', message: 'Failed to download input image' } } as const;
          }
          imageData = Buffer.from(await response.arrayBuffer());
          // Try to determine type from Content-Type header
          const contentType = response.headers.get('content-type');
          if (contentType && contentType.startsWith('image/')) {
            inputImageType = contentType;
          }
        }

        // Prepare parameters
        const parameters: any = { prompt };

        if (typeof flags['guidance'] === 'number') {
          parameters.guidance_scale = Number(flags['guidance']);
        }
        if (typeof flags['negative_prompt'] === 'string') {
          parameters.negative_prompt = flags['negative_prompt'] as string;
        }
        if (typeof flags['steps'] === 'number') {
          parameters.num_inference_steps = Math.max(1, Math.min(100, Number(flags['steps'])));
        }

        // Handle size/ratio
        if (typeof flags['size'] === 'string' || typeof flags['ratio'] === 'string') {
          const sizeStr = getSizeString(flags, modelId);
          if (sizeStr && sizeStr.includes('x')) {
            const sizeParts = sizeStr.split('x').map(n => parseInt(n));
            const width = sizeParts[0];
            const height = sizeParts[1];
            if (width && height && !isNaN(width) && !isNaN(height)) {
              parameters.target_size = { width, height };
            }
          }
        }

        const result = await client.imageToImage({
          provider: "auto",
          model: modelId,
          inputs: new Blob([new Uint8Array(imageData)], { type: inputImageType }),
          parameters
        });

        // Upload to blob storage if configured
        let finalUrl: string;
        if (process.env.S3_API && process.env.S3_PUBLIC_URL && process.env.S3_ACCESS_KEY && process.env.S3_SECRET_KEY) {
          try {
            const { uploadBlobToStorage } = await import('../shared/bucket.mts');
            const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
            finalUrl = await uploadBlobToStorage(result, timestamp);
          } catch (blobError) {
            console.warn('Failed to upload to bucket, using base64:', blobError);
            // Fallback to base64 conversion
            const arrayBuffer = await result.arrayBuffer();
            const buffer = Buffer.from(arrayBuffer);
            const base64 = buffer.toString('base64');
            const outputImageType = result.type || inputImageType || 'image/jpeg';
            finalUrl = `data:${outputImageType};base64,${base64}`;
          }
        } else {
          // Convert to base64 URL when not uploading to storage
          const arrayBuffer = await result.arrayBuffer();
          const buffer = Buffer.from(arrayBuffer);
          const base64 = buffer.toString('base64');
          const outputImageType = result.type || inputImageType || 'image/jpeg';
          finalUrl = `data:${outputImageType};base64,${base64}`;
        }

        const usage = { input_tokens: 0, output_tokens: 0, total_tokens: 0 };
        return { ok: true, text: toMarkdownImage(finalUrl), usage, downloadLink: finalUrl, taskId } as const;
      } catch (e: any) {
        return { ok: false, error: { code: 'network_error', message: e?.message || 'Hugging Face API failed' } } as const;
      }
    };

    return { ok: true, wait, taskId };
  }

  if (model.startsWith('image/')) {
    const modelId = model.replace(/^image\//, '');
    let apiKey: string | null = null;
    try {
      const pk = await getProviderKeys(headers as any, authHeader, isPasswordAuth);
      const keys = pk['modelscope'] || [];
      if (keys.length > 0) { const idx = Math.floor(Math.random() * keys.length); apiKey = keys[idx] || null; }
    } catch { }
    if (!apiKey) return { ok: false, error: { code: 'no_api_key', message: 'Missing ModelScope API key' }, status: 401 };
    const base = SUPPORTED_PROVIDERS.modelscope.baseURL;

    // Special handling for Qwen miratsu
    let effectiveModel = modelId;
    if (/qwen\/?qwen-image/i.test(modelId) && /\b(miratsu style|chibi)\b/i.test(prompt)) {
      effectiveModel = 'MTWLDFC/miratsu_style';
    }

    let sizeStr: string | undefined = getSizeString(flags, effectiveModel);

    const guidance = typeof flags['guidance'] === 'number' ? Number(flags['guidance']) : guidanceFromTopP(top_p, temperature) ?? 3.5;
    const negative_prompt = typeof flags['negative_prompt'] === 'string' ? (flags['negative_prompt'] as string) : undefined;
    const steps = typeof flags['steps'] === 'number' ? Math.max(1, Math.min(100, Number(flags['steps']))) : undefined;
    const seedVal = typeof flags['seed'] === 'number' ? Math.max(0, Number(flags['seed'])) : undefined;

    const payload: any = { model: effectiveModel, prompt };
    if (sizeStr) payload.size = sizeStr;
    if (typeof guidance === 'number') payload.guidance = guidance;
    if (negative_prompt) payload.negative_prompt = negative_prompt.replace(/"/g, '').trim();
    if (steps !== undefined) payload.steps = steps;
    if (seedVal !== undefined) payload.seed = seedVal;

    // Create the task first to get the ID immediately
    try {
      const res = await fetch(`${base}/images/generations`, { method: 'POST', headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json', 'X-ModelScope-Async-Mode': 'true' }, body: JSON.stringify(payload) });
      const j: any = await res.json().catch(() => ({} as any));
      if (!res.ok) return { ok: false, error: j?.error || { code: res.status, message: j?.message || res.statusText } } as const;
      const taskId = j?.task_id as string;

      const wait = async (_signal: AbortSignal) => {
        try {
          const started = Date.now();
          while (true) {
            await sleep(1000);
            const r = await fetch(`${base}/tasks/${taskId}`, { headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json', 'X-ModelScope-Task-Type': 'image_generation' } });
            const dj: any = await r.json().catch(() => ({} as any));
            if (dj.task_status === 'SUCCEED') {
              const url = dj.output_images?.[0];
              return { ok: true, text: toMarkdownImage(url), usage: { input_tokens: 0, output_tokens: 0, total_tokens: 0 }, downloadLink: url, taskId } as const;
            } else if (dj.task_status === 'FAILED') {
              return { ok: false, error: { code: 'failed', message: 'Image Generation Failed.' } } as const;
            }
            if (Date.now() - started > 5 * 60_000) return { ok: false, error: { code: 'timeout', message: 'Image generation timeout' } } as const;
          }
        } catch (e: any) {
          return { ok: false, error: { code: 'network_error', message: e?.message || 'fetch failed' } } as const;
        }
      };
      return { ok: true, wait, taskId };
    } catch (e: any) {
      return { ok: false, error: { code: 'network_error', message: e?.message || 'fetch failed' } };
    }
  }

  return { ok: false, error: { code: 'unsupported_model', message: 'Unsupported image model' }, status: 400 };
}