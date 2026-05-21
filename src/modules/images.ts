import { type WaitResult, lastUserPromptFromMessages, responsesBase, streamChatSingleText, streamResponsesSingleText, streamChatGenerationElapsed, streamResponsesGenerationElapsed, findLinks, hasImageInMessages, sleep } from './utils.js';
import { SUPPORTED_PROVIDERS } from '../shared/providers.js';
import { generateImage } from 'ai';

export type ImageResult = {
  usage: { input_tokens: number; output_tokens: number; total_tokens: number } | null;
  data: any;
};

const toMarkdownImage = (url: string): string => {
  return `![Generated Image](${url})`;
}

const getHelpForModel = (model: string) => {
  if (model.startsWith('image/doubao') || model.startsWith('image/seedream')) {
    return 'Use **Seedream** unified t2i / i2i model *doubao-seedream-4-5-251128* (multiple reference images supported).\nFlags: `--format url|b64_json`, `--size {WxH}|--ratio {e.g., 16:9}`, `--seed N`, `--guidance F`.\n`/upload` uploads output to storage when base64 is returned.';
  }
  if (model.startsWith('image/huggingface/')) {
    return '**Hugging Face** Text-to-Image and Image-to-Image models.\nFlags: `--guidance F`, `--negative_prompt "text"`, `--steps N (1-100)"`, `--size WxH` or `--ratio A:B`, `--seed N`.\n`/upload` upload input images to storage (output images are uploaded to storage). Input images enable image-to-image mode.\nSpecial prompt trigger for Kontext models:\n`Make a shot in the same scene of...`\n`Remove ...`\n`redepthkontext ...`\n`Place it`\n`Fuse this image into background`\n`Convert this image into pencil drawing art style`\n`Turn this image into the Clay_Toy style.`';
  }
  if (model.startsWith('image/modelscope/')) {
    return '**ModelScope** Text-to-Image and Image-to-Image models.\nFlags: `--negative_prompt "text"`, `--steps N (1-100)`, `--guidance F` (or derived from `top_p`/`temperature`), `--size WxH` or `--ratio A:B`, `--seed N`.\nFLUX.1 uses support any ratio. For Qwen models, supported ratios: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3.\n`/upload` upload input images to storage. Input images enable image-to-image mode.\nIf prompt contains `miratsu style` or `chibi` with Qwen/Qwen-Image, switches to **MTWLDFC/miratsu_style**.';
  }
  if (model.startsWith('image/')) {
    const provider = model.split('/')[1] || 'generic';
    return `**${provider.toUpperCase()}** image model via AI SDK Gateway.
Supported flags:
- \`--n N\`: Number of images to generate
- \`--size WxH\`: Custom dimensions
- \`--aspectRatio {1:1|16:9|9:16|4:3|3:4|2:3|3:2}\`: Aspect ratio
- \`--seed S\`: Generation seed
- \`--image URL\`: Reference image (can specify multiple)
- \`--mask URL\`: Mask image
- \`--headers.Header-Name value\`: Custom request header
- \`--providerOptions.provider.key value\`: Custom provider option

Provider Specifics:
- **Black Forest Labs (BFL)**: \`--imagePrompt\` (base64 image), \`--imagePromptUrl\` (URL to download & base64 encode), \`--guidance F\`. Default: \`safetyTolerance = 6\`, \`outputFormat = 'png'\`.
- **Google**: \`--imageSize\`. Default: \`imageSize = '4k'\` for gemini-3-pro-image.
- **OpenAI**: \`--transparent\` (sets background to transparent). Default: \`quality = 'high'\`, \`outputFormat = 'png'\`.
- **xAI**: \`--resolution\`. Default: \`quality = 'high'\`, \`resolution = '2k'\` for grok-imagine-image-pro.`;
  }
  return 'Supported providers: **Seedream** `image/doubao` (t2i/i2i), **Hugging Face** `image/huggingface/huggingface-model-id` (t2i/i2i), **ModelScope** `image/modelscope/modelscope-model-id` (t2i/i2i), **AI Gateway** `image/model-id`.';
}

export const handleImageForChat = async (args: {
  model: string;
  messages: any[];
  stream?: boolean;
  temperature?: number;
  top_p?: number;
}): Promise<Response> => {
  const { model, messages, stream = false, temperature, top_p } = args;

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

export const handleImageForResponses = async (args: {
  model: string;
  messages: any[];
  stream?: boolean;
  temperature?: number;
  top_p?: number;
  request_id: string;
}): Promise<Response> => {
  const { model, messages, stream = false, temperature, top_p, request_id } = args;
  const now = Date.now();
  const last = lastUserPromptFromMessages(messages);
  let prompt = last.text || '';
  const { cleaned, flags } = extractFlags(prompt);
  prompt = cleaned;

  if (prompt.trim() === '/help') {
    const help = getHelpForModel(model);
    const baseObj = responsesBase(now, request_id, model, null, null, false, undefined, undefined, undefined, undefined);
    if (stream) return streamResponsesSingleText(baseObj, help, `msg_${now}`, true);
    const responsePayload = { ...baseObj, status: 'completed', output: [{ type: 'message', id: `msg_${now}`, status: 'completed', role: 'assistant', content: [{ type: 'output_text', text: help }] }], usage: { input_tokens: 0, output_tokens: 0, total_tokens: 0 } } as any;
    return new Response(JSON.stringify(responsePayload), { headers: { 'Content-Type': 'application/json' } });
  }

  const baseObj = responsesBase(now, request_id, model, null, null, false, undefined, undefined, undefined, undefined);

  try {
    const waiter = await buildImageGenerationWaiter({
      model,
      prompt,
      flags,
      contentParts: last.content || [],
      ...(typeof temperature === 'number' ? { temperature } : {}),
      ...(typeof top_p === 'number' ? { top_p } : {}),
    });
    if (!waiter.ok) {
      return new Response(JSON.stringify({ error: waiter.error }), { status: waiter.status || 400, headers: { 'Content-Type': 'application/json' } });
    }
    if (stream) return streamResponsesGenerationElapsed({ baseObj, requestId: request_id, waitForResult: waiter.wait, taskId: waiter.taskId });
    const res = await waiter.wait(new AbortController().signal);
    if (!res.ok) return new Response(JSON.stringify({ error: res.error }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    // For Responses endpoint, usage format is already correct: { input_tokens, output_tokens, total_tokens }
    const responsePayload = { ...baseObj, status: 'completed', output: [{ type: 'message', id: 'msg_' + Date.now(), status: 'completed', role: 'assistant', content: [{ type: 'output_text', text: res.text }] }], usage: res.usage ?? { input_tokens: 0, output_tokens: 0, total_tokens: 0 } } as any;
    return new Response(JSON.stringify(responsePayload), { headers: { 'Content-Type': 'application/json' } });
  } catch (e: any) {
    return new Response(JSON.stringify({ error: { code: 'network_error', message: e?.message || 'generation failed' } }), { status: 500, headers: { 'Content-Type': 'application/json' } });
  }
}

const extractFlags = (prompt: string) => {
  const flags: Record<string, any> = {};
  let cleaned = prompt;
  const flagRegex = /\s--([a-zA-Z0-9_\-\.]+)(?:\s+([^\s][^\n]*?))?(?=\s--|$)/g;
  cleaned = cleaned.replace(flagRegex, (_m, key, val) => {
    const k = String(key).trim();
    let parsedVal: any;
    if (typeof val === 'string' && val.trim().length > 0) {
      const v = val.trim();
      if (/^\d+$/.test(v)) parsedVal = Number(v);
      else if (/^\d+\.\d+$/.test(v)) parsedVal = Number(v);
      else if (v === 'true' || v === 'false') parsedVal = v === 'true';
      else parsedVal = v;
    } else {
      parsedVal = true;
    }

    if (flags[k] !== undefined) {
      if (Array.isArray(flags[k])) {
        flags[k].push(parsedVal);
      } else {
        flags[k] = [flags[k], parsedVal];
      }
    } else {
      flags[k] = parsedVal;
    }

    const lowerK = k.toLowerCase();
    if (lowerK !== k) {
      if (flags[lowerK] !== undefined) {
        if (Array.isArray(flags[lowerK])) {
          flags[lowerK].push(parsedVal);
        } else {
          flags[lowerK] = [flags[lowerK], parsedVal];
        }
      } else {
        flags[lowerK] = parsedVal;
      }
    }

    return '';
  });
  cleaned = cleaned.replace(/\s+/g, ' ').trim();
  return { cleaned, flags };
}

const guidanceFromTopP = (topP?: number, temperature?: number): number | undefined => {
  if (typeof topP === 'number') {
    const t = typeof temperature === 'number' ? temperature : 1;
    const mapped = 1 + (1 - Math.max(0, Math.min(1, topP))) * 9;
    const adj = Math.max(1.1, Math.min(9, mapped * (t <= 0 ? 1 : 1 / t)));
    return Number(adj.toFixed(2));
  }
  return undefined;
}

const ratioToSize = (r: string, model: string): string | null => {
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

const setNestedProperty = (obj: any, path: string[], value: any) => {
  let current = obj;
  for (let i = 0; i < path.length - 1; i++) {
    const key = path[i];
    if (key === undefined) continue;
    if (current[key] === undefined || typeof current[key] !== 'object') {
      current[key] = {};
    }
    current = current[key];
  }
  const lastKey = path[path.length - 1];
  if (lastKey !== undefined) {
    current[lastKey] = value;
  }
};

const buildImageGenerationWaiter = async (params: {
  model: string;
  prompt: string;
  flags: Record<string, any>;
  contentParts: any[];
  temperature?: number;
  top_p?: number;
}): Promise<{ ok: true; wait: (signal: AbortSignal) => Promise<WaitResult>; taskId: string } | { ok: false; error: any; status?: number }> => {
  const { model, contentParts, flags, temperature, top_p } = params;
  let prompt = params.prompt || '';
  const links = findLinks(prompt);
  const imgs = hasImageInMessages(contentParts || []);
  const hasUploadFlag = prompt.toLowerCase().includes('/upload');
  prompt = prompt.replace(/\/upload/gi, '').trim();

  // Remove image URLs from prompt for all providers
  if (links.length > 0) {
    for (const link of links) {
      prompt = prompt.replace(link + ' ', '').replace(link, '').trim();
    }
  }

  if (model.startsWith('image/seedream') || model === 'image/doubao') {
    let apiKey: string | null = null;
    try {
      const keys = String(process.env.DOUBAO_API_KEY).split(',').map((k: string) => k.trim()) || [];
      if (keys.length > 0) { const idx = Math.floor(Math.random() * keys.length); apiKey = keys[idx] || null; }
    } catch { }
    if (!apiKey) return { ok: false, error: { code: 'no_api_key', message: 'Missing Doubao API key' }, status: 401 };
    const base = SUPPORTED_PROVIDERS.doubao.baseURL;
    const url = `${base}/images/generations`;
    const response_format = (flags['format'] as string) || 'url';
    const watermark = false;
    const actualModel = (model.startsWith('image/seedream-latest')) ? 'doubao-seedream-5-0-260128' : 'doubao-seedream-4-5-251128';

    // Collect all potential reference images (uploaded message images + inline links)
    const referenceImages: string[] = [];
    if (imgs.has && Array.isArray((imgs as any).urls)) {
      for (const u of (imgs as any).urls as string[]) {
        if (u && !referenceImages.includes(u)) referenceImages.push(u);
      }
    }
    for (const l of links) {
      if (!referenceImages.includes(l)) referenceImages.push(l);
    }

    let payload: any;
    if (referenceImages.length > 0) {
      // i2i (single or multi reference). For multi, send array. Remove size/ratio flags (model handles adaptively)
      let cleanPrompt = prompt.replace(/\s--(?:size|ratio)\s+[^\s]+/g, '').trim();
      const guidance = typeof flags['guidance'] === 'number' ? flags['guidance'] : (typeof top_p === 'number' ? Math.max(1, Math.min(10, (1 - top_p) * 9 + 1)) : 5.5);
      payload = {
        model: actualModel,
        prompt: cleanPrompt,
        size: (model.startsWith('image/seedream-latest')) ? '3K' : '4K',
        image: referenceImages.length === 1 ? referenceImages[0] : referenceImages,
        response_format,
        seed: typeof flags['seed'] === 'number' ? flags['seed'] : 21,
        guidance_scale: guidance,
        watermark
      };
    } else {
      // pure t2i
      const size = (() => {
        if (typeof flags['size'] === 'string') return flags['size'];
        if (typeof flags['ratio'] === 'string') {
          const ratio = flags['ratio'] as string;
          const ratioMap: Record<string, string> = {
            '1:1': '4096x4096',
            '4:3': '4704x3520',
            '3:4': '3520x4704',
            '16:9': '5504x3040',
            '9:16': '3040x5504',
            '3:2': '4992x3328',
            '2:3': '3328x4992',
            '21:9': '6240x2656'
          };
          return ratioMap[ratio] || '4096x4096';
        }
        return (model.startsWith('image/seedream-latest')) ? '3K' : '4K';
      })();
      const g = typeof flags['guidance'] === 'number' ? flags['guidance'] : guidanceFromTopP(top_p, temperature) ?? 2.5;
      payload = {
        model: actualModel,
        prompt,
        response_format,
        size,
        seed: typeof flags['seed'] === 'number' ? flags['seed'] : -1,
        guidance_scale: g,
        watermark
      };
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

        if (data?.b64_json && hasUploadFlag && (process.env.URL || process.env.VERCEL_PROJECT_PRODUCTION_URL)) {
          try {
            const { uploadBase64ToStorage } = await import('../shared/bucket.js');
            const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
            const blobUrl = await uploadBase64ToStorage(`data:image/png;base64,${data.b64_json}`, timestamp);
            urlOrB64 = blobUrl;
          } catch (blobError) {
            console.warn('Failed to upload to blob store, using base64:', blobError);
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

  if (model.startsWith('image/huggingface/')) {
    let apiKey: string | null = null;
    try {
      const keys = String(process.env.HUGGINGFACE_API_KEY).split(',').map((k: string) => k.trim()) || [];
      if (keys.length > 0) { const idx = Math.floor(Math.random() * keys.length); apiKey = keys[idx] || null; }
    } catch { }
    if (!apiKey) return { ok: false, error: { code: 'no_api_key', message: 'Missing Hugging Face API key' }, status: 401 };

    let modelId = model.replace(/-vision$/, '').replace("image/huggingface/", '');
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

    const hasInputImage = imgs.has || links.length > 0;

    const wait = async (_signal: AbortSignal) => {
      try {
        const { InferenceClient } = await import('@huggingface/inference');
        const client = new InferenceClient(apiKey);

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

        // Handle size/ratio (only if explicitly specified)
        let sizeStr: string | undefined = undefined;
        if (typeof flags['size'] === 'string') {
          sizeStr = flags['size'] as string;
        } else if (typeof flags['ratio'] === 'string') {
          sizeStr = ratioToSize(flags['ratio'] as string, modelId) || undefined;
        }

        // Add seed parameter for t2i
        if (typeof flags['seed'] === 'number') {
          parameters.seed = Math.max(0, Number(flags['seed']));
        }

        let result: Blob;

        if (hasInputImage) {
          // Image-to-image mode
          const imageUrl = imgs.first || links[0] || '';
          let imageData: Buffer;
          let inputImageType = 'image/jpeg'; // default

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

          // For i2i, use target_size if size is specified
          if (sizeStr && sizeStr.includes('x')) {
            const sizeParts = sizeStr.split('x').map(n => parseInt(n));
            const width = sizeParts[0];
            const height = sizeParts[1];
            if (width && height && !isNaN(width) && !isNaN(height)) {
              parameters.target_size = { width, height };
            }
          }

          result = await client.imageToImage({
            provider: "auto",
            model: modelId,
            inputs: new Blob([new Uint8Array(imageData)], { type: inputImageType }),
            parameters
          });
        } else {
          // Text-to-image mode - use width/height directly
          if (sizeStr && sizeStr.includes('x')) {
            const sizeParts = sizeStr.split('x').map(n => parseInt(n));
            const width = sizeParts[0];
            const height = sizeParts[1];
            if (width && height && !isNaN(width) && !isNaN(height)) {
              parameters.width = width;
              parameters.height = height;
            }
          }

          result = await client.textToImage({
            provider: "auto",
            model: modelId,
            inputs: prompt,
            parameters
          }, { outputType: "blob" });
        }

        // Upload to blob storage; fallback to base64 URL on error
        let finalUrl: string;
        try {
          if (!process.env.URL && !process.env.VERCEL_PROJECT_PRODUCTION_URL) throw new Error('No URL or VERCEL_PROJECT_PRODUCTION_URL configured');
          const { uploadBlobToStorage } = await import('../shared/bucket.js');
          const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
          finalUrl = await uploadBlobToStorage(result, timestamp);
        } catch (blobError) {
          console.warn('Failed to upload to storage, using base64:', blobError);
          // Fallback to base64 conversion
          const arrayBuffer = await result.arrayBuffer();
          const buffer = Buffer.from(arrayBuffer);
          const base64 = buffer.toString('base64');
          const outputImageType = result.type || 'image/jpeg';
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

  if (model.startsWith('image/modelscope/')) {
    const modelId = model.replace(/-vision$/, '').replace("image/modelscope/", '');
    let apiKey: string | null = null;
    try {
      const keys = String(process.env.MODELSCOPE_API_KEY).split(',').map((k: string) => k.trim()) || [];
      if (keys.length > 0) { const idx = Math.floor(Math.random() * keys.length); apiKey = keys[idx] || null; }
    } catch { }
    if (!apiKey) return { ok: false, error: { code: 'no_api_key', message: 'Missing ModelScope API key' }, status: 401 };
    const base = SUPPORTED_PROVIDERS.modelscope.baseURL;

    // Special handling for Qwen miratsu
    let effectiveModel = modelId;
    if (/qwen\/?qwen-image/i.test(modelId) && /\b(miratsu style|chibi)\b/i.test(prompt)) {
      effectiveModel = 'MTWLDFC/miratsu_style';
    }

    // Handle size/ratio (only if explicitly specified)
    let sizeStr: string | undefined = undefined;
    if (typeof flags['size'] === 'string') {
      sizeStr = flags['size'] as string;
    } else if (typeof flags['ratio'] === 'string') {
      sizeStr = ratioToSize(flags['ratio'] as string, effectiveModel) || undefined;
    }

    const guidance = typeof flags['guidance'] === 'number' ? Number(flags['guidance']) : guidanceFromTopP(top_p, temperature);
    const negative_prompt = typeof flags['negative_prompt'] === 'string' ? (flags['negative_prompt'] as string) : undefined;
    const steps = typeof flags['steps'] === 'number' ? Math.max(1, Math.min(100, Number(flags['steps']))) : undefined;
    const seedVal = typeof flags['seed'] === 'number' ? Math.max(0, Number(flags['seed'])) : undefined;

    const payload: any = { model: effectiveModel, prompt };
    if (sizeStr) payload.size = sizeStr;
    if (typeof guidance === 'number') payload.guidance = guidance;
    if (negative_prompt) payload.negative_prompt = negative_prompt.replace(/"/g, '').trim();
    if (steps !== undefined) payload.steps = steps;
    if (seedVal !== undefined) payload.seed = seedVal;

    // Handle image-to-image if input image is present
    if (imgs.has || links.length > 0) {
      const imageUrl = imgs.first || links[0] || '';

      // If it's a base64 image, upload to storage first (requires process.env.URL)
      if (imageUrl.startsWith('data:')) {
        if (!process.env.URL && !process.env.VERCEL_PROJECT_PRODUCTION_URL) {
          return { ok: false, error: { code: 'no_storage_url', message: 'URL or VERCEL_PROJECT_PRODUCTION_URL is required for base64 image upload in ModelScope i2i' }, status: 400 };
        }
        try {
          const { uploadBase64ToStorage } = await import('../shared/bucket.js');
          const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
          const uploadedUrl = await uploadBase64ToStorage(imageUrl, timestamp);
          payload.image_url = uploadedUrl;
        } catch (uploadError: any) {
          return { ok: false, error: { code: 'upload_failed', message: uploadError?.message || 'Failed to upload base64 image to storage' }, status: 500 };
        }
      } else {
        // Direct URL, use as-is
        payload.image_url = imageUrl;
      }
    }

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

  if (model.startsWith('image/')) {
    let apiKey: string | null = null;
    try {
      const gatewayKey = process.env.GATEWAY_API_KEY;
      if (gatewayKey) {
        const keys = gatewayKey.split(',').map((k: string) => k.trim()).filter(Boolean);
        if (keys.length > 0) {
          const idx = Math.floor(Math.random() * keys.length);
          apiKey = keys[idx] || null;
        }
      }
    } catch { }
    if (!apiKey) return { ok: false, error: { code: 'no_api_key', message: 'Missing Gateway API key' }, status: 401 };

    const targetModelId = model.replace('image/', '').replace(/-vision$/, '');
    const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
    const taskId = `${targetModelId.split('/')[0] || 'gateway'}_${timestamp}`;

    const originalKeys = Object.keys(flags).filter(k => {
      const lowerK = k.toLowerCase();
      if (k === lowerK) {
        const hasOriginalUppercase = Object.keys(flags).some(otherKey => otherKey.toLowerCase() === lowerK && otherKey !== lowerK);
        if (hasOriginalUppercase) {
          return false;
        }
      }
      return true;
    });

    const headers: Record<string, string> = {};
    const providerOptions: Record<string, any> = {};

    for (const key of originalKeys) {
      if (key.startsWith('headers.')) {
        const headerName = key.slice('headers.'.length);
        setNestedProperty(headers, [headerName], flags[key]);
      } else if (key.startsWith('providerOptions.')) {
        const path = key.slice('providerOptions.'.length).split('.');
        setNestedProperty(providerOptions, path, flags[key]);
      }
    }

    // BFL defaults & aliases
    if (targetModelId.startsWith('bfl/')) {
      if (!providerOptions.blackForestLabs) {
        providerOptions.blackForestLabs = {};
      }
      if (providerOptions.blackForestLabs.safetyTolerance === undefined) {
        providerOptions.blackForestLabs.safetyTolerance = 6;
      }
      if (providerOptions.blackForestLabs.outputFormat === undefined) {
        providerOptions.blackForestLabs.outputFormat = 'png';
      }

      const imagePromptVal = flags['imagePrompt'] || flags['imageprompt'];
      if (imagePromptVal !== undefined) {
        providerOptions.blackForestLabs.imagePrompt = imagePromptVal;
      } else {
        if (providerOptions.blackForestLabs.imagePrompt === undefined) {
          providerOptions.blackForestLabs.imagePrompt = {};
        }
        if (providerOptions.blackForestLabs.imagePrompt && typeof providerOptions.blackForestLabs.imagePrompt === 'object') {
          if (providerOptions.blackForestLabs.imagePrompt.safetyTolerance === undefined) {
            providerOptions.blackForestLabs.imagePrompt.safetyTolerance = 6;
          }
        }
      }

      const guidanceVal = flags['guidance'];
      if (guidanceVal !== undefined) {
        providerOptions.blackForestLabs.guidance = guidanceVal;
      }

      // Flex model specific flat options (for backwards compatibility)
      const isFlexModel = /flex/i.test(targetModelId);
      if (isFlexModel) {
        if (typeof flags['steps'] === 'number') {
          providerOptions.blackForestLabs.steps = flags['steps'];
        }
      }
      if (flags['promptupsampling'] === true || flags['promptUpsampling'] === true) {
        providerOptions.blackForestLabs.promptUpsampling = true;
      }
      if (flags['raw'] === true) {
        providerOptions.blackForestLabs.raw = true;
      }
      const strength = flags['imagePromptStrength'] || flags['imagepromptstrength'];
      if (typeof strength === 'number') {
        providerOptions.blackForestLabs.imagePromptStrength = Math.max(0, Math.min(1, strength));
      }
    }

    // Google defaults & aliases
    if (targetModelId.startsWith('google/')) {
      if (!providerOptions.google) {
        providerOptions.google = {};
      }
      if (!providerOptions.google.imageConfig) {
        providerOptions.google.imageConfig = {};
      }
      const imageSizeVal = flags['imageSize'] || flags['imagesize'];
      if (imageSizeVal !== undefined) {
        providerOptions.google.imageConfig.imageSize = imageSizeVal;
      }
      if (targetModelId.includes('gemini-3-pro-image')) {
        if (providerOptions.google.imageConfig.imageSize === undefined) {
          providerOptions.google.imageConfig.imageSize = '4k';
        }
      }
    }

    // OpenAI defaults & aliases
    if (targetModelId.startsWith('openai/')) {
      if (!providerOptions.openai) {
        providerOptions.openai = {};
      }
      if (flags['transparent'] === true) {
        providerOptions.openai.background = 'transparent';
      }
      if (providerOptions.openai.quality === undefined) {
        providerOptions.openai.quality = 'high';
      }
      if (providerOptions.openai.outputFormat === undefined) {
        providerOptions.openai.outputFormat = 'png';
      }
    }

    // xAI defaults & aliases
    if (targetModelId.startsWith('xai/')) {
      if (!providerOptions.xai) {
        providerOptions.xai = {};
      }
      const resolutionVal = flags['resolution'];
      if (resolutionVal !== undefined) {
        providerOptions.xai.resolution = resolutionVal;
      }
      if (providerOptions.xai.quality === undefined) {
        providerOptions.xai.quality = 'high';
      }
      if (targetModelId.includes('grok-imagine-image-pro')) {
        if (providerOptions.xai.resolution === undefined) {
          providerOptions.xai.resolution = '2k';
        }
      }
    }

    // Collect input images from message content, links, and all --image flags
    const inputImages: string[] = [];
    if (imgs.has && Array.isArray((imgs as any).urls)) {
      for (const u of (imgs as any).urls as string[]) {
        if (u && !inputImages.includes(u)) inputImages.push(u);
      }
    }
    for (const l of links) {
      if (!inputImages.includes(l)) inputImages.push(l);
    }
    const imageFlagVal = flags['image'];
    if (imageFlagVal !== undefined) {
      const flagImages = Array.isArray(imageFlagVal) ? imageFlagVal : [imageFlagVal];
      for (const fi of flagImages) {
        if (typeof fi === 'string' && fi && !inputImages.includes(fi)) {
          inputImages.push(fi);
        }
      }
    }

    // Optional size, aspectRatio, seed, n, mask parameters
    let sizeParam: `${number}x${number}` | undefined = undefined;
    if (typeof flags['size'] === 'string' && /^\d+x\d+$/.test(flags['size'])) {
      sizeParam = flags['size'] as `${number}x${number}`;
    }

    let aspectRatioParam: any = undefined;
    const aspectVal = flags['aspectRatio'] || flags['aspectratio'];
    if (typeof aspectVal === 'string' && ['1:1', '16:9', '9:16', '4:3', '3:4', '2:3', '3:2'].includes(aspectVal)) {
      aspectRatioParam = aspectVal;
    }

    let seedParam: number | undefined = undefined;
    if (typeof flags['seed'] === 'number') {
      seedParam = flags['seed'];
    }

    let nParam: number | undefined = undefined;
    if (typeof flags['n'] === 'number') {
      nParam = flags['n'];
    }

    // Build prompt parameter.
    // If we have input images or a mask, format prompt as an object.
    let promptParam: any;
    if (inputImages.length > 0 || flags['mask'] !== undefined) {
      const promptObj: any = {
        text: prompt,
        images: inputImages,
      };
      if (flags['mask'] !== undefined && typeof flags['mask'] === 'string') {
        promptObj.mask = flags['mask'];
      }
      promptParam = promptObj;
    } else {
      promptParam = prompt;
    }

    const wait = async (_signal: AbortSignal): Promise<WaitResult> => {
      try {
        const imagePromptUrlVal = flags['imagePromptUrl'] || flags['imageprompturl'];
        if (typeof imagePromptUrlVal === 'string' && imagePromptUrlVal) {
          try {
            const resp = await fetch(imagePromptUrlVal);
            if (!resp.ok) throw new Error(`Failed to fetch imagePromptUrl: ${resp.statusText}`);
            const buffer = await resp.arrayBuffer();
            const base64 = Buffer.from(buffer).toString('base64');
            setNestedProperty(providerOptions, ['blackForestLabs', 'imagePrompt'], base64);
          } catch (e) {
            console.error('Error downloading imagePromptUrl:', e);
          }
        }

        globalThis.process.env.AI_GATEWAY_API_KEY = apiKey;

        const options: any = {
          model: targetModelId,
          prompt: promptParam,
          providerOptions,
          headers,
          abortSignal: _signal,
        };
        if (nParam !== undefined) options.n = nParam;
        if (sizeParam !== undefined) options.size = sizeParam;
        if (aspectRatioParam !== undefined) options.aspectRatio = aspectRatioParam;
        if (seedParam !== undefined) options.seed = seedParam;

        const result = await generateImage(options);

        const imagesToProcess = result.images && result.images.length > 0 ? result.images : (result.image ? [result.image] : []);
        if (imagesToProcess.length === 0) {
          return { ok: false, error: { code: 'no_image', message: 'No image generated' } } as const;
        }

        const urls: string[] = [];
        for (let i = 0; i < imagesToProcess.length; i++) {
          const img = imagesToProcess[i];
          if (!img) continue;
          const base64 = img.base64;
          if (!base64) continue;

          const mediaType = (img as any).mediaType || 'image/png';
          const fileSuffix = imagesToProcess.length > 1 ? `${timestamp}_${i}` : timestamp;

          let finalUrl: string;
          try {
            if (!process.env.URL && !process.env.VERCEL_PROJECT_PRODUCTION_URL) throw new Error('No URL or VERCEL_PROJECT_PRODUCTION_URL configured');
            const { uploadBase64ToStorage } = await import('../shared/bucket.js');
            const dataUrl = `data:${mediaType};base64,${base64}`;
            finalUrl = await uploadBase64ToStorage(dataUrl, fileSuffix);
          } catch (blobError) {
            console.warn('Failed to upload to storage, using base64:', blobError);
            finalUrl = `data:${mediaType};base64,${base64}`;
          }
          urls.push(finalUrl);
        }

        if (urls.length === 0) {
          return { ok: false, error: { code: 'no_image', message: 'No image generated' } } as const;
        }

        const markdownText = urls.map(toMarkdownImage).join('\n\n');
        const usage = result.usage ? {
          input_tokens: result.usage.inputTokens ?? 0,
          output_tokens: result.usage.outputTokens ?? 0,
          total_tokens: result.usage.totalTokens ?? 0
        } : { input_tokens: 0, output_tokens: 0, total_tokens: 0 };

        const waitRes: WaitResult = {
          ok: true,
          text: markdownText,
          usage,
          taskId,
          ...(urls[0] !== undefined ? { downloadLink: urls[0] } : {})
        };
        return waitRes;
      } catch (e: any) {
        let actualError = e;
        if (e instanceof Promise || (e && typeof e.then === 'function')) {
          try {
            actualError = await e;
          } catch (awaitedErr) {
            actualError = awaitedErr;
          }
        }

        let errorMessage = 'Gateway API failed';
        try {
          const errorString = actualError?.toString?.() || String(actualError);
          const colonIndex = errorString.indexOf(':');
          if (colonIndex !== -1) {
            const afterColon = errorString.substring(colonIndex + 1);
            const newlineIndex = afterColon.indexOf('\n');
            if (newlineIndex !== -1) {
              errorMessage = afterColon.substring(0, newlineIndex).trim();
            } else {
              errorMessage = afterColon.trim();
            }
          }
          if (!errorMessage || errorMessage === 'Gateway API failed') {
            errorMessage = actualError?.message || actualError?.name || errorString.substring(0, 200) || 'Gateway API failed';
          }
        } catch {
          errorMessage = 'Gateway API failed';
        }
        return { ok: false, error: { code: actualError?.statusCode || 'network_error', message: errorMessage } } as const;
      }
    };

    return { ok: true, wait, taskId };
  }

  return { ok: false, error: { code: 'unsupported_model', message: 'Unsupported image model' }, status: 400 };
}