import { type WaitResult, lastUserPromptFromMessages, responsesBase, streamChatSingleText, streamResponsesSingleText, streamChatGenerationElapsed, streamResponsesGenerationElapsed, findLinks, hasImageInMessages, sleep } from './utils.mts';
import { SUPPORTED_PROVIDERS } from '../shared/providers.mts';
import { experimental_generateImage as generateImage } from 'ai';

export type ImageResult = {
  usage: { input_tokens: number; output_tokens: number; total_tokens: number } | null;
  data: any;
};

const toMarkdownImage = (url: string): string => {
  return `![Generated Image](${url})`;
}

const getHelpForModel = (model: string) => {
  if (model.startsWith('image/doubao')) {
    return 'Use **Doubao** unified t2i / i2i model *doubao-seedream-4-0-250828* (multiple reference images supported).\nFlags: `--format url|b64_json`, `--size {WxH}|--ratio {e.g., 16:9}`, `--seed N`, `--guidance F`.\n`/upload` uploads output to storage when base64 is returned.';
  }
  if (model.startsWith('image/huggingface/')) {
    return '**Hugging Face** Text-to-Image and Image-to-Image models.\nFlags: `--guidance F`, `--negative_prompt "text"`, `--steps N (1-100)"`, `--size WxH` or `--ratio A:B`, `--seed N`.\n`/upload` upload input images to storage (output images are uploaded to storage). Input images enable image-to-image mode.\nSpecial prompt trigger for Kontext models:\n`Make a shot in the same scene of...`\n`Remove ...`\n`redepthkontext ...`\n`Place it`\n`Fuse this image into background`\n`Convert this image into pencil drawing art style`\n`Turn this image into the Clay_Toy style.`';
  }
  if (model.startsWith('image/modelscope/')) {
    return '**ModelScope** Text-to-Image and Image-to-Image models.\nFlags: `--negative_prompt "text"`, `--steps N (1-100)`, `--guidance F` (or derived from `top_p`/`temperature`), `--size WxH` or `--ratio A:B`, `--seed N`.\nFLUX.1 uses support any ratio. For Qwen models, supported ratios: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3.\n`/upload` upload input images to storage. Input images enable image-to-image mode.\nIf prompt contains `miratsu style` or `chibi` with Qwen/Qwen-Image, switches to **MTWLDFC/miratsu_style**.';
  }
  if (model.startsWith('image/bfl/')) {
    return '**Black Forest Labs** FLUX models via AI SDK Gateway.\nMultiple input images supported.\nFlags:\n`--imagePrompt` Base64-encoded image for additional visual context\n`--imagePromptStrength F` (0.0-1.0) Strength of image prompt influence\n`--promptUpsampling` Enable prompt upsampling\n`--raw` Enable raw mode for natural aesthetics\n`--size WxH` Output dimensions (width and height must be multiples of 16)\n`--steps N` Inference steps (flex models only)\n`--guidance F` Guidance scale (flex models only)';
  }
  return 'Supported providers: **Doubao** `image/doubao` (t2i/i2i), **Hugging Face** `image/huggingface/huggingface-model-id` (t2i/i2i), **ModelScope** `image/modelscope/modelscope-model-id` (t2i/i2i), **Black Forest Labs** `image/bfl/model-id` (t2i/i2i).';
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

  if (model.startsWith('image/doubao')) {
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
    const model = 'doubao-seedream-4-0-250828';

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
        model,
        prompt: cleanPrompt,
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
            '1:1': '2048x2048',
            '4:3': '2304x1728',
            '3:4': '1728x2304',
            '16:9': '2560x1440',
            '9:16': '1440x2560',
            '3:2': '2496x1664',
            '2:3': '1664x2496',
            '21:9': '3024x1296'
          };
          return ratioMap[ratio] || '1280x720';
        }
        return '1280x720';
      })();
      const g = typeof flags['guidance'] === 'number' ? flags['guidance'] : guidanceFromTopP(top_p, temperature) ?? 2.5;
      payload = {
        model,
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

        if (data?.b64_json && hasUploadFlag && process.env.URL) {
          try {
            const { uploadBase64ToStorage } = await import('../shared/bucket.mts');
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
          if (!process.env.URL) throw new Error('No process.env.URL configured');
          const { uploadBlobToStorage } = await import('../shared/bucket.mts');
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
        if (!process.env.URL) {
          return { ok: false, error: { code: 'no_storage_url', message: 'process.env.URL is required for base64 image upload in ModelScope i2i' }, status: 400 };
        }
        try {
          const { uploadBase64ToStorage } = await import('../shared/bucket.mts');
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

  if (model.startsWith('image/bfl/')) {
    // Black Forest Labs via AI SDK Gateway
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

    const bflModelId = model.replace('image/bfl/', 'bfl/').replace(/-vision$/, '');
    const isFlexModel = /flex/i.test(bflModelId);
    const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
    const taskId = `bfl_${timestamp}`;

    // Collect input images from message content and links
    const inputImages: string[] = [];
    if (imgs.has && Array.isArray((imgs as any).urls)) {
      for (const u of (imgs as any).urls as string[]) {
        if (u && !inputImages.includes(u)) inputImages.push(u);
      }
    }
    for (const l of links) {
      if (!inputImages.includes(l)) inputImages.push(l);
    }

    // Build providerOptions for Black Forest Labs
    const providerOptions: Record<string, any> = {
      blackForestLabs: {
        outputFormat: 'png',
        safetyTolerance: 5,
      }
    };

    // Add input images to providerOptions (inputImage, inputImage2, inputImage3, etc.)
    if (inputImages.length > 0) {
      providerOptions.blackForestLabs.inputImage = inputImages[0];
      for (let i = 1; i < Math.min(inputImages.length, 10); i++) {
        providerOptions.blackForestLabs[`inputImage${i + 1}`] = inputImages[i];
      }
    }

    // Handle optional flags
    if (typeof flags['imageprompt'] === 'string') {
      providerOptions.blackForestLabs.imagePrompt = flags['imageprompt'];
    }
    if (typeof flags['imagepromptstrength'] === 'number') {
      providerOptions.blackForestLabs.imagePromptStrength = Math.max(0, Math.min(1, flags['imagepromptstrength']));
    }
    if (flags['promptupsampling'] === true) {
      providerOptions.blackForestLabs.promptUpsampling = true;
    }
    if (flags['raw'] === true) {
      providerOptions.blackForestLabs.raw = true;
    }
    // Handle --size WxH flag
    if (typeof flags['size'] === 'string') {
      const sizeParts = (flags['size'] as string).split('x').map(n => parseInt(n));
      const width = sizeParts[0];
      const height = sizeParts[1];
      if (width && height && !isNaN(width) && !isNaN(height)) {
        // Width and height must be multiples of 16
        providerOptions.blackForestLabs.width = Math.round(width / 16) * 16;
        providerOptions.blackForestLabs.height = Math.round(height / 16) * 16;
      }
    }

    // Flex model specific options
    if (isFlexModel) {
      if (typeof flags['steps'] === 'number') {
        providerOptions.blackForestLabs.steps = flags['steps'];
      }
      if (typeof flags['guidance'] === 'number') {
        providerOptions.blackForestLabs.guidance = flags['guidance'];
      }
    }

    const wait = async (_signal: AbortSignal) => {
      try {
        // const { createGateway } = await import('@ai-sdk/gateway');
        // const gateway = createGateway({ apiKey });
        globalThis.process.env.AI_GATEWAY_API_KEY = apiKey;
        // console.log(`Using BFL model: ${bflModelId} with prompt "${prompt}" and options:`, providerOptions);
        const result = await generateImage({
          // model: gateway.imageModel(bflModelId),
          model: bflModelId,
          prompt,
          providerOptions,
        });

        // Get the first generated image
        const image = result.image;
        if (!image || !image.base64) {
          return { ok: false, error: { code: 'no_image', message: 'No image generated' } } as const;
        }

        const mediaType = image.mediaType || 'image/png';
        let finalUrl: string;

        // Upload to blob storage if available; fallback to base64 URL
        try {
          if (!process.env.URL) throw new Error('No process.env.URL configured');
          const { uploadBase64ToStorage } = await import('../shared/bucket.mts');
          const dataUrl = `data:${mediaType};base64,${image.base64}`;
          finalUrl = await uploadBase64ToStorage(dataUrl, timestamp);
        } catch (blobError) {
          console.warn('Failed to upload to storage, using base64:', blobError);
          finalUrl = `data:${mediaType};base64,${image.base64}`;
        }

        const usage = { input_tokens: 0, output_tokens: 0, total_tokens: 0 };
        return { ok: true, text: toMarkdownImage(finalUrl), usage, downloadLink: finalUrl, taskId } as const;
      } catch (e: any) {
        // Handle case where error might be wrapped in a Promise
        let actualError = e;
        if (e instanceof Promise || (e && typeof e.then === 'function')) {
          try {
            actualError = await e;
          } catch (awaitedErr) {
            actualError = awaitedErr;
          }
        }

        // Extract error message from Gateway errors
        // Gateway errors stringify as: "GatewayInternalServerError: [JSON details]\n    at ..."
        let errorMessage = 'BFL API failed';
        try {
          const errorString = actualError?.toString?.() || String(actualError);
          // Try to extract message between "ErrorName: " and first newline
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
          // If still default or empty, try other properties
          if (!errorMessage || errorMessage === 'BFL API failed') {
            errorMessage = actualError?.message || actualError?.name || errorString.substring(0, 200) || 'BFL API failed';
          }
        } catch {
          errorMessage = 'BFL API failed';
        }
        return { ok: false, error: { code: actualError?.statusCode || 'network_error', message: errorMessage } } as const;
      }
    };

    return { ok: true, wait, taskId };
  }

  return { ok: false, error: { code: 'unsupported_model', message: 'Unsupported image model' }, status: 400 };
}