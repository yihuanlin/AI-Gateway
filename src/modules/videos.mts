import { type WaitResult, lastUserPromptFromMessages, lastUserPromptFromResponsesInput, responsesBase, streamChatSingleText, streamResponsesSingleText, streamResponsesGenerationElapsed, streamChatGenerationElapsed, findLinks, hasImageInMessages, sleep } from './utils.mts';
import { SUPPORTED_PROVIDERS, getProviderKeys } from '../shared/providers.mts';

export function toMarkdownVideo(url: string): string {
    return `[Generated Video](${url})`;
}

function ensureRatioInPrompt(prompt: string): string {
    if (/\s--(rt|ratio)\s/.test(prompt)) return prompt;
    return `${prompt} --ratio 16:9`;
}

function helpForVideo(model: string) {
    if (model.includes('doubao')) {
        return '**Doubao** Video models (supports both t2v and i2v. To use i2v, include an image in your message).\nFlags: `--rs/--resolution 480p|720p|1080p`, `--dur/--duration 3-12`, `--seed -1|[0,2^32-1]`, `--cf/--camerafixed true|false`, `--rt/--ratio 16:9` (default of t2v), 4:3, 1:1, 3:4, 9:16, 21:9, adaptive (default of i2v). Special: `/repeat` (use same image as first and last frame), `/upload` upload input images to storage.';
    } else {
        return '**Hugging Face** Video models (supports both t2v and i2v. To use i2v, include an image in your message).\nFlags: `--frames N` (number of frames), `--guidance F` (guidance scale), `--steps N` (inference steps), `--seed N` (random seed).\nOutput video always uploaded if S3 bucket is configured.';
    }
}

async function buildVideoGenerationWaiter(params: {
    model: string;
    prompt: string;
    headers: Headers;
    authHeader: string | null;
    isPasswordAuth: boolean;
    contentParts: any[];
}): Promise<{ ok: true; wait: (signal: AbortSignal) => Promise<WaitResult>; taskId: string } | { ok: false; error: any; status?: number }> {
    const { model, prompt: rawPrompt, headers, authHeader, isPasswordAuth, contentParts } = params;
    const links = findLinks(rawPrompt || '');
    const imgs = hasImageInMessages(contentParts || []);
    let prompt = rawPrompt || '';

    // Check for non-Doubao models - route to Hugging Face
    if (!model.includes('doubao')) {
        let apiKey: string | null = null;
        try {
            const pk = await getProviderKeys(headers as any, authHeader, isPasswordAuth);
            const keys = pk['huggingface'] || [];
            if (keys.length > 0) { const idx = Math.floor(Math.random() * keys.length); apiKey = keys[idx] || null; }
        } catch { }
        if (!apiKey) return { ok: false, error: { code: 'no_api_key', message: 'Missing Hugging Face API key' }, status: 401 };
        const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
        const taskId = `hf_${timestamp}`;

        // Check if we have images for i2v
        const hasImages = imgs.has || links.length > 0;

        const wait = async (_signal: AbortSignal) => {
            try {
                const { InferenceClient } = await import('@huggingface/inference');
                const client = new InferenceClient(apiKey);

                // Extract image data if available
                let imageData: Buffer | null = null;
                let imageType = 'image/jpeg'; // default
                if (hasImages) {
                    const imageUrl = imgs.first || links[0] || '';
                    if (imageUrl.startsWith('data:')) {
                        // Base64 image - extract type from header
                        const base64Match = imageUrl.match(/^data:([^;]+);base64,(.+)$/);
                        if (base64Match && base64Match[2]) {
                            imageType = base64Match[1] || 'image/jpeg'; // e.g., 'image/png', 'image/jpeg'
                            imageData = Buffer.from(base64Match[2], 'base64');
                        }
                    } else {
                        // Download from URL
                        try {
                            const response = await fetch(imageUrl);
                            if (response.ok) {
                                imageData = Buffer.from(await response.arrayBuffer());
                                // Try to determine type from Content-Type header
                                const contentType = response.headers.get('content-type');
                                if (contentType && contentType.startsWith('image/')) {
                                    imageType = contentType;
                                }
                            }
                        } catch (e) {
                            console.warn('Failed to download input image:', e);
                        }
                    }
                }

                // Prepare parameters
                const parameters: any = {};

                if (typeof prompt === 'string' && prompt.includes('--frames')) {
                    const frameMatch = prompt.match(/--frames\s+(\d+)/);
                    if (frameMatch && frameMatch[1]) {
                        parameters.num_frames = parseInt(frameMatch[1]);
                        prompt = prompt.replace(/--frames\s+\d+/, '').trim();
                    }
                }

                if (typeof prompt === 'string' && prompt.includes('--guidance')) {
                    const guidanceMatch = prompt.match(/--guidance\s+([\d.]+)/);
                    if (guidanceMatch && guidanceMatch[1]) {
                        parameters.guidance_scale = parseFloat(guidanceMatch[1]);
                        prompt = prompt.replace(/--guidance\s+[\d.]+/, '').trim();
                    }
                }

                if (typeof prompt === 'string' && prompt.includes('--steps')) {
                    const stepsMatch = prompt.match(/--steps\s+(\d+)/);
                    if (stepsMatch && stepsMatch[1]) {
                        parameters.num_inference_steps = parseInt(stepsMatch[1]);
                        prompt = prompt.replace(/--steps\s+\d+/, '').trim();
                    }
                }

                if (typeof prompt === 'string' && prompt.includes('--seed')) {
                    const seedMatch = prompt.match(/--seed\s+(\d+)/);
                    if (seedMatch && seedMatch[1]) {
                        parameters.seed = parseInt(seedMatch[1]);
                        prompt = prompt.replace(/--seed\s+\d+/, '').trim();
                    }
                }
                const modelId = model.replace('video/', '').replace(/-vision$/, '').replace(/Qwen-/, '');
                // Use imageToVideo if we have image data, otherwise textToVideo
                let result: Blob;
                if (imageData) {
                    result = await client.imageToVideo({
                        provider: "auto",
                        inputs: new Blob([new Uint8Array(imageData)], { type: imageType }),
                        model: modelId,
                        parameters: { ...parameters, prompt }
                    });
                } else {
                    result = await client.textToVideo({
                        provider: "auto",
                        model: modelId,
                        inputs: prompt,
                        parameters
                    });
                }

                // Upload to S3 bucket if configured (videos are too large for base64 responses)
                let finalUrl: string;
                if (process.env.S3_API && process.env.S3_PUBLIC_URL && process.env.S3_ACCESS_KEY && process.env.S3_SECRET_KEY) {
                    try {
                        const { uploadBlobToStorage } = await import('../shared/bucket.mts');
                        const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
                        finalUrl = await uploadBlobToStorage(result, `vid_${timestamp}`);
                    } catch (blobError) {
                        console.warn('Failed to upload video to bucket, using base64:', blobError);
                        // Fallback to base64 conversion
                        const arrayBuffer = await result.arrayBuffer();
                        const buffer = Buffer.from(arrayBuffer);
                        const base64 = buffer.toString('base64');
                        finalUrl = `data:video/mp4;base64,${base64}`;
                    }
                } else {
                    // Convert to base64 URL when not uploading to storage
                    const arrayBuffer = await result.arrayBuffer();
                    const buffer = Buffer.from(arrayBuffer);
                    const base64 = buffer.toString('base64');
                    finalUrl = `data:video/mp4;base64,${base64}`;
                }

                const usage = { input_tokens: 0, output_tokens: 0, total_tokens: 0 };
                return { ok: true, text: toMarkdownVideo(finalUrl), usage, downloadLink: finalUrl, taskId } as const;
            } catch (e: any) {
                return { ok: false, error: { code: 'network_error', message: e?.message || 'Hugging Face video API failed' } } as const;
            }
        };

        return { ok: true, wait, taskId };
    }

    // Doubao Seedance support
    let apiKey: string | null = null;
    try {
        const pk = await getProviderKeys(headers as any, authHeader, isPasswordAuth);
        const keys = pk['doubao'] || [];
        if (keys.length > 0) { const idx = Math.floor(Math.random() * keys.length); apiKey = keys[idx] || null; }
    } catch { }
    if (!apiKey) return { ok: false, error: { code: 'no_api_key', message: 'Missing Doubao API key' }, status: 401 };
    const base = SUPPORTED_PROVIDERS.doubao.baseURL;

    const content: any[] = [{ type: 'text', text: prompt }];
    let modelId = '';
    if (model.includes('doubao-seedance-pro')) {
        modelId = 'doubao-seedance-1-0-pro-250528';
    } else {
        if (links.length > 0 || imgs.has) modelId = 'doubao-seedance-1-0-lite-i2v-250428'; else modelId = 'doubao-seedance-1-0-lite-t2v-250428';
    }

    if (imgs.has || links.length > 0) {
        let first = imgs.first || links[0] || '';

        const hasUploadFlag = prompt.toLowerCase().includes('/upload');

        if (first.startsWith('data:') && hasUploadFlag && process.env.S3_API && process.env.S3_PUBLIC_URL && process.env.S3_ACCESS_KEY && process.env.S3_SECRET_KEY) {
            try {
                const { uploadBase64ToBlob } = await import('../shared/bucket.mts');
                const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
                first = await uploadBase64ToBlob(first, `${timestamp}_first`);
            } catch (blobError) {
                console.warn('Failed to upload first frame to bucket, using base64:', blobError);
            }
        }

        if (modelId.endsWith('i2v-250428')) {
            content.push({ type: 'image_url', image_url: { url: first }, role: 'first_frame' });

            // Check if prompt contains /repeat and only has one image
            const isRepeatMode = prompt.toLowerCase().includes('/repeat');
            const hasOnlyOneImage = !imgs.second && links.length <= 1;

            if (imgs.second || links.length > 1 || (isRepeatMode && hasOnlyOneImage)) {
                let lastUrl = imgs.second || links[links.length - 1] || '';

                // If /repeat mode with only one image, use the same image as last frame
                if (isRepeatMode && hasOnlyOneImage) {
                    lastUrl = first; // Use the same image as first frame
                }

                if (lastUrl.startsWith('data:') && hasUploadFlag && process.env.S3_API && process.env.S3_PUBLIC_URL && process.env.S3_ACCESS_KEY && process.env.S3_SECRET_KEY) {
                    try {
                        const { uploadBase64ToBlob } = await import('../shared/bucket.mts');
                        const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
                        lastUrl = await uploadBase64ToBlob(lastUrl, `${timestamp}_last`);
                    } catch (blobError) {
                        console.warn('Failed to upload last frame to bucket, using base64:', blobError);
                    }
                }

                content.push({ type: 'image_url', image_url: { url: lastUrl }, role: 'last_frame' });
            }

            // Remove /repeat and /upload from prompt if present
            let cleanPrompt = prompt.replace(imgs.first || links[0] || '', '').trim();
            cleanPrompt = cleanPrompt.replace(/\/repeat/gi, '').replace(/\/upload/gi, '').trim();
            content[0].text = cleanPrompt;
        } else {
            content.push({ type: 'image_url', image_url: { url: first } });
            prompt = prompt.replace(imgs.first || links[0] || '', '').trim();
            content[0].text = prompt;
        }
    } else {
        if (modelId.endsWith('t2v-250428')) {
            prompt = ensureRatioInPrompt(prompt);
            content[0].text = prompt;
        }
    }

    // Create the task first to get the ID immediately
    try {
        const createRes = await fetch(`${base}/contents/generations/tasks`, { method: 'POST', headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` }, body: JSON.stringify({ model: modelId, content }) });
        const createJson: any = await createRes.json().catch(() => ({} as any));
        if (!createRes.ok) return { ok: false, error: createJson?.error || { code: createRes.status, message: createJson?.message || createRes.statusText }, status: createRes.status };
        const taskId = (createJson && createJson.id) as string;

        const wait = async (_signal: AbortSignal) => {
            try {
                const started = Date.now();
                while (true) {
                    await sleep(1000);
                    const r = await fetch(`${base}/contents/generations/tasks/${taskId}`, { headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` } });
                    const j: any = await r.json().catch(() => ({} as any));
                    const status = (j && j.status) || 'queued';
                    if (status === 'succeeded') {
                        const url = j?.content?.video_url;
                        const usage = j?.usage ? {
                            input_tokens: 0,
                            output_tokens: j.usage.completion_tokens || 0,
                            total_tokens: j.usage.total_tokens || 0
                        } : { input_tokens: 0, output_tokens: 0, total_tokens: 0 };
                        return { ok: true, text: toMarkdownVideo(url), usage, downloadLink: url, taskId } as const;
                    } else if (status === 'failed') {
                        const err = j?.error || { code: 'failed', message: 'Video Generation Failed.' };
                        return { ok: false, error: err } as const;
                    } else if (status === 'cancelled') {
                        return { ok: true, text: 'Cancelled', usage: { input_tokens: 0, output_tokens: 0, total_tokens: 0 } } as const;
                    }
                    if (Date.now() - started > 10 * 60_000) return { ok: false, error: { code: 'timeout', message: 'Video generation timeout' } } as const;
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

export async function handleVideoForChat(args: { model: string; messages: any[]; headers: Headers; stream?: boolean; authHeader: string | null; isPasswordAuth: boolean; }): Promise<Response> {
    const { model, messages, headers, stream = false, authHeader, isPasswordAuth } = args;
    const now = Date.now();
    const last = lastUserPromptFromMessages(messages);
    let prompt = last.text || '';

    if (prompt.trim() === '/help') {
        const help = helpForVideo(model);
        if (stream) return streamChatSingleText(model, help);
        const created = Math.floor(now / 1000);
        const payload = { id: `chatcmpl-${now}`, object: 'chat.completion', created, model, choices: [{ index: 0, message: { role: 'assistant', content: help }, finish_reason: 'stop' }], usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } } as any;
        return new Response(JSON.stringify(payload), { headers: { 'Content-Type': 'application/json' } });
    }

    const waiter = await buildVideoGenerationWaiter({ model, prompt, headers, authHeader, isPasswordAuth, contentParts: last.content || [] });
    if (!waiter.ok) {
        return new Response(JSON.stringify({ error: waiter.error }), { status: waiter.status || 400, headers: { 'Content-Type': 'application/json' } });
    }
    if (stream) return streamChatGenerationElapsed(model, waiter.wait, waiter.taskId);
    // Non-stream: since video is async by provider, return a minimal ack
    const res = await waiter.wait(new AbortController().signal);
    if (!res.ok) return new Response(JSON.stringify({ error: res.error }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    const created = Math.floor(now / 1000);
    const chatUsage = res.usage ? {
        prompt_tokens: res.usage.input_tokens,
        completion_tokens: res.usage.output_tokens,
        total_tokens: res.usage.total_tokens
    } : { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
    const payload = { id: `chatcmpl-${now}`, object: 'chat.completion', created, model, choices: [{ index: 0, message: { role: 'assistant', content: res.text }, finish_reason: 'stop' }], usage: chatUsage } as any;
    return new Response(JSON.stringify(payload), { headers: { 'Content-Type': 'application/json' } });
}

export async function handleVideoForResponses(args: { model: string; input: any; headers: Headers; stream?: boolean; request_id: string; authHeader: string | null; isPasswordAuth: boolean; }): Promise<Response> {
    const { model, input, headers, stream = false, request_id, authHeader, isPasswordAuth } = args;
    const now = Date.now();
    const last = lastUserPromptFromResponsesInput(input);
    let prompt = last.text || '';

    if (prompt.trim() === '/help') {
        const help = helpForVideo(model);
        const base = responsesBase(now, request_id, model, input, null, true, undefined, undefined, undefined, undefined);
        if (stream) return streamResponsesSingleText(base, help, `msg_${now}`, true);
        const response = { ...base, status: 'completed', output: [{ type: 'message', id: `msg_${now}`, status: 'completed', role: 'assistant', content: [{ type: 'output_text', text: help }] }], usage: { input_tokens: 0, output_tokens: 0, total_tokens: 0 } };
        return new Response(JSON.stringify(response), { headers: { 'Content-Type': 'application/json' } });
    }

    const waiter = await buildVideoGenerationWaiter({ model, prompt, headers, authHeader, isPasswordAuth, contentParts: last.content || [] });
    if (!waiter.ok) {
        return new Response(JSON.stringify({ error: waiter.error }), { status: waiter.status || 400, headers: { 'Content-Type': 'application/json' } });
    }
    if (stream) {
        const baseObj = responsesBase(now, request_id, model, input, null, true, undefined, undefined, undefined, undefined);
        return streamResponsesGenerationElapsed({ baseObj, requestId: request_id, waitForResult: waiter.wait, taskId: waiter.taskId, headers });
    }
    // Non-stream: wait for completion
    const res = await waiter.wait(new AbortController().signal);
    if (!res.ok) return new Response(JSON.stringify({ error: res.error }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    const baseObj = responsesBase(now, request_id, model, input, null, true, undefined, undefined, undefined, undefined);
    // For Responses endpoint, usage format is already correct: { input_tokens, output_tokens, total_tokens }
    const response = { ...baseObj, status: 'completed', output: [{ type: 'message', id: 'msg_' + Date.now(), status: 'completed', role: 'assistant', content: [{ type: 'output_text', text: res.text }] }], usage: res.usage ?? { input_tokens: 0, output_tokens: 0, total_tokens: 0 } };
    return new Response(JSON.stringify(response), { headers: { 'Content-Type': 'application/json' } });
}
