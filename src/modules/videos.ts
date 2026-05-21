import { type WaitResult, lastUserPromptFromMessages, responsesBase, streamChatSingleText, streamResponsesSingleText, streamResponsesGenerationElapsed, streamChatGenerationElapsed, findLinks, hasImageInMessages, sleep } from './utils.js';
import { SUPPORTED_PROVIDERS } from '../shared/providers.js';
import { createGateway } from '@ai-sdk/gateway';
import { experimental_generateVideo as generateVideo } from 'ai';

export const toMarkdownVideo = (url: string): string => {
    return `[Generated Video](${url})`;
}

interface ParsedFlags {
    prompt: string;
    first?: string | undefined;
    last?: string | undefined;
    reference_images: string[];
    reference_videos: string[];
    reference_audios: string[];
    audio?: boolean | undefined;
    web_search?: boolean | undefined;
    ratio?: string | undefined;
    resolution?: string | undefined;
    duration?: number | undefined;
    seed?: number | undefined;
    camera_fixed?: boolean | undefined;
}

const parsePromptFlags = (rawPrompt: string): ParsedFlags => {
    let prompt = rawPrompt;

    const extractStringFlag = (flags: string[]): string | undefined => {
        for (const flag of flags) {
            const regex = new RegExp(`\\s*${flag}\\s+([^\\s]+)`, 'i');
            const match = prompt.match(regex);
            if (match) {
                const val = match[1];
                prompt = prompt.replace(regex, '').trim();
                return val;
            }
        }
        return undefined;
    };

    const extractMultipleStringFlags = (flags: string[]): string[] => {
        const values: string[] = [];
        for (const flag of flags) {
            const regex = new RegExp(`\\s*${flag}\\s+([^\\s]+)`, 'gi');
            const matches = [...prompt.matchAll(regex)];
            for (const m of matches) {
                if (m[1]) {
                    values.push(m[1]);
                }
            }
            prompt = prompt.replace(regex, '').trim();
        }
        return values;
    };

    const extractBooleanFlag = (flags: string[]): boolean | undefined => {
        for (const flag of flags) {
            const regexVal = new RegExp(`\\s*${flag}\\s+(true|false)`, 'i');
            const matchVal = prompt.match(regexVal);
            if (matchVal && matchVal[1]) {
                const val = matchVal[1].toLowerCase() === 'true';
                prompt = prompt.replace(regexVal, '').trim();
                return val;
            }
            const regexWord = new RegExp(`(?:\\s|^)${flag}(?:\\s|$)`, 'i');
            if (regexWord.test(prompt)) {
                prompt = prompt.replace(regexWord, ' ').trim();
                return true;
            }
        }
        return undefined;
    };

    const first = extractStringFlag(['--first']);
    const last = extractStringFlag(['--last']);

    const reference_images = extractMultipleStringFlags(['--reference_image']);
    const reference_videos = extractMultipleStringFlags(['--reference_video']);
    const reference_audios = extractMultipleStringFlags(['--reference_audio']);

    const audio = extractBooleanFlag(['--audio']);
    const web_search = extractBooleanFlag(['--web_search']);

    const ratio = extractStringFlag(['--rt', '--ratio']);
    const resolution = extractStringFlag(['--rs', '--resolution']);
    const durationStr = extractStringFlag(['--dur', '--duration']);
    const seedStr = extractStringFlag(['--seed']);
    const camera_fixed = extractBooleanFlag(['--cf', '--camerafixed', '--camera_fixed']);

    let seed: number | undefined;
    if (seedStr) {
        seed = parseInt(seedStr, 10);
        if (isNaN(seed)) seed = undefined;
    }

    let duration: number | undefined;
    if (durationStr) {
        duration = parseInt(durationStr, 10);
        if (isNaN(duration)) duration = undefined;
    }

    // Clean up double spaces
    prompt = prompt.replace(/\s+/g, ' ').trim();

    return {
        prompt,
        first,
        last,
        reference_images,
        reference_videos,
        reference_audios,
        audio,
        web_search,
        ratio,
        resolution,
        duration,
        seed,
        camera_fixed
    };
};

const helpForVideo = (model: string) => {
    if (model.includes('doubao') || model.includes('seedance')) {
        return `**Doubao / Seedance** Video models (supports both text-to-video and image-to-video).
**Flags:**
*   \`--rs/--resolution 480p|720p|1080p\`: Resolution. Defaults to \`1080p\` (except \`doubao-seedance-1-0-lite-t2v-250428\` and \`doubao-seedance-2-0-fast-260128\` which default to \`720p\`).
*   \`--dur/--duration <seconds>\`: Duration of video. Defaults to \`-1\` for 2.0 models, and \`5\` for others.
*   \`--seed <number>\`: Random seed.
*   \`--cf/--camerafixed\`: Fix the camera movement.
*   \`--rt/--ratio <ratio>\`: Aspect ratio (e.g., 16:9, 9:16, 1:1, etc.). Optional, handled automatically by default.
*   \`--first <url>\`: Image URL to use as the first frame.
*   \`--last <url>\`: Image URL to use as the last frame.
*   \`--audio\`: Generate audio for the video (ignored for 1.0 models).
*   \`--web_search\`: Enable web search tool for generation context.
*   \`--reference_image <url>\`: Reference image URL (can be specified multiple times).
*   \`--reference_video <url>\`: Reference video URL (can be specified multiple times).
*   \`--reference_audio <url>\`: Reference audio URL (can be specified multiple times).
*   \`Special commands in prompt:\`
    *   \`/repeat\`: Repeat the first frame as the last frame.
    *   \`/upload\`: Upload base64 input images to remote storage.`;
    } else {
        return '**Hugging Face** Video models (supports both t2v and i2v. To use i2v, include an image in your message).\nFlags: `--frames N` (number of frames), `--guidance F` (guidance scale), `--steps N` (inference steps), `--seed N` (random seed).\nOutput videos are uploaded to storage.';
    }
}

const buildVideoGenerationWaiter = async (params: {
    model: string;
    prompt: string;
    contentParts: any[];
}): Promise<{ ok: true; wait: (signal: AbortSignal) => Promise<WaitResult>; taskId: string } | { ok: false; error: any; status?: number }> => {
    const { model, prompt: rawPrompt, contentParts } = params;
    const links = findLinks(rawPrompt || '');
    const imgs = hasImageInMessages(contentParts || []);
    let prompt = rawPrompt || '';

    // Check for non-Doubao and non-Seedance models - route to Hugging Face
    if (!model.includes('doubao') && !model.includes('seedance')) {
        let apiKey: string | null = null;
        try {
            const keys = String(process.env.HUGGINGFACE_API_KEY).split(',').map((k: string) => k.trim()) || [];
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

                // Upload to storage; fallback to base64 URL on error
                let finalUrl: string;
                try {
                    if (!process.env.URL && !process.env.VERCEL_PROJECT_PRODUCTION_URL) throw new Error('No URL or VERCEL_PROJECT_PRODUCTION_URL configured');
                    const { uploadBlobToStorage } = await import('../shared/bucket.js');
                    const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
                    finalUrl = await uploadBlobToStorage(result, `vid_${timestamp}`);
                } catch (blobError) {
                    console.warn('Failed to upload video to storage, using base64:', blobError);
                    // Fallback to base64 conversion
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

    // Doubao Seedance support (direct Volcano Ark task API)
    if (model.includes('doubao-seedance')) {
        let apiKey: string | null = null;
        try {
            const keys = String(process.env.DOUBAO_API_KEY).split(',').map((k: string) => k.trim()) || [];
            if (keys.length > 0) { const idx = Math.floor(Math.random() * keys.length); apiKey = keys[idx] || null; }
        } catch { }
        if (!apiKey) return { ok: false, error: { code: 'no_api_key', message: 'Missing Doubao API key' }, status: 401 };
        const base = SUPPORTED_PROVIDERS.doubao.baseURL;

        const parsedFlags = parsePromptFlags(rawPrompt || '');
        const prompt = parsedFlags.prompt;

        const content: any[] = [{ type: 'text', text: prompt }];
        let modelId = '';
        if (model.includes('doubao-seedance-2.0-fast')) {
            modelId = 'doubao-seedance-2-0-fast-260128';
        } else if (model.includes('doubao-seedance-2.0')) {
            modelId = 'doubao-seedance-2-0-260128';
        } else if (model.includes('doubao-seedance-1.5')) {
            modelId = 'doubao-seedance-1-5-pro-251215';
        } else if (model.includes('doubao-seedance-1.0-pro')) {
            modelId = 'doubao-seedance-1-0-pro-250528';
        } else {
            if (links.length > 0 || imgs.has) modelId = 'doubao-seedance-1-0-lite-i2v-250428'; else modelId = 'doubao-seedance-1-0-lite-t2v-250428';
        }

        // Handle image frames (--first / --last or fallback to messages/links)
        const hasUploadFlag = prompt.toLowerCase().includes('/upload');
        let firstFrameUrl = parsedFlags.first;
        let lastFrameUrl = parsedFlags.last;

        if (firstFrameUrl) {
            if (firstFrameUrl.startsWith('data:') && hasUploadFlag && (process.env.URL || process.env.VERCEL_PROJECT_PRODUCTION_URL)) {
                try {
                    const { uploadBase64ToStorage } = await import('../shared/bucket.js');
                    const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
                    firstFrameUrl = await uploadBase64ToStorage(firstFrameUrl, `${timestamp}_first`);
                } catch (blobError) {
                    console.warn('Failed to upload first frame to blob store, using base64:', blobError);
                }
            }
            content.push({ type: 'image_url', image_url: { url: firstFrameUrl }, role: 'first_frame' });
        }

        if (lastFrameUrl) {
            if (lastFrameUrl.startsWith('data:') && hasUploadFlag && (process.env.URL || process.env.VERCEL_PROJECT_PRODUCTION_URL)) {
                try {
                    const { uploadBase64ToStorage } = await import('../shared/bucket.js');
                    const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
                    lastFrameUrl = await uploadBase64ToStorage(lastFrameUrl, `${timestamp}_last`);
                } catch (blobError) {
                    console.warn('Failed to upload last frame to blob store, using base64:', blobError);
                }
            }
            content.push({ type: 'image_url', image_url: { url: lastFrameUrl }, role: 'last_frame' });
        }

        // If neither --first nor --last was specified, fallback to checking user input images
        if (!parsedFlags.first && !parsedFlags.last) {
            if (imgs.has || links.length > 0) {
                let first = imgs.first || links[0] || '';
                if (first.startsWith('data:') && hasUploadFlag && (process.env.URL || process.env.VERCEL_PROJECT_PRODUCTION_URL)) {
                    try {
                        const { uploadBase64ToStorage } = await import('../shared/bucket.js');
                        const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
                        first = await uploadBase64ToStorage(first, `${timestamp}_first`);
                    } catch (blobError) {
                        console.warn('Failed to upload first frame to blob store, using base64:', blobError);
                    }
                }

                content.push({ type: 'image_url', image_url: { url: first }, role: 'first_frame' });

                const isRepeatMode = prompt.toLowerCase().includes('/repeat');
                const hasOnlyOneImage = !imgs.second && links.length <= 1;

                if (imgs.second || links.length > 1 || (isRepeatMode && hasOnlyOneImage)) {
                    let lastUrl = imgs.second || links[links.length - 1] || '';
                    if (isRepeatMode && hasOnlyOneImage) {
                        lastUrl = first;
                    }
                    if (lastUrl.startsWith('data:') && hasUploadFlag && (process.env.URL || process.env.VERCEL_PROJECT_PRODUCTION_URL)) {
                        try {
                            const { uploadBase64ToStorage } = await import('../shared/bucket.js');
                            const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
                            lastUrl = await uploadBase64ToStorage(lastUrl, `${timestamp}_last`);
                        } catch (blobError) {
                            console.warn('Failed to upload last frame to blob store, using base64:', blobError);
                        }
                    }
                    content.push({ type: 'image_url', image_url: { url: lastUrl }, role: 'last_frame' });
                }
            }
        }

        // Remove first image URL, repeat, and upload flags from clean prompt text
        let cleanPrompt = prompt.trim();
        if (!parsedFlags.first && !parsedFlags.last && (imgs.has || links.length > 0)) {
            cleanPrompt = cleanPrompt.replace(imgs.first || links[0] || '', '').trim();
        }
        cleanPrompt = cleanPrompt.replace(/\/repeat/gi, '').replace(/\/upload/gi, '').trim();
        content[0].text = cleanPrompt;

        // Push reference flags if any
        if (parsedFlags.reference_images) {
            for (const url of parsedFlags.reference_images) {
                content.push({
                    type: 'image_url',
                    image_url: { url },
                    role: 'reference_image'
                });
            }
        }
        if (parsedFlags.reference_videos) {
            for (const url of parsedFlags.reference_videos) {
                content.push({
                    type: 'video_url',
                    video_url: { url },
                    role: 'reference_video'
                });
            }
        }
        if (parsedFlags.reference_audios) {
            for (const url of parsedFlags.reference_audios) {
                content.push({
                    type: 'audio_url',
                    audio_url: { url },
                    role: 'reference_audio'
                });
            }
        }

        // Resolution default logic (1.2)
        let defaultResolution = '1080p';
        if (modelId === 'doubao-seedance-1-0-lite-t2v-250428' || modelId === 'doubao-seedance-2-0-fast-260128') {
            defaultResolution = '720p';
        }
        const finalResolution = parsedFlags.resolution || defaultResolution;

        // Duration default logic (1.3)
        let defaultDuration = 5;
        if (modelId.includes('2-0')) {
            defaultDuration = -1;
        }
        const finalDuration = parsedFlags.duration !== undefined ? parsedFlags.duration : defaultDuration;

        // Audio flag (1.5)
        let generate_audio: boolean | undefined = undefined;
        if (parsedFlags.audio !== undefined) {
            if (!modelId.includes('1-0')) {
                generate_audio = parsedFlags.audio;
            }
        }

        // Build Volcano Task request body
        const requestBody: any = {
            model: modelId,
            content,
            duration: finalDuration,
            resolution: finalResolution
        };

        if (parsedFlags.ratio !== undefined) {
            requestBody.ratio = parsedFlags.ratio;
        }
        if (parsedFlags.seed !== undefined) {
            requestBody.seed = parsedFlags.seed;
        }
        if (parsedFlags.camera_fixed !== undefined) {
            requestBody.camera_fixed = parsedFlags.camera_fixed;
        }
        if (generate_audio !== undefined) {
            requestBody.generate_audio = generate_audio;
        }
        if (parsedFlags.web_search) {
            requestBody.tools = [{ type: 'web_search' }];
        }

        try {
            const createRes = await fetch(`${base}/contents/generations/tasks`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${apiKey}`
                },
                body: JSON.stringify(requestBody)
            });
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

    // Gateway provider for seedance 2.0 (video/seedance-2.0) and seedance 2.0 fast using (video/seedance-2.0-fast) generateVideo
    if (model.includes('seedance-2.0') || model.includes('seedance-2.0-fast')) {
        let gatewayApiKey: string | null = null;
        try {
            const keys = String(process.env.GATEWAY_API_KEY).split(',').map((k: string) => k.trim()) || [];
            if (keys.length > 0) { const idx = Math.floor(Math.random() * keys.length); gatewayApiKey = keys[idx] || null; }
        } catch { }
        if (!gatewayApiKey) return { ok: false, error: { code: 'no_api_key', message: 'Missing Gateway API key' }, status: 401 };

        const parsedFlags = parsePromptFlags(rawPrompt || '');
        const prompt = parsedFlags.prompt;

        let sdkModelId = '';
        if (model.includes('seedance-2.0-fast')) {
            sdkModelId = 'bytedance/seedance-2.0-fast';
        } else {
            sdkModelId = 'bytedance/seedance-2.0';
        }

        const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
        const taskId = `bytedance_${timestamp}`;

        // Get images
        let firstFrameImage = parsedFlags.first;
        let lastFrameImage = parsedFlags.last;

        const hasUploadFlag = prompt.toLowerCase().includes('/upload');

        if (!firstFrameImage && !lastFrameImage) {
            if (imgs.has || links.length > 0) {
                firstFrameImage = imgs.first || links[0] || '';

                const isRepeatMode = prompt.toLowerCase().includes('/repeat');
                const hasOnlyOneImage = !imgs.second && links.length <= 1;

                if (imgs.second || links.length > 1 || (isRepeatMode && hasOnlyOneImage)) {
                    lastFrameImage = imgs.second || links[links.length - 1] || '';
                    if (isRepeatMode && hasOnlyOneImage) {
                        lastFrameImage = firstFrameImage;
                    }
                }
            }
        }

        // Clean prompt
        let cleanPrompt = prompt.trim();
        if (!parsedFlags.first && !parsedFlags.last && (imgs.has || links.length > 0)) {
            cleanPrompt = cleanPrompt.replace(imgs.first || links[0] || '', '').trim();
        }
        cleanPrompt = cleanPrompt.replace(/\/repeat/gi, '').replace(/\/upload/gi, '').trim();

        // Image mapping logic:
        // first frame or if only one reference image then pass as prompt.image
        // last frame as providerOptions.bytedance.lastFrameImage
        // multiple ref images as array in providerOptions.bytedance.referenceImages
        let promptImage: string | undefined = undefined;
        let refImages: string[] = [];

        if (firstFrameImage) {
            promptImage = firstFrameImage;
            refImages = parsedFlags.reference_images || [];
        } else if (parsedFlags.reference_images && parsedFlags.reference_images.length === 1) {
            promptImage = parsedFlags.reference_images[0];
        } else if (parsedFlags.reference_images && parsedFlags.reference_images.length > 1) {
            refImages = parsedFlags.reference_images;
        }

        // Duration default logic
        const finalDuration = parsedFlags.duration !== undefined ? parsedFlags.duration : -1;

        const wait = async (_signal: AbortSignal) => {
            try {
                // Upload promptImage / lastFrameImage / refImages to storage if base64 and /upload flag present
                if (hasUploadFlag && (process.env.URL || process.env.VERCEL_PROJECT_PRODUCTION_URL)) {
                    const { uploadBase64ToStorage } = await import('../shared/bucket.js');
                    if (promptImage && promptImage.startsWith('data:')) {
                        try {
                            const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
                            promptImage = await uploadBase64ToStorage(promptImage, `${timestamp}_first`);
                        } catch (err) {
                            console.warn('Failed to upload prompt image to storage:', err);
                        }
                    }
                    if (lastFrameImage && lastFrameImage.startsWith('data:')) {
                        try {
                            const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
                            lastFrameImage = await uploadBase64ToStorage(lastFrameImage, `${timestamp}_last`);
                        } catch (err) {
                            console.warn('Failed to upload last frame image to storage:', err);
                        }
                    }
                    for (let i = 0; i < refImages.length; i++) {
                        const img = refImages[i];
                        if (img && img.startsWith('data:')) {
                            try {
                                const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
                                const uploaded = await uploadBase64ToStorage(img, `${timestamp}_ref_${i}`);
                                refImages[i] = uploaded;
                            } catch (err) {
                                console.warn(`Failed to upload reference image ${i} to storage:`, err);
                            }
                        }
                    }
                }

                const gateway = createGateway({ apiKey: gatewayApiKey! });

                const options: any = {
                    model: gateway(sdkModelId),
                    prompt: {
                        text: cleanPrompt,
                        ...(promptImage && { image: promptImage })
                    },
                    duration: finalDuration,
                    providerOptions: {
                        bytedance: {
                            watermark: false,
                            ...(lastFrameImage && { lastFrameImage }),
                            ...(refImages.length > 0 && { referenceImages: refImages }),
                            ...(parsedFlags.reference_videos && parsedFlags.reference_videos.length > 0 && { referenceVideos: parsedFlags.reference_videos }),
                            ...(parsedFlags.reference_audios && parsedFlags.reference_audios.length > 0 && { referenceAudios: parsedFlags.reference_audios }),
                            ...(parsedFlags.audio !== undefined && { generateAudio: parsedFlags.audio }),
                            ...(parsedFlags.camera_fixed !== undefined && { cameraFixed: parsedFlags.camera_fixed }),
                        }
                    }
                };

                const generationResult = await generateVideo(options);
                const generatedVideo = generationResult.video;

                let videoBuffer: Buffer | null = null;
                if (generatedVideo?.uint8Array) {
                    videoBuffer = Buffer.from(generatedVideo.uint8Array);
                } else if (generatedVideo?.base64) {
                    videoBuffer = Buffer.from(generatedVideo.base64, 'base64');
                }

                if (!videoBuffer) {
                    return { ok: false, error: { code: 'generation_failed', message: 'No video data was returned from the generator.' } } as const;
                }

                const blob = new Blob([videoBuffer as any], { type: 'video/mp4' });
                let finalUrl: string;
                try {
                    if (!process.env.URL && !process.env.VERCEL_PROJECT_PRODUCTION_URL) throw new Error('No URL or VERCEL_PROJECT_PRODUCTION_URL configured');
                    const { uploadBlobToStorage } = await import('../shared/bucket.js');
                    const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
                    finalUrl = await uploadBlobToStorage(blob, `vid_${timestamp}`);
                } catch (blobError) {
                    console.warn('Failed to upload video to storage, using base64:', blobError);
                    const base64 = videoBuffer.toString('base64');
                    finalUrl = `data:video/mp4;base64,${base64}`;
                }

                const usage = { input_tokens: 0, output_tokens: 0, total_tokens: 0 };
                return { ok: true, text: toMarkdownVideo(finalUrl), usage, downloadLink: finalUrl, taskId } as const;
            } catch (e: any) {
                return { ok: false, error: { code: 'generation_failed', message: e?.message || 'Video generation via AI SDK failed' } } as const;
            }
        };

        return { ok: true, wait, taskId };
    }

    return { ok: false, error: { code: 'unsupported_model', message: `Model ${model} is not supported.` } };
}

export const handleVideoForChat = async (args: { model: string; messages: any[]; stream?: boolean; }): Promise<Response> => {
    const { model, messages, stream = false } = args;
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

    const waiter = await buildVideoGenerationWaiter({ model, prompt, contentParts: last.content || [] });
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

export const handleVideoForResponses = async (args: { model: string; messages: any[]; stream?: boolean; request_id: string; }): Promise<Response> => {
    const { model, messages, stream = false, request_id } = args;
    const now = Date.now();
    const last = lastUserPromptFromMessages(messages);
    let prompt = last.text || '';

    if (prompt.trim() === '/help') {
        const help = helpForVideo(model);
        const base = responsesBase(now, request_id, model, null, null, true, undefined, undefined, undefined, undefined);
        if (stream) return streamResponsesSingleText(base, help, `msg_${now}`, true);
        const response = { ...base, status: 'completed', output: [{ type: 'message', id: `msg_${now}`, status: 'completed', role: 'assistant', content: [{ type: 'output_text', text: help }] }], usage: { input_tokens: 0, output_tokens: 0, total_tokens: 0 } };
        return new Response(JSON.stringify(response), { headers: { 'Content-Type': 'application/json' } });
    }

    const waiter = await buildVideoGenerationWaiter({ model, prompt, contentParts: last.content || [] });
    if (!waiter.ok) {
        return new Response(JSON.stringify({ error: waiter.error }), { status: waiter.status || 400, headers: { 'Content-Type': 'application/json' } });
    }
    if (stream) {
        const baseObj = responsesBase(now, request_id, model, null, null, true, undefined, undefined, undefined, undefined);
        return streamResponsesGenerationElapsed({ baseObj, requestId: request_id, waitForResult: waiter.wait, taskId: waiter.taskId });
    }
    // Non-stream: wait for completion
    const res = await waiter.wait(new AbortController().signal);
    if (!res.ok) return new Response(JSON.stringify({ error: res.error }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    const baseObj = responsesBase(now, request_id, model, null, null, true, undefined, undefined, undefined, undefined);
    // For Responses endpoint, usage format is already correct: { input_tokens, output_tokens, total_tokens }
    const response = { ...baseObj, status: 'completed', output: [{ type: 'message', id: 'msg_' + Date.now(), status: 'completed', role: 'assistant', content: [{ type: 'output_text', text: res.text }] }], usage: res.usage ?? { input_tokens: 0, output_tokens: 0, total_tokens: 0 } };
    return new Response(JSON.stringify(response), { headers: { 'Content-Type': 'application/json' } });
}
