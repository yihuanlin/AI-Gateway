import { getStoreWithConfig } from '../shared/store.mts';
import { responsesBase, streamChatSingleText, streamResponsesSingleText } from './utils.mts';

const handleAdminRequest = async (args: {
  text: string;
  messages?: any[];
}): Promise<string> => {
  let { text } = args;
  const { messages = [] } = args;
  text = (text || '').trim();

  const responsesStore = getStoreWithConfig('responses');
  const filesStore = getStoreWithConfig('files');

  // Help
  if (/^\/help$/i.test(text)) {
    return 'Commands: `refresh` (Copilot token) | `list` (responses) | `list [prefix]` (`-r`/`r` responses, `-m`/`m` media, `-c`/`c` chat, `-f`/`f`/`file` files) | `ls` (== `list all`) | `delete all` | `delete [id]` | `rm -f [fileKey]` (delete single file) | `rm -f` (delete all files) | `upload [url?]` (from last user files or URL) | `[id]` to view.';
  }

  // Refresh Copilot token
  if (/^refresh$/i.test(text)) {
    const result = await forceRefreshCopilotToken();
    return result.message;
  }

  // Delete all (responses store)
  if (/^delete\s+all$|^deleteall$|^rm\s+-rf$/i.test(text)) {
    const listResult: any = await (responsesStore as any).list();
    let blobs: any[] = [];
    if (listResult && 'blobs' in listResult && Array.isArray(listResult.blobs)) blobs = listResult.blobs; else {
      for await (const item of listResult) { if (item.blobs) blobs.push(...item.blobs); }
    }
    await Promise.all(blobs.map((b: any) => (responsesStore as any).delete(b.key)));
    return `Deleted ${blobs.length} item(s).`;
  }

  // Delete files: rm -f <key>, delete file <key>, or rm -f (all files)
  {
    const m = text.match(/^(?:delete|rm)\s+(?:-f|file)(?:\s+(\S+))?$/i);
    if (m) {
      const key = m[1];
      if (key) {
        const existing = await (filesStore as any).getWithMetadata(key, { type: 'blob' });
        if (!existing) return `File not found: ${key}`;
        await (filesStore as any).delete(key);
        return `Deleted file: ${key}.`;
      }

      // Delete all files when no key provided
      const listResult: any = await (filesStore as any).list({});
      let blobs: any[] = [];
      if (listResult && 'blobs' in listResult && Array.isArray(listResult.blobs)) blobs = listResult.blobs; else {
        for await (const item of listResult) { if (item.blobs) blobs.push(...item.blobs); }
      }
      if (blobs.length === 0) return 'No files found to delete.';
      await Promise.all(blobs.map((b: any) => (filesStore as any).delete(b.key)));
      return `Deleted ${blobs.length} file(s).`;
    }
  }

  // Delete a response: delete <id> / rm <id>
  {
    const m = text.match(/^(?:delete|rm)\s+(\S+)$/i);
    if (m && m[1]) {
      const id = m[1];
      const existing = await (responsesStore as any).get(id, { type: 'json' });
      if (!existing) return 'Response not found';
      await (responsesStore as any).delete(id);
      return `Deleted ${id}.`;
    }
  }

  // List
  if (/^(?:list|ls)(?:\s+(.+))?$/i.test(text)) {
    const match = text.match(/^(?:list|ls)(?:\s+(.+))?$/i);
    const prefix = match?.[1]?.trim();
    // If files prefix
    if (prefix && ['-f', 'f', 'file'].includes(prefix)) {
      try {
        const listResult: any = await (filesStore as any).list({});
        let blobs: any[] = [];
        if (listResult && 'blobs' in listResult && Array.isArray(listResult.blobs)) blobs = listResult.blobs; else {
          for await (const item of listResult) { if (item.blobs) blobs.push(...item.blobs); }
        }
        const ids = blobs.map((b: any) => b.key);
        return ids.length > 0 ? ids.map((id: string) => `- ${id}`).join('\n') : 'No items found.';
      } catch (e: any) {
        return `List files failed: ${e?.message || 'Unknown error'}`;
      }
    }

    let listOptions: any = {};
    if (prefix === 'r' || prefix === '-r') listOptions.prefix = 'resp';
    else if (prefix === 'm' || prefix === '-m') listOptions.prefix = 'media';
    else if (prefix === 'c' || prefix === '-c') listOptions.prefix = 'chatcmpl';
    else if (prefix && prefix !== 'all') listOptions.prefix = prefix;

    const listResult: any = await (responsesStore as any).list(listOptions);
    let blobs: any[] = [];
    if (listResult && 'blobs' in listResult && Array.isArray(listResult.blobs)) blobs = listResult.blobs; else {
      for await (const item of listResult) { if (item.blobs) blobs.push(...item.blobs); }
    }
    const ids = blobs.map((b: any) => b.key);
    return ids.length > 0 ? ids.map((id: string) => `- ${id}`).join('\n') : 'No items found.';
  }

  // Upload files from last user message or via URL
  const uploadMatch = text.match(/^\/?upload(?:\s+(https?:\/\/\S+))?$/i);
  if (uploadMatch) {
    const urlFromCommand = uploadMatch[1]?.trim();
    // Find last user message with file parts
    const lastUser = (() => {
      for (let i = messages.length - 1; i >= 0; i--) {
        const m = messages[i];
        if (m?.role === 'user' && Array.isArray(m.content)) return m;
      }
      return null;
    })();
    if (!lastUser && !urlFromCommand) return 'No file found in your message.';
    const uploads: string[] = [];
    try {
      const { uploadBase64ToStorage, uploadBlobToStorage, buildPublicUrlForKey } = await import('../shared/bucket.mts');
      if (lastUser) {
        for (const part of lastUser.content) {
          if (part?.type === 'file') {
            try {
              const mediaType = part.mediaType || 'application/pdf';
              if (typeof part.data === 'string') {
                const url = await uploadBase64ToStorage(part.data);
                uploads.push(url);
              } else if (part.data && typeof Blob !== 'undefined') {
                const data = part.data as ArrayBuffer | Uint8Array;
                const blob = new Blob([data as any], { type: mediaType });
                const url = await uploadBlobToStorage(blob);
                uploads.push(url);
              }
            } catch { }
          }
        }
      }

      if (uploads.length === 0 && urlFromCommand) {
        const remoteUrl = urlFromCommand;
        try {
          const response = await fetch(remoteUrl);
          if (!response.ok) {
            const bodyText = await response.text().catch(() => '');
            throw new Error(`Download failed: ${response.status} ${response.statusText}${bodyText ? ` - ${bodyText}` : ''}`);
          }
          const contentType = response.headers.get('content-type') || 'application/octet-stream';
          if (typeof Blob === 'undefined') throw new Error('Blob is not available in this runtime.');
          const buffer = await response.arrayBuffer();
          const blob = new Blob([buffer], { type: contentType });
          const fileKey = deriveFilenameFromLink(remoteUrl, extensionFromContentType(contentType));
          await (filesStore as any).set(fileKey, blob, { metadata: { contentType, sourceUrl: remoteUrl } });
          const url = buildPublicUrlForKey(fileKey);
          uploads.push(url);
        } catch (err: any) {
          return `Upload failed: ${err?.message || 'Unable to download provided URL.'}`;
        }
      }
    } catch (e: any) {
      return `Upload failed: ${e?.message || 'Unknown error'}`;
    }
    return uploads.length > 0 ? uploads.map(u => `Uploaded: ${u}`).join('\n') : 'No valid file parts to upload.';
  }

  // View by ID from responses store
  try {
    const id = text;
    const existing = await (responsesStore as any).get(id, { type: 'json' });
    if (!existing) return 'Item not found';
    return toConversationMarkdown(existing, id);
  } catch (e: any) {
    return `Fetch failed: ${e?.message || 'Unknown error'}`;
  }
}

const forceRefreshCopilotToken = async (): Promise<{ ok: true; message: string } | { ok: false; message: string }> => {
  const { SUPPORTED_PROVIDERS, getProviderKeys } = await import('../shared/providers.mts');
  try {
    const providerKeys = await getProviderKeys();
    const copilotKeys = providerKeys?.copilot || [];
    const randomIndex = Math.floor(Math.random() * copilotKeys.length);
    const apiKey = copilotKeys[randomIndex];
    if (!apiKey) {
      return { ok: false, message: 'Missing Copilot API key. Provide x-copilot-api-key header or set COPILOT_API_KEY in env.' };
    }

    const tokenURL = SUPPORTED_PROVIDERS.copilot.tokenURL;
    const resp = await fetch(tokenURL, {
      method: 'GET',
      headers: { 'Authorization': `Token ${apiKey}`, 'Accept': 'application/json' },
    });
    if (!resp.ok) {
      const errText = await resp.text().catch(() => '');
      return { ok: false, message: `Failed to fetch Copilot token: ${resp.status} ${resp.statusText}${errText ? ` - ${errText}` : ''}` };
    }
    const data = await resp.json() as any;
    const token = data?.token;
    if (!token) {
      return { ok: false, message: 'Token response missing token field.' };
    }

    const now = Date.now();
    const expires = now + data.refresh_in * 1000;

    const store = getStoreWithConfig('copilot-tokens');
    await (store as any).set('token', token, { metadata: { expiration: expires } });

    const expiresStr = new Date(expires).toISOString();
    return { ok: true, message: `Refreshed Copilot token. Expires at ${expiresStr}.` };
  } catch (e: any) {
    return { ok: false, message: e?.message || 'Unexpected error while refreshing Copilot token.' };
  }
}

const toConversationMarkdown = (stored: any, key?: string): string => {
  if (!stored) return 'No content found.';

  // Check if this is a media entry (key starts with media_)
  if (key && key.startsWith('media_')) {
    try {
      const mediaData = typeof stored === 'string' ? JSON.parse(stored) : stored;
      const { id, downloadLink, generatedAt, text } = mediaData;
      let result = `**Media ID:** ${id}\n`;
      if (generatedAt) result += `**Generated at:** ${new Date(generatedAt).toLocaleString()}\n`;
      if (downloadLink) result += `**Download:** [Media Link](${downloadLink})\n`;
      if (text) result += `**Preview:** ${text}\n`;
      return result;
    } catch {
      // If not valid JSON, return raw content
      return typeof stored === 'string' ? stored : JSON.stringify(stored);
    }
  }

  if (key && (key.startsWith('resp_') || key.startsWith('chatcmpl-'))) {
    if (!Array.isArray(stored.messages)) return 'No conversation found.';
    const lines: string[] = [];
    for (const m of stored.messages) {
      const role = (m.role || '').toLowerCase();
      let header = '';
      if (role === 'system') header = 'System';
      else if (role === 'assistant') header = 'Assistant';
      else header = 'User';
      const content = typeof m.content === 'string' ? m.content : (Array.isArray(m.content) ? m.content.map((p: any) => (p?.text || p?.content || '')).filter(Boolean).join('\n') : '');
      lines.push(`### ${header}\n${content}`);
    }
    return lines.join('\n\n');
  }

  // For other keys, return raw content as string
  return typeof stored === 'string' ? stored : JSON.stringify(stored);
}

const lastUserTextFromMessages = (messages: any[]): string => {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m?.role === 'user') {
      if (typeof m.content === 'string') return m.content;
      if (Array.isArray(m.content)) {
        for (let j = m.content.length - 1; j >= 0; j--) {
          const p = m.content[j];
          if (p?.type === 'text' && typeof p.text === 'string') return p.text;
          if (typeof p === 'string') return p;
        }
      }
    }
  }
  return '';
}

const extensionFromContentType = (contentType?: string): string => {
  if (!contentType) return 'bin';
  const main = contentType.split(';')[0]?.trim() || '';
  if (!main.includes('/')) return main || 'bin';
  let subtype = main.split('/')[1] || '';
  if (!subtype && main.startsWith('text/')) return 'txt';
  if (subtype === 'plain') return 'txt';
  if (subtype.includes('+')) subtype = subtype.split('+').pop() || subtype;
  if (subtype.includes('.')) subtype = subtype.split('.').pop() || subtype;
  return subtype || 'bin';
}

const deriveFilenameFromLink = (link: string, fallbackExt: string = 'bin'): string => {
  const safeFallback = `downloaded_${Date.now()}`;
  try {
    const urlObj = new URL(link);
    const segments = urlObj.pathname.split('/').filter(Boolean);
    let candidate = segments.pop() || safeFallback;
    candidate = decodeURIComponent(candidate);
    candidate = candidate.replace(/[^a-zA-Z0-9._-]/g, '_');
    if (!candidate) candidate = safeFallback;
    if (!/\.[A-Za-z0-9]+$/.test(candidate)) candidate = `${candidate}.${fallbackExt}`;
    return candidate;
  } catch {
    return `${safeFallback}.${fallbackExt}`;
  }
}

export const handleAdminForChat = async (args: { messages: any[]; stream?: boolean; model: string }): Promise<Response> => {
  const { messages, stream = false, model } = args;
  const now = Math.floor(Date.now() / 1000);
  const text = lastUserTextFromMessages(messages).trim();
  const md = await handleAdminRequest({ text, messages });
  if (stream) return streamChatSingleText(model, md);
  const payload = { id: `chatcmpl-${now}`, object: 'chat.completion', created: now, model, choices: [{ index: 0, message: { role: 'assistant', content: md } }], usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } } as any;
  return new Response(JSON.stringify(payload), { headers: { 'Content-Type': 'application/json' } });
}

export const handleAdminForResponses = async (args: { messages: any[]; model: string; request_id: string; store?: boolean; stream?: boolean }): Promise<Response> => {
  const { messages, model, request_id, store = false, stream = false } = args;
  const now = Date.now();
  const baseObj = responsesBase(now, request_id, model, null, null, store, undefined, undefined, undefined, undefined);
  const textItemId = `msg_${now}`;
  const text = lastUserTextFromMessages(messages).trim();
  const md = await handleAdminRequest({ text, messages });
  if (!stream) {
    const finalItem = { id: textItemId, type: 'message', status: 'completed', role: 'assistant', content: [{ type: 'output_text', text: md }] };
    const completed = { ...baseObj, status: 'completed', output: [finalItem], usage: { input_tokens: 0, output_tokens: 0, total_tokens: 0 } } as any;
    return new Response(JSON.stringify(completed), { headers: { 'Content-Type': 'application/json' } });
  }
  return streamResponsesSingleText(baseObj, md, textItemId, true);
}

export const listResponsesHttp = async (c: any) => {
  const authHeader = c.req.header('Authorization')?.split(' ')[1] || null;
  const envPassword = process.env.PASSWORD;
  if (!(envPassword && authHeader && envPassword.trim() === authHeader.trim())) return c.text('Unauthorized', 401);

  const prefix = c.req.query('prefix') || '';
  const limit = parseInt(c.req.query('limit') || '20');

  try {
    const headers: Record<string, string> = {};
    c.req.raw.headers.forEach((value: string, key: string) => { headers[key.toLowerCase().replace(/-/g, '_')] = value; });
    const store = getStoreWithConfig('responses');
    const listOptions: any = { ...(prefix && { prefix }) };
    try {
      const listResult: any = await (store as any).list(listOptions);
      let blobs: any[] = [];
      if (listResult && 'blobs' in listResult && Array.isArray((listResult as any).blobs)) {
        blobs = (listResult as any).blobs;
      } else {
        for await (const item of listResult as any) { if (item.blobs) blobs.push(...item.blobs); }
      }
      const responseList = { object: 'list', data: blobs.slice(0, limit).map((b: any) => ({ id: b.key, key: b.key })), has_more: blobs.length > limit };
      return c.json(responseList);
    } catch (listError: any) {
      return c.json({ object: 'list', data: [], has_more: false, error: 'List operation not fully supported' });
    }
  } catch (error: any) {
    return c.json({ error: { message: 'Failed to list responses', type: 'server_error' } }, 500);
  }
}

export const deleteAllResponsesHttp = async (c: any) => {
  const authHeader = c.req.header('Authorization')?.split(' ')[1] || null;
  const envPassword = process.env.PASSWORD;
  if (!(envPassword && authHeader && envPassword.trim() === authHeader.trim())) return c.text('Unauthorized', 401);

  try {
    const headers: Record<string, string> = {};
    c.req.raw.headers.forEach((value: string, key: string) => { headers[key.toLowerCase().replace(/-/g, '_')] = value; });
    const store = getStoreWithConfig('responses');
    try {
      const listResult: any = await (store as any).list();
      let blobs: any[] = [];
      if (listResult && 'blobs' in listResult && Array.isArray((listResult as any).blobs)) blobs = (listResult as any).blobs; else {
        for await (const item of listResult as any) { if (item.blobs) blobs.push(...item.blobs); }
      }
      await Promise.all(blobs.map((b: any) => (store as any).delete(b.key)));
      return c.json({ message: `Deleted ${blobs.length} item(s)`, deleted_count: blobs.length });
    } catch (e: any) {
      return c.json({ error: { message: 'Failed to list responses for deletion', type: 'server_error' } }, 500);
    }
  } catch (error: any) {
    return c.json({ error: { message: 'Failed to delete all responses', type: 'server_error' } }, 500);
  }
}

export const deleteResponseHttp = async (c: any) => {
  const authHeader = c.req.header('Authorization')?.split(' ')[1] || null;
  const envPassword = process.env.PASSWORD;
  if (!(envPassword && authHeader && envPassword.trim() === authHeader.trim())) return c.text('Unauthorized', 401);

  const responseId = c.req.param('response_id');
  try {
    const headers: Record<string, string> = {};
    c.req.raw.headers.forEach((value: string, key: string) => { headers[key.toLowerCase().replace(/-/g, '_')] = value; });
    const store = getStoreWithConfig('responses');
    const existing: any = await (store as any).get(responseId, { type: 'json' });
    if (!existing) return c.json({ error: { message: `Response with ID '${responseId}' not found.`, type: 'invalid_request_error', code: 'response_not_found' } }, 404);
    await (store as any).delete(responseId);
    return c.json({ id: responseId, object: 'response', deleted: true });
  } catch (error: any) {
    return c.json({ error: { message: 'Failed to delete response', type: 'server_error' } }, 500);
  }
}

export const getResponseHttp = async (c: any) => {
  const authHeader = c.req.header('Authorization')?.split(' ')[1] || null;
  const envPassword = process.env.PASSWORD;
  if (!(envPassword && authHeader && envPassword.trim() === authHeader.trim())) return c.text('Unauthorized', 401);

  const responseId = c.req.param('response_id');
  const stream = c.req.query('stream') === 'true';

  try {
    const headers: Record<string, string> = {};
    c.req.raw.headers.forEach((value: string, key: string) => { headers[key.toLowerCase().replace(/-/g, '_')] = value; });
    const store = getStoreWithConfig('responses');
    const storedResponse: any = await (store as any).get(responseId, { type: 'json' });
    if (!storedResponse) return c.json({ error: { message: `Response with ID '${responseId}' not found.`, type: 'invalid_request_error', code: 'response_not_found' } }, 404);

    const output: any[] = [];
    if (storedResponse.messages && Array.isArray(storedResponse.messages)) {
      for (const message of storedResponse.messages) {
        const outputMessage: any = { type: 'message', id: `msg_${Date.now()}`, status: 'completed', role: message.role, content: [] };
        if (typeof message.content === 'string') {
          outputMessage.content.push({ type: message.role === 'assistant' ? 'output_text' : 'input_text', text: message.content, annotations: [] });
        } else if (Array.isArray(message.content)) {
          outputMessage.content = message.content.map((part: any) => {
            if (part.type === 'text') return { type: message.role === 'assistant' ? 'output_text' : 'input_text', text: part.text, annotations: [] };
            if (part.type === 'image') return { type: 'input_image', image_url: { url: part.image }, ...(part.mediaType && { media_type: part.mediaType }) };
            if (part.type === 'file') return { type: 'input_file', data: part.data, media_type: part.mediaType };
            return part;
          });
        }
        output.push(outputMessage);
      }
    }

    const response = { id: responseId, object: 'response', status: 'completed', output, store: true } as any;

    if (stream) {
      const enc = new TextEncoder();
      const streamResponse = new ReadableStream({
        start(controller) {
          let sequenceNumber = 0;
          const emit = (obj: any) => controller.enqueue(enc.encode(`data: ${JSON.stringify(obj)}\n\n`));

          // Start
          const base = { ...response, status: 'in_progress' };
          emit({ type: 'response.created', sequence_number: sequenceNumber++, response: base });
          emit({ type: 'response.in_progress', sequence_number: sequenceNumber++, response: base });

          // If there is at least one assistant message with text, stream it
          const firstMsgIndex = response.output.findIndex((x: any) => x.type === 'message' && x.role === 'assistant');
          if (firstMsgIndex >= 0) {
            const msg = response.output[firstMsgIndex];
            const text = Array.isArray(msg.content)
              ? (msg.content.find((p: any) => p.type === 'output_text')?.text || '')
              : (typeof msg.content === 'string' ? msg.content : '');
            const itemId = msg.id || `msg_${Date.now()}`;
            const outputIndex = firstMsgIndex;
            emit({ type: 'response.output_item.added', sequence_number: sequenceNumber++, output_index: outputIndex, item: { id: itemId, type: 'message', status: 'in_progress', role: 'assistant', content: [] } });
            emit({ type: 'response.content_part.added', sequence_number: sequenceNumber++, item_id: itemId, output_index: outputIndex, content_index: 0, part: { type: 'output_text', text: '' } });
            // Early flush delta
            emit({ type: 'response.output_text.delta', sequence_number: sequenceNumber++, item_id: itemId, output_index: outputIndex, content_index: 0, delta: text ? text : '…' });
            emit({ type: 'response.content_part.done', sequence_number: sequenceNumber++, item_id: itemId, output_index: outputIndex, content_index: 0 });
            const finalItem = { id: itemId, type: 'message', status: 'completed', role: 'assistant', content: [{ type: 'output_text', text }] };
            emit({ type: 'response.output_item.done', sequence_number: sequenceNumber++, output_index: outputIndex, item: finalItem });
          }

          // Completed
          emit({ type: 'response.completed', sequence_number: sequenceNumber++, response });
          controller.enqueue(enc.encode('data: [DONE]\n\n'));
          controller.close();
        }
      });
      return new Response(streamResponse, { headers: { 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' } });
    }

    return c.json(response);
  } catch (error: any) {
    return c.json({ error: { message: 'Failed to retrieve response', type: 'server_error' } }, 500);
  }
}
