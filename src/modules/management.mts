import { getStoreWithConfig } from '../shared/store.mts';
import { responsesBase, streamChatSingleText, streamResponsesSingleText } from './utils.mts';

async function forceRefreshCopilotToken(headers: Headers, isPasswordAuth: boolean = false): Promise<{ ok: true; message: string } | { ok: false; message: string }> {
  const { SUPPORTED_PROVIDERS, getProviderKeys } = await import('../shared/providers.mts');
  try {
    // Resolve Copilot API key from headers or environment (when password auth is enabled)
    const providerKeys = await getProviderKeys(headers as any, null, isPasswordAuth);
    const copilotKeys = providerKeys?.copilot || [];
    const apiKey = copilotKeys[0] || (isPasswordAuth ? (process as any).env?.COPILOT_API_KEY : null);
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

function toConversationMarkdown(stored: any, key?: string): string {
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

function lastUserTextFromMessages(messages: any[]): string {
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

function lastUserTextFromResponses(input: any): string {
  if (typeof input === 'string') return input;
  if (Array.isArray(input)) {
    for (let i = input.length - 1; i >= 0; i--) {
      const m = input[i];
      if (m?.role === 'user') {
        if (typeof m.content === 'string') return m.content;
        if (Array.isArray(m.content)) {
          for (let j = m.content.length - 1; j >= 0; j--) {
            const p = m.content[j];
            if ((p?.type === 'input_text' || p?.type === 'text') && typeof p.text === 'string') return p.text;
            if (typeof p === 'string') return p;
          }
        }
      }
    }
  } else if (input && input.type === 'input_text') {
    return input.text || '';
  }
  return '';
}

export async function handleAdminForChat(args: { messages: any[]; headers: Headers; model: string; stream?: boolean; isPasswordAuth?: boolean }): Promise<Response> {
  const { messages, headers, model, stream = false, isPasswordAuth } = args;
  if (!isPasswordAuth) {
    return new Response(JSON.stringify({ error: { message: 'Unauthorized' } }), { status: 401, headers: { 'Content-Type': 'application/json' } });
  }
  const now = Math.floor(Date.now() / 1000);
  let text = lastUserTextFromMessages(messages).trim();

  const store = getStoreWithConfig('responses');

  if (/^\/help$/.test(text)) {
    const help = 'Commands: `refresh` (Copilot token) | `list` (responses) | `list [prefix]` | `ls` (== `list all`) | `delete all` | `delete [id]` | `rm -f [fileKey]` (delete file in storage) | `[id]` to view.';
    if (stream) {
      return streamChatSingleText(model, help);
    }
    const payload = { id: `chatcmpl-${now}`, object: 'chat.completion', created: now, model, choices: [{ index: 0, message: { role: 'assistant', content: help } }], usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } };
    return new Response(JSON.stringify(payload), { headers: { 'Content-Type': 'application/json' } });
  }

  if (/^refresh$/i.test(text)) {
    const result = await forceRefreshCopilotToken(headers, !!isPasswordAuth);
    const out = result.message;
    if (stream) return streamChatSingleText(model, out);
    const payload = { id: `chatcmpl-${now}`, object: 'chat.completion', created: now, model, choices: [{ index: 0, message: { role: 'assistant', content: out } }], usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } } as any;
    return new Response(JSON.stringify(payload), { headers: { 'Content-Type': 'application/json' } });
  }

  if (/^delete\s+all$|^deleteall$|^rm\s+-rf$/i.test(text)) {
    // Delete all
    try {
      if (stream) {
        let deleted = 0;
        try {
          const listResult: any = await (store as any).list();
          let blobs: any[] = [];
          if (listResult && 'blobs' in listResult && Array.isArray(listResult.blobs)) blobs = listResult.blobs; else {
            for await (const item of listResult) { if (item.blobs) blobs.push(...item.blobs); }
          }
          await Promise.all(blobs.map((b: any) => (store as any).delete(b.key)));
          deleted = blobs.length;
        } catch { }
        return streamChatSingleText(model, `Deleted ${deleted} item(s).`);
      }
      // Non-stream path
      const listResult: any = await (store as any).list();
      let blobs: any[] = [];
      if (listResult && 'blobs' in listResult && Array.isArray(listResult.blobs)) blobs = listResult.blobs; else {
        for await (const item of listResult) { if (item.blobs) blobs.push(...item.blobs); }
      }
      await Promise.all(blobs.map((b: any) => (store as any).delete(b.key)));
      const msg = `Deleted ${blobs.length} item(s).`;
      const payload = { id: `chatcmpl-${now}`, object: 'chat.completion', created: now, model, choices: [{ index: 0, message: { role: 'assistant', content: msg } }], usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } };
      return new Response(JSON.stringify(payload), { headers: { 'Content-Type': 'application/json' } });
    } catch (e: any) {
      return new Response(JSON.stringify({ error: { message: e?.message || 'Delete all failed' } }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    }
  }

  // Delete a file item: rm -f <key> or delete file <key>
  if (/^(?:delete|rm)\s+(?:-f|file)\s+\S+$/i.test(text)) {
    const parts = text.split(/\s+/);
    const key = parts[2];
    try {
      const filesStore = getStoreWithConfig('files');
      const existing = await (filesStore as any).getWithMetadata(key, { type: 'blob' });
      if (!existing) {
        const msg = `File not found: ${key}`;
        if (stream) return streamChatSingleText(model, msg);
        const payload = { id: `chatcmpl-${now}`, object: 'chat.completion', created: now, model, choices: [{ index: 0, message: { role: 'assistant', content: msg } }], usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } } as any;
        return new Response(JSON.stringify(payload), { headers: { 'Content-Type': 'application/json' } });
      }
      await (filesStore as any).delete(key);
      const msg = `Deleted file: ${key}.`;
      if (stream) return streamChatSingleText(model, msg);
      const payload = { id: `chatcmpl-${now}`, object: 'chat.completion', created: now, model, choices: [{ index: 0, message: { role: 'assistant', content: msg } }], usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } } as any;
      return new Response(JSON.stringify(payload), { headers: { 'Content-Type': 'application/json' } });
    } catch (e: any) {
      const msg = `Failed to delete file: ${e?.message || 'Unknown error'}`;
      if (stream) return streamChatSingleText(model, msg);
      return new Response(JSON.stringify({ error: { message: msg } }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    }
  }

  if (/^(?:delete|rm)\s+\S+$/i.test(text)) {
    const id = text.split(/\s+/)[1];
    try {
      if (stream) {
        const existing = await (store as any).get(id, { type: 'json' });
        if (!existing) return streamChatSingleText(model, 'Response not found');
        await (store as any).delete(id);
        return streamChatSingleText(model, `Deleted ${id}.`);
      }
      const existing = await (store as any).get(id, { type: 'json' });
      if (!existing) return new Response(JSON.stringify({ error: { message: 'Response not found' } }), { status: 404, headers: { 'Content-Type': 'application/json' } });
      await (store as any).delete(id);
      const msg = `Deleted ${id}.`;
      const payload = { id: `chatcmpl-${now}`, object: 'chat.completion', created: now, model, choices: [{ index: 0, message: { role: 'assistant', content: msg } }], usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } };
      return new Response(JSON.stringify(payload), { headers: { 'Content-Type': 'application/json' } });
    } catch (e: any) {
      return new Response(JSON.stringify({ error: { message: e?.message || 'Delete failed' } }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    }
  }

  if (/^(?:list|ls)(?:\s+(.+))?$/i.test(text)) {
    const match = text.match(/^(?:list|ls)(?:\s+(.+))?$/i);
    const prefix = match?.[1]?.trim();

    try {
      let listOptions: any = {};
      if (prefix === 'all' || text === 'ls') {
        // List all items
      } else if (prefix === 'm') {
        listOptions.prefix = 'media';
      } else if (prefix === 'c') {
        listOptions.prefix = 'chatcmpl';
      } else if (prefix) {
        listOptions.prefix = prefix;
      } else {
        listOptions.prefix = 'resp'; // Default to responses
      }

      const listResult: any = await (store as any).list(listOptions);
      let blobs: any[] = [];
      if (listResult && 'blobs' in listResult && Array.isArray(listResult.blobs)) blobs = listResult.blobs; else {
        for await (const item of listResult) { if (item.blobs) blobs.push(...item.blobs); }
      }
      const ids = blobs.map((b: any) => b.key);
      const md = ids.length > 0 ? ids.map((id: string) => `- ${id}`).join('\n') : 'No items found.';
      if (stream) {
        return streamChatSingleText(model, md);
      }
      const payload = { id: `chatcmpl-${now}`, object: 'chat.completion', created: now, model, choices: [{ index: 0, message: { role: 'assistant', content: md } }], usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } };
      return new Response(JSON.stringify(payload), { headers: { 'Content-Type': 'application/json' } });
    } catch (e: any) {
      return new Response(JSON.stringify({ error: { message: e?.message || 'List failed' } }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    }
  }

  // Upload command: user message with parts containing files
  if (/^upload$|^\/upload$/i.test(text) && process.env.URL) {
    const { uploadBase64ToStorage, uploadBlobToStorage } = await import('../shared/bucket.mts');
    // Find last user message with file parts
    const lastUser = (() => {
      for (let i = messages.length - 1; i >= 0; i--) {
        const m = messages[i];
        if (m?.role === 'user' && Array.isArray(m.content)) return m;
      }
      return null;
    })();
    if (!lastUser) {
      const payload = { id: `chatcmpl-${now}`, object: 'chat.completion', created: now, model, choices: [{ index: 0, message: { role: 'assistant', content: 'No file found in your message.' } }], usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } } as any;
      return new Response(JSON.stringify(payload), { headers: { 'Content-Type': 'application/json' } });
    }
    const uploads: string[] = [];
    for (const part of lastUser.content) {
      if (part?.type === 'file') {
        try {
          const mediaType = part.mediaType || 'application/octet-stream';
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
    const message = uploads.length > 0 ? uploads.map(u => `Uploaded: ${u}`).join('\n') : 'No valid file parts to upload.';
    if (stream) return streamChatSingleText(model, message);
    const payload = { id: `chatcmpl-${now}`, object: 'chat.completion', created: now, model, choices: [{ index: 0, message: { role: 'assistant', content: message } }], usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } } as any;
    return new Response(JSON.stringify(payload), { headers: { 'Content-Type': 'application/json' } });
  }

  // Treat as ID
  const id = text;
  try {
    if (stream) {
      const existing = await (store as any).get(id, { type: 'json' });
      if (!existing) return streamChatSingleText(model, 'Item not found');
      const md = toConversationMarkdown(existing, id);
      return streamChatSingleText(model, md);
    }
    const existing = await (store as any).get(id, { type: 'json' });
    if (!existing) return new Response(JSON.stringify({ error: { message: 'Item not found' } }), { status: 404, headers: { 'Content-Type': 'application/json' } });
    const md = toConversationMarkdown(existing, id);
    const payload = { id: `chatcmpl-${now}`, object: 'chat.completion', created: now, model, choices: [{ index: 0, message: { role: 'assistant', content: md } }], usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } };
    return new Response(JSON.stringify(payload), { headers: { 'Content-Type': 'application/json' } });
  } catch (e: any) {
    return new Response(JSON.stringify({ error: { message: e?.message || 'Fetch failed' } }), { status: 500, headers: { 'Content-Type': 'application/json' } });
  }
}

export async function handleAdminForResponses(args: { messages: any[]; headers: Headers; model: string; request_id: string; store?: boolean; stream?: boolean, isPasswordAuth?: boolean }): Promise<Response> {
  const { messages, headers, model, request_id, store = false, stream = false, isPasswordAuth = false } = args;
  if (!isPasswordAuth) {
    return new Response(JSON.stringify({ error: { message: 'Unauthorized' } }), { status: 401, headers: { 'Content-Type': 'application/json' } });
  }
  let text = lastUserTextFromMessages(messages).trim();
  const responseStore = getStoreWithConfig('responses');

  const now = Date.now();
  const baseObj = responsesBase(now, request_id, model, null, null, store, undefined, undefined, undefined, undefined);

  const buildCompleted = (messageText: string) => {
    const finalItem = { id: textItemId, type: 'message', status: 'completed', role: 'assistant', content: [{ type: 'output_text', text: messageText }] };
    return { ...baseObj, status: 'completed', output: [finalItem], usage: { input_tokens: 0, output_tokens: 0, total_tokens: 0 } } as any;
  };

  const textItemId = `msg_${now}`;
  const streamTextOnce = (messageText: string) => streamResponsesSingleText(baseObj, messageText, textItemId, true);

  if (/^\/help$/.test(text)) {
    const msg = 'Commands: `refresh` (Copilot token) | `list` (responses) | `list [prefix]` | `ls` (== `list all`) | `delete all` | `delete [id]` | `rm -f [fileKey]` (delete file in storage) | `[id]` to view.';
    if (!stream) return new Response(JSON.stringify(buildCompleted(msg)), { headers: { 'Content-Type': 'application/json' } });
    return streamTextOnce(msg);
  }

  if (/^refresh$/i.test(text)) {
    const result = await forceRefreshCopilotToken(headers, !!isPasswordAuth);
    const msg = result.message;
    if (!stream) return new Response(JSON.stringify(buildCompleted(msg)), { headers: { 'Content-Type': 'application/json' } });
    return streamTextOnce(msg);
  }

  if (/^delete\s+all$|^deleteall$|^rm\s+-rf$/i.test(text)) {
    try {
      const listResult: any = await (responseStore as any).list();
      let blobs: any[] = [];
      if (listResult && 'blobs' in listResult && Array.isArray(listResult.blobs)) blobs = listResult.blobs; else {
        for await (const item of listResult) { if (item.blobs) blobs.push(...item.blobs); }
      }
      await Promise.all(blobs.map((b: any) => (responseStore as any).delete(b.key)));
      const msg = `Deleted ${blobs.length} item(s).`;
      if (!stream) return new Response(JSON.stringify(buildCompleted(msg)), { headers: { 'Content-Type': 'application/json' } });
      return streamTextOnce(msg);
    } catch (e: any) {
      return new Response(JSON.stringify({ error: { message: e?.message || 'Delete all failed' } }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    }
  }

  // Delete a file item: rm -f <key> or delete file <key>
  if (/^(?:delete|rm)\s+(?:-f|file)\s+\S+$/i.test(text)) {
    const parts = text.split(/\s+/);
    const key = parts[2];
    try {
      const filesStore = getStoreWithConfig('files');
      const existing = await (filesStore as any).getWithMetadata(key, { type: 'blob' });
      if (!existing) {
        const msg = `File not found: ${key}`;
        if (!stream) return new Response(JSON.stringify(buildCompleted(msg)), { headers: { 'Content-Type': 'application/json' } });
        return streamTextOnce(msg);
      }
      await (filesStore as any).delete(key);
      const msg = `Deleted file: ${key}.`;
      if (!stream) return new Response(JSON.stringify(buildCompleted(msg)), { headers: { 'Content-Type': 'application/json' } });
      return streamTextOnce(msg);
    } catch (e: any) {
      const msg = `Failed to delete file: ${e?.message || 'Unknown error'}`;
      return new Response(JSON.stringify({ error: { message: msg } }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    }
  }

  if (/^(?:delete|rm)\s+\S+$/i.test(text)) {
    const id = text.split(/\s+/)[1];
    try {
      const existing = await (responseStore as any).get(id, { type: 'json' });
      if (!existing) return new Response(JSON.stringify({ error: { message: 'Response not found' } }), { status: 404, headers: { 'Content-Type': 'application/json' } });
      await (responseStore as any).delete(id);
      const msg = `Deleted ${id}.`;
      if (!stream) return new Response(JSON.stringify(buildCompleted(msg)), { headers: { 'Content-Type': 'application/json' } });
      return streamTextOnce(msg);
    } catch (e: any) {
      return new Response(JSON.stringify({ error: { message: e?.message || 'Delete failed' } }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    }
  }

  if (/^(?:list|ls)(?:\s+(.+))?$/i.test(text)) {
    const match = text.match(/^(?:list|ls)(?:\s+(.+))?$/i);
    const prefix = match?.[1]?.trim();

    try {
      let listOptions: any = {};
      if (prefix === 'all' || text === 'ls') {
        // List all items
      } else if (prefix === 'm') {
        listOptions.prefix = 'media';
      } else if (prefix === 'c') {
        listOptions.prefix = 'chatcmpl';
      } else if (prefix) {
        listOptions.prefix = prefix;
      } else {
        listOptions.prefix = 'resp'; // Default to responses
      }

      const listResult: any = await (responseStore as any).list(listOptions);
      let blobs: any[] = [];
      if (listResult && 'blobs' in listResult && Array.isArray(listResult.blobs)) blobs = listResult.blobs; else {
        for await (const item of listResult) { if (item.blobs) blobs.push(...item.blobs); }
      }
      const ids = blobs.map((b: any) => b.key);
      const md = ids.length > 0 ? ids.map((id: string) => `- ${id}`).join('\n') : 'No items found.';
      if (!stream) return new Response(JSON.stringify(buildCompleted(md)), { headers: { 'Content-Type': 'application/json' } });
      return streamTextOnce(md);
    } catch (e: any) {
      return new Response(JSON.stringify({ error: { message: e?.message || 'List failed' } }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    }
  }

  // Upload command
  if (/^upload$|^\/upload$/i.test(text) && process.env.URL) {
    const { uploadBase64ToStorage, uploadBlobToStorage } = await import('../shared/bucket.mts');
    // Find last user message with file parts
    const lastUser = (() => {
      for (let i = messages.length - 1; i >= 0; i--) {
        const m = messages[i];
        if (m?.role === 'user' && Array.isArray(m.content)) return m;
      }
      return null;
    })();
    const textItemId = `msg_${now}`;
    if (!lastUser) {
      return stream ? streamResponsesSingleText(baseObj, 'No file found in your message.', textItemId, true) : new Response(JSON.stringify(buildCompleted('No file found in your message.')), { headers: { 'Content-Type': 'application/json' } });
    }
    const uploads: string[] = [];
    for (const part of lastUser.content) {
      if (part?.type === 'file') {
        try {
          const mediaType = part.mediaType || 'application/octet-stream';
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
    const messageText = uploads.length > 0 ? uploads.map(u => `Uploaded: ${u}`).join('\n') : 'No valid file parts to upload.';
    if (!stream) return new Response(JSON.stringify(buildCompleted(messageText)), { headers: { 'Content-Type': 'application/json' } });
    return streamResponsesSingleText(baseObj, messageText, textItemId, true);
  }

  // Treat as ID
  try {
    const id = text;
    const existing = await (responseStore as any).get(id, { type: 'json' });
    if (!existing) return new Response(JSON.stringify({ error: { message: 'Item not found' } }), { status: 404, headers: { 'Content-Type': 'application/json' } });
    const md = toConversationMarkdown(existing, id);
    if (!stream) return new Response(JSON.stringify(buildCompleted(md)), { headers: { 'Content-Type': 'application/json' } });
    return streamTextOnce(md);
  } catch (e: any) {
    return new Response(JSON.stringify({ error: { message: e?.message || 'Fetch failed' } }), { status: 500, headers: { 'Content-Type': 'application/json' } });
  }
}

export async function listResponsesHttp(c: any) {
  const authHeader = c.req.header('Authorization');
  const apiKey = authHeader?.split(' ')[1];
  const envPassword = process.env.PASSWORD;
  const isPasswordAuth = !!(envPassword && apiKey && envPassword.trim() === apiKey.trim());
  if (!isPasswordAuth) return c.text('Unauthorized', 401);

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

export async function deleteAllResponsesHttp(c: any) {
  const authHeader = c.req.header('Authorization');
  const apiKey = authHeader?.split(' ')[1];
  const envPassword = process.env.PASSWORD;
  const isPasswordAuth = !!(envPassword && apiKey && envPassword.trim() === apiKey.trim());
  if (!isPasswordAuth) return c.text('Unauthorized', 401);

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

export async function deleteResponseHttp(c: any) {
  const authHeader = c.req.header('Authorization');
  const apiKey = authHeader?.split(' ')[1];
  const envPassword = process.env.PASSWORD;
  const isPasswordAuth = !!(envPassword && apiKey && envPassword.trim() === apiKey.trim());
  if (!isPasswordAuth) return c.text('Unauthorized', 401);

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

export async function getResponseHttp(c: any) {
  const authHeader = c.req.header('Authorization');
  const apiKey = authHeader?.split(' ')[1];
  const envPassword = process.env.PASSWORD;
  const isPasswordAuth = !!(envPassword && apiKey && envPassword.trim() === apiKey.trim());
  if (!isPasswordAuth) return c.text('Unauthorized', 401);

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
