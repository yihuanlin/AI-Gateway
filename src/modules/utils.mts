export type WaitResult = { ok: true; text: string; usage?: { input_tokens: number; output_tokens: number; total_tokens: number }; downloadLink?: string; taskId?: string } | { ok: false; error: any };

export function findLinks(text: string): string[] {
  if (!text) return [];
  const urlRegex = /(https?:\/\/[^\s)]+)(?![^\(]*\))/g;
  const links: string[] = [];
  let m: RegExpExecArray | null;
  while ((m = urlRegex.exec(text)) !== null) {
    const u = m[1] as string;
    if (u) links.push(u);
  }
  return links;
}

export function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

export function hasImageInMessages(content: any): { has: boolean; first?: string | undefined; second?: string | undefined } {
  if (!Array.isArray(content)) return { has: false };
  const urls: string[] = [];
  for (const part of content) {
    if (!part) continue;
    if (part.type === 'input_image') {
      const u = (typeof part.image_url === 'string') ? part.image_url : (part.image_url?.url || part.url);
      if (u) urls.push(u);
    } else if (part.type === 'image') {
      const u = (typeof part.image === 'string') ? part.image : (part.image);
      if (u) urls.push(u);
    }
  }
  return { has: urls.length > 0, first: urls[0], second: urls[1] };
}

export function lastUserPromptFromMessages(messages: any[]): { text: string; content?: any[] } {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m?.role === 'user') {
      if (typeof m.content === 'string') return { text: m.content };
      if (Array.isArray(m.content)) {
        const parts = m.content;
        let text = '';
        for (let j = parts.length - 1; j >= 0; j--) {
          const p = parts[j];
          if (p?.type === 'text' && typeof p.text === 'string') { text = p.text; break; }
          if (typeof p === 'string') { text = p; break; }
        }
        return { text, content: parts };
      }
    }
  }
  return { text: '' };
}

export function responsesBase(createdAt: number, id: string, model: string, input: any, instructions: any, store: boolean, temperature: any, tool_choice: any, tools: any, top_p: any) {
  return {
    id,
    object: 'response',
    created_at: Math.floor(createdAt / 1000),
    status: 'in_progress',
    background: false,
    error: null,
    incomplete_details: null,
    input: typeof input === 'string' ? [{ type: 'input_text', text: input }] : (Array.isArray(input) ? input : (input ? [input] : [])),
    instructions: instructions ?? null,
    max_output_tokens: null,
    max_tool_calls: null,
    model,
    output: [],
    parallel_tool_calls: true,
    previous_response_id: null,
    prompt_cache_key: 'ai-gateway',
    reasoning: null,
    store,
    temperature: temperature ?? 1,
    text: { format: { type: 'text' }, verbosity: 'medium' },
    tool_choice: tool_choice || 'auto',
    tools: tools || [],
    top_p: top_p ?? 1,
    truncation: 'disabled',
    usage: null,
    user: null,
  } as any;
}

export function streamChatSingleText(model: string, text: string): Response {
  const now = Date.now();
  const created = Math.floor(now / 1000);
  const baseChunk: any = { id: `chatcmpl-${now}`, object: 'chat.completion.chunk', created, model, choices: [] };
  const enc = new TextEncoder();
  const rs = new ReadableStream({
    start(controller) {
      const chunk = { ...baseChunk, choices: [{ index: 0, delta: { content: text }, finish_reason: null }] };
      controller.enqueue(enc.encode(`data: ${JSON.stringify(chunk)}\n\n`));
      controller.enqueue(enc.encode('data: [DONE]\n\n'));
      controller.close();
    }
  });
  return new Response(rs, { headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' } });
}

export function streamResponsesSingleText(baseObj: any, messageText: string, textItemId?: string, includeInProgress: boolean = true): Response {
  const enc = new TextEncoder();
  let sequenceNumber = 0;
  const itemId = textItemId || `msg_${Date.now()}`;
  const rs = new ReadableStream({
    start(controller) {
      const emit = (obj: any) => controller.enqueue(enc.encode(`data: ${JSON.stringify(obj)}\n\n`));
      // created and optional in_progress
      emit({ type: 'response.created', sequence_number: sequenceNumber++, response: { ...baseObj } });
      if (includeInProgress) emit({ type: 'response.in_progress', sequence_number: sequenceNumber++, response: { ...baseObj } });
      // message item start
      emit({ type: 'response.output_item.added', sequence_number: sequenceNumber++, output_index: 0, item: { id: itemId, type: 'message', status: 'in_progress', role: 'assistant', content: [] } });
      emit({ type: 'response.content_part.added', sequence_number: sequenceNumber++, item_id: itemId, output_index: 0, content_index: 0, part: { type: 'output_text', text: '' } });
      // single delta
      emit({ type: 'response.output_text.delta', sequence_number: sequenceNumber++, item_id: itemId, output_index: 0, content_index: 0, delta: messageText });
      // close part and item
      emit({ type: 'response.content_part.done', sequence_number: sequenceNumber++, item_id: itemId, output_index: 0, content_index: 0, part: { type: 'output_text', text: messageText } });
      const finalItem = { id: itemId, type: 'message', status: 'completed', role: 'assistant', content: [{ type: 'output_text', text: messageText }] };
      emit({ type: 'response.output_item.done', sequence_number: sequenceNumber++, output_index: 0, item: finalItem });
      const completed = { ...baseObj, status: 'completed', output: [finalItem], usage: { input_tokens: 0, output_tokens: 0, total_tokens: 0 } };
      emit({ type: 'response.completed', sequence_number: sequenceNumber++, response: completed });
      controller.close();
    }
  });
  return new Response(rs, { headers: { 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' } });
}

type Usage = { input_tokens: number; output_tokens: number; total_tokens: number };
type GenResult = { ok: true; text: string; usage?: Usage; downloadLink?: string; taskId?: string } | { ok: false; error: any };

export function streamResponsesGenerationElapsed(params: {
  baseObj: any;
  requestId: string;
  waitForResult: (signal: AbortSignal) => Promise<GenResult>;
  reasoningId?: string;
  textItemId?: string;
  startOutputIndex?: number;
  taskId?: string;
  headers?: Headers;
}): Response {
  const { baseObj, requestId, waitForResult, headers } = params;
  const now = Date.now();
  const enc = new TextEncoder();
  let sequenceNumber = 0;
  const reasoningId = params.reasoningId || `rs_${now}`;
  const textItemId = params.textItemId || `msg_${now}`;
  const startIndex = params.startOutputIndex ?? 0;
  const taskId = params.taskId;

  const ac = new AbortController();

  const rs = new ReadableStream({
    async start(controller) {
      const emit = (obj: any) => controller.enqueue(enc.encode(`data: ${JSON.stringify(obj)}\n\n`));
      // created + in_progress
      emit({ type: 'response.created', sequence_number: sequenceNumber++, response: { ...baseObj } });
      emit({ type: 'response.in_progress', sequence_number: sequenceNumber++, response: { ...baseObj } });
      // reasoning item at startIndex
      emit({ type: 'response.output_item.added', sequence_number: sequenceNumber++, output_index: startIndex, item: { id: reasoningId, type: 'reasoning', summary: [] } });
      emit({ type: 'response.reasoning_summary_part.added', sequence_number: sequenceNumber++, item_id: reasoningId, output_index: startIndex, summary_index: 0, part: { type: 'summary_text', text: '' } });
      if (taskId) {
        emit({ type: 'response.reasoning_summary_text.delta', sequence_number: sequenceNumber++, item_id: reasoningId, output_index: startIndex, summary_index: 0, delta: `${taskId}\n` });
      }

      let done = false;
      const started = Date.now();
      const ticker = (async () => {
        try {
          while (!done) {
            await sleep(1000);
            const seconds = Math.floor((Date.now() - started) / 1000);
            emit({ type: 'response.reasoning_summary_text.delta', sequence_number: sequenceNumber++, item_id: reasoningId, output_index: startIndex, summary_index: 0, delta: `${seconds}s elapsed\n` });
          }
        } catch { }
      })();

      try {
        const result = await waitForResult(ac.signal);
        done = true;
        await ticker; // ensure ticker loop exits
        if (!result.ok) {
          emit({ type: 'response.failed', sequence_number: sequenceNumber++, response: { id: requestId, object: 'response', status: 'failed', error: result.error } });
          controller.close();
          return;
        }

        // Save to blob store if taskId and link are available
        if (taskId && result.downloadLink && headers) {
          try {
            const { getStoreWithConfig } = await import('../shared/store.mts');
            const store = getStoreWithConfig('responses');
            const timestamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, '');
            const mediaKey = `media_${timestamp}`;
            const mediaData = {
              id: taskId,
              downloadLink: result.downloadLink,
              generatedAt: new Date().toISOString(),
              text: result.text
            };
            await (store as any).set(mediaKey, JSON.stringify(mediaData));
          } catch (e) {
            console.error('Failed to save media data to blob store:', e);
          }
        }

        const finalReasoningText = taskId ?
          `${taskId}\n${Math.floor((Date.now() - started) / 1000)}s elapsed\n` :
          `${Math.floor((Date.now() - started) / 1000)}s elapsed\n`;
        const events = generationFinalizeEvents({ baseObj, reasoningId, textItemId, finalText: result.text, finalReasoningText, usage: result.usage ?? { input_tokens: 0, output_tokens: 0, total_tokens: 0 }, reasoningIndex: startIndex, messageIndex: startIndex + 1 });
        for (const ev of events) emit({ ...ev, sequence_number: sequenceNumber++ });
        controller.close();
      } catch (e: any) {
        done = true;
        emit({ type: 'response.failed', sequence_number: sequenceNumber++, response: { id: requestId, object: 'response', status: 'failed', error: { code: 'network_error', message: e?.message || 'stream failed' } } });
        controller.close();
      }
    }
  });

  return new Response(rs, { headers: { 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' } });
}

export function streamChatGenerationElapsed(model: string, waitForResult: (signal: AbortSignal) => Promise<GenResult>, taskId?: string): Response {
  const now = Date.now();
  const created = Math.floor(now / 1000);
  const baseChunk: any = { id: `chatcmpl-${now}`, object: 'chat.completion.chunk', created, model, choices: [] };
  const enc = new TextEncoder();
  const ac = new AbortController();

  const rs = new ReadableStream({
    async start(controller) {
      const started = Date.now();
      let tickerStopped = false;

      // Emit task ID first if provided
      if (taskId) {
        controller.enqueue(enc.encode(`data: ${JSON.stringify({ ...baseChunk, choices: [{ index: 0, delta: { reasoning_content: `${taskId}\n` }, finish_reason: null }] })}\n\n`));
      }

      const tick = async () => {
        try {
          while (!tickerStopped) {
            await sleep(1000);
            if (tickerStopped) break;
            const seconds = Math.floor((Date.now() - started) / 1000);
            controller.enqueue(enc.encode(`data: ${JSON.stringify({ ...baseChunk, choices: [{ index: 0, delta: { reasoning_content: `${seconds}s elapsed\n` }, finish_reason: null }] })}\n\n`));
          }
        } catch {
          // Controller already closed, ignore
        }
      };

      const ticker = tick();
      try {
        const res = await waitForResult(ac.signal);
        tickerStopped = true;
        await ticker; // wait for ticker to stop

        if (!res.ok) {
          controller.enqueue(enc.encode(`data: ${JSON.stringify({ ...baseChunk, choices: [{ index: 0, delta: { content: JSON.stringify({ error: res.error }) }, finish_reason: 'stop' }] })}\n\n`));
          controller.enqueue(enc.encode('data: [DONE]\n\n'));
          controller.close();
          return;
        }
        controller.enqueue(enc.encode(`data: ${JSON.stringify({ ...baseChunk, choices: [{ index: 0, delta: { content: res.text }, finish_reason: null }] })}\n\n`));
        controller.enqueue(enc.encode('data: [DONE]\n\n'));
        controller.close();
      } catch (e: any) {
        tickerStopped = true;
        controller.enqueue(enc.encode(`data: ${JSON.stringify({ ...baseChunk, choices: [{ index: 0, delta: { content: JSON.stringify({ error: { code: 'network_error', message: e?.message || 'fetch failed' } }) }, finish_reason: 'stop' }] })}\n\n`));
        controller.enqueue(enc.encode('data: [DONE]\n\n'));
        controller.close();
      }
    }
  });

  return new Response(rs, { headers: { 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' } });
}

function generationFinalizeEvents(params: {
  baseObj: any;
  reasoningId: string;
  textItemId: string;
  finalText: string;
  finalReasoningText: string;
  usage?: { input_tokens: number; output_tokens: number; total_tokens: number } | null;
  reasoningIndex?: number;
  messageIndex?: number;
}): any[] {
  const { baseObj, reasoningId, textItemId, finalText, finalReasoningText, usage } = params;
  const reasoningIndex = params.reasoningIndex ?? 0;
  const messageIndex = params.messageIndex ?? 1;

  const reasoningItemDone = { id: reasoningId, type: 'reasoning', summary: [{ type: 'summary_text', text: finalReasoningText }] };
  const finalMsgItem = { id: textItemId, type: 'message', status: 'completed', role: 'assistant', content: [{ type: 'output_text', text: finalText, annotations: [] }] };

  const events: any[] = [];
  // 1) Finalize reasoning summary and item first
  events.push({ type: 'response.reasoning_summary_text.done', item_id: reasoningId, output_index: reasoningIndex, summary_index: 0, text: finalReasoningText });
  events.push({ type: 'response.reasoning_summary_part.done', item_id: reasoningId, output_index: reasoningIndex, summary_index: 0, part: { type: 'summary_text', text: finalReasoningText } });
  events.push({ type: 'response.output_item.done', output_index: reasoningIndex, item: reasoningItemDone });

  // 2) Emit the assistant message with the generated media markdown
  events.push({ type: 'response.output_item.added', output_index: messageIndex, item: { id: textItemId, type: 'message', status: 'in_progress', role: 'assistant', content: [] } });
  events.push({ type: 'response.content_part.added', item_id: textItemId, output_index: messageIndex, content_index: 0, part: { type: 'output_text', text: '' } });
  events.push({ type: 'response.output_text.delta', item_id: textItemId, output_index: messageIndex, content_index: 0, delta: finalText });
  events.push({ type: 'response.content_part.done', item_id: textItemId, output_index: messageIndex, content_index: 0, part: { type: 'output_text', text: finalText, annotations: [] } });
  events.push({ type: 'response.output_item.done', output_index: messageIndex, item: finalMsgItem });

  // 3) Completed envelope with both outputs in order: reasoning then message
  const completed = { ...baseObj, status: 'completed', output: [reasoningItemDone, finalMsgItem], usage: usage ?? { input_tokens: 0, output_tokens: 0, total_tokens: 0 } };
  events.push({ type: 'response.completed', response: completed });

  return events;
}
