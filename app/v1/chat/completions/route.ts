import { generateText, streamText, type GenerateTextResult } from 'ai';
import { gateway } from '@ai-sdk/gateway';
import { NextRequest, NextResponse } from 'next/server';

// Helper function to convert Vercel AI SDK result to a non-streaming OpenAI format
function toOpenAIResponse(result: GenerateTextResult<any, any>, model: string) {
  const now = Math.floor(Date.now() / 1000);
  const step = result.steps[0];

  const choices = [];

  const message: {
    role: string;
    content: string | null;
    tool_calls?: Array<{
      id: string;
      type: string;
      function: {
        name: string;
        arguments: string;
      };
    }>;
  } = {
    role: 'assistant',
    content: '',
    tool_calls: [], // Initialize as empty array
  };

  let hasText = false;
  let hasToolCalls = false;

  for (const part of step.content) {
    if (part.type === 'text') {
      message.content = part.text;
      hasText = true;
    } else if (part.type === 'tool-call') {
      hasToolCalls = true;
      // Ensure tool_calls is an array before pushing
      if (!message.tool_calls) {
        message.tool_calls = [];
      }
      message.tool_calls.push({
        id: part.toolCallId,
        type: 'function',
        function: {
          name: part.toolName,
          arguments: JSON.stringify(part.input),
        },
      });
    }
  }

  if (!hasText || hasToolCalls) {
    message.content = null;
  }

  if (!hasToolCalls) {
    delete message.tool_calls;
  }

  choices.push({
    index: 0,
    message: message,
    finish_reason: step.finishReason,
  });

  return {
    id: `chatcmpl-${now}`,
    object: 'chat.completion',
    created: now,
    model: model,
    choices: choices,
    usage: {
      prompt_tokens: (step.usage as any).promptTokens,
      completion_tokens: (step.usage as any).completionTokens,
      total_tokens: (step.usage as any).promptTokens + (step.usage as any).completionTokens,
    },
  };
}

// A new, more robust helper to convert to an OpenAI-compatible stream
function toOpenAIStream(result: any, model: string) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const now = Math.floor(Date.now() / 1000);
      const chunkId = `chatcmpl-${now}`;
      let isFirstTextDelta = true;
      const toolCallStates = new Map<string, { name: string; args: string }>(); // To manage partial tool calls

      for await (const part of result) {
        const delta: { content?: string; role?: string; tool_calls?: any[] } = {};
        let finish_reason: string | null = null;
        let shouldSend = false;

        if (part.type === 'text-delta') {
          delta.content = part.textDelta;
          if (isFirstTextDelta) {
            delta.role = 'assistant';
            isFirstTextDelta = false;
          }
          shouldSend = true;
        } else if (part.type === 'tool-call-delta') {
          let state = toolCallStates.get(part.toolCallId);
          if (!state) {
            state = { name: part.toolName, args: '' };
            toolCallStates.set(part.toolCallId, state);
          }
          state.args += part.argsTextDelta;

          delta.tool_calls = [
            {
              index: 0,
              id: part.toolCallId,
              function: {
                name: part.toolName,
                arguments: part.argsTextDelta,
              },
            },
          ];
          shouldSend = true;
        } else if (part.type === 'finish') {
          finish_reason = part.finishReason;
          shouldSend = true;
        }

        if (shouldSend) {
          const chunk = {
            id: chunkId,
            object: 'chat.completion.chunk',
            created: now,
            model: model,
            choices: [
              {
                index: 0,
                delta: delta,
                finish_reason: finish_reason,
              },
            ],
          };
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
        }
      }

      controller.enqueue(encoder.encode('data: [DONE]\n\n'));
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    },
  });
}


export async function POST(req: NextRequest) {
  const authHeader = req.headers.get('Authorization');
  const AI_GATEWAY_API_KEY = process.env.AI_GATEWAY_API_KEY;

  if (!authHeader || !AI_GATEWAY_API_KEY || authHeader !== `Bearer ${AI_GATEWAY_API_KEY}`) {
    return new NextResponse('Unauthorized', { status: 401 });
  }

  const { model, messages, tools, stream } = await req.json();

  if (stream) {
    const result = await streamText({
      model: gateway(model),
      prompt: messages,
      tools,
    });
    return toOpenAIStream(result, model);
  } else {
    const result = await generateText({
      model: gateway(model),
      prompt: messages,
      tools,
    });
    const openAIResponse = toOpenAIResponse(result, model);
    return NextResponse.json(openAIResponse);
  }
}