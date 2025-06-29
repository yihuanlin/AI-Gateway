import {
  generateText, streamText, type GenerateTextResult,
  // experimental_createMCPClient as createMCPClient 
} from 'ai';
import { createGateway } from '@ai-sdk/gateway';
import { NextRequest, NextResponse } from 'next/server';

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
    tool_calls: [],
  };

  let hasText = false;
  let hasToolCalls = false;

  for (const part of step.content) {
    if (part.type === 'text') {
      message.content = part.text;
      hasText = true;
    } else if (part.type === 'tool-call') {
      hasToolCalls = true;
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

function toOpenAIStream(result: any, model: string) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const now = Math.floor(Date.now() / 1000);
      const chunkId = `chatcmpl-${now}`;
      let isFirstTextDelta = true;
      const toolCallStates = new Map<string, { name: string; args: string }>();

      for await (const part of result.fullStream) {
        const delta: { content?: string; role?: string; tool_calls?: any[]; reasoning_content?: string } = {};
        let finish_reason: string | null = null;
        let shouldSend = false;

        if (part.type === 'text-delta') {
          delta.content = part.textDelta;
          if (isFirstTextDelta) {
            delta.role = 'assistant';
            isFirstTextDelta = false;
          }
          shouldSend = true;
        } else if (part.type === 'text') {
          delta.content = part.text;
          if (isFirstTextDelta) {
            delta.role = 'assistant';
            isFirstTextDelta = false;
          }
          shouldSend = true;
        } else if (part.type === 'tool-call-delta') {
          let state = toolCallStates.get(part.toolCallId);
          const isFirstDeltaForToolCall = !state;
          if (isFirstDeltaForToolCall) {
            state = { name: part.toolName, args: '' };
            toolCallStates.set(part.toolCallId, state);
          }
          state!.args += part.argsTextDelta;

          const toolCall: { index: number; id?: string; function: { name?: string; arguments?: string } } = {
            index: 0,
            function: {
              arguments: part.argsTextDelta,
            },
          };

          if (isFirstDeltaForToolCall) {
            toolCall.id = part.toolCallId;
            toolCall.function.name = part.toolName;
          }

          delta.tool_calls = [toolCall];
          shouldSend = true;
        } else if (part.type === 'tool-call') {
          const toolCall: { index: number; id?: string; function: { name?: string; arguments?: string } } = {
            index: 0,
            id: part.toolCallId,
            function: {
              name: part.toolName,
              arguments: JSON.stringify(part.input),
            },
          };
          delta.tool_calls = [toolCall];
          shouldSend = true;
        } else if (part.type === 'reasoning') {
          delta.reasoning_content = part.text;
          shouldSend = true;
        } else if (part.type === 'finish') {
          finish_reason = part.finishReason;
          if (part.usage) {
            (delta as any).usage = {
              prompt_tokens: part.usage.promptTokens,
              completion_tokens: part.usage.completionTokens,
              total_tokens: part.usage.totalTokens,
            };
          }
          if (part.toolCalls) {
            (delta as any).tool_calls = part.toolCalls.map((tc: any) => ({
              id: tc.toolCallId,
              type: 'function',
              function: {
                name: tc.toolName,
                arguments: JSON.stringify(tc.args),
              },
            }));
          }
          if (part.toolResults) {
            (delta as any).tool_results = part.toolResults.map((tr: any) => ({
              tool_call_id: tr.toolCallId,
              tool_name: tr.toolName,
              result: tr.result,
              is_error: tr.isError,
            }));
          }
          if (part.sources) {
            (delta as any).sources = part.sources;
          }
          if (part.files) {
            (delta as any).files = part.files;
          }
          if (part.warnings) {
            (delta as any).warnings = part.warnings;
          }
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
  const apiKey = authHeader?.split(' ')[1];

  if (!apiKey) {
    return new NextResponse('Unauthorized', { status: 401 });
  }

  let gateway;

  const { model, messages = [], tools, stream, temperature, topP, maxTokens, stopSequences, seed, presencePenalty, frequencyPenalty, tool_choice } = await req.json();

  const validRoles = ['user', 'assistant', 'system', 'tool'];
  const filteredMessages = (messages || []).filter((msg: any) => validRoles.includes(msg.role));

  let aiSdkTools: Record<string, any> = {};
  let shouldFetchMcpTools = false;

  if (tools && Array.isArray(tools)) {
    tools.forEach((tool: any) => {
      if (tool.type === 'function' && tool.function) {
        if (tool.function.name === 'googleSearch') {
          shouldFetchMcpTools = true;
          return;
        }

        let clientParameters = tool.function.parameters || {};

        const finalParameters: Record<string, any> = {
          type: "object",
          properties: clientParameters.properties || clientParameters,
          required: clientParameters.required || [],
        };

        aiSdkTools[tool.function.name] = {
          description: tool.function.description,
          parameters: finalParameters,
        };
      }
    });
  }

  // let mcpClientTools: Record<string, any> | undefined;
  // const AI_GATEWAY_API_KEYS = process.env.AI_GATEWAY_API_KEY?.split(',').map(key => key.trim());
  // const isAuthorizedGatewayKey = AI_GATEWAY_API_KEYS && apiKey && AI_GATEWAY_API_KEYS.includes(apiKey);

  // if (shouldFetchMcpTools && isAuthorizedGatewayKey && process.env.SSE_URL) {
  //   const mcpClient = await createMCPClient({
  //     transport: {
  //       type: 'sse',
  //       url: process.env.SSE_URL,
  //     },
  //   });
  //   mcpClientTools = await mcpClient.tools();
  //   aiSdkTools = { ...aiSdkTools, ...mcpClientTools };
  // }

  gateway = createGateway({
    apiKey: apiKey,
    baseURL: 'https://ai-gateway.vercel.sh/v1/ai',
  });

  try {
    const commonOptions = {
      model: gateway(model),
      messages: filteredMessages,
      tools: aiSdkTools,
      temperature,
      topP,
      maxTokens,
      stopSequences,
      seed,
      presencePenalty,
      frequencyPenalty,
      toolChoice: tool_choice,
      experimental_continueSteps: true,
    };

    if (stream) {
      const result = await streamText(commonOptions);
      return toOpenAIStream(result, model);
    } else {
      const result = await generateText(commonOptions);
      const openAIResponse = toOpenAIResponse(result, model);
      return NextResponse.json(openAIResponse);
    }
  } catch (error: any) {
    console.error('Error processing request:', error);
    const errorMessage = error.message || 'An unknown error occurred';
    const statusCode = error.statusCode || 500;

    if (stream) {
      const encoder = new TextEncoder();
      const errorChunk = {
        id: `chatcmpl-error-${Date.now()}`,
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model: model || 'unknown',
        choices: [
          {
            index: 0,
            delta: { content: `Error: ${errorMessage}` },
            finish_reason: 'stop',
          },
        ],
      };
      const errorStream = new ReadableStream({
        start(controller) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(errorChunk)}\n\n`));
          controller.enqueue(encoder.encode('data: [DONE]\n\n'));
          controller.close();
        },
      });
      return new Response(errorStream, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          Connection: 'keep-alive',
        },
        status: statusCode,
      });
    } else {
      return new NextResponse(JSON.stringify({ error: errorMessage }), {
        status: statusCode,
        headers: { 'Content-Type': 'application/json' },
      });
    }
  }
}