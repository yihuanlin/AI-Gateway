import { generateText, streamText, type GenerateTextResult } from 'ai';
import { createGateway } from '@ai-sdk/gateway';
import { NextRequest, NextResponse } from 'next/server';
import { openai } from '@ai-sdk/openai';
import { google } from '@ai-sdk/google';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

export async function OPTIONS(req: NextRequest) {
  return new NextResponse(null, {
    status: 204,
    headers: corsHeaders,
  });
}

function toOpenAIResponse(result: GenerateTextResult<any, any>, model: string) {
  const now = Math.floor(Date.now() / 1000);
  const choices = result.text
    ? [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: result.text,
          reasoning_content: result.reasoningText,
          tool_calls: result.toolCalls,
          metadata: {
            sources: result.sources
          }
        },
        finish_reason: result.finishReason,
      },
    ]
    : [];

  return {
    id: `chatcmpl-${now}`,
    object: 'chat.completion',
    created: now,
    model: model,
    choices: choices,
    usage: {
      prompt_tokens: result.usage.inputTokens,
      completion_tokens: result.usage.outputTokens,
      total_tokens: result.usage.totalTokens,
    },
  };
}

function toOpenAIStream(result: any, model: string) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const now = Math.floor(Date.now() / 1000);
      const chunkId = `chatcmpl-${now}`;

      for await (const part of result.fullStream) {
        switch (part.type) {
          case 'reasoning-delta': {
            const chunk = {
              id: chunkId,
              object: 'chat.completion.chunk',
              created: now,
              model: model,
              choices: [
                {
                  index: 0,
                  delta: { reasoning_content: part.text },
                  finish_reason: null,
                },
              ],
            };
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
            break;
          }
          case 'text-delta': {
            const chunk = {
              id: chunkId,
              object: 'chat.completion.chunk',
              created: now,
              model: model,
              choices: [
                {
                  index: 0,
                  delta: { content: part.text },
                  finish_reason: null,
                },
              ],
            };
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
            break;
          }
          case 'source': {
            const chunk = {
              id: chunkId,
              object: 'chat.completion.chunk',
              created: now,
              model: model,
              choices: [
                {
                  index: 0,
                  delta: {
                    role: 'assistant',
                    content: null,
                    metadata: {
                      sources: [part.source]
                    }
                  },
                  finish_reason: null,
                },
              ],
            };
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
            break;
          }
          case 'tool-call': {
            const chunk = {
              id: chunkId,
              object: 'chat.completion.chunk',
              created: now,
              model: model,
              choices: [
                {
                  index: 0,
                  delta: {
                    tool_calls: [
                      {
                        index: 0,
                        id: part.toolCallId,
                        type: 'function',
                        function: {
                          name: part.toolName,
                          arguments: JSON.stringify(part.args),
                        },
                      },
                    ],
                  },
                  finish_reason: null,
                },
              ],
            };
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
            break;
          }
          case 'tool-call-delta': {
            const chunk = {
              id: chunkId,
              object: 'chat.completion.chunk',
              created: now,
              model: model,
              choices: [
                {
                  index: 0,
                  delta: {
                    tool_calls: [
                      {
                        index: 0,
                        id: part.toolCallId,
                        type: 'function',
                        function: {
                          name: part.toolName,
                          arguments: part.argsTextDelta,
                        },
                      },
                    ],
                  },
                  finish_reason: null,
                },
              ],
            };
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
            break;
          }
          case 'finish': {
            const chunk = {
              id: chunkId,
              object: 'chat.completion.chunk',
              created: now,
              model: model,
              choices: [
                {
                  index: 0,
                  delta: {},
                  finish_reason: part.finishReason,
                },
              ],
              usage: {
                prompt_tokens: part.usage.inputTokens,
                completion_tokens: part.usage.outputTokens,
                total_tokens: part.usage.totalTokens,
              },
            };
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
            break;
          }
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
      ...corsHeaders,
    },
  });
}


export async function POST(req: NextRequest) {
  const authHeader = req.headers.get('Authorization');
  const apiKey = authHeader?.split(' ')[1];

  if (!apiKey) {
    return new NextResponse('Unauthorized', { status: 401, headers: corsHeaders });
  }

  let gateway;
  let useSearchGrounding = false;
  const { model, messages = [], tools, stream, temperature, top_p, max_completion_tokens, stop_sequences, seed, presence_penalty, frequency_penalty, tool_choice, providerOptions, reasoning_effort, thinking, extra_body } = await req.json();
  let aiSdkTools: Record<string, any> = {};
  if (tools && Array.isArray(tools)) {
    tools.forEach((tool: any) => {
      if (tool.type === 'function' && tool.function) {
        if (tool.function.name === 'googleSearch') {
          const lastMessage = messages[messages.length - 1];
          if (lastMessage && typeof lastMessage.content === 'string' && (lastMessage.content.includes('http://') || lastMessage.content.includes('https://'))) {
            aiSdkTools = {
              ...aiSdkTools,
              url_context: google.tools.urlContext({}),
            };
            useSearchGrounding = true;
          } else {
            aiSdkTools = {
              ...aiSdkTools,
              google_search: google.tools.googleSearch({}),
            };
          }
          return;
        }

        let clientParameters = tool.function.inputSchema || {};

        const finalParameters: Record<string, any> = {
          type: "object",
          properties: clientParameters.properties || clientParameters,
          required: clientParameters.required || [],
        };

        aiSdkTools[tool.function.name] = {
          description: tool.function.description,
          inputSchema: finalParameters,
        };
      }
    });
  }

  if (model.startsWith('openai')) {
    aiSdkTools = {
      ...aiSdkTools,
      web_search_preview: openai.tools.webSearchPreview({}),
    };
  }

  const apiKeys = apiKey.split(',').map(key => key.trim()) || [];
  let lastError: any;

  // Try each API key in sequence until one succeeds
  for (let i = 0; i < apiKeys.length; i++) {
    const currentApiKey = apiKeys[i];

    gateway = createGateway({
      apiKey: currentApiKey,
      baseURL: 'https://ai-gateway.vercel.sh/v1/ai',
    });

    try {
      for (const message of messages) {
        if (message.role === 'user' && Array.isArray(message.content)) {
          console.warn('Processing message content as array');
          message.content = await Promise.all(
            message.content.map(async (part: any) => {
              if (part.type === 'image_url') {
                const url = part.image_url.url;
                if (url.startsWith('data:')) {
                  const [mediaType, base64Data] = url.split(';base64,');
                  const mimeType = mediaType.split(':')[1];
                  const imageBuffer = Buffer.from(base64Data, 'base64');
                  return {
                    type: 'image',
                    image: new Uint8Array(imageBuffer),
                    mimeType: mimeType,
                  };
                } else {
                  const response = await fetch(url);
                  const imageBuffer = await response.arrayBuffer();
                  const mimeType = response.headers.get('content-type');
                  return {
                    type: 'image',
                    image: new Uint8Array(imageBuffer),
                    mimeType: mimeType,
                  };
                }
              }
              return part;
            }),
          );
        }
      }
      const commonOptions = {
        model: gateway(model),
        messages: messages,
        tools: aiSdkTools,
        temperature,
        top_p,
        max_completion_tokens,
        stop_sequences,
        seed,
        presence_penalty,
        frequency_penalty,
        tool_choice: tool_choice,
        providerOptions: providerOptions || {
          anthropic: {
            thinking: thinking || {
              type: "enabled",
              budgetTokens: 8000
            },
            cacheControl: {
              type: "ephemeral"
            }
          },
          openai: {
            reasoningEffort: reasoning_effort || "high",
            reasoningSummary: "auto"
          },
          xai: {
            searchParameters: {
              mode: "auto",
              returnCitations: true
            }
            ,
            ...(reasoning_effort && { reasoningEffort: reasoning_effort })
          },
          google: {
            useSearchGrounding: useSearchGrounding,
            ...(extra_body?.google?.thinking_config && { thinking_config: extra_body.google.thinking_config })
          }
        },
      };

      if (stream) {
        const result = streamText(commonOptions);
        return toOpenAIStream(result, model);
      } else {
        const result = await generateText(commonOptions);
        const openAIResponse = toOpenAIResponse(result, model);
        return NextResponse.json(openAIResponse, { headers: corsHeaders });
      }
    } catch (error: any) {
      console.error(`Error with API key ${i + 1}/${apiKeys.length}:`, error);
      lastError = error;

      if (i < apiKeys.length - 1) {
        continue;
      }

      break;
    }
  }

  console.error('All API keys failed. Last error:', lastError);

  let errorMessage = lastError.message || 'An unknown error occurred';
  let errorType = lastError.type;
  const statusCode = lastError.statusCode || 500;

  if (lastError.cause && lastError.cause.responseBody) {
    try {
      const body = JSON.parse(lastError.cause.responseBody);
      if (body.error) {
        errorMessage = body.error.message || errorMessage;
        errorType = body.error.type || errorType;
      }
    } catch (e) {
      // ignore parsing error
    }
  }

  const errorPayload = {
    error: {
      message: `All ${apiKeys.length} API key(s) failed. Last error: ${errorMessage}`,
      type: errorType,
      statusCode: statusCode,
    },
  };

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
          delta: { content: JSON.stringify(errorPayload) },
          finish_reason: 'stop',
        },
      ],
    };
    const errorStream = new ReadableStream({
      start(controller) {
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify(errorChunk)}\n\n`)
        );
        controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        controller.close();
      },
    });
    return new Response(errorStream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
        ...corsHeaders,
      },
      status: statusCode,
    });
  } else {
    return new NextResponse(JSON.stringify(errorPayload), {
      status: statusCode,
      headers: { 'Content-Type': 'application/json', ...corsHeaders },
    });
  }
}
