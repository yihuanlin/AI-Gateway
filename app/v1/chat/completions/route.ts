import { generateText, stepCountIs, streamText, tool, type GenerateTextResult } from 'ai';
import { createGateway } from '@ai-sdk/gateway';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { NextRequest, NextResponse } from 'next/server';
import { openai } from '@ai-sdk/openai';
import { google } from '@ai-sdk/google';
import { z } from 'zod';

const pythonExecutorTool = tool({
  description: 'Execute Python code remotely via a secure Python execution API. Installed packages include: numpy, pandas.',
  inputSchema: z.object({
    code: z.string().describe('Python code to execute remotely.'),
  }),
  execute: async ({ code }: { code: string }) => {
    console.log(`Executing remote Python code: ${code.substring(0, 100)}...`);
    try {
      if (!pythonUrl) {
        return { error: 'python_url header is not set' };
      }
      if (!pythonApiKey) {
        return { error: 'python_api_key header is not set' };
      }

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s

      const response = await fetch(pythonUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'text/plain',
          'Authorization': `Bearer ${pythonApiKey}`,
        },
        body: code,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const text = await response.text();
      let data: any;
      try {
        data = JSON.parse(text);
      } catch {
        return { error: `Invalid JSON from Python server: ${text.slice(0, 500)}` };
      }

      if (!response.ok || data.error) {
        return {
          success: false,
          error: data?.error || `Python server error (${response.status})`,
          output: data?.output ?? '',
          result: data?.result ?? null,
          status: response.status,
        };
      }

      return {
        success: true,
        output: data.output ?? '',
        ...(data.result !== undefined && { result: data.result }),
      };
    } catch (error: any) {
      const message = error?.name === 'AbortError' ? 'Request to Python server timed out' : (error?.message || 'Unknown error');
      return { success: false, error: message };
    }
  },
});

const tavilySearchTool = tool({
  description: 'Search the web using Tavily API to get current information and relevant results',
  inputSchema: z.object({
    query: z.string().describe('The search query to find information about'),
    max_results: z.number().optional().describe('Maximum number of results to return (default: 5, max: 20)'),
    include_domains: z.array(z.string()).optional().describe('List of domains to include in the search'),
    exclude_domains: z.array(z.string()).optional().describe('List of domains to exclude from the search'),
  }),
  execute: async ({ query, max_results, include_domains, exclude_domains }: {
    query: string;
    max_results?: number;
    include_domains?: string[];
    exclude_domains?: string[];
  }) => {
    console.log(`Tavily search executed with query: ${query}, max_results: ${max_results}, include_domains: ${include_domains}, exclude_domains: ${exclude_domains}`);
    try {
      if (!tavilyApiKey) {
        return {
          error: 'TAVILY_API_KEY environment variable is not set'
        };
      }

      const maxResults = max_results || 5;
      const apiKeys = tavilyApiKey.split(',').map(key => key.trim());
      let lastError: any;

      for (let i = 0; i < apiKeys.length; i++) {
        const currentApiKey = apiKeys[i];

        try {
          const searchPayload = {
            query,
            max_results: Math.min(maxResults, 20),
            include_answer: true,
            include_images: false,
            include_raw_content: false,
            ...(include_domains && { include_domains }),
            ...(exclude_domains && { exclude_domains })
          };

          const response = await fetch('https://api.tavily.com/search', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${currentApiKey}`
            },
            body: JSON.stringify(searchPayload)
          });

          if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Tavily API error (${response.status}): ${errorText}`);
          }
          const data = await response.json();
          return {
            query,
            answer: data.answer || '',
            results: data.results?.map((result: any) => ({
              title: result.title,
              url: result.url,
              content: result.content,
              score: result.score
            })) || [],
            images: data.images || [],
            follow_up_questions: data.follow_up_questions || [],
            search_depth: data.search_depth || 'basic'
          };

        } catch (error: any) {
          console.error(`Error with Tavily API key ${i + 1}/${apiKeys.length}:`, error.message);
          lastError = error;
          if (i < apiKeys.length - 1) {
            continue;
          }

          break;
        }
      }

      return {
        error: `All ${apiKeys.length} Tavily API key(s) failed. Last error: ${lastError?.message || 'Unknown error'}`
      };

    } catch (error: any) {
      return {
        error: `Tavily search failed: ${error.message || 'Unknown error'}`
      };
    }
  }
});

const jinaReaderTool = tool({
  description: 'Fetch and extract clean content from web pages using Jina Reader API',
  inputSchema: z.object({
    url: z.string().describe('The URL of the webpage to fetch content from'),
    format: z.enum(['text', 'markdown', 'json']).optional().describe('Output format (default: text)'),
    include_links: z.boolean().optional().describe('Whether to include links in the output (default: false)'),
    include_images: z.boolean().optional().describe('Whether to include image descriptions (default: false)'),
  }),
  execute: async ({ url, format = 'text', include_links = false, include_images = false }: {
    url: string;
    format?: 'text' | 'markdown' | 'json';
    include_links?: boolean;
    include_images?: boolean;
  }) => {
    console.log(`Jina Reader fetching content from: ${url}, format: ${format}`);
    try {
      try {
        new URL(url);
      } catch {
        return {
          error: 'Invalid URL provided'
        };
      }
      const jinaUrl = new URL(`https://r.jina.ai/${url}`);
      if (format !== 'text') {
        jinaUrl.searchParams.set('format', format);
      }
      if (include_links) {
        jinaUrl.searchParams.set('links', 'true');
      }
      if (include_images) {
        jinaUrl.searchParams.set('images', 'true');
      }

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      const headers: Record<string, string> = {
        'Accept': format === 'json' ? 'application/json' : 'text/plain',
        'User-Agent': 'Vercel-AI-Gateway/1.0'
      };

      if (jinaApiKey) {
        headers['Authorization'] = `Bearer ${jinaApiKey}`;
      }

      const response = await fetch(jinaUrl.toString(), {
        method: 'GET',
        headers,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Jina Reader API error (${response.status}): ${errorText}`);
      }

      const content = await response.text();

      if (format === 'json') {
        try {
          const jsonData = JSON.parse(content);
          return {
            url,
            format,
            success: true,
            data: jsonData
          };
        } catch {
          return {
            url,
            format,
            success: true,
            content: content
          };
        }
      }

      return {
        url,
        format,
        success: true,
        content: content,
        length: content.length
      };

    } catch (error: any) {
      return {
        url,
        error: `Failed to fetch content: ${error.message || 'Unknown error'}`,
        success: false
      };
    }
  },
});

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': '*',
};

// Supported custom providers configuration
const SUPPORTED_PROVIDERS = {
  cerebras: {
    name: 'cerebras',
    baseURL: 'https://api.cerebras.ai/v1',
  },
  groq: {
    name: 'groq',
    baseURL: 'https://api.groq.com/openai/v1',
  },
  gemini: {
    name: 'gemini',
    baseURL: 'https://generativelanguage.googleapis.com/v1beta/openai',
  },
  doubao: {
    name: 'doubao',
    baseURL: 'https://ark.cn-beijing.volces.com/api/v3',
  },
  modelscope: {
    name: 'modelscope',
    baseURL: 'https://api-inference.modelscope.cn/v1',
  },
  github: {
    name: 'github',
    baseURL: 'https://models.inference.ai.azure.com',
  },
  openrouter: {
    name: 'openrouter',
    baseURL: 'https://openrouter.ai/api/v1',
  },
  nvidia: {
    name: 'nvidia',
    baseURL: 'https://integrate.api.nvidia.com/v1',
  },
  mistral: {
    name: 'mistral',
    baseURL: 'https://api.mistral.ai/v1',
  },
  cohere: {
    name: 'cohere',
    baseURL: 'https://api.cohere.ai/compatibility/v1',
  },
  // morph: {
  //   name: 'morph',
  //   baseURL: 'https://api.morphllm.com/v1',
  // },
  infini: {
    name: 'infini',
    baseURL: 'https://cloud.infini-ai.com/maas/v1',
  },
  poixe: {
    name: 'poixe',
    baseURL: 'https://api.poixe.ai/v1',
  },
};

let jinaApiKey: string | null, tavilyApiKey: string | null, pythonApiKey: string | null, pythonUrl: string | null;

// Helper function to create custom provider
function createCustomProvider(providerName: string, apiKey: string) {
  const config = SUPPORTED_PROVIDERS[providerName as keyof typeof SUPPORTED_PROVIDERS];
  if (!config) {
    throw new Error(`Unsupported provider: ${providerName}`);
  }

  return createOpenAICompatible({
    name: 'custom',
    apiKey: apiKey,
    baseURL: config.baseURL,
    includeUsage: true,
  });
}

// Helper function to parse model and determine provider
function parseModelName(model: string) {
  const parts = model.split('/');
  if (parts.length >= 2) {
    const [providerName, ...modelParts] = parts;
    const modelName = modelParts.join('/');
    if (SUPPORTED_PROVIDERS[providerName as keyof typeof SUPPORTED_PROVIDERS]) {
      return {
        provider: providerName,
        model: modelName,
        useCustomProvider: true
      };
    }
  }

  return {
    provider: null,
    model: model,
    useCustomProvider: false
  };
}

function getProviderKeys(req: NextRequest, authHeader: string | null) {
  const providerKeys: Record<string, string[]> = {};

  for (const provider of Object.keys(SUPPORTED_PROVIDERS)) {
    const keyName = `${provider}_api_key`;
    const headerValue = req.headers.get(keyName);
    if (headerValue) {
      const keys = headerValue.split(',').map((k: string) => k.trim());
      providerKeys[provider] = keys;
    }
  }

  // If no provider keys in headers, use auth header for all providers
  if (Object.keys(providerKeys).length === 0 && authHeader) {
    const headerKey = authHeader.split(' ')[1];
    if (headerKey) {
      const keys = headerKey.split(',').map(k => k.trim());
      for (const provider of Object.keys(SUPPORTED_PROVIDERS)) {
        providerKeys[provider] = keys;
      }
    }
  }

  return providerKeys;
}

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
  const excludedTools = ['code_execution', 'python_executor', 'tavily_search', 'jina_reader', 'google_search', 'web_search_preview', 'url_context'];
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
            if (!excludedTools.includes(part.toolName)) {
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
                            arguments: JSON.stringify(part.input),
                          },
                        },
                      ],
                    },
                    finish_reason: null,
                  },
                ],
              };
              controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
            }
            break;
          }
          case 'tool-result': {
            if (!excludedTools.includes(part.toolName)) {
              const chunk = {
                id: chunkId,
                object: 'chat.completion.chunk',
                created: now,
                model: model,
                choices: [
                  {
                    index: 0,
                    delta: {
                      role: 'tool',
                      content: [
                        {
                          type: 'tool_call_output',
                          call_id: part.toolCallId,
                          output: typeof part.result === 'string' ? part.result : JSON.stringify(part.result),
                        }
                      ],
                    },
                    finish_reason: null,
                  },
                ],
              };
              controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
            }
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
                prompt_tokens: part.totalUsage.inputTokens,
                completion_tokens: part.totalUsage.outputTokens,
                total_tokens: part.totalUsage.totalTokens,
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

  const abortController = new AbortController();

  if (req.signal) {
    req.signal.addEventListener('abort', () => {
      abortController.abort();
    });
  }

  let gateway;
  let useSearchGrounding = false;
  jinaApiKey = req.headers.get('jina_api_key');
  tavilyApiKey = req.headers.get('tavily_api_key');
  pythonApiKey = req.headers.get('python_api_key');
  pythonUrl = req.headers.get('python_url');
  const body = await req.json();
  const { model, messages = [], tools, stream, temperature, top_p, top_k, max_tokens, stop_sequences, seed, presence_penalty, frequency_penalty, tool_choice, reasoning_effort, thinking, extra_body } = body;
  // Get provider API keys from request headers
  const providerKeys = getProviderKeys(req, authHeader);

  const vercelCity = req.headers.get('x-vercel-ip-city');

  let contextMessages = [...messages];

  if (vercelCity) {
    const vercelCountry = req.headers.get('x-vercel-ip-country');
    const vercelTimezone = req.headers.get('x-vercel-ip-timezone');
    const forwardedFor = req.headers.get('x-forwarded-for');
    const contextInfo = [
      vercelCity && `City: ${vercelCity}`,
      vercelCountry && `Country: ${vercelCountry}`,
      vercelTimezone && `Time: ${new Date().toLocaleString('en-US', { timeZone: vercelTimezone })}`,
      forwardedFor && `IP: ${forwardedFor}`
    ].filter(Boolean).join(', ');

    const systemMessage = {
      role: 'system' as const,
      content: `Context Information: ${contextInfo}`
    };

    contextMessages = [systemMessage, ...messages];
  }

  let aiSdkTools: Record<string, any> = {};
  if (tools && Array.isArray(tools)) {
    if (model.startsWith('openai')) {
      aiSdkTools.web_search_preview = openai.tools.webSearchPreview({});
    } else if (model.startsWith('xai')) {
      aiSdkTools.python_executor = pythonExecutorTool;
    } else if (!model.startsWith('google')) {
      if (tavilyApiKey) {
        aiSdkTools.tavily_search = tavilySearchTool;
      }
    }
    if (!model.startsWith('google')) {
      aiSdkTools.jina_reader = jinaReaderTool;
      if (pythonApiKey && pythonUrl) {
        aiSdkTools.python_executor = pythonExecutorTool;
      }
    }
    tools.forEach((userTool: any) => {
      if (userTool.type === 'function' && userTool.function) {
        if (userTool.function.name === 'googleSearch') {
          useSearchGrounding = true;
          const lastMessage = contextMessages[contextMessages.length - 1];
          if (lastMessage && typeof lastMessage.content === 'string' && (lastMessage.content.includes('http://') || lastMessage.content.includes('https://'))) {
            aiSdkTools = {
              url_context: google.tools.urlContext({}),
              code_execution: google.tools.codeExecution({}),
            };
            if (tavilyApiKey) {
              aiSdkTools.tavily_search = tavilySearchTool;
            }
          } else {
            aiSdkTools = {
              google_search: google.tools.googleSearch({}),
              jina_reader: jinaReaderTool,
              code_execution: google.tools.codeExecution({}),
            };
          }
          return;
        }

        let clientParameters = userTool.function.parameters || userTool.function.inputSchema || {};

        const finalParameters: Record<string, any> = {
          type: "object",
          properties: clientParameters.properties || clientParameters,
          required: clientParameters.required || [],
        };
        const properties = finalParameters.properties || {};
        const required = finalParameters.required || [];
        const zodFields: Record<string, z.ZodTypeAny> = {};

        for (const [key, prop] of Object.entries(properties)) {
          const propDef = prop as any;
          let zodType: z.ZodTypeAny;

          // Map OpenAI parameter types to Zod types
          switch (propDef.type) {
            case 'string':
              zodType = z.string();
              break;
            case 'number':
              zodType = z.number();
              break;
            case 'integer':
              zodType = z.number().int();
              break;
            case 'boolean':
              zodType = z.boolean();
              break;
            case 'array':
              zodType = z.array(z.any());
              break;
            case 'object':
              zodType = z.object({});
              break;
            default:
              zodType = z.any();
          }

          if (propDef.description) {
            zodType = zodType.describe(propDef.description);
          }

          if (!required.includes(key)) {
            zodType = zodType.optional();
          }

          zodFields[key] = zodType;
        }

        aiSdkTools[userTool.function.name] = tool({
          description: userTool.function.description || `Function ${userTool.function.name}`,
          inputSchema: z.object(zodFields),
        });
      }
    });
  }

  // Parse the model name to determine provider
  const modelInfo = parseModelName(model);
  let providersToTry: Array<{ type: 'gateway' | 'custom', name?: string, apiKey: string, model: string }> = [];

  if (modelInfo.useCustomProvider && modelInfo.provider) {
    // Model is in format provider/model, try custom provider first
    const customProviderKeys = providerKeys[modelInfo.provider];
    if (customProviderKeys && customProviderKeys.length > 0) {
      for (const key of customProviderKeys) {
        providersToTry.push({
          type: 'custom',
          name: modelInfo.provider,
          apiKey: key,
          model: modelInfo.model
        });
      }
    } else {
      // If no specific provider keys, try using auth header keys
      const apiKeys = apiKey.split(',').map(key => key.trim()) || [];
      for (const key of apiKeys) {
        providersToTry.push({
          type: 'custom',
          name: modelInfo.provider,
          apiKey: key,
          model: modelInfo.model
        });
      }
    }
  } else {
    // Use gateway with original model name
    const apiKeys = apiKey.split(',').map(key => key.trim()) || [];
    for (const key of apiKeys) {
      providersToTry.push({
        type: 'gateway',
        apiKey: key,
        model: modelInfo.model
      });
    }
  }

  let lastError: any;

  for (let i = 0; i < providersToTry.length; i++) {
    const provider = providersToTry[i];

    try {
      if (provider.type === 'gateway') {
        gateway = createGateway({
          apiKey: provider.apiKey,
          baseURL: 'https://ai-gateway.vercel.sh/v1/ai',
        });
      } else {
        // Create custom provider
        const customProvider = createCustomProvider(provider.name!, provider.apiKey);
        gateway = customProvider;
      }
      // Process messages to remove tool roles and convert to assistant messages
      const processedMessages = [];

      for (let i = 0; i < contextMessages.length; i++) {
        const message = contextMessages[i];

        if (message.role === 'tool') {
          continue;
        }
        else if (message.role === 'assistant' && message.tool_calls && Array.isArray(message.tool_calls)) {
          let assistantContent = message.content || '';

          for (const toolCall of message.tool_calls) {
            const toolName = toolCall.function.name;
            const args = toolCall.function.arguments;

            assistantContent += `\n<tool_use_result>\n  <name>${toolName}</name>\n  <arguments>${args}</arguments>\n`;

            // Find the corresponding tool result
            const toolResultMessage = contextMessages.find((m: any) =>
              m.role === 'tool' && m.tool_call_id === toolCall.id
            );

            if (toolResultMessage) {
              let resultText = toolResultMessage.content;

              if (typeof resultText === 'string') {
                try {
                  const parsed = JSON.parse(resultText);
                  if (Array.isArray(parsed) && parsed[0] && parsed[0].text) {
                    resultText = parsed[0].text;
                  }
                } catch (e) {
                  // Keep original string
                }
              }

              assistantContent += `\n  <result>${resultText}</result>\n</tool_use_result>`;
            }
          }

          processedMessages.push({
            role: 'assistant',
            content: assistantContent
          });
        }
        else {
          processedMessages.push(message);
        }
      }

      for (const message of processedMessages) {
        if (message.role === 'user' && Array.isArray(message.content)) {
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
      // console.log('Processed messages:', JSON.stringify(processedMessages, null, 2));
      const commonOptions = {
        model: gateway(provider.model),
        messages: processedMessages,
        tools: aiSdkTools,
        temperature,
        topP: top_p,
        topK: top_k,
        maxOutputTokens: max_tokens,
        seed,
        stopSequences: stop_sequences,
        presencePenalty: presence_penalty,
        frequencyPenalty: frequency_penalty,
        toolChoice: tool_choice,
        abortSignal: abortController.signal,
        providerOptions: req.headers.get('provider_options') ? JSON.parse(req.headers.get('provider_options')!) : {
          anthropic: {
            thinking: thinking || {
              type: "enabled",
              budgetTokens: 4000
            },
            cacheControl: {
              type: "ephemeral"
            }
          },
          openai: {
            reasoningEffort: reasoning_effort || "medium",
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
          },
          custom: {
            reasoning_effort: reasoning_effort || "medium"
          },
        },
        stopWhen: [stepCountIs(20)],
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
      console.error(`Error with provider ${i + 1}/${providersToTry.length} (${provider.type}${provider.name ? ':' + provider.name : ''}):`, error);
      lastError = error;

      if (error.name === 'AbortError' || abortController.signal.aborted) {
        console.log(`Request aborted: ${error.message}`);
        const abortPayload = {
          error: {
            message: 'Request was aborted by the user',
            type: 'request_aborted',
            statusCode: 499,
          },
        };

        if (stream) {
          const encoder = new TextEncoder();
          const errorChunk = {
            id: `chatcmpl-abort-${Date.now()}`,
            object: 'chat.completion.chunk',
            created: Math.floor(Date.now() / 1000),
            model: model || 'unknown',
            choices: [
              {
                index: 0,
                delta: { content: JSON.stringify(abortPayload) },
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
            status: 499,
          });
        } else {
          return new NextResponse(JSON.stringify(abortPayload), {
            status: 499,
            headers: { 'Content-Type': 'application/json', ...corsHeaders },
          });
        }
      }

      if (i < providersToTry.length - 1) {
        continue;
      }

      break;
    }
  }

  console.error('All providers failed. Last error:', lastError);

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
      message: `All ${providersToTry.length} provider(s) failed. Last error: ${errorMessage}`,
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
