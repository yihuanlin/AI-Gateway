import { generateText, stepCountIs, streamText, tool, type GenerateTextResult } from 'ai';
import { createGateway } from '@ai-sdk/gateway';
import { NextRequest, NextResponse } from 'next/server';
import { openai } from '@ai-sdk/openai';
import { google } from '@ai-sdk/google';
import { anthropic } from '@ai-sdk/anthropic';
import { z } from 'zod';
const fs = require('fs').promises;
const pathModule = require('path');

const textEditorTool = anthropic.tools.textEditor_20250429({
  execute: async ({
    command,
    path,
    file_text,
    insert_line,
    new_str,
    old_str,
    view_range,
  }) => {
    console.log(`Executing text editor command: ${command}, path: ${path}, file_text: ${file_text}, insert_line: ${insert_line}, new_str: ${new_str}, old_str: ${old_str}, view_range: ${view_range}`);

    try {
      const normalizedPath = pathModule.resolve(path);

      switch (command) {
        case 'view':
          try {
            const stats = await fs.stat(normalizedPath);

            if (stats.isDirectory()) {
              const files = await fs.readdir(normalizedPath);
              return files.map((file: string) => `${file}${stats.isDirectory() ? '/' : ''}`).join('\n');
            } else {
              let content = await fs.readFile(normalizedPath, 'utf8');
              if (view_range && Array.isArray(view_range) && view_range.length === 2) {
                const lines = content.split('\n');
                const [start, end] = view_range;
                const startLine = Math.max(0, start - 1);
                const endLine = end === -1 ? lines.length : Math.min(lines.length, end);
                content = lines.slice(startLine, endLine).join('\n');
              }

              const lines = content.split('\n');
              const numberedContent = lines.map((line: string, index: number) => `${index + 1}: ${line}`).join('\n');

              return numberedContent;
            }
          } catch (error) {
            return `Error reading ${path}: ${error instanceof Error ? error.message : String(error)}`;
          }

        case 'create':
          try {
            const dir = pathModule.dirname(normalizedPath);
            await fs.mkdir(dir, { recursive: true });

            await fs.writeFile(normalizedPath, file_text || '', 'utf8');
            return `Successfully created file: ${path}`;
          } catch (error) {
            return `Error creating file ${path}: ${error instanceof Error ? error.message : String(error)}`;
          }

        case 'str_replace':
          try {
            const content = await fs.readFile(normalizedPath, 'utf8');

            const occurrences = (content.match(new RegExp((old_str || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g')) || []).length;

            if (occurrences === 0) {
              return `Error: Text to replace not found in ${path}`;
            } else if (occurrences > 1) {
              return `Error: Found ${occurrences} matches for replacement text. Please provide more specific text to ensure unique replacement.`;
            }

            const newContent = content.replace(old_str, new_str);
            await fs.writeFile(normalizedPath, newContent, 'utf8');

            return `Successfully replaced text at exactly one location.`;
          } catch (error) {
            return `Error replacing text in ${path}: ${error instanceof Error ? error.message : String(error)}`;
          }

        case 'insert':
          try {
            const content = await fs.readFile(normalizedPath, 'utf8');
            const lines = content.split('\n');

            const insertIndex = Math.max(0, Math.min(insert_line || 0, lines.length));
            lines.splice(insertIndex, 0, new_str);

            const newContent = lines.join('\n');
            await fs.writeFile(normalizedPath, newContent, 'utf8');

            return `Successfully inserted text at line ${insert_line}.`;
          } catch (error) {
            return `Error inserting text in ${path}: ${error instanceof Error ? error.message : String(error)}`;
          }

        default:
          return `Error: Unknown command '${command}'. Supported commands: view, create, str_replace, insert`;
      }
    } catch (error) {
      return `Error: ${error instanceof Error ? error.message : String(error)}`;
    }
  },
});

const bashTool = anthropic.tools.bash_20241022({
  execute: async ({ command, restart }) => {
    console.log(`Executing bash command: ${command}`);
    try {
      const { exec } = require('child_process');
      const { promisify } = require('util');
      const execAsync = promisify(exec);

      const dangerousCommands = ['rm -rf', 'sudo', 'chmod 777', 'mkfs', 'dd if=', 'format', 'del /f', 'deltree'];
      const lowerCommand = command.toLowerCase();

      for (const dangerous of dangerousCommands) {
        if (lowerCommand.includes(dangerous)) {
          return {
            stdout: '',
            stderr: `Command blocked for security reasons: contains '${dangerous}'`,
            return_code: 1
          };
        }
      }

      const options = {
        timeout: 30000, // 30 second timeout
        maxBuffer: 1024 * 1024, // 1MB buffer
        cwd: process.cwd()
      };

      if (restart) {
        return {
          stdout: 'Session restarted successfully',
          stderr: '',
          return_code: 0
        };
      }

      const { stdout, stderr } = await execAsync(command, options);

      return {
        stdout: stdout || '',
        stderr: stderr || '',
        return_code: 0
      };

    } catch (error: any) {
      return {
        stdout: '',
        stderr: error.message || 'Command execution failed',
        return_code: error.code || 1
      };
    }
  },
});

const javascriptExecutorTool = tool({
  description: 'Execute JavaScript. Last line is automatically returned.',
  inputSchema: z.object({
    code: z.string().describe('Plain JavaScript code to execute'),
    context: z.string().optional().describe('Optional context'),
  }),
  execute: async ({ code, context }: { code: string; context?: string }) => {
    console.log(`Executing JavaScript code${context ? ` (${context})` : ''}: ${code.substring(0, 100)}...`);
    try {
      const lines = code.trim().split('\n');

      if (lines.length === 1) {
        const func = new Function(`return ${code.trim()}`);
        const result = func();
        return `Result: ${result}`;
      } else {
        const lastLine = lines[lines.length - 1].trim();

        if (lastLine &&
          !lastLine.includes('=') &&
          !lastLine.startsWith('function') &&
          !lastLine.startsWith('const') &&
          !lastLine.startsWith('let') &&
          !lastLine.startsWith('var') &&
          !lastLine.startsWith('if') &&
          !lastLine.startsWith('for') &&
          !lastLine.startsWith('while') &&
          !lastLine.startsWith('class') &&
          !lastLine.endsWith('{') &&
          !lastLine.endsWith('}')) {

          // Execute all but last line, then return the expression
          const beforeLastLine = lines.slice(0, -1).join('\n');
          const expressionName = lastLine.replace(';', '');

          const func = new Function(`
            ${beforeLastLine}
            return ${expressionName};
          `);

          const result = func();
          return `Result: ${result}`;
        } else {
          // Just execute everything
          const func = new Function(code);
          const result = func();
          return result !== undefined ? `Result: ${result}` : 'Code executed successfully';
        }
      }

    } catch (error: any) {
      console.error(`Code execution failed: ${error.message || 'Unknown error occurred'}`);
      return `Code execution failed: ${error.message || 'Unknown error occurred'}`;
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
      const tavilyApiKey = process.env.TAVILY_API_KEY;
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

      const jinaApiKey = process.env.JINA_API_KEY;
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
  const excludedTools = ['code_execution', 'bash', 'javascript_executor', 'tavily_search', 'jina_reader', 'google_search', 'web_search_preview', 'another_tool', 'url_context'];
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
  const { model, messages = [], tools, stream, temperature, top_p, top_k, max_tokens, stop_sequences, seed, presence_penalty, frequency_penalty, tool_choice, providerOptions, reasoning_effort, thinking, extra_body } = await req.json();

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
      aiSdkTools = {
        web_search_preview: openai.tools.webSearchPreview({}),
        jina_reader: jinaReaderTool,
      };
    } else if (model.startsWith('xai')) {
      aiSdkTools = {
        jina_reader: jinaReaderTool,
      };
    } else if (!model.startsWith('google')) {
      aiSdkTools = {
        tavily_search: tavilySearchTool,
        jina_reader: jinaReaderTool,
      };
    }
    tools.forEach((userTool: any) => {
      if (userTool.type === 'function' && userTool.function) {
        if (userTool.function.name === 'googleSearch') {
          const lastMessage = contextMessages[contextMessages.length - 1];
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
              jina_reader: jinaReaderTool,
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

  if (model.startsWith('anthropic')) {
    aiSdkTools = {
      ...aiSdkTools,
      str_replace_based_edit_tool: textEditorTool,
      bash: bashTool,
    };
  } else if (model.startsWith('google')) {
    aiSdkTools = {
      ...aiSdkTools,
      code_execution: google.tools.codeExecution({}),
    };
  } else {
    aiSdkTools = {
      ...aiSdkTools,
      javascript_executor: javascriptExecutorTool,
    };
  }

  const apiKeys = apiKey.split(',').map(key => key.trim()) || [];
  let lastError: any;

  for (let i = 0; i < apiKeys.length; i++) {
    const currentApiKey = apiKeys[i];

    gateway = createGateway({
      apiKey: currentApiKey,
      baseURL: 'https://ai-gateway.vercel.sh/v1/ai',
    });

    try {
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
        model: gateway(model),
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
        providerOptions: providerOptions || {
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
          }
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
      console.error(`Error with API key ${i + 1}/${apiKeys.length}:`, error);
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
