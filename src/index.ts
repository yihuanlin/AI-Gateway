import { Hono } from 'hono'
import { cors } from 'hono/cors'
import type { Context } from 'hono'
import { generateText, stepCountIs, streamText, tool, type GenerateTextResult } from 'ai'
import { createGateway } from '@ai-sdk/gateway'
import { createOpenAICompatible } from '@ai-sdk/openai-compatible'
import { openai } from '@ai-sdk/openai'
import { google } from '@ai-sdk/google'
import { z } from 'zod'

export const config = { runtime: 'edge' };
const app = new Hono()

// CORS middleware
app.use('*', cors({
	origin: '*',
	allowMethods: ['GET', 'POST', 'OPTIONS'],
	allowHeaders: ['*'],
}))

// Tools definitions
const pythonExecutorTool = tool({
	description: 'Execute Python code remotely via a secure Python execution API. Installed packages include: numpy, pandas.',
	inputSchema: z.object({
		code: z.string().describe('The Python code to execute.'),
	}),
	execute: async ({ code }: { code: string }) => {
		console.log(`Executing remote Python code: ${code.substring(0, 100)}...`);
		try {
			const pythonUrl = process.env.PYTHON_URL || null;
			const pythonApiKey = process.env.PYTHON_API_KEY || null;

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
			const tavilyApiKey = process.env.TAVILY_API_KEY || null;

			if (!tavilyApiKey) {
				return {
					error: 'TAVILY_API_KEY environment variable is not set'
				};
			}

			const maxResults = max_results || 5;
			const apiKeys = tavilyApiKey.split(',').map((key: string) => key.trim());
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
					const data = await response.json() as any;
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
			};

			const jinaApiKey = process.env.JINA_API_KEY || null;
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
	infini: {
		name: 'infini',
		baseURL: 'https://cloud.infini-ai.com/maas/v1',
	},
	poixe: {
		name: 'poixe',
		baseURL: 'https://api.poixe.com/v1',
	},
};

// Pre-compiled constants for performance
const PROVIDER_KEYS = Object.keys(SUPPORTED_PROVIDERS);

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

// Helper function to parse model name for display
function parseModelDisplayName(model: string) {
	let baseName = model.split('/').pop() || model;

	if (baseName.endsWith(':free')) {
		baseName = baseName.slice(0, -5);
	}

	let displayName = baseName.replace(/-/g, ' ');
	displayName = displayName.split(' ').map(word => {
		const lowerWord = word.toLowerCase();
		if (lowerWord === 'deepseek') {
			return 'DeepSeek';
		} else if (lowerWord === 'ernie') {
			return 'ERNIE';
		} else if (lowerWord === 'mai' || lowerWord === 'ds' || lowerWord === 'r1') {
			return word.toUpperCase();
		} else if (lowerWord === 'gpt') {
			return 'GPT';
		} else if (lowerWord === 'oss') {
			return 'OSS';
		} else if (lowerWord === 'glm') {
			return 'GLM';
		} else if (lowerWord.startsWith('o') && lowerWord.length > 1 && /^\d/.test(lowerWord.slice(1))) {
			// Check if word starts with 'o' followed by a number (OpenAI o models)
			return word.toLowerCase();
		} else if (/^a?\d+[bkmae]$/.test(lowerWord)) {
			return word.toUpperCase();
		} else {
			// Capitalize first letter of each word
			return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
		}
	}).join(' ');

	// Handle special cases that need to keep hyphens
	if (displayName === 'MAI DS R1') {
		displayName = 'MAI-DS-R1';
	} else if (displayName.startsWith('GPT ')) {
		// Replace spaces after GPT with hyphens
		displayName = displayName.replace(/^GPT /, 'GPT-');
	}
	return displayName;
}

function getProviderKeys(headers: any, authHeader: string | null, isPasswordAuth: boolean = false) {
	const providerKeys: Record<string, string[]> = {};
	const headerEntries: Record<string, string | null> = {};

	// Read all provider headers at once
	for (const provider of PROVIDER_KEYS) {
		const keyName = `${provider}_api_key`;
		headerEntries[provider] = headers[keyName] || null;
	}

	for (const provider of PROVIDER_KEYS) {
		const headerValue = headerEntries[provider];
		if (headerValue) {
			providerKeys[provider] = headerValue.split(',').map((k: string) => k.trim());
		} else if (isPasswordAuth) {
			// If password auth is enabled and no header key found, try environment variable
			const envKeyName = `${provider.toUpperCase()}_API_KEY`;
			const envValue = process.env[envKeyName];
			if (envValue) {
				providerKeys[provider] = envValue.split(',').map((k: string) => k.trim());
			}
		}
	}

	// If no provider keys in headers, use auth header for all providers (unless password auth)
	if (Object.keys(providerKeys).length === 0 && authHeader && !isPasswordAuth) {
		const headerKey = authHeader.split(' ')[1];
		if (headerKey) {
			const keys = headerKey.split(',').map(k => k.trim());
			for (const provider of PROVIDER_KEYS) {
				providerKeys[provider] = keys;
			}
		}
	}

	return providerKeys;
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
	const TEXT_ENCODER = new TextEncoder();
	const EXCLUDED_TOOLS = new Set(['code_execution', 'python_executor', 'tavily_search', 'jina_reader', 'google_search', 'web_search_preview', 'url_context']);

	const stream = new ReadableStream({
		async start(controller) {
			const now = Math.floor(Date.now() / 1000);
			const chunkId = `chatcmpl-${now}`;

			const baseChunk = {
				id: chunkId,
				object: 'chat.completion.chunk',
				created: now,
				model: model,
			};

			for await (const part of result.fullStream) {
				let chunk;
				switch (part.type) {
					case 'reasoning-delta': {
						chunk = {
							...baseChunk,
							choices: [
								{
									index: 0,
									delta: { reasoning_content: part.text },
									finish_reason: null,
								},
							],
						};
						controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
						break;
					}
					case 'text-delta': {
						chunk = {
							...baseChunk,
							choices: [
								{
									index: 0,
									delta: { content: part.text },
									finish_reason: null,
								},
							],
						};
						controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
						break;
					}
					case 'source': {
						chunk = {
							...baseChunk,
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
						controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
						break;
					}
					case 'tool-call': {
						if (!EXCLUDED_TOOLS.has(part.toolName)) {
							chunk = {
								...baseChunk,
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
							controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
						}
						break;
					}
					case 'tool-result': {
						if (!EXCLUDED_TOOLS.has(part.toolName)) {
							chunk = {
								...baseChunk,
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
							controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
						}
						break;
					}
					case 'finish': {
						chunk = {
							...baseChunk,
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
						controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
						break;
					}
				}
			}

			controller.enqueue(TEXT_ENCODER.encode('data: [DONE]\n\n'));
			controller.close();
		},
	});

	return new Response(stream, {
		headers: {
			'Content-Type': 'text/event-stream',
			'Cache-Control': 'no-cache',
		},
	});
}

// Chat completions endpoint
app.post('/v1/chat/completions', async (c: Context) => {
	const authHeader = c.req.header('Authorization');
	let apiKey = authHeader?.split(' ')[1];

	const envPassword = process.env.PASSWORD;
	const isPasswordAuth = !!(envPassword && apiKey && envPassword.trim() === apiKey.trim());

	if (!apiKey) {
		return c.text('Unauthorized', 401);
	}

	const abortController = new AbortController();

	let gateway;
	let useSearchGrounding = false;

	// Get headers
	const jinaApiKey = c.req.header('jina_api_key') || (isPasswordAuth ? process.env.JINA_API_KEY : null);
	const tavilyApiKey = c.req.header('tavily_api_key') || (isPasswordAuth ? process.env.TAVILY_API_KEY : null);
	const pythonApiKey = c.req.header('python_api_key') || (isPasswordAuth ? process.env.PYTHON_API_KEY : null);
	const pythonUrl = c.req.header('python_url') || (isPasswordAuth ? process.env.PYTHON_URL : null);

	// Read Vercel context headers
	const vercelCity = c.req.header('x-vercel-ip-city');
	const vercelCountry = c.req.header('x-vercel-ip-country');
	const vercelTimezone = c.req.header('x-vercel-ip-timezone');
	const forwardedFor = c.req.header('x-forwarded-for');

	const body = await c.req.json();
	const { model, messages = [], tools, stream, temperature, top_p, top_k, max_tokens, stop_sequences, seed, presence_penalty, frequency_penalty, tool_choice, reasoning_effort, thinking, extra_body } = body;

	// Get provider API keys from request headers
	const headers: Record<string, string> = {}
	c.req.raw.headers.forEach((value, key) => {
		headers[key.toLowerCase().replace(/-/g, '_')] = value
	})
	const providerKeys = getProviderKeys(headers, authHeader || null, isPasswordAuth);

	let contextMessages = messages;

	if (vercelCity) {
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
		} else if (!isPasswordAuth) {
			// If no specific provider keys and not password auth, try using auth header keys
			const apiKeys = apiKey.split(',').map((key: string) => key.trim()) || [];
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
		let apiKeys: string[] = [];

		if (isPasswordAuth) {
			const gatewayKey = process.env.GATEWAY_API_KEY;
			if (gatewayKey) {
				apiKeys = gatewayKey.split(',').map(key => key.trim());
			}
		} else {
			// Use the provided API key
			apiKeys = apiKey.split(',').map((key: string) => key.trim());
		}

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
			const processedMessages: any[] = [];

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
								return {
									type: 'image',
									image: part.image_url.url,
								};
							} else if (part.type === 'input_file') {
								const base64Data = part.file_data;
								let mediaType = 'application/pdf';

								if (base64Data.startsWith('data:')) {
									const match = base64Data.match(/^data:([^;]+)/);
									if (match) {
										mediaType = match[1];
									}
								}

								return {
									type: 'file',
									data: base64Data,
									mediaType: mediaType
								};
							}
							return part;
						}),
					);
				}
			}

			const providerOptionsHeader = c.req.header('provider_options');
			const providerOptions = providerOptionsHeader ? JSON.parse(providerOptionsHeader) : {
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
					},
					...(reasoning_effort && { reasoningEffort: reasoning_effort })
				},
				google: {
					useSearchGrounding: useSearchGrounding,
					...(extra_body?.google?.thinking_config && { thinking_config: extra_body.google.thinking_config })
				},
				custom: {
					reasoning_effort: reasoning_effort || "medium"
				},
			};

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
				providerOptions,
				stopWhen: [stepCountIs(20)],
			};

			if (stream) {
				const result = streamText(commonOptions);
				return toOpenAIStream(result, model);
			} else {
				const result = await generateText(commonOptions);
				const openAIResponse = toOpenAIResponse(result, model);
				return c.json(openAIResponse);
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
						},
						status: 499,
					});
				} else {
					return c.json(abortPayload, 499 as any);
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
			},
			status: statusCode,
		});
	} else {
		return c.json(errorPayload, statusCode);
	}
})

// Helper function to fetch models from custom provider
async function fetchProviderModels(providerName: string, apiKey: string) {
	const config = SUPPORTED_PROVIDERS[providerName as keyof typeof SUPPORTED_PROVIDERS];
	if (!config) {
		throw new Error(`Unsupported provider: ${providerName}`);
	}

	const modelsEndpoint = `${config.baseURL}/models`;
	const response = await fetch(modelsEndpoint, {
		method: 'GET',
		headers: {
			'Authorization': `Bearer ${apiKey}`,
			'Content-Type': 'application/json',
		},
	});

	if (!response.ok) {
		throw new Error(`Provider ${providerName} models API failed: ${response.status} ${response.statusText}`);
	}

	const data = await response.json();
	return data;
}

// Helper function to get provider API keys from request headers
function getProviderKeysFromHeaders(headers: Record<string, string>, isPasswordAuth: boolean = false) {
	const providerKeys: Record<string, string[]> = {};

	for (const provider of Object.keys(SUPPORTED_PROVIDERS)) {
		const keyName = `${provider}_api_key`;
		const headerValue = headers[keyName];
		if (headerValue) {
			const keys = headerValue.split(',').map((k: string) => k.trim());
			providerKeys[provider] = keys;
		}
	}

	// If password auth is enabled, also check for environment variables for all providers
	if (isPasswordAuth) {
		for (const provider of Object.keys(SUPPORTED_PROVIDERS)) {
			const envKeyName = `${provider.toUpperCase()}_API_KEY`;
			const envValue = process.env[envKeyName];
			if (envValue) {
				const keys = envValue.split(',').map((k: string) => k.trim());
				// If provider already has header keys, merge them; otherwise add env keys
				if (providerKeys[provider]) {
					providerKeys[provider].push(...keys);
				} else {
					providerKeys[provider] = keys;
				}
			}
		}
	}

	return providerKeys;
}

// Helper function to filter out unwanted models
function shouldIncludeModel(model: any, providerName?: string) {
	const modelId = model.id.toLowerCase();
	// Common exclusions for all providers
	const commonExclusions = ['gemma', 'rerank', 'distill', 'parse', 'embed', 'bge-', 'tts', 'phi', 'live', 'audio', 'lite', 'qwen2', 'qwen-2', 'qwen1', 'qwq', 'qvq', 'gemini-2.0', 'gemini-1', 'learnlm', 'gemini-exp', 'turbo', 'claude-3', 'voxtral', 'pixtral', 'mixtral', 'ministral', '-24', 'moderation', 'saba', '-ocr-'];
	if (commonExclusions.some(exclusion => modelId.includes(exclusion))) {
		return false;
	}
	if (!modelId.includes('super') && ((['nemotron', 'llama'].some(exclusion => modelId.includes(exclusion))) || modelId.includes('nvidia'))) {
		return false;
	}

	// Provider-specific exclusions
	if (providerName === 'gemini' && ['veo', 'imagen'].some(exclusion => modelId.includes(exclusion))) {
		return false;
	} else if (providerName === 'openrouter' && !modelId.includes(':free')) {
		return false;
	} else if (providerName !== 'mistral' && modelId.includes('mistral')) {
		return false;
	}

	if (!providerName && ['mistral', 'alibaba', 'cohere', 'deepseek', 'moonshotai', 'morph', 'zai'].some(exclusion => modelId.includes(exclusion))) {
		return false;
	}

	return true;
}

async function getModelsResponse(apiKey: string, providerKeys: Record<string, string[]>, isPasswordAuth: boolean = false) {
	let gatewayApiKeys: string[] = [];

	if (isPasswordAuth) {
		const gatewayKey = process.env.GATEWAY_API_KEY;
		if (gatewayKey) {
			gatewayApiKeys = gatewayKey.split(',').map((key: string) => key.trim());
		}
	} else {
		gatewayApiKeys = apiKey.split(',').map((key: string) => key.trim());
	}

	const fetchPromises: Promise<any[]>[] = [];

	if (gatewayApiKeys.length > 0) {
		const randomIndex = Math.floor(Math.random() * gatewayApiKeys.length);
		const currentApiKey = gatewayApiKeys[randomIndex];

		const gatewayPromise = (async () => {
			try {
				const gateway = createGateway({
					apiKey: currentApiKey,
					baseURL: 'https://ai-gateway.vercel.sh/v1/ai',
				});

				const availableModels = await gateway.getAvailableModels();
				const now = Math.floor(Date.now() / 1000);

				return availableModels.models.map(model => ({
					id: model.id,
					name: model.name,
					description: `${model.pricing
						? ` I: $${(Number(model.pricing.input) * 1000000).toFixed(2)}, O: $${(
							Number(model.pricing.output) * 1000000
						).toFixed(2)};`
						: ''
						} ${model.description || ''}`,
					object: 'model',
					created: now,
					owned_by: model.name.split('/')[0],
					pricing: model.pricing || {},
					source: 'gateway',
				})).filter(model => shouldIncludeModel(model));
			} catch (error: any) {
				console.error(`Error with gateway API key:`, error);
				return [];
			}
		})();

		fetchPromises.push(gatewayPromise);
	}

	// Add provider fetch promises
	for (const [providerName, keys] of Object.entries(providerKeys)) {
		if (keys.length === 0) continue;

		// Randomly select one API key for this provider
		const randomIndex = Math.floor(Math.random() * keys.length);
		const providerApiKey = keys[randomIndex];

		const providerPromise = (async () => {
			try {
				let formattedModels: any[] = [];
				const now = Math.floor(Date.now() / 1000);

				// Check if this provider has a custom model list (doesn't support /models endpoint)
				if (CUSTOM_MODEL_LISTS[providerName as keyof typeof CUSTOM_MODEL_LISTS]) {
					const customModels = CUSTOM_MODEL_LISTS[providerName as keyof typeof CUSTOM_MODEL_LISTS];
					formattedModels = customModels.map(model => ({
						id: `${providerName}/${model.id}`,
						name: model.name,
						object: 'model',
						created: now,
						owned_by: providerName,
						pricing: {},
						source: providerName,
					})).filter(model => shouldIncludeModel(model, providerName));
				} else {
					// Use regular API call for providers that support /models endpoint
					const providerModels = await fetchProviderModels(providerName, providerApiKey);
					formattedModels = (providerModels as any).data?.map((model: any) => ({
						id: `${providerName}/${model.id.replace('models/', '')}`,
						name: `${model.name?.replace(' (free)', '') || parseModelDisplayName(model.id)}`,
						description: model.description || '',
						object: 'model',
						created: model.created || now,
						owned_by: model.owned_by || providerName,
						pricing: model.pricing || {},
						source: providerName,
					})).filter((model: any) => {
						if (!shouldIncludeModel(model, providerName)) {
							return false;
						}
						return true;
					}) || [];
				}

				return formattedModels;
			} catch (error: any) {
				console.error(`Error with ${providerName} API key:`, error);
				return [];
			}
		})();

		fetchPromises.push(providerPromise);
	}

	// Fetch all providers in parallel
	const results = await Promise.allSettled(fetchPromises);

	// Collect all successful results
	const allModels: any[] = [];
	results.forEach((result) => {
		if (result.status === 'fulfilled' && result.value.length > 0) {
			allModels.push(...result.value);
		}
	});

	// If we have models from any source, return them
	if (allModels.length > 0) {
		return {
			object: 'list',
			data: allModels,
		};
	}

	throw new Error('All provider(s) failed to return models');
}

// Custom model lists for providers that don't support /models endpoint
const CUSTOM_MODEL_LISTS = {
	poixe: [
		{ id: 'gpt-5:free', name: 'GPT-5' },
		{ id: 'grok-3-mini:free', name: 'Grok 3 Mini' },
		{ id: 'grok-4:free', name: 'Grok 4' },
	],
	doubao: [
		{ id: 'doubao-seed-1-6-flash-250715', name: 'Doubao Seed 1.6 Flash' },
		{ id: 'doubao-seed-1-6-thinking-250715', name: 'Doubao Seed 1.6 Thinking' },
		{ id: 'deepseek-r1-250528', name: 'DeepSeek R1' },
		{ id: 'deepseek-v3-250324', name: 'DeepSeek V3' },
		{ id: 'kimi-k2-250711', name: 'Kimi K2' },
	],
	cohere: [
		{ id: 'command-a-03-2025', name: 'Command A' },
		{ id: 'command-a-vision-07-2025', name: 'Cohere A Vision' },
	],
	github: [
		{ id: 'grok-3-mini', name: 'Grok 3 Mini' },
		{ id: 'grok-3', name: 'Grok 3' },
		{ id: 'gpt-5', name: 'GPT-5' },
		{ id: 'gpt-5-chat', name: 'GPT-5 Chat' },
	],
};

// Models endpoint
app.get('/v1/models', async (c: Context) => {
	const authHeader = c.req.header('Authorization');
	let apiKey = authHeader?.split(' ')[1];

	// Check for password authentication
	const envPassword = process.env.PASSWORD;
	const isPasswordAuth = !!(envPassword && apiKey && envPassword.trim() === apiKey.trim());

	if (!apiKey) {
		return c.text('Unauthorized', 401);
	}

	// Get provider API keys from headers
	const headers: Record<string, string> = {}
	c.req.raw.headers.forEach((value, key) => {
		headers[key.toLowerCase().replace(/-/g, '_')] = value
	})
	const providerKeys = getProviderKeysFromHeaders(headers, isPasswordAuth);

	try {
		const modelsResponse = await getModelsResponse(apiKey, providerKeys, isPasswordAuth);
		return c.json(modelsResponse);
	} catch (error: any) {
		return c.json({
			error: error.message || 'All provider(s) failed to return models'
		}, 500);
	}
})

app.post('/v1/models', async (c: Context) => {
	const authHeader = c.req.header('Authorization');
	let apiKey = authHeader?.split(' ')[1];

	// Check for password authentication
	const envPassword = process.env.PASSWORD;
	const isPasswordAuth = !!(envPassword && apiKey && envPassword.trim() === apiKey.trim());

	if (!apiKey) {
		return c.text('Unauthorized', 401);
	}

	// Get provider API keys from headers
	const headers: Record<string, string> = {}
	c.req.raw.headers.forEach((value, key) => {
		headers[key.toLowerCase().replace(/-/g, '_')] = value
	})
	const providerKeys = getProviderKeysFromHeaders(headers, isPasswordAuth);

	try {
		const modelsResponse = await getModelsResponse(apiKey, providerKeys, isPasswordAuth);
		return c.json(modelsResponse);
	} catch (error: any) {
		return c.json({
			error: error.message || 'All provider(s) failed to return models'
		}, 500);
	}
})

app.get('/*', (c: Context) => {
	return c.text('Running')
})

export default (request: Request) => app.fetch(request)
