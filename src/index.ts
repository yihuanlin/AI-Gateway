import { Hono } from 'hono'
import { cors } from 'hono/cors'
import type { Context } from 'hono'
import { generateText, stepCountIs, streamText, tool, type GenerateTextResult } from 'ai'
import { createGateway } from '@ai-sdk/gateway'
import { createOpenAICompatible } from '@ai-sdk/openai-compatible'
import { openai, createOpenAI } from '@ai-sdk/openai'
import { google, createGoogleGenerativeAI } from '@ai-sdk/google'
import { groq, createGroq } from '@ai-sdk/groq';
import { getStore } from '@netlify/blobs'
import { string, number, boolean, array, object, optional, int, enum as zenum } from 'zod/mini'

const app = new Hono()

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
		baseURL: 'https://generativelanguage.googleapis.com/v1beta',
	},
	chatgpt: {
		name: 'chatgpt',
		baseURL: 'https://api.openai.com/v1',
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
		baseURL: 'https://models.github.ai/inference',
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

// Pre-compiled constants for /v1/chat/completions
const TEXT_ENCODER = new TextEncoder();
const EXCLUDED_TOOLS = new Set(['code_execution', 'python_executor', 'tavily_search', 'jina_reader', 'google_search', 'web_search_preview', 'url_context', 'browser_search']);
const PROVIDER_KEYS = Object.keys(SUPPORTED_PROVIDERS);
const RESEARCH_KEYWORDS = ['scientific', 'biolog', 'research', 'paper'];
const MAX_ATTEMPTS = 3;

let tavilyApiKey: string | null = null;
let pythonApiKey: string | null = null;
let pythonUrl: string | null = null;
let semanticScholarApiKey: string | null = null;
let geo: {
	city?: string;
	country?: { code: string; name: string };
	timezone?: string;
} | null = null;

// Helper functions
function getStoreWithConfig(name: string, headers?: Record<string, string>) {
	// Check for NETLIFY_SITE_ID and NETLIFY_TOKEN in environment variables
	const siteID = process.env.NETLIFY_SITE_ID;
	const token = process.env.NETLIFY_TOKEN;

	// Check for corresponding headers if environment variables are not found
	const headerSiteID = headers?.['x-netlify-site-id'];
	const headerToken = headers?.['x-netlify-token'];

	if ((siteID && token) || (headerSiteID && headerToken)) {
		return getStore({
			name,
			siteID: siteID || headerSiteID!,
			token: token || headerToken!
		});
	} else if (process.env.NETLIFY_BLOBS_CONTEXT) {
		return getStore(name);
	} else {
		// Fallback to default getStore call
		return getStore(name);
	}
}

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

function randomId(prefix: string) {
	try {
		// Prefer Web Crypto for consistency across runtimes
		const bytes = crypto.getRandomValues(new Uint8Array(16));
		const hex = Array.from(bytes).map((b) => b.toString(16).padStart(2, '0')).join('');
		return `${prefix}_${hex}`;
	} catch {
		// Fallback
		const hex = Array.from({ length: 16 }, () => Math.floor(Math.random() * 256).toString(16).padStart(2, '0')).join('');
		return `${prefix}_${hex}`;
	}
}

async function getProviderKeys(headers: any, authHeader: string | null, isPasswordAuth: boolean = false): Promise<Record<string, string[]>> {
	const providerKeys: Record<string, string[]> = {};
	const headerEntries: Record<string, string | null> = {};

	// Batch read all provider headers at once for better performance
	for (const provider of PROVIDER_KEYS) {
		const keyName = `x-${provider}-api-key`;
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

async function toOpenAIResponse(result: GenerateTextResult<any, any>, model: string) {
	const now = Math.floor(Date.now() / 1000);

	const annotations = result.sources ? result.sources.map((source: any) => ({
		type: (source.sourceType || 'url') + '_citation',
		url_citation: {
			url: source.url || '',
			title: source.title || ''
		}
	})) : [];

	const choices = result.text
		? [
			{
				index: 0,
				message: {
					role: 'assistant',
					content: result.text,
					reasoning_content: result.reasoningText,
					tool_calls: result.toolCalls,
					...(annotations.length > 0 ? { annotations } : {})
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

function addContextMessages(messages: any[], c: Context): any[] {
	if (!geo) return messages;

	const { country, city, timezone } = geo;
	const ip = c.req.header('x-forwarded-for');
	const now = new Date();

	const contextParts = [
		(city && country) ? `Location: ${city} (${country.name})` : country && `Country: ${country.name}`,
		`Time: ${now.toLocaleString(country?.code === 'GB' ? 'en-GB' : 'en-US', { timeZone: timezone }).replace(',', '')} (${now.toLocaleDateString('en-US', { weekday: 'long', timeZone: timezone })})`,
		ip && `IP: ${ip}`
	].filter(Boolean);

	if (contextParts.length > 0) {
		const systemMessage = {
			role: 'system' as const,
			content: `Context Information: ${contextParts.join(', ')}`
		};
		return [systemMessage, ...messages];
	}

	return messages;
}

async function processMessages(contextMessages: any[]): Promise<any[]> {
	const processedMessages: any[] = [];

	// First pass: process tool calls and assistant messages
	for (let mi = 0; mi < contextMessages.length; mi++) {
		const message = contextMessages[mi];

		// Skip tool messages (chat completions format) 
		if (message.role === 'tool') continue;

		// Handle function_call_output (responses format)
		if (message.type === 'function_call_output') {
			let resultText = message.output;
			if (typeof resultText === 'string') {
				try {
					const parsed = JSON.parse(resultText);
					if (Array.isArray(parsed) && parsed[0] && parsed[0].text) {
						resultText = parsed[0].text;
					}
				} catch { }
			}

			// Add a tool use message format that AI SDK understands
			processedMessages.push({
				role: 'assistant',
				content: `<tool_use_result>\n  <result>${resultText}</result>\n</tool_use_result>`
			});
			continue;
		}

		if (message.role === 'assistant' && message.tool_calls && Array.isArray(message.tool_calls)) {
			let assistantContent = message.content || '';
			for (const toolCall of message.tool_calls) {
				const toolName = toolCall.function.name;
				const args = toolCall.function.arguments;
				assistantContent += `\n<tool_use_result>\n  <name>${toolName}</name>\n  <arguments>${args}</arguments>\n`;

				// Look for tool result in both formats
				let toolResultMessage = contextMessages.find((m: any) => m.role === 'tool' && m.tool_call_id === toolCall.id);

				// If not found in chat completions format, look for responses format
				if (!toolResultMessage) {
					toolResultMessage = contextMessages.find((m: any) => m.type === 'function_call_output' && m.call_id === toolCall.id);
				}

				if (toolResultMessage) {
					// Handle both content (chat completions) and output (responses) fields
					let resultText = toolResultMessage.content || toolResultMessage.output;
					if (typeof resultText === 'string') {
						try {
							const parsed = JSON.parse(resultText);
							if (Array.isArray(parsed) && parsed[0] && parsed[0].text) {
								resultText = parsed[0].text;
							}
						} catch { }
					}
					assistantContent += `\n  <result>${resultText}</result>\n</tool_use_result>`;
				}
			}
		} else {
			processedMessages.push(message);
		}
	}

	// Second pass: process content arrays asynchronously
	await Promise.all(
		processedMessages.map(async (message) => {
			if (message.role === 'user' && Array.isArray(message.content)) {
				message.content = await Promise.all(
					message.content.map(async (part: any) => {
						if (part.type === 'image_url') {
							return { type: 'image', image: part.image_url.url };
						} else if (part.type === 'input_file') {
							const base64Data = part.file_data;
							let mediaType = 'application/pdf';
							if (base64Data.startsWith('data:')) {
								const match = base64Data.match(/^data:([^;]+)/);
								if (match) mediaType = match[1];
							}
							return { type: 'file', data: base64Data, mediaType };
						}
						return part;
					}),
				);
			}
		})
	);
	return processedMessages;
}

function createCustomProvider(providerName: string, apiKey: string) {
	const config = SUPPORTED_PROVIDERS[providerName as keyof typeof SUPPORTED_PROVIDERS];
	if (!config) {
		throw new Error(`Unsupported provider: ${providerName}`);
	}
	switch (config.name) {
		case 'chatgpt':
			return createOpenAI({
				name: 'custom',
				apiKey: apiKey,
				baseURL: config.baseURL,
			}).responses;
		case 'gemini':
			return createGoogleGenerativeAI({
				apiKey: apiKey,
				baseURL: config.baseURL,
			});
		case 'groq':
			return createGroq({
				apiKey: apiKey,
				baseURL: config.baseURL,
			});
		default:
			return createOpenAICompatible({
				name: 'custom',
				apiKey: apiKey,
				baseURL: config.baseURL,
				includeUsage: true,
			});
	}
}

function buildDefaultProviderOptions(args: {
	providerOptionsHeader?: string | null,
	thinking?: any,
	reasoning_effort?: any,
	extra_body?: any,
	text_verbosity?: any,
	service_tier?: any,
}) {
	const { providerOptionsHeader, thinking, reasoning_effort, extra_body, text_verbosity, service_tier } = args;
	const providerOptions = providerOptionsHeader ? JSON.parse(providerOptionsHeader) : {
		anthropic: {
			thinking: thinking || { type: "enabled", budgetTokens: 4000 },
			cacheControl: { type: "ephemeral" },
		},
		openai: {
			reasoningEffort: reasoning_effort || "medium",
			reasoningSummary: "auto",
			textVerbosity: text_verbosity || "medium",
			serviceTier: service_tier || "auto",
			store: false,
			promptCacheKey: 'ai-gateway',
		},
		xai: {
			searchParameters: { mode: "auto", returnCitations: true },
			...(reasoning_effort && { reasoningEffort: reasoning_effort }),
		},
		google: {
			// Provide sensible defaults; allow extra_body.google to override
			thinkingConfig: {
				thinkingBudget: extra_body?.google?.thinking_config?.thinking_budget || 4000,
				includeThoughts: true,
			},
		},
		custom: {
			reasoning_effort: reasoning_effort || "medium",
			extra_body: extra_body,
		},
	};
	return providerOptions;
}

type Attempt = { type: 'gateway' | 'custom', name?: string, apiKey: string, model: string };
const getGatewayForAttempt = (attempt: Attempt) => {
	if (attempt.type === 'gateway') {
		const gatewayOptions: any = { apiKey: attempt.apiKey, baseURL: 'https://ai-gateway.vercel.sh/v1/ai' };
		if (attempt.model === 'anthropic/claude-sonnet-4') {
			gatewayOptions.headers = { 'anthropic-beta': 'context-1m-2025-08-07' };
		}
		return createGateway(gatewayOptions);
	}
	return createCustomProvider(attempt.name!, attempt.apiKey);
};

function prepareProvidersToTry(args: {
	model: string,
	providerKeys: Record<string, string[]>,
	isPasswordAuth: boolean,
	authApiKey?: string | null,
}) {
	const { model, providerKeys, isPasswordAuth, authApiKey } = args;
	const modelInfo = parseModelName(model);
	const providersToTry: Array<Attempt> = [];

	if (modelInfo.useCustomProvider && modelInfo.provider) {
		let keys: string[] = providerKeys[modelInfo.provider] || [];
		if (keys.length === 0) {
			if (!isPasswordAuth && authApiKey) {
				keys = authApiKey.split(',').map((k: string) => k.trim()).filter(Boolean);
			} else if (isPasswordAuth) {
				const envKeyName = `${modelInfo.provider.toUpperCase()}_API_KEY`;
				const envVal = process.env[envKeyName];
				if (envVal) keys = envVal.split(',').map((k: string) => k.trim()).filter(Boolean);
			}
		}

		if (keys.length > 0) {
			const shuffledKeys = keys.slice().sort(() => Math.random() - 0.5);
			for (const key of shuffledKeys) {
				providersToTry.push({ type: 'custom', name: modelInfo.provider, apiKey: key, model: modelInfo.model });
			}
		} else {
			let gatewayKeys: string[] = [];
			if (isPasswordAuth) {
				const gatewayKey = process.env.GATEWAY_API_KEY;
				if (gatewayKey) gatewayKeys = gatewayKey.split(',').map(k => k.trim()).filter(Boolean);
			} else if (authApiKey) {
				gatewayKeys = authApiKey.split(',').map((k: string) => k.trim()).filter(Boolean);
			}

			if (gatewayKeys.length > 0) {
				const start = Math.floor(Math.random() * gatewayKeys.length);
				for (let idx = 0; idx < gatewayKeys.length; idx++) {
					const k = gatewayKeys[(start + idx) % gatewayKeys.length];
					if (k) providersToTry.push({ type: 'gateway', apiKey: k, model: modelInfo.model });
				}
			}
		}
	} else {
		let gatewayKeys: string[] = [];
		if (isPasswordAuth) {
			const gatewayKey = process.env.GATEWAY_API_KEY;
			if (gatewayKey) gatewayKeys = gatewayKey.split(',').map(k => k.trim()).filter(Boolean);
		} else if (authApiKey) {
			gatewayKeys = authApiKey.split(',').map((k: string) => k.trim()).filter(Boolean);
		}

		if (gatewayKeys.length > 0) {
			const start = Math.floor(Math.random() * gatewayKeys.length);
			for (let idx = 0; idx < gatewayKeys.length; idx++) {
				const k = gatewayKeys[(start + idx) % gatewayKeys.length];
				if (k) providersToTry.push({ type: 'gateway', apiKey: k, model });
			}
		}
	}

	return { modelInfo, providersToTry };
}

// Convert OpenAI Responses API input format to AI SDK messages
function responsesInputToAiSdkMessages(input: any): any[] {
	if (!input) return [];

	// Simple string => user text
	if (typeof input === 'string') {
		return [{ role: 'user', content: input }];
	}

	// Single input_text object
	if (input && typeof input === 'object' && input.type === 'input_text' && typeof input.text === 'string') {
		return [{ role: 'user', content: input.text }];
	}

	// Array of role/content objects
	if (Array.isArray(input)) {
		const messages: any[] = [];
		for (const item of input) {
			// Handle function_call_output messages (responses format)
			if (item?.type === 'function_call_output') {
				messages.push(item); // Pass through unchanged
				continue;
			}

			const role = item?.role;

			// Handle assistant messages with string content
			if (role === 'assistant' && typeof item?.content === 'string') {
				messages.push({ role: 'assistant', content: item.content });
				continue;
			}

			const contentArr = Array.isArray(item?.content) ? item.content : [];
			if (!role || contentArr.length === 0) continue;

			if (role === 'system') {
				// AI SDK expects a string content for system
				const text = contentArr
					.map((part: any) =>
						part?.type === 'input_text' && typeof part?.text === 'string'
							? part.text
							: typeof part?.text === 'string'
								? part.text
								: typeof part === 'string'
									? part
									: ''
					)
					.filter(Boolean)
					.join('\n');
				if (text) messages.push({ role: 'system', content: text });
				continue;
			}

			const parts: any[] = [];
			for (const part of contentArr) {
				if (!part) continue;
				if (part.type === 'input_text' && typeof part.text === 'string') {
					parts.push({ type: 'text', text: part.text });
				} else if (part.type === 'text' && typeof part.text === 'string') {
					parts.push({ type: 'text', text: part.text });
				} else if (part.type === 'input_image') {
					const imageSrc = part?.image_url?.url || part?.url || part?.image || part?.data;
					if (imageSrc) {
						const mediaType = part?.media_type || part?.mediaType;
						parts.push(mediaType ? { type: 'image', image: imageSrc, mediaType } : { type: 'image', image: imageSrc });
					}
				} else if (part.type === 'input_file') {
					const data = part?.data || part?.file_data || part?.url;
					const mediaType = part?.media_type || part?.mediaType || 'application/octet-stream';
					if (data) parts.push({ type: 'file', data, mediaType });
				} else if (typeof part === 'string') {
					parts.push({ type: 'text', text: part });
				}
			}

			if (parts.length > 0) {
				const finalRole = role === 'assistant' || role === 'user' || role === 'tool' ? role : 'user';
				messages.push({ role: finalRole, content: parts });
			}
		}
		return messages;
	}

	return [];
}

// Build AI SDK tools from OpenAI tools array with shared heuristics
function buildAiSdkTools(model: string, userTools: any[] | undefined, messages: any[]): Record<string, any> {
	let aiSdkTools: Record<string, any> = {};

	// Only build tools if userTools is explicitly provided as an array (even if empty)
	if (Array.isArray(userTools)) {
		userTools.forEach((userTool: any) => {
			// Support both OpenAI Chat-style and flat Responses-style tool schemas
			const isFunctionType = userTool?.type === 'function';
			const fn = userTool.function || (isFunctionType ? { name: userTool.name, parameters: userTool.parameters } : null);
			if (isFunctionType && fn && (fn.name || userTool.name)) {
				let clientParameters = fn.parameters || fn.inputSchema;
				if (!clientParameters) return;
				const finalParameters: Record<string, any> = {
					type: 'object',
					properties: clientParameters.properties || clientParameters,
					required: clientParameters.required || [],
				};
				const properties = finalParameters.properties || {};
				const required = finalParameters.required || [];
				const zodFields: Record<string, any> = {};

				for (const [key, prop] of Object.entries(properties)) {
					const propDef = prop as any;
					let zodType: any;
					switch (propDef.type) {
						case 'string': zodType = string({ message: propDef.description || 'String parameter' }); break;
						case 'number': zodType = number({ message: propDef.description || 'Number parameter' }); break;
						case 'integer': zodType = int({ message: propDef.description || 'Integer parameter' }); break;
						case 'boolean': zodType = boolean({ message: propDef.description || 'Boolean parameter' }); break;
						case 'array': zodType = array(string({ message: 'Array item' }), { message: propDef.description || 'Array parameter' }); break;
						case 'object': zodType = object({}, { message: propDef.description || 'Object parameter' }); break;
						default: zodType = string({ message: 'Any parameter' });
					}
					if (!required.includes(key)) zodType = optional(zodType);
					zodFields[key] = zodType;
				}

				const fnName = fn.name || userTool.name;
				const description = fn.description || userTool.description || `Function ${fnName}`;
				aiSdkTools[fnName] = tool({ description, inputSchema: object(zodFields) });
			}
		});

		const googleIncompatible = (!['google', 'gemini'].some(prefix => model.startsWith(prefix)) || Object.keys(aiSdkTools).length > 0);

		const messageText = messages.map((msg: any) =>
			typeof msg.content === 'string'
				? msg.content.toLowerCase()
				: Array.isArray(msg.content)
					? msg.content.map((p: any) => (p?.text || '')).join(' ').toLowerCase()
					: ''
		).join(' ');

		const containsResearchKeywords = RESEARCH_KEYWORDS.some(keyword => messageText.includes(keyword));

		if (model.startsWith('openai')) {
			aiSdkTools.web_search_preview = openai.tools.webSearchPreview({});
			aiSdkTools.code_interpreter = openai.tools.codeInterpreter({});
		} else if (model.startsWith('groq/openai')) {
			aiSdkTools.browser_search = groq.tools.browserSearch({});
		} else if (googleIncompatible && !model.startsWith('xai')) {
			if (tavilyApiKey) aiSdkTools.tavily_search = tavilySearchTool;
		}
		if (containsResearchKeywords) {
			aiSdkTools.ensembl_api = ensemblApiTool;
			aiSdkTools.semantic_scholar_search = semanticScholarSearchTool;
			aiSdkTools.semantic_scholar_recommendations = semanticScholarRecommendationsTool;
		}
		if (googleIncompatible) {
			aiSdkTools.jina_reader = jinaReaderTool;
			if (!model.startsWith('openai') && pythonApiKey && pythonUrl) {
				aiSdkTools.python_executor = pythonExecutorTool;
			}
		} else {
			aiSdkTools = {
				google_search: google.tools.googleSearch({}),
				url_context: google.tools.urlContext({}),
				code_execution: google.tools.codeExecution({}),
			};
		}
	}

	return aiSdkTools;
}

const buildCommonOptions = (
	gw: any,
	attempt: Attempt,
	params: {
		messages: any[],
		aiSdkTools: Record<string, any>,
		temperature?: number,
		top_p?: number,
		top_k?: number,
		max_tokens?: number,
		seed?: number,
		stop_sequences?: string[],
		presence_penalty?: number,
		frequency_penalty?: number,
		tool_choice?: any,
		abortSignal: AbortSignal,
		providerOptions: any,
	}
) => ({
	model: gw(attempt.model),
	messages: params.messages,
	tools: params.aiSdkTools,
	temperature: params.temperature,
	topP: params.top_p,
	topK: params.top_k,
	maxOutputTokens: params.max_tokens,
	seed: params.seed,
	stopSequences: params.stop_sequences,
	presencePenalty: params.presence_penalty,
	frequencyPenalty: params.frequency_penalty,
	toolChoice: params.tool_choice,
	abortSignal: params.abortSignal,
	providerOptions: params.providerOptions,
	stopWhen: [stepCountIs(20)],
	onError: () => { }
}) as any;

// Tools definitions
const pythonExecutorTool = tool({
	description: 'Execute Python code remotely via a secure Python execution API. Installed packages include: numpy, pandas.',
	inputSchema: object({
		code: string({ message: 'The Python code to execute.' }),
	}),
	providerOptions: {
		anthropic: {
			cacheControl: { type: 'ephemeral' },
		},
	},
	execute: async ({ code }: { code: string }) => {
		console.log(`Executing remote Python code: ${code.substring(0, 100)}...`);
		try {
			if (!pythonUrl) {
				return { error: 'Python URL is not configured' };
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
	description: 'Web search tool using Tavily AI search engine',
	inputSchema: object({
		query: string({ message: 'Search query' }),
		max_results: optional(number({ message: 'Maximum number of results to return (default: 5, max: 20)' })),
		include_raw_content: optional(boolean({ message: 'Include the cleaned and parsed HTML content of each search result (default: false)' })),
		include_domains: optional(array(string({ message: 'Domain to include' }), { message: 'List of domains to include in the search' })),
		exclude_domains: optional(array(string({ message: 'Domain to exclude' }), { message: 'List of domains to exclude from the search' })),
	}),
	providerOptions: {
		anthropic: {
			cacheControl: { type: 'ephemeral' },
		},
	},
	execute: async (params) => {
		const { query, max_results, include_domains, exclude_domains, include_raw_content } = params;
		console.log(`Tavily search with query: ${query}`);
		try {
			if (!tavilyApiKey) {
				return { error: 'Tavily API key is not configured' };
			}

			const maxResults = max_results || 5;
			const includeRawContent = include_raw_content || false;
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
						include_raw_content: includeRawContent,
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
	inputSchema: object({
		url: string({ message: 'The URL of the webpage to fetch content from' }),
	}),
	providerOptions: {
		anthropic: {
			cacheControl: { type: 'ephemeral' },
		},
	},
	execute: async ({ url }: {
		url: string;
	}) => {
		console.log(`Jina Reader fetching content from: ${url}`);
		try {
			try {
				new URL(url);
			} catch {
				return {
					error: 'Invalid URL provided'
				};
			}
			const jinaUrl = new URL(`https://r.jina.ai/${url}`);

			const controller = new AbortController();
			const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

			const headers: Record<string, string> = {
				'X-Base': 'final'
			};

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

			return await response.text();

		} catch (error: any) {
			return {
				url,
				error: `Failed to fetch content: ${error.message || 'Unknown error'}`,
				success: false
			};
		}
	},
});

const ensemblApiTool = tool({
	description: 'Access Ensembl REST API for genomic data and bioinformatics information',
	inputSchema: object({
		path: string({ message: 'API endpoint path (without base URL). Examples: xrefs/symbol/Danio_rerio/sox6_201?object_type=transcript, lookup/id/ENSG00000139618, sequence/id/ENSG00000139618' }),
	}),
	providerOptions: {
		anthropic: {
			cacheControl: { type: 'ephemeral' },
		},
	},
	execute: async ({ path }: { path: string }) => {
		console.log(`Ensembl API request to path: ${path}`);
		try {
			const cleanPath = path.startsWith('/') ? path.slice(1) : path;
			const baseUrl = 'https://rest.ensembl.org';
			const fullUrl = `${baseUrl}/${cleanPath}`;

			const controller = new AbortController();
			const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

			const response = await fetch(fullUrl, {
				method: 'GET',
				headers: {
					'Content-Type': 'application/json',
					'Accept': 'application/json'
				},
				signal: controller.signal
			});

			clearTimeout(timeoutId);

			if (!response.ok) {
				const errorText = await response.text();
				return {
					error: `Ensembl API error (${response.status}): ${errorText}`,
					path: path,
					url: fullUrl
				};
			}

			return await response.json();

		} catch (error: any) {
			const message = error?.name === 'AbortError' ? 'Request to Ensembl API timed out' : (error?.message || 'Unknown error');
			return {
				error: `Ensembl API request failed: ${message}`,
				path: path
			};
		}
	},
});

const semanticScholarSearchTool = tool({
	description: 'Search Semantic Scholar for academic papers or authors',
	inputSchema: object({
		query: string({ message: 'Search query text' }),
		type: optional(zenum(['paper', 'author'], { message: 'Type of search: "paper" for papers or "author" for authors (default: "paper")' })),
		fields: optional(string({ message: 'Comma-separated list of fields to return (e.g., "title,authors,year,abstract")' })),
		limit: optional(number({ message: 'Maximum number of results (default: 10, max: 100 for papers, max: 1000 for authors)' })),
		offset: optional(number({ message: 'Used for pagination to get more results (default: 0)' })),
		year: optional(string({ message: 'Filter by publication year or range (e.g., "2020", "2018-2022")' })),
		venue: optional(string({ message: 'Filter by publication venue (e.g., "Nature", "Science")' })),
		fieldsOfStudy: optional(string({ message: 'Filter by fields of study (e.g., "Computer Science,Biology")' })),
		minCitationCount: optional(number({ message: 'Minimum number of citations required' })),
		publicationTypes: optional(string({ message: 'Filter by publication types (e.g., "Review,JournalArticle")' })),
	}),
	providerOptions: {
		anthropic: {
			cacheControl: { type: 'ephemeral' },
		},
	},
	execute: async (params) => {
		const { query, type = 'paper', fields, limit = 10, offset = 0, year, venue, fieldsOfStudy, minCitationCount, publicationTypes } = params;
		console.log(`Semantic Scholar ${type} search: ${query}`);
		try {
			const baseUrl = 'https://api.semanticscholar.org/graph/v1';
			const endpoint = type === 'paper' ? 'paper/search' : 'author/search';

			const params = new URLSearchParams();
			params.append('query', query);

			// Set appropriate limit based on search type and API constraints
			const maxLimit = type === 'paper' ? 100 : 1000;
			params.append('limit', Math.min(limit, maxLimit).toString());

			if (offset > 0) params.append('offset', offset.toString());
			if (fields) params.append('fields', fields);
			if (year) params.append('year', year);
			if (venue) params.append('venue', venue);
			if (fieldsOfStudy) params.append('fieldsOfStudy', fieldsOfStudy);
			if (minCitationCount) params.append('minCitationCount', minCitationCount.toString());
			if (publicationTypes) params.append('publicationTypes', publicationTypes);

			const fullUrl = `${baseUrl}/${endpoint}?${params.toString()}`;

			const controller = new AbortController();
			const timeoutId = setTimeout(() => controller.abort(), 30000);

			const headers: Record<string, string> = {
				'Content-Type': 'application/json',
				'Accept': 'application/json'
			};

			if (semanticScholarApiKey) {
				headers['x-api-key'] = semanticScholarApiKey;
			}

			const response = await fetch(fullUrl, {
				method: 'GET',
				headers,
				signal: controller.signal
			});

			clearTimeout(timeoutId);

			if (!response.ok) {
				const errorText = await response.text();
				console.log(`Semantic Scholar API error: (${response.status}): ${errorText}`);
				return {
					error: `Semantic Scholar API error (${response.status}): ${errorText}`,
					query: query,
					type: type,
					url: fullUrl
				};
			}

			const data = await response.json() as any;
			return {
				query: query,
				type: type,
				url: fullUrl,
				total: data.total || 0,
				offset: data.offset || 0,
				next: data.next,
				results: data.data || data
			};

		} catch (error: any) {
			const message = error?.name === 'AbortError' ? 'Request to Semantic Scholar API timed out' : (error?.message || 'Unknown error');
			console.log(`Semantic Scholar search failed: ${message}`);
			return {
				error: `Semantic Scholar search failed: ${message}`,
				query: query,
				type: type
			};
		}
	},
});

const semanticScholarRecommendationsTool = tool({
	description: 'Get paper recommendations from Semantic Scholar based on example papers',
	inputSchema: object({
		paperId: optional(string({ message: 'Single paper ID to get recommendations for' })),
		positivePaperIds: optional(array(string({ message: 'Paper ID' }), { message: 'Array of paper IDs that represent positive examples (for batch recommendations)' })),
		negativePaperIds: optional(array(string({ message: 'Paper ID' }), { message: 'Array of paper IDs that represent negative examples (for batch recommendations)' })),
		fields: optional(string({ message: 'Comma-separated list of fields to return (e.g., "title,authors,year,abstract")' })),
		limit: optional(number({ message: 'Maximum number of recommendations (default: 10, max: 100)' })),
		from: optional(zenum(['recent', 'all-cs'], { message: 'Pool of papers to recommend from (default: recent)' })),
	}),
	providerOptions: {
		anthropic: {
			cacheControl: { type: 'ephemeral' },
		},
	},
	execute: async (params) => {
		const { paperId, positivePaperIds, negativePaperIds, fields, limit = 10, from = 'recent' } = params;
		console.log(`Semantic Scholar recommendations for: ${paperId || `${positivePaperIds?.length || 0} positive papers`}`);
		try {
			const baseUrl = 'https://api.semanticscholar.org/recommendations/v1';
			let url: string;
			let method: string = 'GET';
			let body: any = null;

			if (paperId) {
				// Single paper recommendation
				const params = new URLSearchParams();
				params.append('limit', Math.min(limit, 100).toString());
				params.append('from', from);
				if (fields) params.append('fields', fields);

				url = `${baseUrl}/papers/forpaper/${paperId}?${params.toString()}`;
			} else if (positivePaperIds && positivePaperIds.length > 0) {
				// Batch recommendations
				const params = new URLSearchParams();
				params.append('limit', Math.min(limit, 100).toString());
				if (fields) params.append('fields', fields);

				url = `${baseUrl}/papers?${params.toString()}`;
				method = 'POST';
				body = {
					positivePaperIds: positivePaperIds,
					...(negativePaperIds && { negativePaperIds: negativePaperIds })
				};
			} else {
				return {
					error: 'Either paperId or positivePaperIds must be provided'
				};
			}

			const controller = new AbortController();
			const timeoutId = setTimeout(() => controller.abort(), 30000);

			const headers: Record<string, string> = {
				'Content-Type': 'application/json',
				'Accept': 'application/json'
			};

			if (semanticScholarApiKey) {
				headers['x-api-key'] = semanticScholarApiKey;
			}

			const requestOptions: RequestInit = {
				method: method,
				headers,
				signal: controller.signal
			};

			if (body) {
				requestOptions.body = JSON.stringify(body);
			}

			const response = await fetch(url, requestOptions);

			clearTimeout(timeoutId);

			if (!response.ok) {
				const errorText = await response.text();
				console.log(`Semantic Scholar Recommendations API error: (${response.status}): ${errorText}`);
				return {
					error: `Semantic Scholar Recommendations API error (${response.status}): ${errorText}`,
					url: url
				};
			}

			const data = await response.json() as any;
			return {
				url: url,
				method: method,
				recommendations: data.recommendedPapers || data.data || data
			};

		} catch (error: any) {
			const message = error?.name === 'AbortError' ? 'Request to Semantic Scholar Recommendations API timed out' : (error?.message || 'Unknown error');
			console.log(`Semantic Scholar recommendations failed: ${message}`);
			return {
				error: `Semantic Scholar recommendations failed: ${message}`
			};
		}
	},
});

// CORS middleware
app.use('*', cors({
	origin: '*',
	allowMethods: ['GET', 'POST', 'OPTIONS'],
	allowHeaders: ['*'],
}))

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

	// Get headers
	tavilyApiKey = c.req.header('x-tavily-api-key') || (isPasswordAuth ? process.env.TAVILY_API_KEY || null : null);
	pythonApiKey = c.req.header('x-python-api-key') || (isPasswordAuth ? process.env.PYTHON_API_KEY || null : null);
	pythonUrl = c.req.header('x-python-url') || (isPasswordAuth ? process.env.PYTHON_URL || null : null);
	semanticScholarApiKey = c.req.header('x-semantic-scholar-api-key') || (isPasswordAuth ? process.env.SEMANTIC_SCHOLAR_API_KEY || null : null);

	const body = await c.req.json();
	const { model, messages = [], tools, stream, temperature, top_p, top_k, max_tokens, stop_sequences, seed, presence_penalty, frequency_penalty, tool_choice, reasoning_effort, thinking, extra_body, text_verbosity, service_tier } = body;

	// Get provider API keys from request headers
	const headers: Record<string, string> = {}
	c.req.raw.headers.forEach((value, key) => {
		headers[key.toLowerCase().replace(/-/g, '_')] = value
	})

	// Add context messages using shared function
	const contextMessages = addContextMessages(messages, c);

	// Use async provider keys function for better performance
	const providerKeys = await getProviderKeys(headers, authHeader || null, isPasswordAuth);
	const aiSdkTools: Record<string, any> = buildAiSdkTools(model, tools, contextMessages);

	// Parse the model name to determine provider(s)
	const { providersToTry } = prepareProvidersToTry({ model, providerKeys, isPasswordAuth, authApiKey: apiKey });

	const processedMessages = await processMessages(contextMessages);

	const providerOptionsHeader = c.req.header('x-provider-options');
	const providerOptions = buildDefaultProviderOptions({ providerOptionsHeader: providerOptionsHeader ?? null, thinking, reasoning_effort, extra_body, text_verbosity, service_tier });

	// If streaming, handle retries within a single ReadableStream so we can switch keys on error mid-stream
	if (stream) {
		const streamResponse = new ReadableStream({
			async start(controller) {
				const now = Math.floor(Date.now() / 1000);
				const chunkId = `chatcmpl-${now}`;
				const baseChunk = { id: chunkId, object: 'chat.completion.chunk', created: now, model } as any;
				const maxAttempts = Math.min(providersToTry.length, MAX_ATTEMPTS);
				let attemptsTried = 0;
				let lastStreamError: any = null;
				let accumulatedCitations: Array<{ type: string, url_citation: { url: string, title: string } }> = [];
				for (let i = 0; i < maxAttempts; i++) {
					if (abortController.signal.aborted) break;
					const attempt = providersToTry[i];
					if (!attempt) continue;
					try {
						attemptsTried++;
						const gw = getGatewayForAttempt(attempt);
						const commonOptions = buildCommonOptions(gw, attempt, {
							messages: processedMessages,
							aiSdkTools,
							temperature,
							top_p,
							top_k,
							max_tokens,
							seed,
							stop_sequences,
							presence_penalty,
							frequency_penalty,
							tool_choice,
							abortSignal: abortController.signal,
							providerOptions,
						});

						const result = streamText(commonOptions);
						// Forward chunks; on error, try next key/provider
						for await (const rawPart of (result as any).fullStream) {
							const part: any = rawPart;
							if (abortController.signal.aborted) throw new Error('aborted');
							let chunk: any;
							switch (part.type) {
								case 'error':
									if ([429, 401].includes((part as any)?.error?.statusCode)) {
										throw new Error((part as any)?.error || 'Streaming provider error');
									} else {
										chunk = { ...baseChunk, choices: [{ index: 0, delta: { error: part.error }, finish_reason: null }] };
										controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
									}
									break;
								case 'reasoning-delta':
									chunk = { ...baseChunk, choices: [{ index: 0, delta: { reasoning_content: part.text }, finish_reason: null }] };
									controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
									break;
								case 'text-delta':
									chunk = { ...baseChunk, choices: [{ index: 0, delta: { content: part.text }, finish_reason: null }] };
									controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
									break;
								case 'source':
									// Accumulate citations for later inclusion in finish
									accumulatedCitations.push({
										type: part.sourceType + '_citation',
										url_citation: {
											url: part.url || '',
											title: part.title || ''
										}
									});
									break;
								case 'tool-call':
									if (!EXCLUDED_TOOLS.has(part.toolName)) {
										chunk = { ...baseChunk, choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: part.toolCallId, type: 'function', function: { name: part.toolName, arguments: JSON.stringify(part.input) } }] }, finish_reason: null }] };
										controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
									}
									break;
								case 'tool-result':
									if (!EXCLUDED_TOOLS.has(part.toolName)) {
										chunk = { ...baseChunk, choices: [{ index: 0, delta: { role: 'tool', content: [{ type: 'tool_call_output', call_id: part.toolCallId, output: typeof part.result === 'string' ? part.result : JSON.stringify(part.result) }] }, finish_reason: null }] };
										controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
									}
									break;
								case 'finish':
									if (accumulatedCitations.length > 0) {
										const citationsChunk = { ...baseChunk, choices: [{ index: 0, delta: { annotations: accumulatedCitations }, finish_reason: null }] };
										controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(citationsChunk)}\n\n`));
									}
									chunk = { ...baseChunk, choices: [{ index: 0, delta: {}, finish_reason: part.finishReason }], usage: { prompt_tokens: part.totalUsage.inputTokens, completion_tokens: part.totalUsage.outputTokens, total_tokens: part.totalUsage.totalTokens } };
									controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
									break;
							}
						}
						// If finished streaming without throwing, end the SSE and return
						controller.enqueue(TEXT_ENCODER.encode('data: [DONE]\n\n'));
						controller.close();
						return;
					} catch (error: any) {
						if (abortController.signal.aborted || error?.message === 'aborted' || error?.name === 'AbortError') {
							// Aborted by client
							const abortPayload = { error: { message: 'Request was aborted by the user', type: 'request_aborted', statusCode: 499 } };
							const abortChunk = { ...baseChunk, choices: [{ index: 0, delta: { content: JSON.stringify(abortPayload) }, finish_reason: 'stop' }] };
							controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(abortChunk)}\n\n`));
							controller.enqueue(TEXT_ENCODER.encode('data: [DONE]\n\n'));
							controller.close();
							return;
						}
						lastStreamError = error;
						console.error(`Streaming error with provider ${i + 1}/${providersToTry.length} (${attempt.type}${attempt.name ? ':' + attempt.name : ''}):`, error);
						// Otherwise, try next key/provider in the next loop iteration
						continue;
					}
				}

				// If all attempts failed
				const statusCode = lastStreamError?.statusCode || 500;
				const errMsg = lastStreamError?.message || 'An unknown error occurred';
				const errorPayload = { error: { message: `All ${attemptsTried} attempt(s) failed. Last error: ${errMsg}`, type: lastStreamError?.type, statusCode } };
				const errorChunk = { ...baseChunk, choices: [{ index: 0, delta: { content: JSON.stringify(errorPayload) }, finish_reason: 'stop' }] };
				controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(errorChunk)}\n\n`));
				controller.enqueue(TEXT_ENCODER.encode('data: [DONE]\n\n'));
				controller.close();
			},
		});
		return new Response(streamResponse, { headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' } });
	}

	// Non-streaming: try providers sequentially and return first success
	let lastError: any;
	const maxAttempts = Math.min(providersToTry.length, MAX_ATTEMPTS);
	let attemptsTried = 0;
	for (let i = 0; i < maxAttempts; i++) {
		const provider = providersToTry[i];
		if (!provider) continue;
		try {
			attemptsTried++;
			const gw = getGatewayForAttempt(provider);
			const commonOptions = buildCommonOptions(gw, provider, {
				messages: processedMessages,
				aiSdkTools,
				temperature,
				top_p,
				top_k,
				max_tokens,
				seed,
				stop_sequences,
				presence_penalty,
				frequency_penalty,
				tool_choice,
				abortSignal: abortController.signal,
				providerOptions,
			});

			const result = await generateText(commonOptions);
			const openAIResponse = await toOpenAIResponse(result, model);
			return c.json(openAIResponse);
		} catch (error: any) {
			console.error(`Error with provider ${i + 1}/${providersToTry.length} (${provider.type}${provider.name ? ':' + provider.name : ''}):`, error.message || error);
			lastError = error;

			if (error.name === 'AbortError' || abortController.signal.aborted) {
				const abortPayload = { error: { message: 'Request was aborted by the user', type: 'request_aborted', statusCode: 499 } };
				return c.json(abortPayload, 499 as any);
			}
			if (i < maxAttempts - 1) continue;
			break;
		}
	}

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
			message: `All ${attemptsTried} attempt(s) failed. Last error: ${errorMessage}`,
			type: errorType,
			statusCode: statusCode,
		},
	};
	return c.json(errorPayload, statusCode);
})

app.post('/v1/responses', async (c: Context) => {
	const authHeader = c.req.header('Authorization');
	const apiKey = authHeader?.split(' ')[1] || null;
	const envPassword = process.env.PASSWORD;
	const isPasswordAuth = !!(envPassword && apiKey && envPassword.trim() === apiKey.trim());
	if (!apiKey) return c.text('Unauthorized', 401);

	const abortController = new AbortController();

	// Headers and aux keys
	tavilyApiKey = c.req.header('x-tavily-api-key') || (isPasswordAuth ? process.env.TAVILY_API_KEY || null : null);
	pythonApiKey = c.req.header('x-python-api-key') || (isPasswordAuth ? process.env.PYTHON_API_KEY || null : null);
	pythonUrl = c.req.header('x-python-url') || (isPasswordAuth ? process.env.PYTHON_URL || null : null);
	semanticScholarApiKey = c.req.header('x-semantic-scholar-api-key') || (isPasswordAuth ? process.env.SEMANTIC_SCHOLAR_API_KEY || null : null);

	const body = await c.req.json();
	const {
		model,
		input,
		instructions = null,
		stream = false,
		temperature,
		top_p,
		top_k,
		max_output_tokens,
		stop_sequences,
		seed,
		presence_penalty,
		frequency_penalty,
		tool_choice,
		tools,
		// Reasoning: body.reasoning?.effort
		reasoning,
		previous_response_id,
		request_id,
		extra_body,
		text_verbosity,
		service_tier,
	} = body || {};

	// Provider keys and headers map
	const headers: Record<string, string> = {};
	c.req.raw.headers.forEach((value, key) => {
		headers[key.toLowerCase().replace(/-/g, '_')] = value;
	});
	const providerKeys = await getProviderKeys(headers, authHeader || null, isPasswordAuth);
	// Build tools using shared helper
	let aiSdkTools: Record<string, any> = buildAiSdkTools(model, tools, []);

	// Convert Responses API input => AI SDK messages using shared helper
	const toAiSdkMessages = async (): Promise<any[]> => {
		// Seed from previous stored conversation if provided
		let history: any[] = [];
		if (previous_response_id) {
			try {
				const store = getStoreWithConfig('responses', headers);
				const existing: any = await store.get(previous_response_id, { type: 'json' as any });
				if (existing && existing.messages && Array.isArray(existing.messages)) history = existing.messages;
			} catch { }
		}

		// Map `input` into messages
		const mapped = responsesInputToAiSdkMessages(input);

		// Prepend instructions as system only for first request (no continuation)
		if (!previous_response_id) {
			if (instructions) {
				history = [{ role: 'system', content: String(instructions) }, ...history];
			}
			const combined = [...history, ...mapped];
			// Add context messages using shared function
			const contextMessages = addContextMessages(combined, c);
			return processMessages(contextMessages);
		}
		return processMessages(history);
	};

	const messages = await toAiSdkMessages();

	// Provider options (map differences)
	const providerOptionsHeader = c.req.header('x-provider-options');
	const reasoning_effort = reasoning?.effort ?? undefined;
	const providerOptions = buildDefaultProviderOptions({
		providerOptionsHeader: providerOptionsHeader ?? null,
		thinking: undefined,
		reasoning_effort,
		extra_body,
		text_verbosity,
		service_tier,
	});

	// Rebuild tools based on actual messages context
	aiSdkTools = buildAiSdkTools(model, tools, messages);

	// Prepare providers to try
	const { providersToTry } = prepareProvidersToTry({ model, providerKeys, isPasswordAuth, authApiKey: apiKey });

	const commonParams = {
		messages,
		aiSdkTools,
		temperature,
		top_p,
		top_k,
		max_tokens: max_output_tokens,
		seed,
		stop_sequences,
		presence_penalty,
		frequency_penalty,
		tool_choice,
		abortSignal: abortController.signal,
		providerOptions,
	};

	// Storage preparation
	const responseId = request_id || randomId('resp');

	if (stream) {
		// Streaming SSE per OpenAI Responses API
		const streamResponse = new ReadableStream({
			async start(controller) {
				const createdAt = Math.floor(Date.now() / 1000);
				let sequenceNumber = 0;
				let outputIndex = 0;
				const outputItems: any[] = [];

				const baseResponseObj = {
					id: responseId,
					object: 'response',
					created_at: createdAt,
					status: 'in_progress',
					background: false,
					error: null,
					incomplete_details: null,
					instructions,
					max_output_tokens: max_output_tokens ?? null,
					max_tool_calls: null,
					model,
					output: outputItems,
					parallel_tool_calls: true,
					previous_response_id: previous_response_id || null,
					prompt_cache_key: null,
					reasoning: { effort: reasoning_effort ?? null, summary: null },
					safety_identifier: null,
					service_tier: "priority",
					store: true,
					temperature: temperature ?? 1,
					text: { format: { type: 'text' }, verbosity: "medium" },
					tool_choice: tool_choice || 'auto',
					tools: tools || [],
					top_logprobs: 0,
					top_p: top_p ?? 1,
					truncation: 'disabled',
					usage: null,
					user: null,
				} as any;

				const emit = (obj: any) => controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(obj)}\n\n`));

				// response.created
				emit({
					type: 'response.created',
					sequence_number: sequenceNumber++,
					response: baseResponseObj
				});

				// response.in_progress (optional second event)
				emit({
					type: 'response.in_progress',
					sequence_number: sequenceNumber++,
					response: { ...baseResponseObj, status: 'in_progress' }
				});

				const maxAttempts = Math.min(providersToTry.length, MAX_ATTEMPTS);
				let attemptsTried = 0;
				let lastStreamError: any = null;
				let textItemId: string | null = null;
				let collectedText = '';
				let accumulatedSources: Array<{ title: string, url: string, type: string }> = [];
				let reasoningItemId: string | null = null;
				let reasoningText = '';
				let reasoningSummaryIndex = 0;
				let functionCallItems: Map<string, { id: string, name: string, call_id: string, args: string, outputIndex: number }> = new Map();

				for (let i = 0; i < maxAttempts; i++) {
					if (abortController.signal.aborted) break;
					const attempt = providersToTry[i];
					if (!attempt) continue;
					try {
						attemptsTried++;
						const gw = getGatewayForAttempt(attempt);
						const commonOptions = buildCommonOptions(gw, attempt, commonParams);
						const result = streamText(commonOptions);

						for await (const part of (result as any).fullStream) {
							if (abortController.signal.aborted) throw new Error('aborted');

							switch (part.type) {
								case 'source': {
									// Accumulate sources for later inclusion in content_part.done
									accumulatedSources.push({
										title: part.title || '',
										url: part.url || '',
										type: part.sourceType + '_citation',
									});
									break;
								}
								case 'reasoning-start': {
									if (!reasoningItemId) {
										reasoningItemId = randomId('rs');
										reasoningSummaryIndex = 0;
										const reasoningItem = {
											id: reasoningItemId,
											type: 'reasoning',
											summary: []
										};
										outputItems.push(reasoningItem);

										// Emit output_item.added for reasoning start
										emit({
											type: 'response.output_item.added',
											sequence_number: sequenceNumber++,
											output_index: outputIndex,
											item: reasoningItem
										});

										// Emit reasoning_summary_part.added for the text part
										emit({
											type: 'response.reasoning_summary_part.added',
											sequence_number: sequenceNumber++,
											item_id: reasoningItemId,
											output_index: outputIndex,
											summary_index: reasoningSummaryIndex,
											part: {
												type: 'summary_text',
												text: ''
											}
										});
									}
									break;
								}
								case 'reasoning-delta': {
									const delta = (part.delta ?? part.text ?? '');
									reasoningText += delta;

									// Emit reasoning_summary_text.delta
									emit({
										type: 'response.reasoning_summary_text.delta',
										sequence_number: sequenceNumber++,
										item_id: reasoningItemId,
										output_index: outputIndex,
										summary_index: reasoningSummaryIndex,
										delta: delta
									});
									break;
								}
								case 'reasoning-end': {
									if (reasoningItemId) {
										// Emit reasoning_summary_text.done
										emit({
											type: 'response.reasoning_summary_text.done',
											sequence_number: sequenceNumber++,
											item_id: reasoningItemId,
											output_index: outputIndex,
											summary_index: reasoningSummaryIndex,
											text: reasoningText
										});

										// Emit reasoning_summary_part.done
										emit({
											type: 'response.reasoning_summary_part.done',
											sequence_number: sequenceNumber++,
											item_id: reasoningItemId,
											output_index: outputIndex,
											summary_index: reasoningSummaryIndex,
											part: {
												type: 'summary_text',
												text: reasoningText
											}
										});

										// Create final reasoning item with summary
										const reasoningItem = {
											id: reasoningItemId,
											type: 'reasoning',
											summary: [{
												type: 'summary_text',
												text: reasoningText
											}]
										};

										// Update the item in outputItems
										const itemIndex = outputItems.findIndex(item => item.id === reasoningItemId);
										if (itemIndex >= 0) outputItems[itemIndex] = reasoningItem;

										// Emit output_item.done
										emit({
											type: 'response.output_item.done',
											sequence_number: sequenceNumber++,
											output_index: outputIndex,
											item: reasoningItem
										});

										outputIndex++;
									}
									break;
								}
								case 'text-start': {
									if (!textItemId) {
										textItemId = randomId('msg');
										const textOutputIndex = outputIndex + 1;
										const textItem = {
											id: textItemId,
											type: 'message',
											status: 'in_progress',
											role: 'assistant',
											content: []
										};
										outputItems.push(textItem);
										emit({
											type: 'response.output_item.added',
											sequence_number: sequenceNumber++,
											output_index: textOutputIndex,
											item: textItem
										});

										// Emit content_part.added for output_text
										emit({
											type: 'response.content_part.added',
											sequence_number: sequenceNumber++,
											item_id: textItemId,
											output_index: textOutputIndex,
											content_index: 0,
											part: {
												type: 'output_text',
												text: '',
											}
										});

										outputIndex = textOutputIndex;
									}
									break;
								}
								case 'text-delta': {
									collectedText += part.text;
									emit({
										type: 'response.output_text.delta',
										sequence_number: sequenceNumber++,
										item_id: textItemId,
										output_index: outputIndex,
										content_index: 0,
										delta: part.text
									});
									break;
								}
								case 'tool-input-start': {
									if (!EXCLUDED_TOOLS.has(part.toolName)) {
										// Use a unique tracking key that won't conflict
										const trackingKey = part.toolCallId || part.id;
										const funcItemId = randomId('fc');
										const callId = part.toolCallId || randomId('call');
										const currentOutputIndex = outputIndex + 1;

										const functionItem = {
											id: funcItemId,
											type: 'function_call',
											status: 'in_progress',
											arguments: '',
											call_id: callId,
											name: part.toolName
										};

										functionCallItems.set(trackingKey, {
											id: funcItemId,
											name: part.toolName,
											call_id: callId,
											args: '',
											outputIndex: currentOutputIndex
										});
										outputItems.push(functionItem);

										emit({
											type: 'response.output_item.added',
											sequence_number: sequenceNumber++,
											output_index: currentOutputIndex,
											item: functionItem
										});
									}
									break;
								}
								case 'tool-call': {
									// Final tool call with complete input - finalize arguments
									if (!EXCLUDED_TOOLS.has(part.toolName)) {
										const trackingKey = part.toolCallId || part.id;
										const funcCall = functionCallItems.get(trackingKey);

										if (funcCall) {
											const finalArgs = JSON.stringify(part.input ?? {});

											emit({
												type: 'response.function_call_arguments.done',
												sequence_number: sequenceNumber++,
												item_id: funcCall.id,
												output_index: funcCall.outputIndex,
												arguments: finalArgs
											});

											// Update function call item with final arguments
											const updatedItem = {
												id: funcCall.id,
												type: 'function_call',
												status: 'in_progress',
												arguments: finalArgs,
												call_id: funcCall.call_id,
												name: funcCall.name
											};

											// Update the item in outputItems and store final args
											const itemIndex = outputItems.findIndex(item => item.id === funcCall.id);
											if (itemIndex >= 0) outputItems[itemIndex] = updatedItem;
											funcCall.args = finalArgs; // Store for tool-result
										}
									}
									break;
								}
								case 'tool-input-delta': {
									// Find the function call by tool call ID or part ID
									const trackingKey = part.toolCallId || part.id;
									const funcCall = functionCallItems.get(trackingKey);

									if (funcCall && !EXCLUDED_TOOLS.has(funcCall.name)) {
										const delta = part.delta ?? '';
										funcCall.args += delta;

										emit({
											type: 'response.function_call_arguments.delta',
											sequence_number: sequenceNumber++,
											item_id: funcCall.id,
											output_index: funcCall.outputIndex,
											delta
										});
									}
									break;
								}
								case 'tool-result': {
									// Find the function call by tool call ID or tool name
									const trackingKey = part.toolCallId || part.id;
									const funcCall = functionCallItems.get(trackingKey);

									if (funcCall && !EXCLUDED_TOOLS.has(funcCall.name)) {
										// Mark function call as completed with the result
										const completedItem = {
											id: funcCall.id,
											type: 'function_call',
											status: 'completed',
											arguments: funcCall.args || '{}',
											call_id: funcCall.call_id,
											name: funcCall.name
										};

										// Update the item in outputItems
										const itemIndex = outputItems.findIndex(item => item.id === funcCall.id);
										if (itemIndex >= 0) outputItems[itemIndex] = completedItem;

										emit({
											type: 'response.output_item.done',
											sequence_number: sequenceNumber++,
											output_index: funcCall.outputIndex,
											item: completedItem
										});

										// Remove from tracking and increment global output index
										functionCallItems.delete(trackingKey);
										outputIndex = Math.max(outputIndex, funcCall.outputIndex);
									}
									break;
								}
								case 'finish': {
									// Finalize text item if exists
									if (textItemId) {
										// Emit content_part.done with accumulated annotations
										emit({
											type: 'response.content_part.done',
											sequence_number: sequenceNumber++,
											item_id: textItemId,
											output_index: outputIndex,
											content_index: 0,
											part: {
												type: 'output_text',
												text: collectedText,
												annotations: accumulatedSources
											}
										});

										const completedTextItem = {
											id: textItemId,
											type: 'message',
											status: 'completed',
											role: 'assistant',
											content: [{ type: 'output_text', text: collectedText, annotations: accumulatedSources }]
										};

										// Update the item in outputItems
										const itemIndex = outputItems.findIndex(item => item.id === textItemId);
										if (itemIndex >= 0) outputItems[itemIndex] = completedTextItem;

										emit({
											type: 'response.output_item.done',
											sequence_number: sequenceNumber++,
											output_index: outputIndex,
											item: completedTextItem
										});
									}

									const filteredOutput = outputItems.filter(item => item.type !== 'function_call');

									const completed = {
										...baseResponseObj,
										status: 'completed',
										output: filteredOutput,
										usage: part.totalUsage ? {
											input_tokens: part.totalUsage.inputTokens,
											output_tokens: part.totalUsage.outputTokens,
											total_tokens: part.totalUsage.totalTokens
										} : null
									};
									emit({
										type: 'response.completed',
										sequence_number: sequenceNumber++,
										response: completed
									});

									// Save conversation
									try {
										const store = getStoreWithConfig('responses', headers);
										await store.setJSON(responseId, {
											id: responseId,
											messages: [...messages, { role: 'assistant', content: collectedText }],
											assistant: collectedText
										});
									} catch { }
									controller.close();
									return;
								}
								case 'error': {
									if ([429, 401].includes((part as any)?.error?.statusCode)) throw new Error((part as any)?.error || 'Streaming provider error');
									emit({
										type: 'error',
										sequence_number: sequenceNumber++,
										code: (part as any)?.error?.statusCode || 'ERR',
										message: part.error,
										param: null
									});
									break;
								}
							}
						}
					} catch (err: any) {
						if (abortController.signal.aborted || err?.message === 'aborted' || err?.name === 'AbortError') {
							emit({
								type: 'response.failed',
								sequence_number: sequenceNumber++,
								response: {
									id: responseId,
									object: 'response',
									status: 'failed',
									error: { code: 'request_aborted', message: 'Request was aborted by the user' }
								}
							});
							controller.close();
							return;
						}
						lastStreamError = err;
						continue;
					}
				}

				// All attempts failed
				const msg = lastStreamError?.message || 'An unknown error occurred';
				emit({
					type: 'response.failed',
					sequence_number: sequenceNumber++,
					response: {
						id: responseId,
						object: 'response',
						status: 'failed',
						error: { code: 'server_error', message: msg }
					}
				});
				controller.close();
			},
		});
		return new Response(streamResponse, { headers: { 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' } });
	}

	// Non-streaming path
	const maxAttempts = Math.min(providersToTry.length, MAX_ATTEMPTS);
	let lastError: any;
	for (let i = 0; i < maxAttempts; i++) {
		const attempt = providersToTry[i];
		if (!attempt) continue;
		try {
			const gw = getGatewayForAttempt(attempt);
			const commonOptions = buildCommonOptions(gw, attempt, commonParams);
			const result = await generateText(commonOptions);
			// Transform result.sources to annotations format  
			const annotations = result.sources ? result.sources.map((source: any) => ({
				title: source.title || '',
				url: source.url || '',
				type: (source.sourceType || 'url') + '_citation'
			})) : [];
			// Construct Responses API output
			const inputNormalized = typeof input === 'string'
				? [{ type: 'input_text', text: input }]
				: (Array.isArray(input) ? input : (input ? [input] : []));
			const output = result.text ? [{
				type: 'message',
				id: randomId('msg'),
				status: 'completed',
				role: 'assistant',
				content: [{ type: 'output_text', text: result.text, annotations }]
			}] : [];
			const responsePayload = {
				id: responseId,
				object: 'response',
				created_at: Math.floor(Date.now() / 1000),
				status: 'completed',
				error: null,
				incomplete_details: null,
				input: inputNormalized,
				instructions,
				max_output_tokens: max_output_tokens ?? null,
				model,
				output,
				previous_response_id: previous_response_id || null,
				reasoning: { effort: reasoning_effort ?? null, summary: null },
				parallel_tool_calls: true,
				store: true,
				temperature: temperature ?? 1,
				text: { format: { type: 'text' } },
				tool_choice: tool_choice || 'auto',
				tools: tools || [],
				top_p: top_p ?? 1,
				truncation: 'disabled',
				usage: result.usage ? { input_tokens: result.usage.inputTokens, output_tokens: result.usage.outputTokens, total_tokens: result.usage.totalTokens } : null,
				user: null,
			} as any;

			try {
				const store = getStoreWithConfig('responses', headers);
				// Append assistant reply to the stored message history
				await store.setJSON(responseId, { id: responseId, messages: [...messages, { role: 'assistant', content: result.text }], assistant: result.text });
			} catch { }

			return c.json(responsePayload);
		} catch (error: any) {
			lastError = error;
			if (error.name === 'AbortError' || abortController.signal.aborted) return c.json({ error: { message: 'Request was aborted by the user', type: 'request_aborted', statusCode: 499 } }, 499 as any);
			if (i < maxAttempts - 1) continue;
			break;
		}
	}

	const statusCode = lastError?.statusCode || 500;
	const errorPayload = { error: { message: lastError?.message || 'All attempts failed', type: lastError?.type, statusCode } };
	return c.json(errorPayload, statusCode);
});

// Shared models handler function
async function handleModelsRequest(c: Context) {
	// Helper function to fetch models from custom provider
	async function fetchProviderModels(providerName: string, apiKey: string) {
		const config = SUPPORTED_PROVIDERS[providerName as keyof typeof SUPPORTED_PROVIDERS];
		if (!config) {
			throw new Error(`Unsupported provider: ${providerName}`);
		}
		let modelsEndpoint;
		if (providerName === 'github') {
			modelsEndpoint = config.baseURL.replace('inference', 'catalog/models');
		} else {
			modelsEndpoint = `${config.baseURL}/models`;
		}

		let response;
		if (providerName === 'gemini') {
			modelsEndpoint = modelsEndpoint + '?key=' + apiKey;
			response = await fetch(modelsEndpoint);
		} else {
			response = await fetch(modelsEndpoint, {
				method: 'GET',
				headers: {
					'Authorization': `Bearer ${apiKey}`,
					'Content-Type': 'application/json',
				},
			});
		}

		if (!response.ok) {
			throw new Error(`Provider ${providerName} models API failed: ${response.status} ${response.statusText}`);
		}

		const data = await response.json() as any;
		if (providerName === 'gemini') {
			return {
				data: data.models.map((model: any) => ({
					id: model.name,
					name: model.displayName,
					description: model.description || '',
				}))
			}
		} else if (providerName === 'github') {
			return {
				data: data
			}
		}

		return data;
	}

	// Helper function to get provider API keys from request headers
	function getProviderKeysFromHeaders(headers: Record<string, string>, isPasswordAuth: boolean = false) {
		const providerKeys: Record<string, string[]> = {};

		for (const provider of Object.keys(SUPPORTED_PROVIDERS)) {
			const keyName = `x-${provider}-api-key`;
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
		const commonExclusions = [
			'gemma', 'rerank', 'distill', 'parse', 'embed', 'bge-', 'tts', 'phi', 'live', 'audio', 'lite',
			'qwen2', 'qwen-2', 'qwen1', 'qwq', 'qvq', 'gemini-2.0', 'gemini-1', 'learnlm', 'gemini-exp',
			'turbo', 'claude-3', 'voxtral', 'pixtral', 'mixtral', 'ministral', '-24', 'moderation', 'saba', '-ocr-',
			'transcribe', 'image', 'dall', 'davinci', 'babbage'
		];
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
		} else if (providerName === 'chatgpt' && (modelId.split('-').length - 1) > 2) {
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
					if (!currentApiKey) {
						throw new Error('No valid gateway API key found');
					}

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
					if (!providerApiKey) {
						throw new Error(`No valid API key found for provider: ${providerName}`);
					}

					let formattedModels: any[] = [];

					// Check if this provider has a custom model list (doesn't support /models endpoint)
					if (CUSTOM_MODEL_LISTS[providerName as keyof typeof CUSTOM_MODEL_LISTS]) {
						const customModels = CUSTOM_MODEL_LISTS[providerName as keyof typeof CUSTOM_MODEL_LISTS];
						formattedModels = customModels.map(model => ({
							id: `${providerName}/${model.id}`,
							name: model.name,
							object: 'model',
							created: 0,
							owned_by: providerName,
						})).filter(model => shouldIncludeModel(model, providerName));
					} else {
						// Use regular API call for providers that support /models endpoint
						const providerModels = await fetchProviderModels(providerName, providerApiKey);
						formattedModels = (providerModels as any).data?.map((model: any) => ({
							id: `${providerName}/${model.id.replace('models/', '')}`,
							name: `${model.name?.replace(' (free)', '') || parseModelDisplayName(model.id)}`,
							description: model.description || model.summary || '',
							object: 'model',
							created: model.created || 0,
							owned_by: model.owned_by || providerName,
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
	};
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
}

app.get('/v1/models', handleModelsRequest);
app.post('/v1/models', handleModelsRequest);

app.get('/*', (c: Context) => {
	return c.text('Running')
})

export default (request: Request, context: any) => {
	geo = context.geo || null;
	return app.fetch(request);
}
