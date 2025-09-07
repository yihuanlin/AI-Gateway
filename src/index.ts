import { Hono } from 'hono'
import { cors } from 'hono/cors'
import type { Context } from 'hono'
import { generateText, stepCountIs, streamText, tool } from 'ai'
import { createGateway } from '@ai-sdk/gateway'
import { createOpenAICompatible } from '@ai-sdk/openai-compatible'
import { openai, createOpenAI } from '@ai-sdk/openai'
import { google, createGoogleGenerativeAI } from '@ai-sdk/google'
import { groq, createGroq } from '@ai-sdk/groq';
import { SUPPORTED_PROVIDERS, getProviderKeys } from './shared/providers.mts'
import { getStoreWithConfig } from './shared/store.mts'
import { string, number, boolean, array, object, optional, int, enum as zenum } from 'zod/mini'

const app = new Hono()
const TEXT_ENCODER = new TextEncoder();
const SEARCH_TOOLS = new Set(['web_search', 'fetch', 'google_search', 'web_search_preview', 'url_context', 'browser_search', 'scholar_search', 'paper_recommendations', 'ensembl_api']);
const CODE_TOOLS = new Set(['code_execution', 'python_executor']);
const EXCLUDED_TOOLS = new Set([...SEARCH_TOOLS, ...CODE_TOOLS]);
const RESEARCH_KEYWORDS = ['research', 'paper'];
const MAX_ATTEMPTS = 3;

let accumulatedSources: Array<{ title: string, url: string, type: string, start_index: number, end_index: number }> = [];
let tavilyApiKey: string | null = null;
let pythonApiKey: string | null = null;
let pythonUrl: string | null = null;
let semanticScholarApiKey: string | null = null;
let geo: {
	city?: string;
	country?: { code: string; name: string };
	timezone?: string;
	subdivision?: { code: string; name: string };
} | null = null;
let isResearchMode: boolean = false;

// Helper functions
async function fetchCopilotToken(apiKey: string): Promise<string> {
	const store = await getStoreWithConfig('copilot-tokens');
	const res = await store.getWithMetadata('token', { type: 'text' }) as { data?: string | null, metadata?: { expiration?: number } | null } | null;
	const token = res?.data ?? null;
	const expiration = res?.metadata?.expiration ?? null;
	const now = Date.now();
	if (token && expiration && now < expiration) {
		return token;
	}
	const config = SUPPORTED_PROVIDERS.copilot;
	const response = await fetch(config.tokenURL, {
		method: 'GET',
		headers: {
			'Authorization': `Token ${apiKey}`,
			'Accept': 'application/json',
		},
	});

	if (!response.ok) {
		throw new Error(`Failed to fetch Copilot token: ${response.status} ${response.statusText}`);
	}

	const data = await response.json() as any;
	const key = data.token;
	const expires = now + data.refresh_in * 1000; // (data.expires_in - 60) * 1000 || (now + 86400000);
	console.log('Fetched new Copilot token, expires at', expires);
	await store.set('token', key, {
		metadata: {
			expiration: expires
		}
	});
	return key;
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

function startsWithThinking(text: string): boolean {
	// Remove leading whitespace and empty newlines
	const lines = text.split('\n');
	const nonEmptyLines: string[] = [];

	for (const line of lines) {
		const trimmed = line.trim();
		if (trimmed !== '') {
			nonEmptyLines.push(trimmed);
		}
	}

	// Check if the first or second non-empty line starts with >
	if (nonEmptyLines.length >= 1 && nonEmptyLines[0] && nonEmptyLines[0].startsWith('>')) {
		return true;
	}
	if (nonEmptyLines.length >= 2 && nonEmptyLines[1] && nonEmptyLines[1].startsWith('>')) {
		return true;
	}

	return false;
}

function findThinkingIndex(text: string): number {
	// Find where the actual reasoning content (>) starts
	// Look for the first line that starts with > and return the position of the >
	const lines = text.split('\n');
	let currentIndex = 0;

	for (let i = 0; i < lines.length; i++) {
		const line = lines[i];
		if (!line) {
			currentIndex += 1; // Just the newline
			continue;
		}

		const trimmedLine = line.trim();

		// If this line starts with >, find the exact position of the >
		if (trimmedLine.startsWith('>')) {
			return currentIndex;
		}

		// Add the length of this line plus the newline character
		currentIndex += line.length + 1; // +1 for the \n
	}

	return -1; // No reasoning content found
}

function cleanPoeReasoningDelta(delta: string, isFirstDelta: boolean = false): string {
	let cleanedDelta = delta;

	// For first delta, remove everything before the first line that starts with >
	if (isFirstDelta) {
		const lines = cleanedDelta.split('\n');
		let firstQuoteLineIndex = -1;

		// Find the first line that starts with >
		for (let i = 0; i < lines.length; i++) {
			const line = lines[i];
			if (line && line.trim().startsWith('>')) {
				firstQuoteLineIndex = i;
				break;
			}
		}

		// If we found a quote line, remove everything before it
		if (firstQuoteLineIndex >= 0) {
			cleanedDelta = lines.slice(firstQuoteLineIndex).join('\n');
		}
	}

	// Remove ">" quotation markdown from the beginning of lines
	const lines = cleanedDelta.split('\n');
	const cleanedLines = lines.map(line => {
		if (line.startsWith('> ')) {
			return line.substring(2); // Remove "> "
		} else if (line.startsWith('>')) {
			return line.substring(1); // Remove ">"
		}
		return line;
	});

	return cleanedLines.join('\n');
}

function cleanPoeReasoning(reasoning: string): string {
	// Remove everything before the first line that starts with >
	const lines = reasoning.split('\n');
	let firstQuoteLineIndex = -1;

	// Find the first line that starts with >
	for (let i = 0; i < lines.length; i++) {
		const line = lines[i];
		if (line && line.trim().startsWith('>')) {
			firstQuoteLineIndex = i;
			break;
		}
	}

	// If no quote line found, return empty
	if (firstQuoteLineIndex < 0) {
		return '';
	}

	// Take only the lines from the first quote line onwards
	const reasoningLines = lines.slice(firstQuoteLineIndex);

	// Remove ">" quotation markdown from the beginning of lines
	const cleanedLines = reasoningLines.map(line => {
		if (line.startsWith('> ')) {
			return line.substring(2); // Remove "> "
		} else if (line.startsWith('>')) {
			return line.substring(1); // Remove ">"
		}
		return line;
	});

	return cleanedLines.join('\n').trim();
}

// Helper function to extract Poe reasoning content from text
function extractPoeReasoning(text: string): { reasoning: string, content: string } {
	if (!startsWithThinking(text)) {
		return { reasoning: '', content: text };
	}

	// Find where "Thinking..." starts
	const thinkingIndex = findThinkingIndex(text);
	if (thinkingIndex < 0) {
		return { reasoning: '', content: text };
	}

	const beforeThinking = text.substring(0, thinkingIndex);
	const afterThinking = text.substring(thinkingIndex);

	// Find the end of reasoning (two consecutive newlines where the second doesn't start with >)
	const lines = afterThinking.split('\n');
	let reasoningEndIndex = -1;

	for (let i = 0; i < lines.length - 1; i++) {
		const currentLine = lines[i];
		const nextLine = lines[i + 1];
		if (currentLine === '' && nextLine && nextLine !== '' && !nextLine.startsWith('>')) {
			// Found the end, calculate the position in the original text
			const linesBefore = lines.slice(0, i).join('\n');
			reasoningEndIndex = thinkingIndex + linesBefore.length + 1; // +1 for the newline
			break;
		}
	}

	if (reasoningEndIndex >= 0) {
		let reasoning = text.substring(thinkingIndex, reasoningEndIndex);
		const content = beforeThinking + text.substring(reasoningEndIndex + 1); // +1 to skip the newline

		// Clean up reasoning content
		reasoning = cleanPoeReasoning(reasoning);

		return { reasoning, content };
	} else {
		// No clear end found, treat everything after Thinking... as reasoning
		let reasoning = afterThinking;

		// Clean up reasoning content
		reasoning = cleanPoeReasoning(reasoning);

		return { reasoning, content: beforeThinking };
	}
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

async function processMessages(contextMessages: any[], options?: { extractMarkdownImages?: boolean }): Promise<any[]> {
	const processedMessages: any[] = [];

	const extractFromText = async (text: string): Promise<{ cleaned: string; files: any[] }> => {
		const imgRegex = /!\[Generated Image\]\(([^\)]+)\)/g;
		let match: RegExpExecArray | null;
		let cleaned = text;
		const files: any[] = [];
		while ((match = imgRegex.exec(text)) !== null) {
			const urlStr = String(match[1] || '').trim();
			if (!urlStr) continue;
			let mediaType: string = 'image/png';
			let data: Uint8Array | null = null;
			if (urlStr.startsWith('data:')) {
				const dm = urlStr.match(/^data:([^;]+);base64,(.+)$/);
				if (dm && dm[1] && dm[2]) {
					mediaType = String(dm[1]);
					const b64 = dm[2];
					const bin = atob(b64);
					const bytes = new Uint8Array(bin.length);
					for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
					data = bytes;
				}
			} else if (urlStr.startsWith('http://') || urlStr.startsWith('https://')) {
				try {
					const resp = await fetch(urlStr);
					const ct = resp.headers.get('content-type');
					if (ct) mediaType = ct.split(';')[0] as string;
					const ab = await resp.arrayBuffer();
					data = new Uint8Array(ab);
				} catch { }
			}
			if (data) {
				files.push({ type: 'file', data, mediaType });
				cleaned = cleaned.replace(match[0], '').trim();
			}
		}
		return { cleaned, files };
	};

	const appendToolCallsToContent = async (content: any, tool_calls: any[]): Promise<any> => {
		if (!Array.isArray(tool_calls) || tool_calls.length === 0) return content;
		// Build appended string for each tool call including optional results
		const buildBlock = async (toolCall: any): Promise<string> => {
			const toolName = toolCall?.function?.name;
			const args = toolCall?.function?.arguments;
			let block = `\n<tool_use_result>\n  <name>${toolName}</name>\n  <arguments>${args}</arguments>`;
			// Find matching tool result in chat or responses format
			const toolResultMessage = contextMessages.find((m: any) => m.role === 'tool' && m.tool_call_id === toolCall.id);
			if (toolResultMessage) {
				let resultText = toolResultMessage.content || toolResultMessage.output;
				if (typeof resultText === 'string') {
					try {
						const parsed = JSON.parse(resultText);
						if (Array.isArray(parsed) && parsed[0] && parsed[0].text) {
							resultText = parsed[0].text;
						}
					} catch { }
				}
				block += `\n  <result>${resultText}</result>`;
			}
			block += `\n</tool_use_result>`;
			return block;
		};
		if (content === undefined || content === null) {
			let appended = '';
			for (const tc of tool_calls) appended += await buildBlock(tc);
			return appended;
		}
		if (typeof content === 'string') {
			let appended = content;
			for (const tc of tool_calls) appended += await buildBlock(tc);
			return appended;
		}
		if (Array.isArray(content)) {
			// Ensure there is a text part to append to; if not, create one
			let lastText = content.length > 0 ? content[content.length - 1] : null;
			if (!lastText || lastText.type !== 'text' || typeof lastText.text !== 'string') {
				lastText = { type: 'text', text: '' };
				content.push(lastText);
			}
			for (const tc of tool_calls) {
				lastText.text += await buildBlock(tc);
			}
			return content;
		}
		if (typeof content === 'object' && content && typeof content.text === 'string') {
			let appended = content.text;
			for (const tc of tool_calls) appended += await buildBlock(tc);
			return { ...content, text: appended };
		}
		return content;
	};

	for (let mi = 0; mi < contextMessages.length; mi++) {
		const message = contextMessages[mi];
		if (!message) continue;
		if (message.role === 'tool') continue;
		if (message.type === 'function_call') continue;
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
			const id = message.call_id;
			let toolName = 'unknown_tool';
			let args = 'NA';
			for (const msg of contextMessages) {
				if (msg.type === 'function_call' && msg.call_id === id) {
					toolName = msg.name || 'unknown_tool';
					args = msg.arguments || 'NA';
					break;
				}
			}
			processedMessages.push({
				role: 'assistant',
				content: `<tool_use_result>\n  <name>${toolName}</name>\n  <<arguments>>${args}</<arguments>>\n  <result>${resultText}</result>\n</tool_use_result>`
			});
			continue;
		}

		let nextMessage = message;

		if (message.role === 'assistant') {
			if (Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
				nextMessage.content = await appendToolCallsToContent(message.content, message.tool_calls);
			}
			if (options?.extractMarkdownImages) {
				if (typeof nextMessage.content === 'string') {
					const { cleaned, files } = await extractFromText(nextMessage.content);
					if (files.length > 0) {
						const parts: any[] = [];
						if (cleaned) parts.push({ type: 'text', text: cleaned });
						parts.push(...files);
						nextMessage.content = parts;
					}
				} else if (Array.isArray(nextMessage.content)) {
					const updatedParts: any[] = [];
					const appendedFiles: any[] = [];
					for (const part of nextMessage.content) {
						if (part && part.type === 'text' && typeof part.text === 'string') {
							const { cleaned, files } = await extractFromText(part.text);
							updatedParts.push({ ...part, text: cleaned });
							if (files.length > 0) appendedFiles.push(...files);
						} else {
							updatedParts.push(part);
						}
					}
					if (appendedFiles.length > 0) {
						nextMessage.content = [...updatedParts, ...appendedFiles];
					} else {
						nextMessage.content = updatedParts;
					}
				} else if (typeof nextMessage.content === 'object' && nextMessage.content && typeof nextMessage.content.text === 'string') {
					const { cleaned, files } = await extractFromText(nextMessage.content.text);
					if (files.length > 0) {
						const parts: any[] = [];
						if (cleaned) parts.push({ type: 'text', text: cleaned });
						parts.push(...files);
						nextMessage.content = parts;
					}
				}
			}
		} else if (message.role === 'user' && Array.isArray(message.content)) {
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

		if (typeof nextMessage.content === 'object' && !Array.isArray(nextMessage.content) && nextMessage.content?.text) {
			nextMessage.content = [nextMessage.content];
		}

		processedMessages.push(nextMessage);
	}
	return processedMessages;
}

async function createCustomProvider(providerName: string, apiKey: string) {
	const config = SUPPORTED_PROVIDERS[providerName as keyof typeof SUPPORTED_PROVIDERS];
	if (!config) {
		throw new Error(`Unsupported provider: ${providerName}`);
	}

	switch (providerName) {
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
		case 'copilot':
			const copilotToken = await fetchCopilotToken(apiKey);
			return createOpenAICompatible({
				name: 'custom',
				apiKey: copilotToken,
				baseURL: config.baseURL,
				includeUsage: true,
				headers: {
					"editor-version": "vscode/1.103.1",
					"copilot-vision-request": "true",
					"editor-plugin-version": "copilot-chat/0.30.1",
					"user-agent": "GitHubCopilotChat/0.30.1"
				},
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
	reasoning_summary?: any,
	store?: any,
	model?: any,
	search?: boolean,
}) {
	const { providerOptionsHeader, thinking, reasoning_effort, extra_body, text_verbosity, service_tier, reasoning_summary, store, model, search } = args;
	if (model.startsWith('anthropic/')) {
		return {
			anthropic:
			{
				thinking: thinking || { type: "enabled", budgetTokens: isResearchMode ? 32000 : 4000 },
				cacheControl: { type: "ephemeral" },
			}
		};
	}
	if (model.startsWith('xai/')) {
		return {
			xai: {
				...(search && {
					searchParameters: {
						mode: 'auto',
						returnCitations: true,
						maxSearchResults: isResearchMode ? 30 : 15,
						sources: [
							{
								type: 'web',
								...(!isResearchMode && geo?.country?.code ?
									{ country: geo.country.code } : {})
							},
							...(!isResearchMode ? [{
								type: 'x'
							},
							{
								type: 'news',
								...(!isResearchMode && geo?.country?.code ?
									{ country: geo.country.code } : {})
							}] : []),
						]
					}
				}),
				...((model.startsWith('xai/grok-3') && (reasoning_effort || isResearchMode)) && { reasoningEffort: reasoning_effort || (isResearchMode ? 'high' : 'low') }),
			}
		};
	}
	if (model.startsWith('perplexity/')) {
		if (isResearchMode) {
			return {
				perplexity: {
					search_mode: 'academic',
					web_search_options: {
						search_context_size: 'high',
					},
				}
			};
		}
		return {};
	}
	const providerOptions = providerOptionsHeader ? JSON.parse(providerOptionsHeader) : {
		openai: {
			reasoningEffort: reasoning_effort || (isResearchMode ? 'high' : "medium"),
			reasoningSummary: reasoning_summary || "auto",
			textVerbosity: text_verbosity || "medium",
			serviceTier: service_tier || "auto",
			store: model.startsWith('chatgpt/') ? store : false,
			promptCacheKey: 'ai-gateway',
		},
		google: {
			...(!model.toLowerCase().includes('image') ? {
				thinkingConfig: {
					thinkingBudget: extra_body?.google?.thinking_config?.thinking_budget || -1,
					includeThoughts: true,
				}
			} : {
				responseModalities: ["IMAGE", "TEXT"]
			}),
		},
		custom: {
			reasoning_effort: reasoning_effort || (isResearchMode ? 'high' : "medium"),
			...(extra_body && { extra_body }),
			...(thinking && { thinking }),
		},
	};
	return providerOptions;
}

type Attempt = { type: 'gateway' | 'custom', name?: string, apiKey: string, model: string };

const getGatewayForAttempt = async (attempt: Attempt) => {
	if (attempt.type === 'gateway') {
		const gatewayOptions: any = { apiKey: attempt.apiKey };
		if (attempt.model === 'anthropic/claude-sonnet-4') {
			gatewayOptions.headers = { 'anthropic-beta': 'context-1m-2025-08-07' };
		}
		return createGateway(gatewayOptions);
	}
	return await createCustomProvider(attempt.name!, attempt.apiKey);
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
			if (item?.type?.includes('function')) {
				messages.push(item); // Handled by processMessages()
				continue;
			}

			const role = item?.role;
			if (typeof item?.content === 'string') {
				if (role === 'assistant' || role === 'user') {
					messages.push({ role, content: { type: 'text', text: item.content } });
					continue;
				} else if (role === 'system') {
					messages.push({ role, content: item.content });
					continue;
				}
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
				if (part.type.includes('text') && typeof part.text === 'string') {
					parts.push({ type: 'text', text: part.text });
				} else if (part.type === 'input_image' && role === 'assistant') {
					const imageSrc: string | undefined = typeof part?.image_url === 'string' ? part.image_url : (part?.image_url?.url || part?.url);
					if (imageSrc) {
						let mediaType = part?.media_type || part?.mediaType;
						if (!mediaType && typeof imageSrc === 'string' && imageSrc.startsWith('data:')) {
							const m = imageSrc.match(/^data:([^;]+);/);
							if (m) mediaType = m[1];
						}
						parts.push(mediaType ? { type: 'file', data: imageSrc, mediaType } : { type: 'file', data: imageSrc, mediaType: 'image/png' });
					}
				} else if (part.type.includes('image')) {
					const imageSrc = part?.image_url?.url || part?.url || part?.image || part?.data || (typeof part?.image_url === 'string' ? part.image_url : undefined);
					if (imageSrc) {
						const mediaType = part?.media_type || part?.mediaType;
						parts.push(mediaType ? { type: 'image', image: imageSrc, mediaType } : { type: 'image', image: imageSrc });
					}
				} else if (part.type.includes('file')) {
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
function buildAiSdkTools(model: string, userTools: any[] | undefined): Record<string, any> {
	let aiSdkTools: Record<string, any> = {};
	function buildResearchTools() {
		if (isResearchMode) {
			aiSdkTools.ensembl_api = ensemblApiTool;
			aiSdkTools.scholar_search = semanticScholarSearchTool;
			aiSdkTools.paper_recommendations = semanticScholarRecommendationsTool;
		}
	}

	if (model.startsWith('perplexity')) {
		buildResearchTools();
	} else if (Array.isArray(userTools)) {
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

		buildResearchTools();

		const googleIncompatible = (!['google', 'gemini'].some(prefix => model.startsWith(prefix)) || Object.keys(aiSdkTools).length > 0);

		if (model.startsWith('openai')) {
			aiSdkTools.web_search_preview = openai.tools.webSearchPreview({
				searchContextSize: isResearchMode ? 'high' : 'medium',
				...(!isResearchMode && geo ? {
					userLocation: {
						type: 'approximate',
						...(geo.city && { city: geo.city }),
						...(geo.subdivision?.name && { region: geo.subdivision.name }),
						...(geo.country?.code && { country: geo.country.code }),
						...(geo.timezone && { timezone: geo.timezone }),
					}
				} : {})
			});
			aiSdkTools.code_interpreter = openai.tools.codeInterpreter({});
		} else if (model.startsWith('groq/openai')) {
			aiSdkTools.browser_search = groq.tools.browserSearch({});
		} else if (googleIncompatible && !model.startsWith('xai')) {
			if (tavilyApiKey) aiSdkTools.web_search = tavilySearchTool;
		}
		if (googleIncompatible) {
			aiSdkTools.fetch = jinaReaderTool;
			if (!isResearchMode && !model.startsWith('openai') && pythonApiKey && pythonUrl) {
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

// Helper function to modify messages for Poe provider
function modifyMessagesForPoe(messages: any[], reasoning_effort?: string): any[] {
	if (!reasoning_effort) return messages;

	// Find the last user message
	const modifiedMessages = [...messages];
	for (let i = modifiedMessages.length - 1; i >= 0; i--) {
		const message = modifiedMessages[i];
		if (message.role === 'user') {
			// Clone the message to avoid mutating the original
			const modifiedMessage = { ...message };

			// Handle different content types
			if (typeof modifiedMessage.content === 'string') {
				modifiedMessage.content = `${modifiedMessage.content} --reasoning_effort "${reasoning_effort}"`;
			} else if (Array.isArray(modifiedMessage.content)) {
				// Find the last text part and append to it
				const contentCopy = [...modifiedMessage.content];
				for (let j = contentCopy.length - 1; j >= 0; j--) {
					const part = contentCopy[j];
					if (part.type === 'text' && typeof part.text === 'string') {
						contentCopy[j] = {
							...part,
							text: `${part.text} --reasoning_effort "${reasoning_effort}"`
						};
						break;
					}
				}
				modifiedMessage.content = contentCopy;
			}

			modifiedMessages[i] = modifiedMessage;
			break;
		}
	}

	return modifiedMessages;
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
		reasoning_effort?: string,
	}
) => {
	// Modify messages for Poe provider if reasoning_effort is provided
	const finalMessages = attempt.name === 'poe' ?
		modifyMessagesForPoe(params.messages, params.reasoning_effort) :
		params.messages;

	return {
		model: gw(attempt.model),
		messages: finalMessages,
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
		maxRetries: 0,
		onError: () => { }
	} as any;
};

const pythonExecutorTool = tool({
	description: 'Execute Python code remotely. Installed packages: numpy, pandas',
	inputSchema: object({
		code: string({ message: 'The Python code to execute.' }),
	}),
	providerOptions: {
		anthropic: {
			cacheControl: { type: 'ephemeral' },
		},
	},
	execute: async ({ code }: { code: string }, { abortSignal }: { abortSignal?: AbortSignal }) => {
		console.log(`Executing remote Python code: ${code.substring(0, 100)}...`);
		try {
			if (!pythonUrl) {
				return { error: 'Python URL is not configured' };
			}

			const controller = new AbortController();
			const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s

			if (abortSignal) {
				abortSignal.addEventListener('abort', () => {
					controller.abort();
				});
				if (abortSignal.aborted) {
					controller.abort();
				}
			}

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
				console.error(`Python server error (${response.status}): ${data?.error || text.slice(0, 500)}`);
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
			const message = error?.name === 'AbortError' ? 'Request to Python server timed out or was cancelled' : (error?.message || 'Unknown error');
			return { success: false, error: message };
		}
	},
});

const tavilySearchTool = tool({
	description: 'Web search using Tavily',
	inputSchema: object({
		query: string({ message: 'Search query' }),
		max_results: optional(number({ message: `Maximum number of results to return (default: ${isResearchMode ? '10' : '5'}, max: 20)` })),
		include_raw_content: optional(boolean({ message: 'Include the cleaned and parsed HTML content of each search result (default: false)' })),
		include_domains: optional(array(string({ message: 'Domain to include' }), { message: 'List of domains to include in the search' })),
		exclude_domains: optional(array(string({ message: 'Domain to exclude' }), { message: 'List of domains to exclude from the search' })),
		start_date: optional(string({ message: 'Start date for search results (format: YYYY-MM-DD)' })),
		deep_search: optional(boolean({ message: `Enable deep search for more comprehensive results (default: ${isResearchMode ? 'true' : 'false'})` })),
		include_images: optional(boolean({ message: 'Include images in the search results (default: false)' })),
	}),
	providerOptions: {
		anthropic: {
			cacheControl: { type: 'ephemeral' },
		},
	},
	execute: async (params, { abortSignal }: { abortSignal?: AbortSignal }) => {
		const { query, max_results, include_domains, exclude_domains, include_raw_content, start_date, deep_search, include_images } = params;
		console.log(`Tavily search with query: ${query}`);
		try {
			if (!tavilyApiKey) {
				return { error: 'Tavily API key is not configured' };
			}

			const maxResults = max_results || (isResearchMode ? 10 : 5);
			const includeRawContent = include_raw_content || false;
			const topic = query.toLowerCase().includes('news') ? 'news' : 'general';
			const search_depth = deep_search || isResearchMode ? 'advanced' : 'basic';
			const apiKeys = tavilyApiKey.split(',').map((key: string) => key.trim());
			let lastError: Error | null = null;
			const country: string | undefined = !isResearchMode
				? geo?.country?.name?.toLocaleLowerCase()
				: undefined;

			for (let i = 0; i < apiKeys.length; i++) {
				const currentApiKey = apiKeys[i];

				try {
					let searchPayload = {
						query,
						topic,
						start_date,
						search_depth,
						max_results: Math.min(maxResults, 20),
						include_raw_content: includeRawContent,
						auto_parameters: true,
						...(include_domains && { include_domains }),
						...(exclude_domains && { exclude_domains }),
						...(include_images && { include_images, include_image_descriptions: true }),
						...(country && { country })
					};

					const controller = new AbortController();
					const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout
					if (abortSignal) {
						abortSignal.addEventListener('abort', () => {
							controller.abort();
						});
						if (abortSignal.aborted) {
							controller.abort();
						}
					}

					const response = await fetch('https://api.tavily.com/search', {
						method: 'POST',
						headers: {
							'Content-Type': 'application/json',
							'Authorization': `Bearer ${currentApiKey}`
						},
						body: JSON.stringify(searchPayload),
						signal: controller.signal
					});
					clearTimeout(timeoutId);
					if (!response.ok) {
						const errorText = await response.text();
						throw new Error(`Tavily API error (${response.status}): ${errorText}`);
					}
					const data = await response.json() as any;
					const results = data.results || [];
					const images = data.images || [];

					// Add search results to accumulatedSources
					if (results && Array.isArray(results)) {
						results.forEach((result: any) => {
							if (result.url) {
								accumulatedSources.push({
									title: result.title || '',
									url: result.url,
									type: 'url_citation',
									start_index: 0,
									end_index: 0
								});
							}
						});
					}

					return {
						query,
						results: results || [],
						...(images.length > 0 && { images })
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
	description: 'Fetch web pages as markdown',
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
	}, { abortSignal }: { abortSignal?: AbortSignal }) => {
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

			if (abortSignal) {
				abortSignal.addEventListener('abort', () => {
					controller.abort();
				});
				if (abortSignal.aborted) {
					controller.abort();
				}
			}

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

			const text = await response.text();
			// Use the first non-empty line as the title
			let title = '';
			for (const line of text.split('\n')) {
				const trimmed = line.trim();
				if (trimmed !== '') {
					title = trimmed;
					break;
				}
			}
			accumulatedSources.push({
				title,
				url,
				type: 'url_citation',
				start_index: 0,
				end_index: 0
			});

			return text;

		} catch (error: any) {
			console.error(`Jina Reader error: ${error.message || 'Unknown error'}`);
			const message = error?.name === 'AbortError' ? 'Request was cancelled or timed out' : (error?.message || 'Unknown error');
			return {
				url,
				error: `Failed to fetch content: ${message}`,
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
	execute: async ({ path }: { path: string }, { abortSignal }: { abortSignal?: AbortSignal }) => {
		console.log(`Ensembl API request to path: ${path}`);
		try {
			const cleanPath = path.startsWith('/') ? path.slice(1) : path;
			const baseUrl = 'https://rest.ensembl.org';
			const fullUrl = `${baseUrl}/${cleanPath}`;

			const controller = new AbortController();
			const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

			if (abortSignal) {
				abortSignal.addEventListener('abort', () => {
					controller.abort();
				});
				if (abortSignal.aborted) {
					controller.abort();
				}
			}

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

			// Add fetched URL to accumulatedSources
			accumulatedSources.push({
				title: 'Ensembl API for ' + path,
				url: fullUrl,
				type: 'url_citation',
				start_index: 0,
				end_index: 0
			});

			return await response.json();

		} catch (error: any) {
			const message = error?.name === 'AbortError' ? 'Request to Ensembl API timed out or was cancelled' : (error?.message || 'Unknown error');
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
	execute: async (params, { abortSignal }: { abortSignal?: AbortSignal }) => {
		const { query, type = 'paper', limit = 10, offset = 0, year, venue, fieldsOfStudy, minCitationCount, publicationTypes } = params;
		console.log(`Semantic Scholar ${type} search: ${query}`);
		try {
			const baseUrl = 'https://api.semanticscholar.org/graph/v1';
			const endpoint = type === 'paper' ? 'paper/search' : 'author/search';
			const fields = type === 'paper'
				? 'url,title,abstract,citationCount,influentialCitationCount,openAccessPdf,publicationTypes,publicationDate,journal,authors.name,authors.hIndex,authors.affiliations'
				: 'url,name,affiliations,homepage,paperCount,citationCount,hIndex,papers.title,papers.year,papers.journal,papers.citationCount,papers.influentialCitationCount';

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
			const timeoutId = setTimeout(() => controller.abort(), 60000);

			if (abortSignal) {
				abortSignal.addEventListener('abort', () => {
					controller.abort();
				});
				if (abortSignal.aborted) {
					controller.abort();
				}
			}

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

			// Add URLs to accumulatedSources based on search type
			if (data.data && Array.isArray(data.data)) {
				data.data.forEach((item: any) => {
					if (type === 'paper') {
						const paperUrl = item.url || item.openAccessPdf?.url;
						if (paperUrl) {
							accumulatedSources.push({
								title: item.title || '',
								url: paperUrl,
								type: 'url_citation',
								start_index: 0,
								end_index: 0
							});
						}
					} else if (type === 'author') {
						if (item.url) {
							accumulatedSources.push({
								title: item.name || ' - Semantic Scholar',
								url: item.url,
								type: 'url_citation',
								start_index: 0,
								end_index: 0
							});
						}
						if (item.homepage) {
							accumulatedSources.push({
								title: item.name + ' - Homepage',
								url: item.homepage,
								type: 'url_citation',
								start_index: 0,
								end_index: 0
							});
						}
					}
				});
			}
			return data;

		} catch (error: any) {
			const message = error?.name === 'AbortError' ? 'Request to Semantic Scholar API timed out or was cancelled' : (error?.message || 'Unknown error');
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
		paperId: optional(string({ message: 'Single paper ID to get recommendations for, can be Semantic Scholar ID, DOI, or PMID etc.' })),
		positivePaperIds: optional(array(string({ message: 'Paper ID' }), { message: 'Array of paper IDs that represent positive examples (for batch recommendations)' })),
		negativePaperIds: optional(array(string({ message: 'Paper ID' }), { message: 'Array of paper IDs that represent negative examples (for batch recommendations)' })),
		limit: optional(number({ message: 'Maximum number of recommendations (default: 10, max: 100)' })),
		from: optional(zenum(['recent', 'all'], { message: 'Pool of papers to recommend from (default: recent; only works with single paper recommendations using paperId)' })),
	}),
	providerOptions: {
		anthropic: {
			cacheControl: { type: 'ephemeral' },
		},
	},
	execute: async (params, { abortSignal }: { abortSignal?: AbortSignal }) => {
		const { paperId, positivePaperIds, negativePaperIds, limit = 10, from = 'recent' } = params;
		console.log(`Semantic Scholar recommendations for: ${paperId || `${positivePaperIds?.length || 0} positive papers`}`);
		try {
			const baseUrl = 'https://api.semanticscholar.org/recommendations/v1';
			const fields = 'url,title,abstract,citationCount,influentialCitationCount,openAccessPdf,publicationTypes,publicationDate,journal,author';
			let url: string;
			let method: string = 'GET';
			let body: any = null;

			if (paperId) {
				// Single paper recommendation
				const params = new URLSearchParams();
				params.append('limit', Math.min(limit, 100).toString());
				params.append('from', from === 'all' ? 'all-cs' : 'recent');
				params.append('fields', fields);

				url = `${baseUrl}/papers/forpaper/${paperId}?${params.toString()}`;
			} else if (positivePaperIds && positivePaperIds.length > 0) {
				// Batch recommendations
				const params = new URLSearchParams();
				params.append('limit', Math.min(limit, 100).toString());
				params.append('fields', fields);

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

			if (abortSignal) {
				abortSignal.addEventListener('abort', () => {
					controller.abort();
				});
				if (abortSignal.aborted) {
					controller.abort();
				}
			}

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

			// Add recommendation URLs to accumulatedSources
			const recommendations = data.recommendedPapers || data.data || data;
			if (recommendations && Array.isArray(recommendations)) {
				recommendations.forEach((paper: any) => {
					const paperUrl = paper.url || paper.openAccessPdf?.url;
					if (paperUrl) {
						accumulatedSources.push({
							title: paper.title || '',
							url: paperUrl,
							type: 'url_citation',
							start_index: 0,
							end_index: 0
						});
					}
				});
			}

			return recommendations;

		} catch (error: any) {
			const message = error?.name === 'AbortError' ? 'Request to Semantic Scholar Recommendations API timed out or was cancelled' : (error?.message || 'Unknown error');
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

app.post('/v1/responses', async (c: Context) => {
	const authHeader = c.req.header('Authorization');
	const apiKey = authHeader?.split(' ')[1] || null;
	const envPassword = process.env.PASSWORD;
	const isPasswordAuth = !!(envPassword && apiKey && envPassword.trim() === apiKey.trim());
	if (!apiKey) return c.text('Unauthorized', 401);

	const abortController = new AbortController();
	if (c.req.raw?.signal) {
		c.req.raw.signal.addEventListener('abort', () => {
			abortController.abort();
		});
		// If already aborted, abort immediately
		if (c.req.raw.signal.aborted) {
			abortController.abort();
		}
	}

	const body = await c.req.json();
	const {
		model,
		input,
		instructions = null,
		stream = false,
		temperature,
		top_p,
		max_output_tokens,
		tool_choice,
		tools,
		// Reasoning: body.reasoning?.effort
		reasoning,
		previous_response_id,
		request_id,
		extra_body,
		text,
		service_tier,
		store = true,
	} = body || {};
	const now = Date.now();
	const responseId = request_id || 'resp_' + new Date(now).toISOString().slice(0, 16).replace(/[-:T]/g, '');
	if (typeof model === 'string' && model.startsWith('image/')) {
		const { handleImageForResponses } = await import('./modules/images.mts');
		return await handleImageForResponses({ model, input, headers: c.req.raw.headers as any, stream: !!stream, temperature, top_p, request_id: responseId, authHeader: authHeader || null, isPasswordAuth });
	}
	if (typeof model === 'string' && model.startsWith('video/')) {
		const { handleVideoForResponses } = await import('./modules/videos.mts');
		return await handleVideoForResponses({ model, input, headers: c.req.raw.headers as any, stream: !!stream, request_id: responseId, authHeader: authHeader || null, isPasswordAuth });
	}
	if (model === 'admin/magic') {
		const { handleAdminForResponses } = await import('./modules/management.mts');
		return await handleAdminForResponses({ input, headers: c.req.raw.headers as any, model, request_id: responseId, stream: !!stream, isPasswordAuth });
	}
	// Headers and aux keys
	tavilyApiKey = c.req.header('x-tavily-api-key') || (isPasswordAuth ? process.env.TAVILY_API_KEY || null : null);
	pythonApiKey = c.req.header('x-python-api-key') || (isPasswordAuth ? process.env.PYTHON_API_KEY || null : null);
	pythonUrl = c.req.header('x-python-url') || (isPasswordAuth ? process.env.PYTHON_URL || null : null);
	semanticScholarApiKey = c.req.header('x-semantic-scholar-api-key') || (isPasswordAuth ? process.env.SEMANTIC_SCHOLAR_API_KEY || null : null);
	// Provider keys and headers map
	const headers: Record<string, string> = {};
	c.req.raw.headers.forEach((value, key) => {
		headers[key.toLowerCase().replace(/-/g, '_')] = value;
	});
	const providerKeys = await getProviderKeys(headers, authHeader || null, isPasswordAuth);
	const toAiSdkMessages = async (): Promise<any[]> => {
		// Seed from previous stored conversation if provided
		let history: any[] = [];

		const mapped = responsesInputToAiSdkMessages(input);
		const messages = await processMessages(mapped);

		if (previous_response_id) {
			try {
				const blobStore = getStoreWithConfig('responses');
				const existing: any = await blobStore.get(previous_response_id, { type: 'json' as any });
				if (existing && existing.messages && Array.isArray(existing.messages)) {
					history = existing.messages.filter((m: any) => m?.role !== 'system');
				}
			} catch { }
		}
		if (typeof model === 'string' && !model.toLowerCase().includes('image')) {
			history = addContextMessages(history, c);
		}
		if (instructions) {
			history = [{ role: 'system', content: String(instructions) }, ...history];
		}
		return [...history, ...messages];
	};

	const messages = await toAiSdkMessages();
	let modelId: string = model;
	let thinking: Record<string, any> | undefined = undefined;
	let extraBody: Record<string, any> = extra_body;
	let search: boolean = false;
	if (modelId.startsWith('doubao/')) {
		if (Array.isArray(messages) && messages.length > 0) {
			const lastMsg = messages[messages.length - 1];
			const parts = Array.isArray(lastMsg?.content) ? lastMsg.content : [];
			const hasImage = parts.some((p: any) => p?.type === 'image');
			if (hasImage) {
				modelId = 'doubao/doubao-seed-1-6-vision-250815';
			}
		}
	}
	const messageText = messages.map((msg: any) =>
		typeof msg.content === 'string'
			? msg.content.toLowerCase()
			: Array.isArray(msg.content)
				? msg.content.map((p: any) => (p?.text || '')).join(' ').toLowerCase()
				: ''
	).join(' ');
	isResearchMode = RESEARCH_KEYWORDS.some(keyword => messageText.includes(keyword));
	const aiSdkTools: Record<string, any> = buildAiSdkTools(modelId, tools);
	if (Object.keys(aiSdkTools).length === 0) {
		if (modelId.startsWith('doubao/deepseek-v3-1')) {
			thinking = {
				type: 'enabled',
			};
		} else if (modelId.startsWith('modelscope/deepseek-ai/DeepSeek-V3.1')) {
			extraBody = {
				enable_thinking: true,
			};
		}
	} else {
		search = true;
	}

	const { providersToTry } = prepareProvidersToTry({ model: modelId, providerKeys, isPasswordAuth, authApiKey: apiKey });
	const providerOptionsHeader = c.req.header('x-provider-options');
	const providerOptions = buildDefaultProviderOptions({
		providerOptionsHeader: providerOptionsHeader ?? null,
		thinking,
		reasoning_effort: reasoning?.effort || undefined,
		extra_body: extraBody,
		text_verbosity: text?.verbosity || undefined,
		service_tier,
		reasoning_summary: reasoning?.summary || undefined,
		store,
		model: modelId,
		search,
	});
	const commonParams = {
		messages,
		aiSdkTools,
		temperature,
		top_p,
		max_tokens: max_output_tokens,
		tool_choice,
		abortSignal: abortController.signal,
		providerOptions,
		reasoning_effort: reasoning?.effort || undefined,
	};
	const maxAttempts = Math.min(providersToTry.length, MAX_ATTEMPTS);
	// Storage preparation already computed above

	if (stream) {
		// Streaming SSE per OpenAI Responses API
		const streamResponse = new ReadableStream({
			async start(controller) {
				const createdAt = Math.floor(now / 1000);
				let sequenceNumber = 0;
				let outputIndex = 0;
				const outputItems: any[] = [];

				let storedImageMarkdown = '';
				let savedTextContent = '';

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
					model: modelId,
					output: outputItems,
					parallel_tool_calls: true,
					previous_response_id: previous_response_id || null,
					prompt_cache_key: 'ai-gateway',
					reasoning: reasoning,
					store: store,
					temperature: temperature ?? 1,
					text: text || { format: { type: 'text' }, verbosity: 'medium' },
					tool_choice: tool_choice || 'auto',
					tools: tools || [],
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

				let attemptsTried = 0;
				let lastStreamError: any = null;
				let textItemId: string | null = null;
				let collectedText = '';
				let reasoningItemId: string | null = null;
				let reasoningText = '';
				let reasoningSummaryIndex = 0;
				let functionCallItems: Map<string, { id: string, name: string, call_id: string, args: string, outputIndex: number }> = new Map();

				// Poe-specific reasoning detection state
				let isPoeProvider = false;
				let poeReasoningMode = false;
				let poeReasoningBuffer = '';

				for (let i = 0; i < maxAttempts; i++) {
					if (abortController.signal.aborted) break;
					const attempt = providersToTry[i];
					if (!attempt) continue;
					try {
						attemptsTried++;
						const gw = await getGatewayForAttempt(attempt);

						// Check if this is Poe provider
						isPoeProvider = attempt.name === 'poe';

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
										start_index: 0,
										end_index: 0
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
										reasoningItemId = null;

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
												annotations: accumulatedSources
											}
										});

										outputIndex = textOutputIndex;
									}
									break;
								}
								case 'text-delta': {
									let text: string = part.text;
									// Handle Poe-specific reasoning detection
									if (isPoeProvider) {
										poeReasoningBuffer += text;

										// Check if reasoning is starting
										if (!poeReasoningMode && startsWithThinking(poeReasoningBuffer)) {
											poeReasoningMode = true;

											// Start reasoning item if not already started
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

											const thinkingIndex = findThinkingIndex(poeReasoningBuffer);
											if (thinkingIndex >= 0) {
												const beforeThinking = poeReasoningBuffer.substring(0, thinkingIndex);
												const reasoningStart = poeReasoningBuffer.substring(thinkingIndex);

												if (beforeThinking && textItemId) {
													collectedText += beforeThinking;
													emit({
														type: 'response.output_text.delta',
														sequence_number: sequenceNumber++,
														item_id: textItemId,
														output_index: outputIndex,
														content_index: 0,
														delta: beforeThinking
													});
												}

												const cleanedReasoningStart = cleanPoeReasoningDelta(reasoningStart, true);
												reasoningText += cleanedReasoningStart;
												emit({
													type: 'response.reasoning_summary_text.delta',
													sequence_number: sequenceNumber++,
													item_id: reasoningItemId,
													output_index: outputIndex,
													summary_index: reasoningSummaryIndex,
													delta: cleanedReasoningStart
												});

												// Reset buffer to contain only processed reasoning content
												poeReasoningBuffer = reasoningStart;
											}
											continue;
										}

										// Check if reasoning is ending (two consecutive newlines without >)
										if (poeReasoningMode) {
											const lines = poeReasoningBuffer.split('\n');
											let reasoningEnded = false;

											// Look for two consecutive lines where the second doesn't start with >
											for (let i = lines.length - 2; i >= 0; i--) {
												const currentLine = lines[i];
												const nextLine = lines[i + 1];
												if (currentLine === '' && nextLine && nextLine !== '' && !nextLine.startsWith('>')) {
													reasoningEnded = true;
													break;
												}
											}

											if (reasoningEnded) {
												// Find where reasoning content ends
												const reasoningEndIndex = poeReasoningBuffer.lastIndexOf('\n\n');
												let reasoningContent = '';
												let postReasoningContent = '';

												if (reasoningEndIndex >= 0) {
													reasoningContent = poeReasoningBuffer.substring(0, reasoningEndIndex);
													postReasoningContent = poeReasoningBuffer.substring(reasoningEndIndex + 2);
												} else {
													reasoningContent = poeReasoningBuffer;
												}

												// Emit final reasoning content
												if (reasoningContent) {
													const newReasoningText = reasoningContent.substring(reasoningText.length);
													if (newReasoningText) {
														const cleanedNewReasoningText = cleanPoeReasoningDelta(newReasoningText);
														reasoningText += cleanedNewReasoningText;
														emit({
															type: 'response.reasoning_summary_text.delta',
															sequence_number: sequenceNumber++,
															item_id: reasoningItemId,
															output_index: outputIndex,
															summary_index: reasoningSummaryIndex,
															delta: cleanedNewReasoningText
														});
													}
												}

												// End reasoning
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
													reasoningItemId = null;
													reasoningText = '';
												}

												poeReasoningMode = false;
												poeReasoningBuffer = postReasoningContent;

												// Continue with post-reasoning content as regular text
												if (postReasoningContent) {
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

													collectedText += postReasoningContent;
													emit({
														type: 'response.output_text.delta',
														sequence_number: sequenceNumber++,
														item_id: textItemId,
														output_index: outputIndex,
														content_index: 0,
														delta: postReasoningContent
													});
												}
												continue;
											} else {
												// Still in reasoning mode, add to reasoning
												const newReasoningText = text;
												const cleanedNewReasoningText = cleanPoeReasoningDelta(newReasoningText);
												reasoningText += cleanedNewReasoningText;
												emit({
													type: 'response.reasoning_summary_text.delta',
													sequence_number: sequenceNumber++,
													item_id: reasoningItemId,
													output_index: outputIndex,
													summary_index: reasoningSummaryIndex,
													delta: cleanedNewReasoningText
												});
												continue;
											}
										}

										// If not in reasoning mode and no reasoning detected, treat as regular text
										if (!poeReasoningMode) {
											// Reset buffer periodically to prevent memory issues
											if (poeReasoningBuffer.length > 1000) {
												poeReasoningBuffer = poeReasoningBuffer.slice(-500);
											}
										}
									}

									// Regular text handling (non-Poe or non-reasoning content)
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
									if (!collectedText && modelId === 'cerebras/qwen-3-235b-a22b-thinking-2507') {
										text = '<think>' + text;
									}
									collectedText += text;
									emit({
										type: 'response.output_text.delta',
										sequence_number: sequenceNumber++,
										item_id: textItemId,
										output_index: outputIndex,
										content_index: 0,
										delta: text
									});
									break;
								}
								case 'text-end': {
									if (textItemId) {
										for (const annotation of accumulatedSources) {
											emit({
												type: 'response.output_text.annotation.added',
												sequence_number: sequenceNumber++,
												item_id: textItemId,
												output_index: outputIndex,
												content_index: 0,
												annotation
											});
										}
										emit({
											type: 'response.output_text.done',
											sequence_number: sequenceNumber++,
											item_id: textItemId,
											output_index: outputIndex,
											content_index: 0,
											text: collectedText
										});

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
										savedTextContent += collectedText;
										collectedText = '';
										accumulatedSources = [];
										textItemId = null;
									}
									break;
								}
								case 'file': {
									// Stream image file as image_generation_call output item/events
									try {
										const imageItemId = randomId('img');
										const imageOutputIndex = outputIndex + 1;
										const base64Data: string | null = (part as any)?.file?.base64Data || null;
										const mediaType: string = (part as any)?.file?.mediaType || 'image/png';

										const imageItem = {
											id: imageItemId,
											type: 'image_generation_call',
											status: 'in_progress',
											result: null,
										};
										outputItems.push(imageItem);

										// Announce the new output item
										emit({
											type: 'response.output_item.added',
											sequence_number: sequenceNumber++,
											output_index: imageOutputIndex,
											item: imageItem,
										});

										// Emit in_progress and generating signals
										emit({
											type: 'response.image_generation_call.in_progress',
											output_index: imageOutputIndex,
											item_id: imageItemId,
											sequence_number: sequenceNumber++,
										});
										emit({
											type: 'response.image_generation_call.generating',
											output_index: imageOutputIndex,
											item_id: imageItemId,
											sequence_number: sequenceNumber++,
										});

										// If we have any image bytes now, surface as a partial image
										if (base64Data) {
											emit({
												type: 'response.image_generation_call.partial_image',
												output_index: imageOutputIndex,
												item_id: imageItemId,
												sequence_number: sequenceNumber++,
												partial_image_index: 0,
												partial_image_b64: base64Data,
											});
										}

										// Complete the image generation call
										emit({
											type: 'response.image_generation_call.completed',
											output_index: imageOutputIndex,
											item_id: imageItemId,
											sequence_number: sequenceNumber++,
										});

										const completedImageItem = {
											id: imageItemId,
											type: 'image_generation_call',
											status: 'completed',
											result: base64Data || null,
										};
										const imgItemIdx = outputItems.findIndex((it) => it.id === imageItemId);
										if (imgItemIdx >= 0) outputItems[imgItemIdx] = completedImageItem; else outputItems.push(completedImageItem);

										emit({
											type: 'response.output_item.done',
											sequence_number: sequenceNumber++,
											output_index: imageOutputIndex,
											item: completedImageItem,
										});

										// Advance output index past the image item
										outputIndex = imageOutputIndex;

										// If storing, upload to bucket and accumulate markdown to store (not to response)
										if (store && base64Data && process.env.S3_API && process.env.S3_PUBLIC_URL && process.env.S3_ACCESS_KEY && process.env.S3_SECRET_KEY) {
											try {
												const { uploadBlobToStorage } = await import('./shared/bucket.mts');
												const bin = atob(base64Data);
												const bytes = new Uint8Array(bin.length);
												for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
												const blob = new Blob([bytes], { type: mediaType });
												const url = await uploadBlobToStorage(blob);
												const md = `![Generated Image](${url})`;
												storedImageMarkdown += '\n\n' + md;
											} catch { }
										}
									} catch (e) {
										// ignore malformed file parts
									}
									break;
								}
								case 'tool-input-start': {
									const trackingKey = part.id;
									const funcItemId = randomId('fc');
									const currentOutputIndex = outputIndex + 1;
									if (CODE_TOOLS.has(part.toolName)) {
										functionCallItems.set(trackingKey, {
											id: funcItemId,
											name: part.toolName,
											call_id: `ci_${trackingKey}`,
											args: '',
											outputIndex: currentOutputIndex
										});
										emit({
											type: 'response.output_item.added',
											sequence_number: sequenceNumber++,
											output_index: currentOutputIndex,
											item: {
												id: funcItemId,
												type: 'code_interpreter_call',
												status: 'in_progress'
											}
										});
										emit({
											type: 'response.code_interpreter_call.in_progress',
											sequence_number: sequenceNumber++,
											output_index: currentOutputIndex,
											item_id: trackingKey
										});
										break;
									}
									if (SEARCH_TOOLS.has(part.toolName)) {
										functionCallItems.set(trackingKey, {
											id: funcItemId,
											name: part.toolName,
											call_id: `ws_${trackingKey}`,
											args: '',
											outputIndex: currentOutputIndex
										});
										emit({
											type: 'response.output_item.added',
											sequence_number: sequenceNumber++,
											output_index: currentOutputIndex,
											item: {
												id: funcItemId,
												type: 'web_search_call',
												status: 'in_progress'
											}
										});
										emit({
											type: 'response.web_search_call.in_progress',
											sequence_number: sequenceNumber++,
											output_index: currentOutputIndex,
											item_id: trackingKey
										});
										break;
									}
									const callId = `call_${trackingKey}`;
									functionCallItems.set(trackingKey, {
										id: funcItemId,
										name: part.toolName,
										call_id: callId,
										args: '',
										outputIndex: currentOutputIndex
									});
									const functionItem = {
										id: funcItemId,
										type: 'function_call',
										arguments: '',
										call_id: callId,
										name: part.toolName
									};

									outputItems.push(functionItem);

									emit({
										type: 'response.output_item.added',
										sequence_number: sequenceNumber++,
										output_index: currentOutputIndex,
										item: functionItem
									});
									break;
								}
								case 'tool-call': {
									const trackingKey = part.toolCallId;
									const funcCall = functionCallItems.get(trackingKey);
									if (funcCall) {
										outputIndex = Math.max(outputIndex, funcCall.outputIndex);
										if (CODE_TOOLS.has(part.toolName)) {
											emit({
												type: 'response.code_interpreter_call.interpreting',
												sequence_number: sequenceNumber++,
												output_index: funcCall.outputIndex,
												item_id: trackingKey
											});
											break;
										}
										if (SEARCH_TOOLS.has(part.toolName)) {
											emit({
												type: 'response.web_search_call.searching',
												sequence_number: sequenceNumber++,
												output_index: funcCall.outputIndex,
												item_id: trackingKey
											});
											break;
										}
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
											arguments: finalArgs,
											call_id: funcCall.call_id,
											name: funcCall.name
										};

										emit({
											type: 'response.output_item.done',
											sequence_number: sequenceNumber++,
											output_index: funcCall.outputIndex,
											item: updatedItem
										});
										functionCallItems.delete(trackingKey);
										// Update the item in outputItems and store final args
										const itemIndex = outputItems.findIndex(item => item.id === funcCall.id);
										if (itemIndex >= 0) outputItems[itemIndex] = updatedItem;
										funcCall.args = finalArgs; // Store for tool-result
									}
									break;
								}
								case 'tool-input-delta': {
									// Find the function call by tool call ID or part ID
									const trackingKey = part.id;
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
									const trackingKey = part.toolCallId;
									const funcCall = functionCallItems.get(trackingKey);

									if (funcCall) {
										outputIndex = Math.max(outputIndex, funcCall.outputIndex);
										if (CODE_TOOLS.has(part.toolName)) {
											emit({
												type: 'response.code_interpreter_call.completed',
												sequence_number: sequenceNumber++,
												output_index: funcCall.outputIndex,
												item_id: trackingKey
											});
											emit({
												type: 'response.output_item.done',
												sequence_number: sequenceNumber++,
												output_index: funcCall.outputIndex,
												item: {
													id: funcCall.id,
													type: 'code_interpreter_call',
													status: 'completed'
												}
											});
											break;
										}
										if (SEARCH_TOOLS.has(part.toolName)) {
											emit({
												type: 'response.web_search_call.completed',
												sequence_number: sequenceNumber++,
												output_index: funcCall.outputIndex,
												item_id: trackingKey
											});
											emit({
												type: 'response.output_item.done',
												sequence_number: sequenceNumber++,
												output_index: funcCall.outputIndex,
												item: {
													id: funcCall.id,
													type: 'web_search_call',
													status: 'completed'
												}
											});
											break;
										}
									}
									break;
								}
								case 'finish': {
									const reason = part.finishReason.replace("-", "_") || 'stop';
									if (reason !== 'stop' && reason !== 'tool_calls') {
										emit({
											type: 'error',
											sequence_number: sequenceNumber++,
											code: reason,
											message: `**Unexpected finish**: ${reason}`,
											param: null
										});
									}
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
										collectedText = '';
										accumulatedSources = [];
										textItemId = null;
									}

									const filteredOutput = outputItems.filter(item => item.type !== 'function_call');

									const completed = {
										...baseResponseObj,
										status: 'completed',
										output: filteredOutput,
										usage: part.totalUsage ? {
											input_tokens: part.totalUsage.inputTokens,
											output_tokens: part.totalUsage.outputTokens,
											total_tokens: part.totalUsage.totalTokens,
											input_tokens_details: {
												cached_tokens: part.totalUsage.cachedInputTokens || 0,
											},
											output_tokens_details: {
												reasoning_tokens: part.totalUsage.reasoningTokens,
											}
										} : null
									};
									if (store && (savedTextContent || storedImageMarkdown)) {
										try {
											const blobStore = getStoreWithConfig('responses');
											await blobStore.setJSON(responseId, {
												id: responseId,
												messages: [...messages, { role: 'assistant', content: savedTextContent }]
											});
											if (storedImageMarkdown) {
												await blobStore.setJSON(responseId + '_image', {
													id: responseId,
													messages: [{ role: 'assistant', content: storedImageMarkdown }]
												});
											}
										} catch { }
									}
									emit({
										type: 'response.completed',
										sequence_number: sequenceNumber++,
										response: completed
									});
									controller.enqueue(TEXT_ENCODER.encode('data: [DONE]\n\n'));
									controller.close();
									return;
								}
								case 'error': {
									const errInfo = (part as any)?.error || {};
									const code = errInfo?.statusCode || 'ERR';
									const message = JSON.parse(errInfo.responseBody || '{}').error?.metadata?.raw || errInfo.message || errInfo;
									if ([429, 401, 402].includes(code)) {
										const e = new Error(message);
										(e as any).statusCode = code;
										(e as any).type = errInfo?.type || 'provider_error';
										throw e;
									}
									i = maxAttempts;
									emit({
										type: 'error',
										sequence_number: sequenceNumber++,
										code,
										message,
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
							controller.enqueue(TEXT_ENCODER.encode('data: [DONE]\n\n'));
							controller.close();
							return;
						}
						console.error(`Error with provider: ${attempt.name} (${i + 1}/${maxAttempts}): ${err.message}`);
						lastStreamError = err;
						continue;
					}
				}

				// All attempts failed
				const msg = lastStreamError?.message || 'An unknown error occurred';
				const statusCode = lastStreamError?.statusCode || 500;
				const message = `${statusCode} All ${attemptsTried} attempt(s) failed. Last error: ${msg}`
				if (lastStreamError) {
					emit({
						type: 'error',
						sequence_number: sequenceNumber++,
						code: statusCode,
						message
					});
				}
				emit({
					type: 'response.failed',
					sequence_number: sequenceNumber++,
					response: {
						id: responseId,
						object: 'response',
						status: 'failed',
						error: { code: 'server_error', message }
					}
				});
				controller.enqueue(TEXT_ENCODER.encode('data: [DONE]\n\n'));
				controller.close();
			},
		});
		return new Response(streamResponse, { headers: { 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' } });
	}

	// Non-streaming path
	let attemptsTried = 0;
	let lastError: any;
	for (let i = 0; i < maxAttempts; i++) {
		attemptsTried++;
		const attempt = providersToTry[i];
		if (!attempt) continue;
		try {
			const gw = await getGatewayForAttempt(attempt);
			const commonOptions = buildCommonOptions(gw, attempt, commonParams);
			const result = await generateText(commonOptions);
			const toolCalls = result.toolCalls;

			// Handle Poe-specific reasoning extraction for non-streaming
			let content = result.text || '';
			let reasoningContent = result.reasoningText || '';

			if (attempt.name === 'poe' && content && !reasoningContent) {
				const extracted = extractPoeReasoning(content);
				content = extracted.content;
				reasoningContent = extracted.reasoning;
			}

			// Transform result.sources to annotations format  
			const sources = result.sources ? result.sources.map((source: any) => ({
				title: source.title || '',
				url: source.url || '',
				type: (source.sourceType || 'url') + '_citation',
				start_index: source.start_index || 0,
				end_index: source.end_index || 0
			})) : [];
			accumulatedSources.push(...sources);

			// Construct Responses API output
			const inputNormalized = typeof input === 'string'
				? [{ type: 'input_text', text: input }]
				: (Array.isArray(input) ? input : (input ? [input] : []));

			// Build output items
			const output: any[] = [];

			if (reasoningContent) {
				output.push({
					type: 'reasoning',
					id: randomId('rs'),
					summary: [{
						type: 'summary_text',
						text: reasoningContent
					}]
				});
			}
			if (content) {
				output.push({
					type: 'message',
					id: randomId('msg'),
					status: 'completed',
					role: 'assistant',
					content: [{ type: 'output_text', text: content, annotations: accumulatedSources }]
				});
			}
			if (result.steps && result.steps[0] && result.steps[0].content) {
				for (const msg of result.steps[0].content) {
					if (msg.type === 'tool-result') {
						const name = msg.toolName || '';
						if (CODE_TOOLS.has(name)) {
							output.push({
								type: 'code_interpreter_call',
								id: msg.toolCallId,
								status: 'completed',
							});
							continue;
						}
						if (SEARCH_TOOLS.has(name)) {
							output.push({
								type: 'web_search_call',
								id: msg.toolCallId,
								status: 'completed',
							});
							continue;
						}
					}
				}
			}
			if (toolCalls && Array.isArray(toolCalls)) {
				for (const tc of toolCalls) {
					output.push({
						type: 'function_call',
						id: randomId('fc'),
						call_id: `call_${tc.toolCallId}`,
						name: tc.toolName,
						arguments: JSON.stringify(tc.input ?? {}),
					});
				}
			}

			// Add image_generation_call items for any generated files
			if (Array.isArray((result as any).files) && (result as any).files.length > 0) {
				for (const f of (result as any).files) {
					try {
						const fileObj = (f as any)?.file;
						const base64 = fileObj?.base64Data as string | undefined;
						// We only need to return base64 result per your spec
						output.push({ id: randomId('img'), type: 'image_generation_call', status: 'completed', result: base64 || null });
					} catch { }
				}
			}
			const responsePayload = {
				id: responseId,
				object: 'response',
				created_at: Math.floor(now / 1000),
				status: 'completed',
				error: null,
				incomplete_details: null,
				input: inputNormalized,
				instructions,
				max_output_tokens: max_output_tokens ?? null,
				model: modelId,
				output,
				previous_response_id: previous_response_id || null,
				reasoning: reasoning,
				parallel_tool_calls: true,
				store: store,
				temperature: temperature ?? 1,
				text: { format: { type: 'text' } },
				tool_choice: tool_choice || 'auto',
				tools: tools || [],
				top_p: top_p ?? 1,
				truncation: 'disabled',
				usage: result.usage ? {
					input_tokens: result.usage.inputTokens,
					output_tokens: result.usage.outputTokens,
					total_tokens: result.usage.totalTokens,
					input_tokens_details: {
						cached_tokens: result.usage.cachedInputTokens || 0,
					},
					output_tokens_details: {
						reasoning_tokens: result.usage.reasoningTokens,
					}
				} : null,
				user: null,
			} as any;

			if (store) {
				try {
					let extraMd = '';
					if (Array.isArray((result as any).files) && (result as any).files.length > 0 && process.env.S3_API && process.env.S3_PUBLIC_URL && process.env.S3_ACCESS_KEY && process.env.S3_SECRET_KEY) {
						const { uploadBlobToStorage } = await import('./shared/bucket.mts');
						for (const f of (result as any).files) {
							try {
								const fileObj = (f as any)?.file;
								const b64 = fileObj?.base64Data as string | undefined;
								const mt = fileObj?.mediaType as string | undefined;
								if (b64 && mt) {
									const bin = atob(b64);
									const bytes = new Uint8Array(bin.length);
									for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
									const blob = new Blob([bytes], { type: mt });
									const url = await uploadBlobToStorage(blob);
									extraMd += (extraMd ? '\n\n' : '') + `![Generated Image](${url})`;
								}
							} catch { }
						}
					}
					const blobStore = getStoreWithConfig('responses');
					await blobStore.setJSON(responseId, { id: responseId, messages: [...messages, { role: 'assistant', content }] });
					if (extraMd) await blobStore.setJSON(responseId + '_image', { id: responseId, messages: [...messages, { role: 'assistant', content: extraMd }] });
				} catch { }
			}

			return c.json(responsePayload);
		} catch (error: any) {
			lastError = error;
			if (error.name === 'AbortError' || abortController.signal.aborted) return c.json({ error: { message: 'Request was aborted by the user', type: 'request_aborted', statusCode: 499 } }, 499 as any);
			console.error(`Error with provider: ${attempt.name} (${i + 1}/${maxAttempts}): ${JSON.parse(error.responseBody || '{}').error?.metadata?.raw || error.message || error}`);
			// Only retry for specific status codes
			const statusCode = error.statusCode;
			if (![429, 401, 402].includes(statusCode) || i >= maxAttempts - 1) {
				break;
			}
			continue;
		}
	}

	const statusCode = lastError?.statusCode || 500;
	let message = lastError?.message || 'All attempts failed';
	const rb = lastError?.responseBody;
	if (rb) {
		try {
			const obj = typeof rb === 'string' ? JSON.parse(rb) : rb;
			message = obj?.error?.metadata?.raw || obj?.error?.message || message;
		} catch { /* ignore JSON parse errors */ }
	}
	const errorPayload = { error: { message: `${statusCode} All ${attemptsTried} attempt(s) failed. Last error: ${message}`, type: lastError?.type, statusCode } };
	return c.json(errorPayload, statusCode);
});

app.post('/v1/chat/completions', async (c: Context) => {
	const authHeader = c.req.header('Authorization');
	let apiKey = authHeader?.split(' ')[1];

	const envPassword = process.env.PASSWORD;
	const isPasswordAuth = !!(envPassword && apiKey && envPassword.trim() === apiKey.trim());

	if (!apiKey) {
		return c.text('Unauthorized', 401);
	}

	const abortController = new AbortController();
	if (c.req.raw?.signal) {
		c.req.raw.signal.addEventListener('abort', () => {
			abortController.abort();
		});
		// If already aborted, abort immediately
		if (c.req.raw.signal.aborted) {
			abortController.abort();
		}
	}

	let gateway;

	// Get headers
	tavilyApiKey = c.req.header('x-tavily-api-key') || (isPasswordAuth ? process.env.TAVILY_API_KEY || null : null);
	pythonApiKey = c.req.header('x-python-api-key') || (isPasswordAuth ? process.env.PYTHON_API_KEY || null : null);
	pythonUrl = c.req.header('x-python-url') || (isPasswordAuth ? process.env.PYTHON_URL || null : null);
	semanticScholarApiKey = c.req.header('x-semantic-scholar-api-key') || (isPasswordAuth ? process.env.SEMANTIC_SCHOLAR_API_KEY || null : null);

	const body = await c.req.json();
	const {
		model,
		messages = [],
		tools,
		stream,
		temperature,
		top_p,
		top_k,
		max_tokens,
		stop_sequences,
		seed,
		presence_penalty,
		frequency_penalty,
		tool_choice,
		reasoning_effort,
		thinking,
		extra_body,
		text_verbosity,
		service_tier,
		store = true
	} = body;
	const contextMessages = (typeof model === 'string' && model.toLowerCase().includes('image')) ? messages : addContextMessages(messages, c);
	const processedMessages = await processMessages(contextMessages, { extractMarkdownImages: true });

	if (typeof model === 'string' && model.startsWith('image/')) {
		const { handleImageForChat } = await import('./modules/images.mts');
		return await handleImageForChat({ model, messages: processedMessages, headers: c.req.raw.headers as any, stream: !!stream, temperature, top_p, authHeader: authHeader || null, isPasswordAuth });
	}
	if (typeof model === 'string' && model.startsWith('video/')) {
		const { handleVideoForChat } = await import('./modules/videos.mts');
		return await handleVideoForChat({ model, messages: processedMessages, headers: c.req.raw.headers as any, stream: !!stream, authHeader: authHeader || null, isPasswordAuth });
	}
	if (model === 'admin/magic') {
		const { handleAdminForChat } = await import('./modules/management.mts');
		return await handleAdminForChat({ messages: processedMessages, headers: c.req.raw.headers as any, model, stream: !!stream, isPasswordAuth });
	}

	const headers: Record<string, string> = {}
	c.req.raw.headers.forEach((value, key) => {
		headers[key.toLowerCase().replace(/-/g, '_')] = value
	})
	const providerKeys = await getProviderKeys(headers, authHeader || null, isPasswordAuth);
	let modelId: string = model;
	let thinkingConfig: Record<string, any> = thinking;
	let extraBody: Record<string, any> = extra_body;
	let search: boolean = false;
	if (modelId.startsWith('doubao/')) {
		if (Array.isArray(processedMessages) && processedMessages.length > 0) {
			const lastMsg = processedMessages[processedMessages.length - 1];
			const parts = Array.isArray(lastMsg?.content) ? lastMsg.content : [];
			const hasImage = parts.some((p: any) => p?.type === 'image');
			if (hasImage) {
				modelId = 'doubao/doubao-seed-1-6-vision-250815';
			}
		}
	}
	const messageText = processedMessages.map((msg: any) =>
		typeof msg.content === 'string'
			? msg.content.toLowerCase()
			: Array.isArray(msg.content)
				? msg.content.map((p: any) => (p?.text || '')).join(' ').toLowerCase()
				: ''
	).join(' ');
	isResearchMode = RESEARCH_KEYWORDS.some(keyword => messageText.includes(keyword));
	const aiSdkTools: Record<string, any> = buildAiSdkTools(modelId, tools);
	if (Object.keys(aiSdkTools).length === 0) {
		if (modelId.startsWith('doubao/deepseek-v3-1')) {
			thinkingConfig = {
				type: 'enabled',
			};
		} else if (modelId.startsWith('modelscope/deepseek-ai/DeepSeek-V3.1')) {
			extraBody = {
				enable_thinking: true,
			};
		}
	} else {
		search = true;
	}
	const { providersToTry } = prepareProvidersToTry({ model: modelId, providerKeys, isPasswordAuth, authApiKey: apiKey });
	const providerOptionsHeader = c.req.header('x-provider-options');
	const providerOptions = buildDefaultProviderOptions({
		providerOptionsHeader: providerOptionsHeader ?? null,
		thinking: thinkingConfig,
		reasoning_effort,
		extra_body: extraBody,
		text_verbosity,
		service_tier,
		store,
		model: modelId,
		search,
	});
	const commonParams = {
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
		reasoning_effort,
	};
	const now = Math.floor(Date.now() / 1000);
	const chunkId = `chatcmpl-${now}`;
	const maxAttempts = Math.min(providersToTry.length, MAX_ATTEMPTS);
	// If streaming, handle retries within a single ReadableStream so we can switch keys on error mid-stream
	if (stream) {
		const streamResponse = new ReadableStream({
			async start(controller) {
				const baseChunk = { id: chunkId, object: 'chat.completion.chunk', created: now, model: modelId } as any;
				let attemptsTried = 0;
				let lastStreamError: any = null;
				let accumulatedText = '';

				// Poe-specific reasoning detection state
				let isPoeProvider = false;
				let poeReasoningMode = false;
				let poeReasoningBuffer = '';
				let poeAccumulatedReasoning = '';

				for (let i = 0; i < maxAttempts; i++) {
					if (abortController.signal.aborted) break;
					const attempt = providersToTry[i];
					if (!attempt) continue;
					try {
						attemptsTried++;
						const gw = await getGatewayForAttempt(attempt);

						// Check if this is Poe provider
						isPoeProvider = attempt.name === 'poe';

						const commonOptions = buildCommonOptions(gw, attempt, commonParams);

						const result = streamText(commonOptions);
						// Forward chunks; on error, try next key/provider
						for await (const part of (result as any).fullStream) {

							if (abortController.signal.aborted) throw new Error('aborted');
							let chunk: any;
							switch (part.type) {
								case 'error':
									const errInfo = (part as any)?.error || {};
									const code = errInfo?.statusCode || 'ERR';
									const message = JSON.parse(errInfo.responseBody || '{}').error?.metadata?.raw || errInfo.message || errInfo;
									if ([429, 401, 402].includes(code)) {
										const e = new Error(message);
										(e as any).statusCode = code;
										(e as any).type = errInfo?.type || 'provider_error';
										throw e;
									}
									i = maxAttempts;
									chunk = { ...baseChunk, choices: [{ index: 0, delta: { refusal: message, content: '**Error**: ' + message }, finish_reason: 'stop' }] };
									controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
									break;
								case 'reasoning-delta':
									chunk = { ...baseChunk, choices: [{ index: 0, delta: { reasoning_content: part.text }, finish_reason: null }] };
									controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
									break;
								case 'text-delta':
									let text: string = part.text;
									// Handle Poe-specific reasoning detection
									if (isPoeProvider) {
										poeReasoningBuffer += text;

										// Check if reasoning is starting
										if (!poeReasoningMode && startsWithThinking(poeReasoningBuffer)) {
											poeReasoningMode = true;

											// Extract the reasoning text (remove the part before "Thinking...")
											const thinkingIndex = findThinkingIndex(poeReasoningBuffer);
											if (thinkingIndex >= 0) {
												const beforeThinking = poeReasoningBuffer.substring(0, thinkingIndex);
												const reasoningStart = poeReasoningBuffer.substring(thinkingIndex);

												// Emit any text before "Thinking..." as regular content
												if (beforeThinking) {
													chunk = { ...baseChunk, choices: [{ index: 0, delta: { content: beforeThinking }, finish_reason: null }] };
													controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
												}

												// Clean and add reasoning text
												const cleanedReasoningStart = cleanPoeReasoningDelta(reasoningStart, true);
												poeAccumulatedReasoning += cleanedReasoningStart;
												chunk = { ...baseChunk, choices: [{ index: 0, delta: { reasoning_content: cleanedReasoningStart }, finish_reason: null }] };
												controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));

												// Reset buffer to contain only processed reasoning content
												poeReasoningBuffer = reasoningStart;
											}
											break;
										}

										// Check if reasoning is ending (two consecutive newlines without >)
										if (poeReasoningMode) {
											const lines = poeReasoningBuffer.split('\n');
											let reasoningEnded = false;

											// Look for two consecutive lines where the second doesn't start with >
											for (let i = lines.length - 2; i >= 0; i--) {
												const currentLine = lines[i];
												const nextLine = lines[i + 1];
												if (currentLine === '' && nextLine && nextLine !== '' && !nextLine.startsWith('>')) {
													reasoningEnded = true;
													break;
												}
											}

											if (reasoningEnded) {
												// Find where reasoning content ends
												const reasoningEndIndex = poeReasoningBuffer.lastIndexOf('\n\n');
												let reasoningContent = '';
												let postReasoningContent = '';

												if (reasoningEndIndex >= 0) {
													reasoningContent = poeReasoningBuffer.substring(0, reasoningEndIndex);
													postReasoningContent = poeReasoningBuffer.substring(reasoningEndIndex + 2);
												} else {
													reasoningContent = poeReasoningBuffer;
												}

												// Emit final reasoning content
												if (reasoningContent) {
													const newReasoningText = reasoningContent.substring(poeAccumulatedReasoning.length);
													if (newReasoningText) {
														const cleanedNewReasoningText = cleanPoeReasoningDelta(newReasoningText);
														poeAccumulatedReasoning += cleanedNewReasoningText;
														chunk = { ...baseChunk, choices: [{ index: 0, delta: { reasoning_content: cleanedNewReasoningText }, finish_reason: null }] };
														controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
													}
												}

												poeReasoningMode = false;
												poeReasoningBuffer = postReasoningContent;

												// Continue with post-reasoning content as regular text
												if (postReasoningContent) {
													accumulatedText += postReasoningContent;
													chunk = { ...baseChunk, choices: [{ index: 0, delta: { content: postReasoningContent }, finish_reason: null }] };
													controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
												}
												break;
											} else {
												// Still in reasoning mode, add to reasoning
												const newReasoningText = text;
												const cleanedNewReasoningText = cleanPoeReasoningDelta(newReasoningText);
												poeAccumulatedReasoning += cleanedNewReasoningText;
												chunk = { ...baseChunk, choices: [{ index: 0, delta: { reasoning_content: cleanedNewReasoningText }, finish_reason: null }] };
												controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
												break;
											}
										}

										// If not in reasoning mode and no reasoning detected, treat as regular text
										if (!poeReasoningMode) {
											// Reset buffer periodically to prevent memory issues
											if (poeReasoningBuffer.length > 1000) {
												poeReasoningBuffer = poeReasoningBuffer.slice(-500);
											}
										}
									}
									if (!accumulatedText && model === 'cerebras/qwen-3-235b-a22b-thinking-2507') {
										text = '<think>' + text;
									}
									// Regular text handling (non-Poe or non-reasoning content)
									accumulatedText += text;
									chunk = { ...baseChunk, choices: [{ index: 0, delta: { content: text }, finish_reason: null }] };
									controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
									break;
								case 'source':
									// Accumulate citations for later inclusion in finish
									accumulatedSources.push({
										type: (part.sourceType || 'url') + '_citation',
										url: part.url || '',
										title: part.title || '',
										start_index: 0,
										end_index: 0
									});
									break;
								case 'file': {
									try {
										const b64 = (part as any)?.file?.base64Data as string | undefined;
										const mt = (part as any)?.file?.mediaType as string | undefined;
										if (b64 && mt) {
											const url = `data:${mt};base64,${b64}`;
											const chunk = {
												...baseChunk,
												choices: [{
													index: 0,
													delta: {
														images: [
															{ type: 'image_url', image_url: { url } }
														]
													},
													finish_reason: null
												}]
											};
											controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
											// Also upload and stream markdown if storing
											if (store && process.env.S3_API && process.env.S3_PUBLIC_URL && process.env.S3_ACCESS_KEY && process.env.S3_SECRET_KEY) {
												try {
													const { uploadBlobToStorage } = await import('./shared/bucket.mts');
													const bin = atob(b64);
													const bytes = new Uint8Array(bin.length);
													for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
													const blob = new Blob([bytes], { type: mt });
													const publicUrl = await uploadBlobToStorage(blob);
													const md = `![Generated Image](${publicUrl})`;
													const mdChunk = { ...baseChunk, choices: [{ index: 0, delta: { content: (accumulatedText ? '\n\n' : '') + md }, finish_reason: null }] };
													controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(mdChunk)}\n\n`));
													accumulatedText += (accumulatedText ? '\n\n' : '') + md;
												} catch { }
											}
										}
									} catch { }
									break;
								}
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
									const reason = part.finishReason.replace("-", "_") || 'stop';
									if (reason !== 'stop' && reason !== 'tool_calls') {
										chunk = { ...baseChunk, choices: [{ index: 0, delta: { refusal: reason, content: `**Unexpected Finish**: ${reason}` }, finish_reason: reason }] };
										controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
									}
									if (accumulatedSources.length > 0) {
										const accumulatedCitations = accumulatedSources.map(source => ({
											type: source.type === 'url_citation' ? 'url_citation' : `${source.type}_citation`,
											url_citation: {
												url: source.url,
												title: source.title
											}
										}));
										const citationsChunk = { ...baseChunk, choices: [{ index: 0, delta: { annotations: accumulatedCitations }, finish_reason: null }] };
										controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(citationsChunk)}\n\n`));
									}
									chunk = {
										...baseChunk,
										choices: [
											{ index: 0, delta: {}, finish_reason: reason }
										],
										usage: {
											prompt_tokens: part.totalUsage.inputTokens,
											completion_tokens: part.totalUsage.outputTokens,
											total_tokens: part.totalUsage.totalTokens,
											prompt_tokens_details: {
												cached_tokens: part.totalUsage.cachedInputTokens || 0
											},
											completion_tokens_details: {
												reasoning_tokens: part.totalUsage.reasoningTokens || 0
											}
										}
									};
									controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(chunk)}\n\n`));
									break;
							}
						}
						// If finished streaming without throwing, end the SSE and return
						controller.enqueue(TEXT_ENCODER.encode('data: [DONE]\n\n'));
						if (store && accumulatedText) {
							try {
								const blobStore = getStoreWithConfig('responses');
								await blobStore.setJSON(chunkId, {
									id: chunkId,
									messages: [...messages, { role: 'assistant', content: accumulatedText }]
								});
							} catch { }
						}
						controller.close();
						return;
					} catch (error: any) {
						if (abortController.signal.aborted || error?.message === 'aborted' || error?.name === 'AbortError') {
							// Aborted by client
							const abortChunk = { ...baseChunk, choices: [{ index: 0, delta: { refusal: 'Request was aborted by the user', content: '**Aborted**' }, finish_reason: 'stop' }] };
							controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(abortChunk)}\n\n`));
							controller.enqueue(TEXT_ENCODER.encode('data: [DONE]\n\n'));
							controller.close();
							return;
						}
						lastStreamError = error;
						console.error(`Error with provider: ${attempt.name} (${i + 1}/${maxAttempts}): ${error.message}`);
						// Otherwise, try next key/provider in the next loop iteration
						continue;
					}
				}

				// If all attempts failed
				if (lastStreamError) {
					const statusCode = lastStreamError?.statusCode || 500;
					const errMsg = lastStreamError?.message || 'An unknown error occurred';
					const refusal = `${statusCode} All ${attemptsTried} attempt(s) failed. Last error: ${errMsg}`;
					const errorChunk = { ...baseChunk, choices: [{ index: 0, delta: { refusal, content: '**Error**: ' + refusal }, finish_reason: 'stop' }] };
					controller.enqueue(TEXT_ENCODER.encode(`data: ${JSON.stringify(errorChunk)}\n\n`));
				}
				controller.enqueue(TEXT_ENCODER.encode('data: [DONE]\n\n'));
				controller.close();
			},
		});
		return new Response(streamResponse, { headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' } });
	}

	// Non-streaming: try providers sequentially and return first success
	let lastError: any;
	let attemptsTried = 0;
	for (let i = 0; i < maxAttempts; i++) {
		const provider = providersToTry[i];
		if (!provider) continue;
		try {
			attemptsTried++;
			const gw = await getGatewayForAttempt(provider);
			const commonOptions = buildCommonOptions(gw, provider, commonParams);

			const result = await generateText(commonOptions);

			const annotations = result.sources ? result.sources.map((source: any) => ({
				type: (source.sourceType || 'url') + '_citation',
				url: source.url || '',
				title: source.title || '',
				start_index: 0,
				end_index: 0
			})) : [];
			accumulatedSources.push(...annotations);
			const accumulatedCitations = accumulatedSources.map(source => ({
				type: source.type === 'url_citation' ? 'url_citation' : `${source.type}_citation`,
				url_citation: {
					url: source.url,
					title: source.title,
					start_index: source.start_index,
					end_index: source.end_index
				}
			}));

			let content = result.text || '';
			let reasoningContent = result.reasoningText || '';

			// Handle Poe-specific reasoning extraction for non-streaming
			if (provider.name === 'poe' && content && !reasoningContent) {
				const extracted = extractPoeReasoning(content);
				content = extracted.content;
				reasoningContent = extracted.reasoning;
			}

			// Collect any generated files into images array per client convention
			let imagesExt: any[] | undefined = undefined;
			if (Array.isArray((result as any).files) && (result as any).files.length > 0) {
				imagesExt = [];
				for (const f of (result as any).files) {
					try {
						const fileObj = (f as any)?.file;
						const b64 = fileObj?.base64Data as string | undefined;
						const mt = fileObj?.mediaType as string | undefined;
						if (b64 && mt) {
							imagesExt.push({ type: 'image_url', image_url: { url: `data:${mt};base64,${b64}` } });
						}
					} catch { }
				}
			}

			// If storing, upload images and append markdown to response content (chat requirement)
			if (store && Array.isArray((result as any).files) && (result as any).files.length > 0 && process.env.S3_API && process.env.S3_PUBLIC_URL && process.env.S3_ACCESS_KEY && process.env.S3_SECRET_KEY) {
				for (const f of (result as any).files) {
					try {
						const { uploadBlobToStorage } = await import('./shared/bucket.mts');
						const fileObj = (f as any)?.file;
						const b64 = fileObj?.base64Data as string | undefined;
						const mt = fileObj?.mediaType as string | undefined;
						if (b64 && mt) {
							const bin = atob(b64);
							const bytes = new Uint8Array(bin.length);
							for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
							const blob = new Blob([bytes], { type: mt });
							const url = await uploadBlobToStorage(blob);
							content += (content ? '\n\n' : '') + `![Generated Image](${url})`;
						}
					} catch { }
				}
			}

			const choices = content
				? [
					{
						index: 0,
						message: {
							role: 'assistant',
							content,
							reasoning_content: reasoningContent || undefined,
							tool_calls: result.toolCalls,
							...(accumulatedCitations.length > 0 ? { annotations: accumulatedCitations } : {}),
							...(imagesExt && imagesExt.length > 0 ? { images: imagesExt } : {})
						},
						finish_reason: result.finishReason,
					},
				]
				: [];

			if (store && content) {
				try {
					const blobStore = getStoreWithConfig('responses');
					await blobStore.setJSON(chunkId, { id: chunkId, messages: [...messages, { role: 'assistant', content }] });
				} catch { }
			}

			return c.json({
				id: chunkId,
				object: 'chat.completion',
				created: now,
				model: model,
				choices: choices,
				usage: {
					prompt_tokens: result.usage.inputTokens,
					completion_tokens: result.usage.outputTokens,
					total_tokens: result.usage.totalTokens,
					prompt_tokens_details: {
						cached_tokens: result.usage.cachedInputTokens || 0
					},
					completion_tokens_details: {
						reasoning_tokens: result.usage.reasoningTokens
					}
				},
			});

		} catch (error: any) {
			lastError = error;

			if (error.name === 'AbortError' || abortController.signal.aborted) {
				const abortPayload = { error: { message: 'Request was aborted by the user', type: 'request_aborted', statusCode: 499 } };
				return c.json(abortPayload, 499 as any);
			}
			console.error(`Error with provider: ${provider.name} (${i + 1}/${maxAttempts}): ${JSON.parse(error.responseBody || '{}').error?.metadata?.raw || error.message || error}`);
			// Only retry for specific status codes
			const statusCode = error.statusCode;
			if (![429, 401, 402].includes(statusCode) || i >= maxAttempts - 1) {
				break;
			}
			continue;
		}
	}

	let errorMessage = lastError.message || 'An unknown error occurred';
	let errorType = lastError.type;
	const statusCode = lastError.statusCode || 500;

	// Extract error message from metadata.raw if available
	const rb = lastError.responseBody || lastError.cause?.responseBody;
	if (rb) {
		try {
			const obj = typeof rb === 'string' ? JSON.parse(rb) : rb;
			errorMessage = obj?.error?.metadata?.raw || obj?.error?.message || errorMessage;
			errorType = obj?.error?.type || errorType;
		} catch {
			// ignore JSON parse errors
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

const CUSTOM_MODEL_LISTS = {
	poixe: [
		{ id: 'gpt-5:free', name: 'GPT-5 4K/2K' },
		{ id: 'gpt-5-mini:free', name: 'GPT-5 Mini 4K' },
		{ id: 'grok-3-mini:free', name: 'Grok 3 Mini 4K' },
		{ id: 'grok-4:free', name: 'Grok 4 4K/2K' },
		{ id: 'claude-sonnet-4-20250514:free', name: 'Claude Sonnet 4 4K/2K' },
		{ id: 'doubao-seed-1-6-thinking-250615:free', name: 'Doubao Seed 1.6 Thinking 4K/2K' },
	],
	doubao: [
		{ id: 'doubao-seed-1-6-flash-250715', name: 'Doubao Seed 1.6 Flash' },
		{ id: 'doubao-seed-1-6-thinking-250715', name: 'Doubao Seed 1.6 Thinking' },
		{ id: 'deepseek-v3-1-250821', name: 'DeepSeek V3.1' },
		{ id: 'deepseek-r1-250528', name: 'DeepSeek R1' },
		{ id: 'kimi-k2-250711', name: 'Kimi K2' },
	],
	cohere: [
		{ id: 'command-a-03-2025', name: 'Command A' },
		{ id: 'command-a-vision-07-2025', name: 'Cohere A Vision' },
		{ id: 'command-a-reasoning-08-2025', name: 'Command A Reasoning' },
		{ id: 'command-a-translate-08-2025', name: 'Command A Translation' },
	],
};

function isSupportedProvider(name: string): name is keyof typeof SUPPORTED_PROVIDERS {
	return Object.prototype.hasOwnProperty.call(SUPPORTED_PROVIDERS, name);
}

function shouldIncludeModel(model: any, providerName?: string) {
	const modelId = String(model.id || '').toLowerCase();
	const commonExclusions = [
		'gemma', 'rerank', 'distill', 'parse', 'embed', 'bge-', 'tts', 'phi', 'live', 'audio', 'lite',
		'qwen2', 'qwen-2', 'qwen1', 'qwq', 'qvq', 'gemini-1', 'learnlm', 'gemini-exp', 'gpt-4', 'gpt-3',
		'turbo', 'claude-3', 'voxtral', 'pixtral', 'mixtral', 'ministral', '-24', 'moderation', 'saba', '-ocr-',
		'transcribe', 'dall', 'davinci', 'babbage', 'hailuo', 'kling', 'wan', 'ideogram', 'background'
	];
	if (commonExclusions.some((e) => modelId.includes(e))) return false;
	if (!modelId.includes('super') && ((['nemotron', 'llama'].some((e) => modelId.includes(e))) || modelId.includes('nvidia'))) return false;
	if (providerName === 'openrouter' && !modelId.includes(':free')) return false;
	if (providerName !== 'mistral' && modelId.includes('mistral')) return false;
	if (providerName === 'chatgpt' && modelId.split('-').length > 4) return false;
	if (!providerName && ['mistral', 'alibaba', 'cohere', 'deepseek', 'moonshotai', 'morph', 'zai'].some((e) => modelId.includes(e))) return false;
	return true;
}

async function fetchProviderModels(providerName: string, apiKey: string) {
	if (!isSupportedProvider(providerName)) {
		throw new Error(`Unsupported provider: ${providerName}`);
	}
	const config = SUPPORTED_PROVIDERS[providerName];
	if (!config) throw new Error(`Unsupported provider: ${providerName}`);
	let modelsEndpoint: string;
	if (providerName === 'github') modelsEndpoint = config.baseURL.replace('inference', 'catalog/models');
	else modelsEndpoint = `${config.baseURL}/models`;

	let response: Response;
	if (providerName === 'gemini') {
		modelsEndpoint = modelsEndpoint + '?key=' + apiKey;
		response = await fetch(modelsEndpoint);
	} else if (providerName === 'copilot') {
		const copilotToken = await fetchCopilotToken(apiKey);
		response = await fetch(modelsEndpoint, {
			method: 'GET',
			headers: {
				'Authorization': `Bearer ${copilotToken}`,
				'Content-Type': 'application/json',
				"editor-version": "vscode/1.103.1",
				"copilot-vision-request": "true",
				"editor-plugin-version": "copilot-chat/0.30.1",
				"user-agent": "GitHubCopilotChat/0.30.1"
			},
		});
	} else {
		response = await fetch(modelsEndpoint, { method: 'GET', headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' } });
	}
	if (!response.ok) throw new Error(`Provider ${providerName} models API failed: ${response.status} ${response.statusText}`);
	const data = (await response.json()) as any;
	if (providerName === 'gemini') {
		return { data: data.models.map((m: any) => ({ id: m.name, name: m.displayName, description: m.description || '' })) };
	} else if (providerName === 'github') {
		return { data };
	}
	return data;
}

async function getModelsResponse(apiKey: string, providerKeys: Record<string, string[]>, isPasswordAuth: boolean = false) {
	let gatewayApiKeys: string[] = [];
	if (isPasswordAuth) {
		const gatewayKey = process.env.GATEWAY_API_KEY;
		if (gatewayKey) gatewayApiKeys = gatewayKey.split(',').map((k) => k.trim());
	} else {
		gatewayApiKeys = apiKey.split(',').map((k) => k.trim());
	}

	// Helper to enforce 5s timeout, returning [] on timeout or error
	const withTimeout = <T>(p: Promise<T>, provider: string): Promise<T> =>
		new Promise((resolve) => {
			const fallback = [] as T;
			const id = setTimeout(() => {
				console.warn(`Timeout fetching models from ${provider}`);
				resolve(fallback);
			}, 5000);
			p.then((v) => {
				clearTimeout(id);
				resolve(v);
			}).catch(() => {
				clearTimeout(id);
				resolve(fallback);
			});
		});

	const fetchPromises: Promise<any[]>[] = [];

	if (gatewayApiKeys.length > 0) {
		const randomIndex = Math.floor(Math.random() * gatewayApiKeys.length);
		const currentApiKey = gatewayApiKeys[randomIndex];
		const gatewayPromise = (async () => {
			try {
				if (!currentApiKey) throw new Error('No valid gateway API key found');
				const gateway = createGateway({ apiKey: currentApiKey });
				const availableModels = await gateway.getAvailableModels();
				const now = Math.floor(Date.now() / 1000);
				return availableModels.models
					.map((model: any) => ({
						id: model.id,
						name: model.name,
						description: model.pricing ? ` I: $${(Number(model.pricing.input) * 1000000).toFixed(2)}, O: $${(Number(model.pricing.output) * 1000000).toFixed(2)}; ${model.description || ''}` : (model.description || ''),
						object: 'model',
						created: now,
						owned_by: model.name.split('/')[0],
					}))
					.filter((m: any) => shouldIncludeModel(m));
			} catch (e) {
				return [] as any[];
			}
		})();
		fetchPromises.push(withTimeout(gatewayPromise, 'gateway'));
	}

	for (const [providerName, keys] of Object.entries(providerKeys)) {
		if (!keys || keys.length === 0) continue;
		if (!isSupportedProvider(providerName)) continue;
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
				console.error(`Error getting models for ${providerName}:`, error);
				return [];
			}
		})();
		fetchPromises.push(withTimeout(providerPromise, providerName));
	}

	const results = await Promise.allSettled(fetchPromises);
	const allModels: any[] = [];
	results.forEach((r) => { if (r.status === 'fulfilled' && r.value.length > 0) allModels.push(...r.value); });

	const curated = [
		{ id: 'admin/magic', name: 'Responses Management', object: 'model', created: 0, owned_by: 'admin' },
		{ id: 'image/doubao-vision', name: 'Seed Image', object: 'model', created: 0, owned_by: 'doubao' },
		{ id: 'image/MusePublic/14_ckpt_SD_XL', name: 'Anything XL', object: 'model', created: 0, owned_by: 'modelscope' },
		{ id: 'image/MusePublic/489_ckpt_FLUX_1', name: 'FLUX.1 [dev]', object: 'model', created: 0, owned_by: 'modelscope' },
		{ id: 'image/MusePublic/flux-high-res', name: 'FLUX.1 [dev] High-Res', object: 'model', created: 0, owned_by: 'modelscope' },
		{ id: 'image/black-forest-labs/FLUX.1-Krea-dev', name: 'FLUX.1 Krea [dev]', object: 'model', created: 0, owned_by: 'modelscope' },
		{ id: 'image/black-forest-labs/FLUX.1-Kontext-dev-vision', name: 'FLUX.1 Kontext [dev]', object: 'model', created: 0, owned_by: 'huggingface' },
		{ id: 'image/Qwen/Qwen-Image', name: 'Qwen-Image', object: 'model', created: 0, owned_by: 'modelscope' },
		{ id: 'image/Qwen/Qwen-Image-Edit-vision', name: 'Qwen-Image-Edit', object: 'model', created: 0, owned_by: 'huggingface' },
		{ id: 'video/doubao-seedance-pro-vision', name: 'Seedance 1.0 Pro', object: 'model', created: 0, owned_by: 'doubao' },
		{ id: 'video/doubao-seedance-lite-vision', name: 'Seedance 1.0 Lite', object: 'model', created: 0, owned_by: 'doubao' },
		{ id: 'video/Wan-AI/Qwen-Wan2.2-I2V-A14B-vision', name: 'Qwen Wan2.2 I2V', object: 'model', created: 0, owned_by: 'huggingface' },
		{ id: 'video/Wan-AI/Qwen-Wan2.2-T2V-A14B', name: 'Qwen Wan2.2 T2V', object: 'model', created: 0, owned_by: 'huggingface' },
		{ id: 'video/Wan-AI/Qwen-Wan2.2-TI2V-5B', name: 'Qwen Wan2.2 T2V 5B', object: 'model', created: 0, owned_by: 'huggingface' },
		{ id: 'video/Lightricks/LTX-Video-vision', name: 'LTX Video TI2V', object: 'model', created: 0, owned_by: 'huggingface' },
	];
	const existingIds = new Set(allModels.map((m) => m.id));
	for (const m of curated) if (!existingIds.has(m.id)) allModels.push(m);

	if (allModels.length > 0) return { object: 'list', data: allModels };
	throw new Error('All provider(s) failed to return models');
}

async function handleModelsRequest(c: any) {
	const authHeader = c.req.header('Authorization');
	let apiKey = authHeader?.split(' ')[1];
	const envPassword = process.env.PASSWORD;
	const isPasswordAuth = !!(envPassword && apiKey && envPassword.trim() === apiKey.trim());
	if (!apiKey) return c.text('Unauthorized', 401);

	const headers: Record<string, string> = {};
	c.req.raw.headers.forEach((value: string, key: string) => {
		headers[key.toLowerCase().replace(/-/g, '_')] = value;
	});
	const providerKeys = await getProviderKeys(headers as any, authHeader || null, isPasswordAuth);

	try {
		const modelsResponse = await getModelsResponse(apiKey, providerKeys, isPasswordAuth);
		return c.json(modelsResponse);
	} catch (error: any) {
		return c.json({ error: error?.message || 'All provider(s) failed to return models' }, 500);
	}
}

function parseModelDisplayName(model: string) {
	let baseName = model.split('/').pop() || model;
	if (baseName.endsWith(':free')) baseName = baseName.slice(0, -5);
	let displayName = baseName.replace(/-/g, ' ');
	displayName = displayName.split(' ').map(word => {
		const lowerWord = word.toLowerCase();
		if (lowerWord === 'deepseek') return 'DeepSeek';
		if (lowerWord === 'ernie') return 'ERNIE';
		if (['mai', 'ds', 'r1'].includes(lowerWord)) return word.toUpperCase();
		if (lowerWord === 'gpt') return 'GPT';
		if (lowerWord === 'oss') return 'OSS';
		if (lowerWord === 'glm') return 'GLM';
		if (lowerWord.startsWith('o') && lowerWord.length > 1 && /^\d/.test(lowerWord.slice(1))) return word.toLowerCase();
		if (/^a?\d+[bkmae]$/.test(lowerWord)) return word.toUpperCase();
		return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
	}).join(' ');
	if (displayName === 'MAI DS R1') displayName = 'MAI-DS-R1';
	else if (displayName.startsWith('GPT ')) displayName = displayName.replace(/^GPT /, 'GPT-');
	return displayName;
}

export function parseModelName(model: string) {
	const parts = model.split('/');
	if (parts.length >= 2) {
		const [providerName, ...modelParts] = parts as [keyof typeof SUPPORTED_PROVIDERS, ...string[]];
		const modelName = modelParts.join('/');
		if (Object.prototype.hasOwnProperty.call(SUPPORTED_PROVIDERS, providerName)) {
			return { provider: String(providerName), model: modelName, useCustomProvider: true };
		}
	}
	return { provider: null, model: model, useCustomProvider: false };
}

app.get('/v1/models', async (c: Context) => {
	return handleModelsRequest(c);
});
app.post('/v1/models', async (c: Context) => {
	return handleModelsRequest(c);
});

// Get a model response
app.get('/v1/responses/:response_id', async (c: Context) => {
	const { getResponseHttp } = await import('./modules/management.mts');
	return getResponseHttp(c);
});

// List responses
app.get('/v1/responses', async (c: Context) => {
	const { listResponsesHttp } = await import('./modules/management.mts');
	return listResponsesHttp(c);
});

// Delete all responses
app.delete('/v1/responses/all', async (c: Context) => {
	const { deleteAllResponsesHttp } = await import('./modules/management.mts');
	return deleteAllResponsesHttp(c);
});

// Delete a model response
app.delete('/v1/responses/:response_id', async (c: Context) => {
	const { deleteResponseHttp } = await import('./modules/management.mts');
	return deleteResponseHttp(c);
});

app.get('/*', (c: Context) => {
	return c.text('Running')
})

export default (request: Request, context: any) => {
	geo = context.geo || null;
	return app.fetch(request);
}
