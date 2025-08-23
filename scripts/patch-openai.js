#!/usr/bin/env node

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { createRequire } from 'module';
import { dirname, resolve } from 'path';

// Use Node.js module resolution to find the actual OpenAI module path
const require = createRequire(import.meta.url);
let OPENAI_DIST_PATHS = [];

try {
  // Get the directory containing both .js and .mjs files
  const openaiModulePath = require.resolve('@ai-sdk/openai');
  const distDir = dirname(openaiModulePath);

  // Patch both CommonJS (.js) and ES module (.mjs) versions, plus TypeScript definitions
  OPENAI_DIST_PATHS = [
    resolve(distDir, 'index.js'),
    resolve(distDir, 'index.mjs'),
    resolve(distDir, 'index.d.ts'),
    resolve(distDir, 'index.d.mts')
  ];

  console.log('Found OpenAI module files:', OPENAI_DIST_PATHS);
} catch (error) {
  console.error('Could not resolve @ai-sdk/openai module:', error.message);
  process.exit(1);
}// Check if files exist
for (const path of OPENAI_DIST_PATHS) {
  if (!existsSync(path)) {
    console.error('OpenAI module not found at:', path);
    process.exit(1);
  }
}

console.log('Patching OpenAI module to support doubao.web_search...');

let patchedCount = 0;

for (const OPENAI_DIST_PATH of OPENAI_DIST_PATHS) {
  console.log(`Processing ${OPENAI_DIST_PATH}...`);

  let content = readFileSync(OPENAI_DIST_PATH, 'utf8');

  // Check if already patched to avoid double-patching
  if (content.includes('doubao.web_search') || content.includes('doubaoWebSearch')) {
    console.log(`Already patched - skipping ${OPENAI_DIST_PATH}`);
    continue;
  }

  const isESModule = OPENAI_DIST_PATH.endsWith('.mjs');
  const isTypeScript = OPENAI_DIST_PATH.endsWith('.d.ts') || OPENAI_DIST_PATH.endsWith('.d.mts');

  if (isTypeScript) {
    // Handle TypeScript definition files
    const doubaoTypeInsertion = `    doubaoWebSearch: _ai_sdk_provider_utils.ProviderDefinedToolFactory<{
        limit?: number;
        userLocation?: {
            type: string;
            country?: string;
            region?: string;
            city?: string;
        };
        maxToolCalls?: number;
    }, {}>;`;

    // Find the specific pattern: "    }>;\n};" and replace with the insertion
    content = content.replace('    }>;\n};', '    }>;\n' + doubaoTypeInsertion + '\n};');

    writeFileSync(OPENAI_DIST_PATH, content);
    console.log(`✅ Successfully patched ${OPENAI_DIST_PATH} (TypeScript definitions)`);
    patchedCount++;
    continue;
  }

  // Handle JavaScript files
  const importPattern = isESModule ? 'z4' : 'import_v44';

  // Add the Doubao web search args schema
  const doubaoSchemaInsertion = `var doubaoWebSearchArgsSchema = ${importPattern}.object({
  limit: ${importPattern}.number().int().min(1).max(50).optional(),
  userLocation: ${importPattern}.object({
    type: ${importPattern}.string(),
    country: ${importPattern}.string().optional(),
    region: ${importPattern}.string().optional(),
    city: ${importPattern}.string().optional()
  }).optional(),
  maxToolCalls: ${importPattern}.number().int().min(1).max(10).optional()
});
`;

  // Add the doubaoWebSearch tool definition
  const doubaoToolInsertion = `var doubaoWebSearch = ${isESModule ? 'createProviderDefinedToolFactory2' : '(0, import_provider_utils4.createProviderDefinedToolFactory)'}({
  id: "doubao.web_search",
  name: "web_search",
  inputSchema: ${importPattern}.object({
    limit: ${importPattern}.number().int().min(1).max(50).optional(),
    userLocation: ${importPattern}.object({
      type: ${importPattern}.string(),
      country: ${importPattern}.string().optional(),
      region: ${importPattern}.string().optional(),
      city: ${importPattern}.string().optional()
    }).optional(),
    maxToolCalls: ${importPattern}.number().int().min(1).max(10).optional()
  })
});
`;

  // Find where to insert (after webSearchPreview)
  const schemaInsertPoint = content.indexOf(isESModule ?
    'var webSearchPreview = createProviderDefinedToolFactory2({' :
    'var webSearchPreview = (0, import_provider_utils4.createProviderDefinedToolFactory)({'
  );

  if (schemaInsertPoint === -1) {
    console.error(`Could not find insertion point in ${OPENAI_DIST_PATH}`);
    continue;
  }

  // Insert the schema and tool definition
  content = content.slice(0, schemaInsertPoint) + doubaoSchemaInsertion + doubaoToolInsertion + content.slice(schemaInsertPoint);

  // Add doubaoWebSearch to openaiTools object
  const toolsObjectPattern = /var openaiTools = \{[\s\S]*?webSearchPreview[\s\S]*?\};/;
  content = content.replace(toolsObjectPattern, (match) => {
    return match.replace('webSearchPreview\n};', 'webSearchPreview,\n  doubaoWebSearch\n};');
  });  // Add the doubao.web_search case in tool processing
  const doubaoCase = `          case "doubao.web_search": {
            const args = doubaoWebSearchArgsSchema.parse(tool.args);
            ${isESModule ? 'openaiTools2' : 'openaiTools2'}.push({
              type: "web_search",
              limit: args.limit,
              userLocation: args.userLocation,
              maxToolCalls: args.maxToolCalls
            });
            break;
          }`;

  // Find and replace both occurrences of the switch statement
  const switchPattern = /(case "openai\.web_search_preview": {[\s\S]*?break;\s*})/g;
  let matches = 0;
  content = content.replace(switchPattern, (match) => {
    matches++;
    return match + '\n' + doubaoCase;
  });

  if (matches === 0) {
    console.error(`Could not find openai.web_search_preview cases in ${OPENAI_DIST_PATH}`);
    continue;
  }

  writeFileSync(OPENAI_DIST_PATH, content);
  console.log(`✅ Successfully patched ${OPENAI_DIST_PATH} (${matches} locations)`);
  patchedCount++;
}

if (patchedCount > 0) {
  console.log(`✅ Successfully patched ${patchedCount} OpenAI module files to support doubao.web_search`);
} else {
  console.log('No files were patched');
}