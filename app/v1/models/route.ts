import { createGateway } from '@ai-sdk/gateway';
import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'edge';

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

export async function GET(req: NextRequest) {
  const authHeader = req.headers.get('Authorization');
  const apiKey = authHeader?.split(' ')[1];

  if (!apiKey) {
    return new NextResponse('Unauthorized', {
      status: 401,
      headers: corsHeaders,
    });
  }

  const apiKeys = apiKey.split(',').map(key => key.trim()) || [];
  let lastError: any;

  for (let i = 0; i < apiKeys.length; i++) {
    const currentApiKey = apiKeys[i];

    const gateway = createGateway({
      apiKey: currentApiKey,
      baseURL: 'https://ai-gateway.vercel.sh/v1/ai',
    });

    try {
      const availableModels = await gateway.getAvailableModels();

      const now = Math.floor(Date.now() / 1000);
      const data = availableModels.models.map(model => ({
        id: model.id,
        name: model.name,
        description: `${model.pricing
          ? ` I: ${(Number(model.pricing.input) * 1000000).toFixed(2)}$, O: ${(
            Number(model.pricing.output) * 1000000
          ).toFixed(2)}$`
          : ''
          } ${model.description || ''}`,
        object: 'model',
        created: now,
        owned_by: model.name.split('/')[0],
        pricing: model.pricing || {},
      }));

      return NextResponse.json(
        {
          object: 'list',
          data: data,
        },
        {
          headers: corsHeaders,
        },
      );
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
  const errorMessage = lastError.message || 'An unknown error occurred';
  const statusCode = lastError.statusCode || 500;
  return new NextResponse(JSON.stringify({
    error: `All ${apiKeys.length} API key(s) failed. Last error: ${errorMessage}`
  }), {
    status: statusCode,
    headers: {
      'Content-Type': 'application/json',
      ...corsHeaders,
    },
  });
}