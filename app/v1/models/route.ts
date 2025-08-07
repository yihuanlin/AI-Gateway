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

  let finalApiKey = apiKey;
  if (apiKey === process.env.PASSWORD) {
    const apiKeys =
      process.env.AI_GATEWAY_API_KEY?.split(',').map(key => key.trim()) || [];
    if (apiKeys.length > 0) {
      const randomIndex = Math.floor(Math.random() * apiKeys.length);
      finalApiKey = apiKeys[randomIndex];
    }
  }

  const gateway = createGateway({
    apiKey: finalApiKey,
    baseURL: 'https://ai-gateway.vercel.sh/v1/ai',
  });

  try {
    const availableModels = await gateway.getAvailableModels();

    const now = Math.floor(Date.now() / 1000);
    const data = availableModels.models.map(model => ({
      id: model.id,
      object: 'model',
      created: now,
      owned_by: 'vercel-ai-gateway',
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
    console.error('Error fetching models:', error);
    const errorMessage = error.message || 'An unknown error occurred';
    const statusCode = error.statusCode || 500;
    return new NextResponse(JSON.stringify({ error: errorMessage }), {
      status: statusCode,
      headers: {
        'Content-Type': 'application/json',
        ...corsHeaders,
      },
    });
  }
}