import { createGateway } from '@ai-sdk/gateway';
import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'edge';

export async function GET(req: NextRequest) {
  const authHeader = req.headers.get('Authorization');
  const apiKey = authHeader?.split(' ')[1];

  if (!apiKey) {
    return new NextResponse('Unauthorized', { status: 401 });
  }

  let finalApiKey = apiKey;
  if (apiKey === process.env.PASSWORD) {
    const apiKeys = process.env.AI_GATEWAY_API_KEY?.split(',').map(key => key.trim()) || [];
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

    return NextResponse.json({
      object: 'list',
      data: data,
    });
  } catch (error: any) {
    console.error('Error fetching models:', error);
    const errorMessage = error.message || 'An unknown error occurred';
    const statusCode = error.statusCode || 500;
    return new NextResponse(JSON.stringify({ error: errorMessage }), {
      status: statusCode,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}