function getStoreTokenAndId(): { token: string; storeId: string } {
  const token = process.env.BLOB_READ_WRITE_TOKEN;
  if (!token) {
    throw new Error('Missing BLOB_READ_WRITE_TOKEN environment variable');
  }
  const parts = token.split('_');
  const storeId = parts[3] || '';
  return { token, storeId };
}

async function blobPut(
  pathname: string,
  body: any,
  options?: { contentType?: string }
): Promise<any> {
  const { token, storeId } = getStoreTokenAndId();
  const baseUrl = process.env.VERCEL_BLOB_API_URL || 'https://vercel.com/api/blob';

  const headers: Record<string, string> = {
    'authorization': `Bearer ${token}`,
    'x-vercel-blob-store-id': storeId,
    'x-api-version': '12',
    'x-vercel-blob-access': 'private',
    'x-add-random-suffix': '0',
    'x-allow-overwrite': '1',
  };

  let contentType = options?.contentType;
  if (!contentType && body && typeof body === 'object' && 'type' in body && body.type) {
    contentType = body.type;
  }

  if (contentType) {
    headers['x-content-type'] = contentType;
    headers['content-type'] = contentType;
  }

  const url = `${baseUrl}/?pathname=${encodeURIComponent(pathname)}`;
  const response = await fetch(url, {
    method: 'PUT',
    headers,
    body,
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => '');
    throw new Error(`Vercel Blob PUT failed: ${response.status} ${response.statusText} ${errorText}`);
  }

  return await response.json();
}

async function blobGet(
  pathname: string
): Promise<Response | null> {
  const { token, storeId } = getStoreTokenAndId();
  const blobUrl = `https://${storeId}.private.blob.vercel-storage.com/${pathname}`;

  const response = await fetch(blobUrl, {
    method: 'GET',
    headers: {
      'authorization': `Bearer ${token}`,
    },
  });

  if (response.status === 404) {
    return null;
  }

  if (!response.ok) {
    throw new Error(`Vercel Blob GET failed: ${response.status} ${response.statusText}`);
  }

  return response;
}

async function blobDelete(pathnames: string[]): Promise<void> {
  const { token, storeId } = getStoreTokenAndId();
  const baseUrl = process.env.VERCEL_BLOB_API_URL || 'https://vercel.com/api/blob';

  const headers: Record<string, string> = {
    'authorization': `Bearer ${token}`,
    'x-vercel-blob-store-id': storeId,
    'x-api-version': '12',
    'content-type': 'application/json',
  };

  const response = await fetch(`${baseUrl}/delete`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ urls: pathnames }),
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => '');
    throw new Error(`Vercel Blob DELETE failed: ${response.status} ${response.statusText} ${errorText}`);
  }
}

async function blobList(options: { prefix?: string; cursor?: string | undefined; limit?: number }): Promise<any> {
  const { token, storeId } = getStoreTokenAndId();
  const baseUrl = process.env.VERCEL_BLOB_API_URL || 'https://vercel.com/api/blob';

  const searchParams = new URLSearchParams();
  if (options.prefix) searchParams.set('prefix', options.prefix);
  if (options.cursor) searchParams.set('cursor', options.cursor);
  if (options.limit) searchParams.set('limit', options.limit.toString());

  const headers: Record<string, string> = {
    'authorization': `Bearer ${token}`,
    'x-vercel-blob-store-id': storeId,
    'x-api-version': '12',
  };

  const url = `${baseUrl}?${searchParams.toString()}`;
  const response = await fetch(url, {
    method: 'GET',
    headers,
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => '');
    throw new Error(`Vercel Blob LIST failed: ${response.status} ${response.statusText} ${errorText}`);
  }

  return await response.json();
}

class VercelBlobStore {
  constructor(private name: string) {}

  private getFullKey(key: string): string {
    return `${this.name}/${key}`;
  }

  private getMetadataKey(key: string): string {
    return `${this.name}/${key}.metadata.json`;
  }

  async set(key: string, value: any, options?: { metadata?: Record<string, any> }) {
    const fullKey = this.getFullKey(key);
    await blobPut(fullKey, value);
    if (options?.metadata) {
      const metaKey = this.getMetadataKey(key);
      await blobPut(metaKey, JSON.stringify(options.metadata), { contentType: 'application/json' });
    }
  }

  async setJSON(key: string, value: any) {
    const fullKey = this.getFullKey(key);
    await blobPut(fullKey, JSON.stringify(value), { contentType: 'application/json' });
  }

  async get(key: string, options?: { type?: 'text' | 'json' | 'blob' | 'arrayBuffer' | 'stream' }) {
    const fullKey = this.getFullKey(key);
    try {
      const response = await blobGet(fullKey);
      if (!response) return null;
      const type = options?.type || 'text';
      if (type === 'json') {
        return await response.json();
      } else if (type === 'blob') {
        return await response.blob();
      } else if (type === 'arrayBuffer') {
        return await response.arrayBuffer();
      } else if (type === 'stream') {
        return response.body;
      } else {
        return await response.text();
      }
    } catch {
      return null;
    }
  }

  async getWithMetadata(key: string, options?: { type?: 'text' | 'json' | 'blob' | 'arrayBuffer' | 'stream' }) {
    const fullKey = this.getFullKey(key);
    const metaKey = this.getMetadataKey(key);
    try {
      const response = await blobGet(fullKey);
      if (!response) return null;

      const type = options?.type || 'text';
      let data: any;
      if (type === 'json') {
        data = await response.json();
      } else if (type === 'blob') {
        data = await response.blob();
      } else if (type === 'arrayBuffer') {
        data = await response.arrayBuffer();
      } else if (type === 'stream') {
        data = response.body;
      } else {
        data = await response.text();
      }

      let metadata: Record<string, any> | undefined;
      try {
        const metaResponse = await blobGet(metaKey);
        if (metaResponse) {
          metadata = (await metaResponse.json()) as Record<string, any>;
        }
      } catch {
        // No metadata file
      }

      return { data, metadata };
    } catch {
      return null;
    }
  }

  async delete(key: string) {
    const fullKey = this.getFullKey(key);
    const metaKey = this.getMetadataKey(key);
    try {
      await blobDelete([fullKey, metaKey]);
    } catch {
      // Best effort
    }
  }

  async list(options?: { prefix?: string }) {
    try {
      const vercelPrefix = `${this.name}/${options?.prefix || ''}`;
      let cursor: string | undefined;
      let hasMore = true;
      const blobsList: any[] = [];

      while (hasMore) {
        const result = await blobList({ prefix: vercelPrefix, cursor });

        if (result.blobs) {
          for (const b of result.blobs) {
            if (b.pathname.endsWith('.metadata.json')) continue;

            const key = b.pathname.substring(this.name.length + 1);
            blobsList.push({
              key,
              size: b.size,
              uploadedAt: b.uploadedAt ? new Date(b.uploadedAt) : new Date(),
            });
          }
        }

        hasMore = result.hasMore;
        cursor = result.cursor;
      }

      return { blobs: blobsList };
    } catch {
      return { blobs: [] };
    }
  }
}

export const getStoreWithConfig = (name: string) => {
  return new VercelBlobStore(name);
};
