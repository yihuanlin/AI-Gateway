import { getStoreWithConfig } from './store.mts';

function extFromContentType(contentType: string): string {
    if (!contentType) return 'bin';
    const main = (contentType.split?.(';')[0]) || contentType;
    const parts = main.includes('/') ? main.split('/') : ['application', 'octet-stream'];
    return parts[1] || 'bin';
}

function nowKey(ext: string, timestamp?: string): string {
    const ts = timestamp || new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
    return `${ts}.${ext}`;
}

export function buildPublicUrlForKey(key: string): string {
    const base = (process.env.URL || '').replace(/^http:\/\//, 'https://');
    const prefix = base || '';
    return `${prefix}/v1/files/${encodeURIComponent(key)}`;
}

export async function uploadBase64ToStorage(base64Data: string, timestamp?: string): Promise<string> {
    // Parse base64 data and determine content type
    let contentType = 'application/octet-stream';
    let bytes: Uint8Array | null = null;

    if (base64Data.startsWith('data:')) {
        const matches = base64Data.match(/^data:([^;]+);base64,(.+)$/);
        if (matches && matches[2]) {
            contentType = matches[1] || 'application/octet-stream';
            const binaryString = atob(matches[2]);
            bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
        } else {
            throw new Error('Invalid base64 data URL format');
        }
    } else {
        // Plain base64, best-effort: require caller to include mediaType elsewhere if needed
        const binaryString = atob(base64Data);
        bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
        contentType = 'application/octet-stream';
    }

    const ext = extFromContentType(contentType);
    const key = nowKey(ext, timestamp);
    const store = await getStoreWithConfig('files');
    const blob = new Blob([bytes!], { type: contentType });
    await store.set(key, blob, { metadata: { contentType } });
    return buildPublicUrlForKey(key);
}

export async function uploadBlobToStorage(blob: Blob, timestamp?: string): Promise<string> {
    const contentType = blob.type || 'application/octet-stream';
    const ext = extFromContentType(contentType);
    const key = nowKey(ext, timestamp);
    const store = await getStoreWithConfig('files');
    await store.set(key, blob, { metadata: { contentType } });
    return buildPublicUrlForKey(key);
}

export async function getFileWithMetadata<T extends 'arrayBuffer' | 'blob' | 'json' | 'stream' | 'text' = 'blob'>(key: string, type?: T): Promise<{ data: any; metadata?: Record<string, any> } | null> {
    const store = await getStoreWithConfig('files');
    try {
        const res = await store.getWithMetadata(key, { type: (type || 'blob') as any });
        if (!res) return null;
        return { data: (res as any).data ?? res, metadata: (res as any).metadata };
    } catch {
        return null;
    }
}
