import { getStoreWithConfig } from './store.mts';

const extFromContentType = (contentType: string): string => {
    if (!contentType) return 'bin';
    const main = (contentType.split?.(';')[0]) || contentType;
    const parts = main.includes('/') ? main.split('/') : ['image', 'jpeg'];
    return parts[1] || 'bin';
}

const nowKey = (ext: string, timestamp?: string): string => {
    const ts = timestamp || new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
    return `${ts}.${ext}`;
}

export const buildPublicUrlForKey = (key: string): string => {
    return `${process.env.URL}/v1/files/${encodeURIComponent(key)}`;
}

export const uploadBase64ToStorage = async (base64Data: string, timestamp?: string): Promise<string> => {
    // Parse base64 data and determine content type
    let contentType = 'image/jpeg';
    let bytes: Uint8Array | null = null;

    if (base64Data.startsWith('data:')) {
        const matches = base64Data.match(/^data:([^;]+);base64,(.+)$/);
        if (matches && matches[2]) {
            contentType = matches[1] || 'image/jpeg';
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
        contentType = 'image/jpeg';
    }

    const ext = extFromContentType(contentType);
    const key = nowKey(ext, timestamp);
    const store = await getStoreWithConfig('files');
    const blob = new Blob([bytes!], { type: contentType });
    await store.set(key, blob, { metadata: { contentType } });
    return buildPublicUrlForKey(key);
}

export const uploadBlobToStorage = async (blob: Blob, timestamp?: string): Promise<string> => {
    const contentType = blob.type || 'image/jpeg';
    const ext = extFromContentType(contentType);
    const key = nowKey(ext, timestamp);
    const store = await getStoreWithConfig('files');
    await store.set(key, blob, { metadata: { contentType } });
    return buildPublicUrlForKey(key);
}

export const getFileWithMetadata = async <T extends 'arrayBuffer' | 'blob' | 'json' | 'stream' | 'text' = 'blob'>(key: string, type?: T): Promise<{ data: any; metadata?: Record<string, any> } | null> => {
    const store = await getStoreWithConfig('files');
    try {
        const res = await store.getWithMetadata(key, { type: (type || 'blob') as any });
        if (!res) return null;
        return { data: (res as any).data ?? res, metadata: (res as any).metadata };
    } catch {
        return null;
    }
}
