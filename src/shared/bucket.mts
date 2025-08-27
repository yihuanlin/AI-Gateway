async function signRequest(method: string, url: string, headers: Record<string, string>, body: Uint8Array, accessKey: string, secretKey: string, region: string = 'auto') {
    const urlObj = new URL(url);
    const pathname = urlObj.pathname;

    const now = new Date();
    const dateStamp = now.toISOString().slice(0, 10).replace(/-/g, '');
    const timeStamp = now.toISOString().slice(0, 19).replace(/[-:]/g, '') + 'Z';

    // Create canonical headers
    const canonicalHeaders = Object.keys(headers)
        .sort()
        .map(key => `${key.toLowerCase()}:${headers[key]}\n`)
        .join('');

    const signedHeaders = Object.keys(headers)
        .sort()
        .map(key => key.toLowerCase())
        .join(';');

    // Create payload hash
    const bodyBuffer = new ArrayBuffer(body.length);
    const bodyView = new Uint8Array(bodyBuffer);
    bodyView.set(body);
    const payloadHash = await crypto.subtle.digest('SHA-256', bodyBuffer).then(buf =>
        Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2, '0')).join('')
    );

    // Create canonical request
    const canonicalRequest = [
        method,
        pathname,
        '', // query string
        canonicalHeaders,
        signedHeaders,
        payloadHash
    ].join('\n');

    // Create string to sign
    const algorithm = 'AWS4-HMAC-SHA256';
    const credentialScope = `${dateStamp}/${region}/s3/aws4_request`;
    const canonicalRequestHash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(canonicalRequest)).then(buf =>
        Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2, '0')).join('')
    );

    const stringToSign = [
        algorithm,
        timeStamp,
        credentialScope,
        canonicalRequestHash
    ].join('\n');

    // Calculate signature
    const signingKey = await getSignatureKey(secretKey, dateStamp, region, 's3');
    const signature = await crypto.subtle.sign('HMAC', signingKey, new TextEncoder().encode(stringToSign)).then(buf =>
        Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2, '0')).join('')
    );

    // Create authorization header
    const authorization = `${algorithm} Credential=${accessKey}/${credentialScope}, SignedHeaders=${signedHeaders}, Signature=${signature}`;

    return {
        ...headers,
        'Authorization': authorization,
        'X-Amz-Date': timeStamp,
        'X-Amz-Content-Sha256': payloadHash
    };
}

async function getSignatureKey(key: string, dateStamp: string, regionName: string, serviceName: string) {
    const kDate = await crypto.subtle.importKey('raw', new TextEncoder().encode('AWS4' + key), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']).then(k =>
        crypto.subtle.sign('HMAC', k, new TextEncoder().encode(dateStamp))
    );
    const kRegion = await crypto.subtle.importKey('raw', kDate, { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']).then(k =>
        crypto.subtle.sign('HMAC', k, new TextEncoder().encode(regionName))
    );
    const kService = await crypto.subtle.importKey('raw', kRegion, { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']).then(k =>
        crypto.subtle.sign('HMAC', k, new TextEncoder().encode(serviceName))
    );
    const kSigning = await crypto.subtle.importKey('raw', kService, { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']).then(k =>
        crypto.subtle.sign('HMAC', k, new TextEncoder().encode('aws4_request'))
    );
    return crypto.subtle.importKey('raw', kSigning, { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
}

export async function uploadBase64ToStorage(base64Data: string, timestamp?: string): Promise<string> {
    if (!process.env.S3_API || !process.env.S3_PUBLIC_URL || !process.env.S3_ACCESS_KEY || !process.env.S3_SECRET_KEY) {
        throw new Error('S3_API, S3_PUBLIC_URL, S3_ACCESS_KEY, and S3_SECRET_KEY environment variables are required');
    }

    // Parse base64 data and determine content type
    let contentType = 'image/jpeg';
    let imageBuffer: Uint8Array;

    if (base64Data.startsWith('data:')) {
        // Extract content type from data URL
        const matches = base64Data.match(/^data:([^;]+);base64,(.+)$/);
        if (matches && matches[2]) {
            contentType = matches[1] || 'image/jpeg';
            // Convert base64 to Uint8Array using browser-compatible method
            const binaryString = atob(matches[2]);
            imageBuffer = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                imageBuffer[i] = binaryString.charCodeAt(i);
            }
        } else {
            throw new Error('Invalid base64 data URL format');
        }
    } else {
        // Plain base64 string, assume JPEG
        const binaryString = atob(base64Data);
        imageBuffer = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            imageBuffer[i] = binaryString.charCodeAt(i);
        }
    }

    // Determine file extension from content type
    const extension = contentType.split('/')[1] || 'jpg';

    const fileTimestamp = timestamp || new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
    const filename = `${fileTimestamp}.${extension}`;

    const url = process.env.S3_API + '/' + filename;

    // Prepare headers
    const headers: Record<string, string> = {
        'Content-Type': contentType,
        'Content-Length': imageBuffer.length.toString(),
        'Host': new URL(process.env.S3_API).hostname
    };

    // Sign the request
    const signedHeaders = await signRequest('PUT', url, headers, imageBuffer, process.env.S3_ACCESS_KEY, process.env.S3_SECRET_KEY);

    // Create a new ArrayBuffer for edge function compatibility
    const bodyBuffer = new ArrayBuffer(imageBuffer.length);
    const bodyView = new Uint8Array(bodyBuffer);
    bodyView.set(imageBuffer);

    const response = await fetch(url, {
        method: 'PUT',
        headers: signedHeaders,
        body: bodyBuffer
    });

    if (!response.ok) {
        throw new Error(`Failed to upload to bucket: ${response.status} ${response.statusText}`);
    }

    // Return the public URL
    const publicUrl = process.env.S3_PUBLIC_URL.endsWith('/')
        ? process.env.S3_PUBLIC_URL + filename
        : process.env.S3_PUBLIC_URL + '/' + filename;

    return publicUrl;
}

export async function uploadBlobToStorage(blob: Blob, timestamp?: string): Promise<string> {
    if (!process.env.S3_API || !process.env.S3_PUBLIC_URL || !process.env.S3_ACCESS_KEY || !process.env.S3_SECRET_KEY) {
        throw new Error('S3_API, S3_PUBLIC_URL, S3_ACCESS_KEY, and S3_SECRET_KEY environment variables are required');
    }

    // Get content type from blob
    const contentType = blob.type || 'application/octet-stream';

    // Determine file extension from content type
    let extension = 'bin';
    if (contentType.startsWith('image/')) {
        extension = contentType.split('/')[1] || 'jpg';
    } else if (contentType.startsWith('video/')) {
        extension = contentType.split('/')[1] || 'mp4';
    }

    const fileTimestamp = timestamp || new Date().toISOString().replace(/[-:T]/g, '').slice(0, 12);
    const filename = `${fileTimestamp}.${extension}`;

    const url = process.env.S3_API + '/' + filename;

    // Convert blob to Uint8Array for signing
    const arrayBuffer = await blob.arrayBuffer();
    const uint8Array = new Uint8Array(arrayBuffer);

    // Prepare headers
    const headers: Record<string, string> = {
        'Content-Type': contentType,
        'Content-Length': blob.size.toString(),
        'Host': new URL(process.env.S3_API).hostname
    };

    // Sign the request
    const signedHeaders = await signRequest('PUT', url, headers, uint8Array, process.env.S3_ACCESS_KEY, process.env.S3_SECRET_KEY);

    const response = await fetch(url, {
        method: 'PUT',
        headers: signedHeaders,
        body: arrayBuffer
    });

    if (!response.ok) {
        throw new Error(`Failed to upload to bucket: ${response.status} ${response.statusText}`);
    }

    // Return the public URL
    const publicUrl = process.env.S3_PUBLIC_URL.endsWith('/')
        ? process.env.S3_PUBLIC_URL + filename
        : process.env.S3_PUBLIC_URL + '/' + filename;

    return publicUrl;
}
