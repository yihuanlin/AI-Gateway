async function signRequest(method: string, url: string, headers: Record<string, string>, body: Buffer, accessKey: string, secretKey: string, region: string = 'auto') {
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
    const payloadHash = await crypto.subtle.digest('SHA-256', body).then(buf =>
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

export async function uploadBase64ToBlob(base64Data: string, timestamp?: string): Promise<string> {
    if (!process.env.S3_API || !process.env.S3_PUBLIC_URL || !process.env.S3_ACCESS_KEY || !process.env.S3_SECRET_KEY) {
        throw new Error('S3_API, S3_PUBLIC_URL, S3_ACCESS_KEY, and S3_SECRET_KEY environment variables are required');
    }

    // Parse base64 data and determine content type
    let contentType = 'image/jpeg';
    let imageBuffer: Buffer;

    if (base64Data.startsWith('data:')) {
        // Extract content type from data URL
        const matches = base64Data.match(/^data:([^;]+);base64,(.+)$/);
        if (matches && matches[2]) {
            contentType = matches[1] || 'image/jpeg';
            imageBuffer = Buffer.from(matches[2], 'base64');
        } else {
            throw new Error('Invalid base64 data URL format');
        }
    } else {
        // Plain base64 string, assume JPEG
        imageBuffer = Buffer.from(base64Data, 'base64');
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

    const response = await fetch(url, {
        method: 'PUT',
        headers: signedHeaders,
        body: imageBuffer
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
