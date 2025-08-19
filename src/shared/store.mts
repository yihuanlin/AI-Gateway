import { getStore } from '@netlify/blobs';

function readHeader(headers: Headers | Record<string, string> | undefined, name: string): string | undefined {
  if (!headers) return undefined;
  try {
    if (typeof (headers as any).get === 'function') {
      const h = headers as Headers;
      return h.get(name) || h.get(name.toLowerCase()) || undefined;
    }
  } catch {}
  const obj = headers as Record<string, string>;
  const underscore = name.replace(/-/g, '_');
  return obj?.[name] || obj?.[name.toLowerCase()] || obj?.[underscore] || obj?.[underscore.toLowerCase()];
}

export function getStoreWithConfig(name: string, headers?: Headers | Record<string, string>) {
  const siteID = process.env.NETLIFY_SITE_ID;
  const token = process.env.NETLIFY_TOKEN;
  const headerSiteID = readHeader(headers, 'x-netlify-site-id');
  const headerToken = readHeader(headers, 'x-netlify-token');

  if ((siteID && token) || (headerSiteID && headerToken)) {
    return getStore({ name, siteID: siteID || headerSiteID!, token: token || headerToken! });
  } else if (process.env.NETLIFY_BLOBS_CONTEXT) {
    return getStore(name);
  } else {
    return getStore(name);
  }
}
