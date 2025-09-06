import { getStore } from '@netlify/blobs';

export function getStoreWithConfig(name: string) {
  const siteID = process.env.NETLIFY_SITE_ID;
  const token = process.env.NETLIFY_TOKEN;
  if ((siteID && token)) {
    return getStore({ name, siteID, token });
  } else if (process.env.NETLIFY_BLOBS_CONTEXT) {
    return getStore(name);
  } else {
    return getStore(name);
  }
}
