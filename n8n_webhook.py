import os
import json
from typing import Any, Dict, Optional

import httpx

DEFAULT_WEBHOOK_URL = os.getenv(
    "N8N_WEBHOOK_URL",
    "https://utk.app.n8n.cloud/webhook-test/dcf2f8b7-bec4-41f6-9a4b-a6d178540908",
)


async def trigger_n8n_workflow(
    document: Dict[str, Any],
    webhook_url: Optional[str] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    timeout_seconds: float = 30.0,
) -> Dict[str, Any]:
    """
    Invoke the upstream n8n webhook (GET) with the extracted document payload.

    The document JSON is serialized and passed via the querystring parameter `payload`.
    """
    if not document:
        raise ValueError("Document payload is required.")

    url = webhook_url or DEFAULT_WEBHOOK_URL
    params = extra_params.copy() if extra_params else {}
    params["payload"] = json.dumps(document, ensure_ascii=False)

    timeout = httpx.Timeout(timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, params=params)

    if resp.status_code >= 400:
        raise RuntimeError(f"n8n webhook error {resp.status_code}: {resp.text}")

    try:
        body = resp.json()
    except ValueError:
        body = {"raw": resp.text}

    return {
        "status_code": resp.status_code,
        "body": body,
        "url": str(resp.url),
    }
