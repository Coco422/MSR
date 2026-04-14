from __future__ import annotations

import secrets

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader


def api_key_scheme(header_name: str) -> APIKeyHeader:
    return APIKeyHeader(name=header_name, auto_error=False)


async def require_api_key(request: Request, api_key: str | None = Security(APIKeyHeader(name="X-API-Key", auto_error=False))) -> str:
    expected = request.app.state.container.settings.security.api_key
    if not api_key or not secrets.compare_digest(api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
    return api_key
