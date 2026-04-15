"""
utils/rate_limiter.py - In-memory per-IP sliding window rate limiter.

Implements a simple sliding window counter:
  - Each IP address has a list of request timestamps.
  - On each request, timestamps older than the window are pruned.
  - If the remaining count >= max_requests, the request is rejected (HTTP 429).

This is thread-safe via threading.Lock and suitable for single-process deployments.
For multi-process (gunicorn workers), use Redis-backed rate limiting instead.
"""

import time
import threading
from typing import Dict, List
from collections import defaultdict

from fastapi import HTTPException, Request

from utils.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Per-IP sliding window rate limiter.

    Args:
        max_requests:  Maximum number of requests allowed within the window.
        window_seconds: Duration of the sliding window in seconds.
    """

    def __init__(self, max_requests: int = 5, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from the request (supports proxied requests)."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def check(self, request: Request) -> None:
        """
        Check whether the request is within rate limits.

        Raises:
            HTTPException(429): If the client has exceeded the rate limit.
        """
        client_ip = self._get_client_ip(request)
        now = time.time()
        window_start = now - self.window_seconds

        with self._lock:
            # Prune expired timestamps
            self._requests[client_ip] = [
                ts for ts in self._requests[client_ip] if ts > window_start
            ]

            if len(self._requests[client_ip]) >= self.max_requests:
                retry_after = int(
                    self._requests[client_ip][0] + self.window_seconds - now
                ) + 1
                logger.warning(
                    f"Rate limit exceeded for IP {client_ip}: "
                    f"{len(self._requests[client_ip])}/{self.max_requests} "
                    f"in {self.window_seconds}s window."
                )
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "max_requests": self.max_requests,
                        "window_seconds": self.window_seconds,
                        "retry_after_seconds": retry_after,
                    },
                )

            # Record this request
            self._requests[client_ip].append(now)

        logger.debug(
            f"Rate limit OK for {client_ip}: "
            f"{len(self._requests[client_ip])}/{self.max_requests}"
        )


# Singleton limiter instance (configured via settings at import time)
# Actual initialization happens in main.py to avoid circular imports
rate_limiter: RateLimiter | None = None


def init_rate_limiter(max_requests: int, window_seconds: int) -> RateLimiter:
    """Initialize the global rate limiter singleton."""
    global rate_limiter
    rate_limiter = RateLimiter(max_requests=max_requests, window_seconds=window_seconds)
    return rate_limiter


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    if rate_limiter is None:
        raise RuntimeError("Rate limiter not initialized. Call init_rate_limiter() first.")
    return rate_limiter
