from __future__ import annotations

import logging
import time
from typing import Any, Callable, TypeVar

import requests

from modules.utils.exceptions import InsufficientCreditsException

logger = logging.getLogger(__name__)

# Total attempts (initial try + retries) for any recoverable operation.
MAX_ATTEMPTS = 3

# Base multiplier for the exponential backoff between attempts (seconds).
BACKOFF_BASE = 1.5

T = TypeVar("T")


def backoff_delay(attempt: int, base: float = BACKOFF_BASE) -> float:
    """Seconds to wait before attempt ``attempt`` (0-indexed)."""
    return base * (attempt + 1)


def is_transient_error(exc: BaseException) -> bool:
    """Return True for errors worth retrying: network glitches and server-side
    (5xx) or rate-limit (429) responses. Auth/quota/client errors are not."""
    if isinstance(exc, InsufficientCreditsException):
        return False
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        response = getattr(exc, "response", None)
        code = response.status_code if response is not None else 500
        return code >= 500 or code == 429
    return False


def run_with_retry(
    func: Callable[[], T],
    *,
    max_attempts: int = MAX_ATTEMPTS,
    base_delay: float = BACKOFF_BASE,
    label: str = "operation",
) -> T:
    """Run ``func`` up to ``max_attempts`` times.

    Transient errors (see :func:`is_transient_error`) are retried with an
    exponential backoff. ``InsufficientCreditsException`` and any non-retryable
    error are re-raised immediately. Returns the first successful result.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return func()
        except InsufficientCreditsException:
            raise
        except Exception as exc:
            last_exc = exc
            if is_transient_error(exc) and attempt < max_attempts - 1:
                delay = backoff_delay(attempt, base_delay)
                logger.warning(
                    "%s failed (attempt %d/%d), retrying in %.1fs: %s",
                    label,
                    attempt + 1,
                    max_attempts,
                    delay,
                    exc,
                )
                time.sleep(delay)
                continue
            logger.warning("%s failed after %d attempts: %s", label, attempt + 1, exc)
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{label} produced no result")


def retry_ocr(
    ocr_engine: Any,
    image: Any,
    blocks: list,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    base_delay: float = BACKOFF_BASE,
    label: str = "OCR",
) -> bool:
    """Run OCR on ``blocks``, retrying both transient server errors and any
    bubbles that come back empty.

    ``blocks`` are mutated in place by ``ocr_engine.process``. Returns when OCR
    completed, possibly with a few bubbles still empty (the caller should report
    those rather than skip the page). Raises the underlying exception when the
    whole OCR call kept failing with non-recoverable errors, so the caller's
    error handling can skip the page.
    """
    targets = list(blocks)
    for attempt in range(max_attempts):
        try:
            ocr_engine.process(image, targets)
        except InsufficientCreditsException:
            raise
        except Exception as exc:
            if not is_transient_error(exc) or attempt == max_attempts - 1:
                # Non-recoverable error (or out of attempts): let the caller's
                # error handling decide how to surface it.
                logger.exception("%s failed after %d attempts: %s", label, attempt + 1, exc)
                raise
            delay = backoff_delay(attempt, base_delay)
            logger.warning(
                "%s transient error (attempt %d/%d), retrying in %.1fs: %s",
                label,
                attempt + 1,
                max_attempts,
                delay,
                exc,
            )
            time.sleep(delay)
            continue

        # Success: narrow to bubbles still missing text and retry those.
        targets = [b for b in targets if not (getattr(b, "text", "") or "").strip()]
        if not targets:
            return True
        if attempt < max_attempts - 1:
            delay = backoff_delay(attempt, base_delay)
            logger.info(
                "%s: %d empty bubble(s) (attempt %d/%d), retrying in %.1fs",
                label,
                len(targets),
                attempt + 1,
                max_attempts,
                delay,
            )
            time.sleep(delay)
    return True


def retry_translate(
    translator: Any,
    blk_list: list,
    image: Any,
    extra_context: str,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    base_delay: float = BACKOFF_BASE,
    context_blocks: list | None = None,
    batch_size: int | None = None,
    scene_description: str | None = None,
    scene_block_metadata: dict | None = None,
    label: str = "Translation",
) -> None:
    """Translate ``blk_list`` with up to ``max_attempts`` recovery passes.

    The first pass translates the whole page (with sliding-window
    ``context_blocks``); later passes re-translate only the bubbles left empty,
    which is cheaper and matches the historical behaviour. Transient server
    errors are retried with backoff. Mutates ``blk_list`` in place.

    Raises ``InsufficientCreditsException`` and any non-retryable error after
    all attempts are exhausted so the caller can skip the page. Residual empty
    bubbles after the final attempt are left as-is (the caller reports them).
    """
    pending = list(blk_list)
    for attempt in range(max_attempts):
        use_context = context_blocks if attempt == 0 else None
        try:
            translator.translate(
                pending,
                image,
                extra_context,
                context_blocks=use_context,
                batch_size=batch_size,
                scene_description=scene_description,
                scene_block_metadata=scene_block_metadata,
            )
        except InsufficientCreditsException:
            raise
        except Exception as exc:
            if is_transient_error(exc) and attempt < max_attempts - 1:
                delay = backoff_delay(attempt, base_delay)
                logger.warning(
                    "%s transient error (attempt %d/%d), retrying in %.1fs: %s",
                    label,
                    attempt + 1,
                    max_attempts,
                    delay,
                    exc,
                )
                time.sleep(delay)
                pending = [b for b in pending if not (getattr(b, "translation", "") or "").strip()] or pending
                continue
            logger.exception("%s failed after %d attempts: %s", label, attempt + 1, exc)
            raise

        empty = [b for b in pending if not (getattr(b, "translation", "") or "").strip()]
        if not empty:
            return
        if attempt < max_attempts - 1:
            delay = backoff_delay(attempt, base_delay)
            logger.info(
                "%s: %d empty bubble(s) (attempt %d/%d), retrying in %.1fs",
                label,
                len(empty),
                attempt + 1,
                max_attempts,
                delay,
            )
            time.sleep(delay)
            pending = empty
            continue
        return
