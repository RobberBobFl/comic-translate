import logging
import time

import numpy as np
import requests
import json

from .base import OCREngine
from ..translation.reasoning import get_reasoning_off_params, is_ollama_endpoint
from ..utils.textblock import TextBlock, adjust_text_line_coordinates

logger = logging.getLogger(__name__)

# Retry config for transient server errors (rate limits, OOM, etc.)
_MAX_RETRIES = 3
_RETRY_BACKOFF = [1.0, 2.0, 4.0]  # seconds between attempts


class CustomOCR(OCREngine):
    """OCR engine using a custom OpenAI-compatible vision API endpoint.

    The user supplies their own API base URL, API key (optional for local
    servers) and model name. Any server that exposes an OpenAI-compatible
    ``/chat/completions`` endpoint with vision support can be used, e.g.
    Ollama, LM Studio, vLLM, or cloud providers.
    """

    DEFAULT_API_URL = "http://localhost:11434/v1"

    def __init__(self):
        self.api_key = ""
        self.model = ""
        self.api_base_url = f"{self.DEFAULT_API_URL}/chat/completions"
        self.expansion_percentage = 0
        self.max_tokens = 5000

    def initialize(
        self,
        api_key: str = "",
        api_url: str = DEFAULT_API_URL,
        model: str = "",
        expansion_percentage: int = 0,
    ) -> None:
        """Initialize the custom OCR engine.

        Args:
            api_key: API key for the endpoint (empty for local servers).
            api_url: Base API URL, e.g. ``http://localhost:11434/v1``.
                ``/chat/completions`` is appended automatically if missing.
            model: Model name as exposed by the endpoint.
            expansion_percentage: Percentage to expand text bounding boxes.
        """
        self.api_key = api_key or ""
        self.model = model or ""
        self.expansion_percentage = expansion_percentage

        base = (api_url or self.DEFAULT_API_URL).rstrip("/")
        if base.endswith("/chat/completions"):
            self.api_base_url = base
        else:
            self.api_base_url = f"{base}/chat/completions"

    def process_image(self, img: np.ndarray, blk_list: list[TextBlock]) -> list[TextBlock]:
        """Process an image with the custom OCR by processing individual text regions."""
        h, w = img.shape[:2]
        for blk in blk_list:
            # Use the text bbox for cropping, not the bubble bbox.
            # bubble_xyxy is for rendering (where to place translated text).
            # When two text regions share one bubble, using bubble_xyxy would
            # give both blocks the same crop and the same OCR text.
            x1, y1, x2, y2 = adjust_text_line_coordinates(
                blk.xyxy,
                self.expansion_percentage,
                self.expansion_percentage,
                img,
            )

            # Bounding boxes may be numpy floats (e.g. from detection / webtoon
            # coordinate maths). Slicing requires integers, so round + clamp to
            # the image bounds like the other OCR engines do.
            x1 = int(round(float(x1)))
            y1 = int(round(float(y1)))
            x2 = int(round(float(x2)))
            y2 = int(round(float(y2)))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x1 < x2 and y1 < y2:
                cropped_img = img[y1:y2, x1:x2]
                img_to_ocr = self.encode_image(cropped_img)
                blk.text = self._get_ocr(img_to_ocr)
                logger.debug("[CustomOCR] crop=(%d,%d,%d,%d) size=%dx%d text=%r",
                             x1, y1, x2, y2, x2-x1, y2-y1, (blk.text or '')[:80])
            else:
                logger.debug("[CustomOCR] invalid crop xyxy=%s -> empty", [x1,y1,x2,y2])

        return blk_list

    def _get_ocr(self, base64_image: str) -> str:
        """Get OCR result from the custom vision model via REST API call.

        Retries up to ``_MAX_RETRIES`` times with exponential back-off on
        transient errors (connection failures, server 4xx/5xx) so that a
        single hiccup does not silently drop a bubble.
        """
        if not self.model:
            raise ValueError("Model not initialized. Call initialize() first.")

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Write out the text in this image. Do NOT Translate. Do not write anything else",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            "max_completion_tokens": self.max_tokens,
        }

        payload.update(
            get_reasoning_off_params(
                self.api_base_url, self.model, is_ollama_endpoint(self.api_base_url)
            )
        )

        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = requests.post(
                    self.api_base_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=60,
                )
                response.raise_for_status()
                response_json = response.json()
                text = response_json["choices"][0]["message"]["content"]
                return text.replace("\n", " ") if "\n" in text else text
            except requests.exceptions.RequestException as e:
                last_err = e
                detail = ""
                if hasattr(e, "response") and e.response is not None:
                    try:
                        error_details = e.response.json()
                        detail = json.dumps(error_details)
                    except Exception:
                        detail = f"Status code: {e.response.status_code}"
                # Non-retryable errors: auth failures, invalid model, etc.
                lowered = (str(e) + detail).lower()
                if any(kw in lowered for kw in ("multimodal", "does not support", "invalid_request_error", "authorization", "invalid api key")):
                    break
                # Retryable: connection errors, 429, 5xx
                if attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                    logger.warning("Custom OCR attempt %d/%d failed (%s), retrying in %.1fs...",
                                   attempt + 1, _MAX_RETRIES, str(e)[:120], wait)
                    time.sleep(wait)

        # All retries exhausted — log and return empty
        error_msg = f"Custom OCR API request failed after {_MAX_RETRIES} attempts: {str(last_err)}"
        detail = ""
        if hasattr(last_err, "response") and last_err.response is not None:
            try:
                error_details = last_err.response.json()
                detail = json.dumps(error_details)
                error_msg += f" - {detail}"
            except Exception:
                error_msg += f" - Status code: {last_err.response.status_code}"
        logger.error(error_msg)
        return ""
