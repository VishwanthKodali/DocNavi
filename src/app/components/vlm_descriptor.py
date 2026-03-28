"""
VLM descriptor — async GPT-4o vision calls for all image blocks.

Why async:
  - 115 images × ~4s per call = ~8 min sequential
  - With asyncio.gather + semaphore(10): ~1-2 min
  - AsyncOpenAI is the drop-in async client from the openai package

Format handling:
  - GPT-4o accepts: PNG, JPEG, GIF, WEBP only
  - NASA handbook contains TIFF images (Im100.tiff etc.)
  - TIFFs are converted to PNG in-memory via Pillow before upload
  - BadRequestError (400) is non-retryable — skip, never crash the pipeline
"""
from __future__ import annotations

import asyncio
import base64
import io
import uuid
from dataclasses import dataclass

from openai import AsyncOpenAI, BadRequestError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.app.components.pdf_parser import ImageBlock
from src.app.common.settings import settings
from src.app.common.logger import get_logger
from src.app.common.exceptions import VLMError

logger = get_logger("components.vlm_descriptor")

# Max concurrent GPT-4o vision calls — stays within rate limits
VLM_CONCURRENCY = 10

SUPPORTED_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

VLM_SYSTEM_PROMPT = """You are a technical document analyst specialising in systems engineering.
Given an image from the NASA Systems Engineering Handbook (SP-2016-6105 Rev2):

1. Identify the diagram/figure type (Vee model, lifecycle diagram, flowchart, process flow,
   decision tree, block diagram, table, photo, or other)
2. Describe the main concepts, process steps, or data shown
3. Extract all visible text labels, acronyms, and annotations
4. Describe any decision logic, arrows, or flow directions
5. State which handbook sections or topics this likely relates to

Be precise and technical. Output plain prose — no markdown headers or bullet symbols."""


# ── Format helpers ─────────────────────────────────────────────────────────

def _infer_mime(data: bytes) -> str:
    """Infer MIME type from magic bytes — never trust the file extension."""
    if data[:2] == b'\xff\xd8':
        return "image/jpeg"
    if data[:4] == b'\x89PNG':
        return "image/png"
    if data[:4] in (b'GIF8', b'GIF9'):
        return "image/gif"
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return "image/webp"
    if data[:2] in (b'II', b'MM'):      # TIFF little-endian / big-endian
        return "image/tiff"
    return "image/jpeg"                 # safe fallback


def _ensure_supported(data: bytes, mime: str) -> tuple[bytes, str]:
    """
    Convert unsupported formats (TIFF, BMP, etc.) to PNG in-memory.
    Returns (bytes, mime). No-op for already-supported formats.
    """
    if mime in SUPPORTED_MIMES:
        return data, mime
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        converted = buf.getvalue()
        logger.debug("Converted %s → PNG (%d → %d bytes)", mime, len(data), len(converted))
        return converted, "image/png"
    except ImportError:
        raise VLMError("Pillow required to convert TIFF images. Run: pip install Pillow")
    except Exception as exc:
        raise VLMError(f"Image format conversion failed ({mime}): {exc}") from exc


# ── Data model ─────────────────────────────────────────────────────────────

@dataclass
class ImageDescription:
    figure_id: str
    page_num: int
    description: str
    image_name: str = ""
    image_path: str = ""
    section_id: str = ""
    width: float = 0.0
    height: float = 0.0


# ── Async core ─────────────────────────────────────────────────────────────

async def _call_vlm_async(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    b64: str,
    mime: str,
    page_num: int,
    image_name: str,
    attempt: int = 0,
) -> str:
    """
    Single async GPT-4o vision call, guarded by a semaphore.
    Retries up to 3 times on transient errors (rate limit / 5xx).
    Returns description string on success, raises VLMError on failure.
    """
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=settings.vlm_model,
                max_tokens=600,
                messages=[
                    {"role": "system", "content": VLM_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{b64}",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    f"This image is from page {page_num} of the "
                                    "NASA Systems Engineering Handbook. Describe it."
                                ),
                            },
                        ],
                    },
                ],
            )
            return response.choices[0].message.content or ""

        except BadRequestError as exc:
            # 400 = invalid image format — non-retryable
            raise VLMError(
                f"GPT-4o rejected '{image_name}' (400 non-retryable): {exc}"
            ) from exc

        except Exception as exc:
            # Transient error — retry with exponential back-off (max 3 attempts)
            if attempt < 2:
                wait = 2 ** (attempt + 1)   # 2s, 4s
                logger.debug(
                    "VLM transient error for '%s', retry %d in %ds: %s",
                    image_name, attempt + 1, wait, exc,
                )
                await asyncio.sleep(wait)
                return await _call_vlm_async(
                    client, semaphore, b64, mime, page_num, image_name, attempt + 1
                )
            raise VLMError(
                f"GPT-4o failed for '{image_name}' after 3 attempts: {exc}"
            ) from exc


async def _describe_one_async(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    image_block: ImageBlock,
    section_id: str,
    index: int,
    total: int,
) -> ImageDescription | None:
    """
    Describe a single ImageBlock asynchronously.
    Returns None on any failure so the batch never aborts.
    """
    if not image_block.image_bytes or len(image_block.image_bytes) < 500:
        logger.warning(
            "VLM %d/%d — SKIP '%s' page %d: image too small",
            index, total, image_block.name, image_block.page_num,
        )
        return None

    try:
        data = image_block.image_bytes
        mime = _infer_mime(data)
        data, mime = _ensure_supported(data, mime)       # TIFF → PNG if needed
        b64 = base64.b64encode(data).decode()

        description = await _call_vlm_async(
            client, semaphore, b64, mime,
            image_block.page_num, image_block.name,
        )

        fig_id = str(uuid.uuid5(
            uuid.NAMESPACE_DNS,
            f"fig:{image_block.page_num}:{image_block.block_index}:{image_block.name}",
        ))
        logger.info(
            "  VLM %d/%d — page %d '%s' (%d chars)",
            index, total, image_block.page_num, image_block.name, len(description),
        )
        return ImageDescription(
            figure_id=fig_id,
            page_num=image_block.page_num,
            description=description,
            image_name=image_block.name,
            section_id=section_id,
            width=image_block.width,
            height=image_block.height,
        )

    except VLMError as exc:
        logger.warning(
            "  VLM %d/%d — SKIP page %d '%s': %s",
            index, total, image_block.page_num, image_block.name, exc,
        )
        return None

    except Exception as exc:
        logger.warning(
            "  VLM %d/%d — SKIP page %d '%s' (unexpected): %s",
            index, total, image_block.page_num, image_block.name, exc,
        )
        return None


async def _describe_all_async(
    image_blocks: list[ImageBlock],
    section_map: dict[int, str],
) -> list[ImageDescription]:
    """
    Describe all images concurrently, capped at VLM_CONCURRENCY parallel calls.
    Uses asyncio.gather so all tasks launch together and complete as fast as
    the semaphore + API rate limits allow.
    """
    client    = AsyncOpenAI(api_key=settings.openai_api_key)
    semaphore = asyncio.Semaphore(VLM_CONCURRENCY)
    total     = len(image_blocks)

    tasks = [
        _describe_one_async(
            client, semaphore,
            ib,
            section_map.get(ib.page_num, "unknown"),
            idx + 1,
            total,
        )
        for idx, ib in enumerate(image_blocks)
    ]

    raw_results = await asyncio.gather(*tasks, return_exceptions=False)

    # Filter out None (skipped) entries
    results = [r for r in raw_results if r is not None]
    skipped = total - len(results)
    logger.info(
        "VLM async batch complete: %d described, %d skipped out of %d",
        len(results), skipped, total,
    )
    return results


# ── Public API ─────────────────────────────────────────────────────────────

def describe_images_batch(
    image_blocks: list[ImageBlock],
    section_map: dict[int, str],
) -> list[ImageDescription]:
    """
    Synchronous entry point called by ingestion_pipeline.py.
    Runs the async batch inside its own event loop so the caller
    doesn't need to be async itself.
    """
    if not image_blocks:
        return []

    logger.info(
        "Starting async VLM batch: %d images, concurrency=%d",
        len(image_blocks), VLM_CONCURRENCY,
    )

    try:
        # If an event loop is already running (e.g. inside FastAPI's async context),
        # use asyncio.ensure_future / nest_asyncio. For background tasks (our case),
        # there is no running loop so asyncio.run() is safe.
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Running inside an existing loop (e.g. pytest-asyncio, Jupyter)
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(_describe_all_async(image_blocks, section_map))
        else:
            return asyncio.run(_describe_all_async(image_blocks, section_map))
    except Exception as exc:
        logger.exception("Async VLM batch failed entirely: %s", exc)
        return []
