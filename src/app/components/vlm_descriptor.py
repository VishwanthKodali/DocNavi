"""
VLM descriptor — sends image blocks to GPT-4o vision and returns
structured diagram descriptions for storage in image_store.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.app.components.pdf_parser import ImageBlock
from src.app.common.settings import settings
from src.app.common.logger import get_logger
from src.app.common.exceptions import VLMError

logger = get_logger("components.vlm_descriptor")

VLM_SYSTEM_PROMPT = """You are a technical document analyst specialising in systems engineering diagrams.
Given an image from the NASA Systems Engineering Handbook, provide a structured description.
Include:
1. Diagram type (flowchart / Vee model / lifecycle / decision tree / table / figure / other)
2. Main concepts or process steps shown
3. Decision logic or branching paths (if any)
4. Key labels, acronyms, and text visible in the image
5. Which handbook sections or topics this likely relates to
Be precise and technical. Output plain text, no markdown headers."""


@dataclass
class ImageDescription:
    figure_id: str
    page_num: int
    description: str
    image_path: str = ""   # set later when saved to disk
    section_id: str = ""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def describe_image(image_block: ImageBlock, section_id: str = "") -> ImageDescription:
    """Call GPT-4o vision to describe a diagram. Retries up to 3 times."""
    client = OpenAI(api_key=settings.openai_api_key)
    try:
        response = client.chat.completions.create(
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
                                "url": f"data:image/png;base64,{image_block.image_b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": "Describe this diagram from the NASA handbook."},
                    ],
                },
            ],
        )
        description = response.choices[0].message.content or ""
        fig_id = str(uuid.uuid5(
            uuid.NAMESPACE_DNS,
            f"fig:{image_block.page_num}:{image_block.block_index}",
        ))
        logger.debug("VLM described figure on page %d (%d chars)", image_block.page_num, len(description))
        return ImageDescription(
            figure_id=fig_id,
            page_num=image_block.page_num,
            description=description,
            section_id=section_id,
        )
    except Exception as exc:
        raise VLMError(f"GPT-4o VLM call failed for page {image_block.page_num}: {exc}") from exc


def describe_images_batch(
    image_blocks: list[ImageBlock],
    section_map: dict[int, str],
) -> list[ImageDescription]:
    """Describe all images sequentially (rate-limit safe)."""
    results: list[ImageDescription] = []
    for idx, ib in enumerate(image_blocks):
        sec = section_map.get(ib.page_num, "unknown")
        try:
            desc = describe_image(ib, section_id=sec)
            results.append(desc)
            logger.info("  VLM %d/%d — page %d", idx + 1, len(image_blocks), ib.page_num)
        except VLMError as exc:
            logger.warning("Skipping image on page %d: %s", ib.page_num, exc)
    return results
