# from fastapi import APIRouter,HTTPException
# from fastapi import Query as QParam
# import fitz

# router = APIRouter()

# @router.get("/debug/parse-page", tags=["diagnostics"])
# def debug_parse_page(
#     pdf_path: str = QParam(..., description="Absolute path to PDF on server"),
#     page_num: int = QParam(1, description="1-indexed page number"),
# ):
#     """
#     Inspect raw block extraction for a single page.
#     Useful for diagnosing why blocks or chunks are 0.
#     """
#     from pathlib import Path as _Path
#     p = _Path(pdf_path)
#     if not p.exists():
#         raise HTTPException(status_code=404, detail=f"File not found: {pdf_path}")

#     doc = fitz.open(str(p))
#     try:
#         page = doc[page_num - 1]
#         raw = page.get_text("dict")
#         blocks = raw.get("blocks", [])
#         summary = []
#         for b in blocks[:20]:   # first 20 blocks
#             btype = b.get("type")
#             if btype == 0:
#                 text_sample = ""
#                 max_fs = 0.0
#                 for line in b.get("lines", []):
#                     for span in line.get("spans", []):
#                         text_sample += span.get("text", "")
#                         fs = span.get("size", 0)
#                         if fs > max_fs:
#                             max_fs = fs
#                 summary.append({
#                     "type": "text",
#                     "font_size": round(max_fs, 1),
#                     "text_preview": text_sample[:120].strip(),
#                     "bbox": [round(x, 1) for x in b["bbox"]],
#                 })
#             elif btype == 1:
#                 summary.append({"type": "image", "bbox": [round(x, 1) for x in b["bbox"]]})
#         plain_text = page.get_text("text")[:500]
#     finally:
#         doc.close()

#     return {
#         "page_num": page_num,
#         "total_blocks_on_page": len(blocks),
#         "blocks_sample": summary,
#         "plain_text_preview": plain_text,
#     }