"""
Query pipeline — LangGraph StateGraph (Diagram 2).

Nodes (in order):
  intent_router → [direct_answer | query_expansion]
  query_expansion → [dense_retriever ‖ sparse_retriever]  (parallel)
  dense_retriever + sparse_retriever → rrf_fusion
  rrf_fusion → reranker
  reranker → reference_detector
  reference_detector → [augmentation_fetcher | pass_through]
  augmentation_fetcher + pass_through → context_assembler
  context_assembler → generator
  generator → citation_validator → END
"""
from __future__ import annotations

import re
from typing import TypedDict, Annotated, Literal
from operator import add

from langgraph.graph import StateGraph, END
from openai import OpenAI
from qdrant_client.http import models as qm

from src.app.components.acronym_resolver import get_resolver
from src.app.components.embedder import embed_single
from src.app.components.bm25_index import get_bm25_index
from src.app.database.qdrant_client import get_qdrant_client
from src.app.common.settings import settings
from src.app.common.logger import get_logger

logger = get_logger("components.query_pipeline")


# ── Cross-encoder singleton ───────────────────────────────────────────────────

_cross_encoder = None

def _get_cross_encoder():
    """Load cross-encoder once and reuse across all queries."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder model (first time)...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Cross-encoder loaded.")
    return _cross_encoder


# ── State schema ──────────────────────────────────────────────────────────────

class QueryState(TypedDict):
    query: str
    expanded_query: str
    intent: str                                     # "retrieve" | "direct"
    dense_results: list[dict]
    sparse_results: list[dict]
    fused_results: list[dict]
    reranked_chunks: list[dict]
    augmented_context: list[dict]
    final_context: str
    answer: str
    citations: Annotated[list[dict], add]           # accumulate across nodes
    confidence_score: float
    retry_count: int


# ── Node implementations ──────────────────────────────────────────────────────

def node_intent_router(state: QueryState) -> dict:
    """LangGraph node: classify query as 'retrieve' or 'direct'."""
    client = OpenAI(api_key=settings.openai_api_key)
    prompt = (
        "Does answering the following question accurately require looking up information "
        "in the NASA Systems Engineering Handbook? "
        "Respond with exactly one word: YES or NO.\n\n"
        f"Question: {state['query']}"
    )
    resp = client.chat.completions.create(
        model=settings.intent_model,
        max_tokens=5,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    answer_text = (resp.choices[0].message.content or "YES").strip().upper()
    intent = "retrieve" if "YES" in answer_text else "direct"
    logger.debug("Intent router → %s", intent)
    return {"intent": intent}


def node_direct_answer(state: QueryState) -> dict:
    """LangGraph node: answer without retrieval for off-topic/simple queries."""
    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.generation_model,
        max_tokens=400,
        messages=[
            {"role": "system", "content": "You are a helpful systems engineering assistant."},
            {"role": "user", "content": state["query"]},
        ],
    )
    return {
        "answer": resp.choices[0].message.content or "",
        "citations": [],
        "confidence_score": 1.0,
    }


def node_query_expansion(state: QueryState) -> dict:
    """LangGraph node: expand acronyms and build embedding."""
    resolver = get_resolver()
    expanded = resolver.expand_query(state["query"])
    logger.debug("Query expanded: '%s' → '%s'", state["query"], expanded)
    return {"expanded_query": expanded}


def node_dense_retriever(state: QueryState) -> dict:
    """LangGraph node: Qdrant cosine similarity search."""
    client = get_qdrant_client()
    vector = embed_single(state["expanded_query"])

    response = client.query_points(
        collection_name=settings.collection_text,
        query=vector,
        limit=settings.dense_top_k,
        with_payload=True,
        with_vectors=False,
    )
    dense = [
        {
            "chunk_id": str(r.id),
            "score": r.score,
            "text": r.payload.get("text", ""),
            "payload": r.payload,
            "rank": idx + 1,
            "source": "dense",
        }
        for idx, r in enumerate(response.points)
    ]
    logger.debug("Dense retriever: %d results", len(dense))
    return {"dense_results": dense}


def node_sparse_retriever(state: QueryState) -> dict:
    """LangGraph node: BM25 keyword search."""
    bm25 = get_bm25_index()
    if not bm25.is_ready():
        logger.warning("BM25 index not ready — skipping sparse retrieval")
        return {"sparse_results": []}

    results = bm25.query(state["expanded_query"], top_k=settings.sparse_top_k)
    client = get_qdrant_client()
    enriched: list[dict] = []
    for r in results:
        try:
            points = client.retrieve(
                collection_name=settings.collection_text,
                ids=[r["chunk_id"]],
                with_payload=True,
            )
            payload = points[0].payload if points else {}
        except Exception:
            payload = {"text": r["text"]}
        enriched.append({
            "chunk_id": r["chunk_id"],
            "score": r["score"],
            "text": r["text"],
            "payload": payload,
            "rank": r["rank"],
            "source": "sparse",
        })
    logger.debug("Sparse retriever: %d results", len(enriched))
    return {"sparse_results": enriched}


def node_rrf_fusion(state: QueryState) -> dict:
    """LangGraph node: Reciprocal Rank Fusion of dense + sparse results."""
    k = settings.rrf_k
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}

    for item in state["dense_results"]:
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + item["rank"])
        meta[cid] = item

    for item in state["sparse_results"]:
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + item["rank"])
        if cid not in meta:
            meta[cid] = item

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:settings.dense_top_k]
    fused = [
        {**meta[cid], "rrf_score": score, "rank": idx + 1}
        for idx, (cid, score) in enumerate(ranked)
    ]
    logger.debug("RRF fusion: %d candidates", len(fused))
    return {"fused_results": fused}


def node_reranker(state: QueryState) -> dict:
    """LangGraph node: cross-encoder reranking of top-K fused results."""
    try:
        model = _get_cross_encoder()  # ← reuse singleton, no re-download
        pairs = [(state["query"], item["text"]) for item in state["fused_results"]]
        ce_scores = model.predict(pairs).tolist()

        boosted: list[tuple[float, dict]] = []
        for score, item in zip(ce_scores, state["fused_results"]):
            cross_refs = item.get("payload", {}).get("cross_refs", [])
            boost = 0.05 * len(cross_refs) if cross_refs else 0.0
            boosted.append((score + boost, item))

        boosted.sort(key=lambda x: x[0], reverse=True)
        top = boosted[:settings.rerank_top_k]
        reranked = [{**item, "rerank_score": score} for score, item in top]
        top_score = top[0][0] if top else 0.0

    except Exception as exc:
        logger.warning("Cross-encoder failed (%s), using RRF order", exc)
        reranked = state["fused_results"][:settings.rerank_top_k]
        top_score = reranked[0].get("rrf_score", 0.0) if reranked else 0.0

    logger.debug("Reranker: top-%d, confidence=%.3f", len(reranked), top_score)
    return {"reranked_chunks": reranked, "confidence_score": float(top_score)}


def node_reference_detector(state: QueryState) -> dict:
    """LangGraph node: inspect payloads for image_ref / table_ref / cross_refs."""
    enriched: list[dict] = []
    for chunk in state["reranked_chunks"]:
        payload = chunk.get("payload", {})
        chunk["has_image_ref"] = bool(payload.get("image_ref"))
        chunk["has_table_ref"] = bool(payload.get("table_ref"))
        chunk["has_cross_refs"] = bool(payload.get("cross_refs"))
        enriched.append(chunk)
    return {"reranked_chunks": enriched}


def node_augmentation_fetcher(state: QueryState) -> dict:
    """LangGraph node: fetch supplementary context from image_store / table_store / text_chunks."""
    client = get_qdrant_client()
    augmented: list[dict] = []

    for chunk in state["reranked_chunks"]:
        payload = chunk.get("payload", {})
        extra: list[str] = []

        if chunk.get("has_image_ref") and payload.get("image_ref"):
            try:
                pts = client.retrieve(
                    collection_name=settings.collection_images,
                    ids=[payload["image_ref"]],
                    with_payload=True,
                )
                if pts:
                    extra.append(f"[Diagram context]: {pts[0].payload.get('description', '')}")
            except Exception as exc:
                logger.debug("image_ref fetch failed: %s", exc)

        if chunk.get("has_table_ref") and payload.get("table_ref"):
            try:
                pts = client.retrieve(
                    collection_name=settings.collection_tables,
                    ids=[payload["table_ref"]],
                    with_payload=True,
                )
                if pts:
                    extra.append(f"[Table context]: {pts[0].payload.get('text', '')}")
            except Exception as exc:
                logger.debug("table_ref fetch failed: %s", exc)

        if chunk.get("has_cross_refs"):
            for ref_sec_id in (payload.get("cross_refs") or [])[:2]:
                try:
                    results = client.scroll(
                        collection_name=settings.collection_text,
                        scroll_filter=qm.Filter(
                            must=[qm.FieldCondition(
                                key="section_id",
                                match=qm.MatchValue(value=ref_sec_id),
                            )]
                        ),
                        limit=1,
                        with_payload=True,
                    )
                    hits = results[0]
                    if hits:
                        ref_text = hits[0].payload.get("text", "")
                        extra.append(f"[Cross-ref Section {ref_sec_id}]: {ref_text[:400]}")
                except Exception as exc:
                    logger.debug("cross_ref fetch %s failed: %s", ref_sec_id, exc)

        augmented.append({**chunk, "extra_context": extra})

    logger.debug("Augmentation fetcher: enriched %d chunks", len(augmented))
    return {"augmented_context": augmented}


def node_pass_through(state: QueryState) -> dict:
    """LangGraph node: no augmentation needed, pass chunks through."""
    return {"augmented_context": [{**c, "extra_context": []} for c in state["reranked_chunks"]]}


def node_context_assembler(state: QueryState) -> dict:
    """LangGraph node: build the final prompt context string."""
    parts: list[str] = []
    citations: list[dict] = []

    for idx, chunk in enumerate(state["augmented_context"]):
        payload = chunk.get("payload", {})
        section_id = payload.get("section_id", "?")
        page_num = payload.get("page_num", "?")
        score = chunk.get("rerank_score", chunk.get("rrf_score", 0.0))

        header = f"[Source {idx+1}: Section {section_id}, Page {page_num}, confidence={score:.3f}]"
        body = chunk.get("text", "")
        extra = "\n".join(chunk.get("extra_context", []))
        block = f"{header}\n{body}"
        if extra:
            block += f"\n{extra}"
        parts.append(block)

        citations.append({
            "section_id": section_id,
            "page_num": page_num,
            "confidence": round(score, 3),
            "source_type": payload.get("content_type", "text"),
        })

    return {"final_context": "\n\n---\n\n".join(parts), "citations": citations}


def node_generator(state: QueryState) -> dict:
    """LangGraph node: GPT-4o generation with assembled context."""
    system_prompt = (
        "You are a precise technical QA assistant for the NASA Systems Engineering Handbook. "
        "Answer the question using ONLY the provided source sections. "
        "Always cite your sources using the format: (Section X.X, Page Y). "
        "If the answer is not in the sources, say so explicitly. "
        "Do not speculate beyond what is written."
    )
    user_prompt = (
        f"Context:\n{state['final_context']}\n\n"
        f"Question: {state['query']}\n\n"
        "Answer (cite your sources):"
    )
    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.generation_model,
        max_tokens=800,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = resp.choices[0].message.content or ""
    logger.debug("Generator produced %d chars", len(answer))
    return {"answer": answer}


def node_citation_validator(state: QueryState) -> dict:
    """LangGraph node: flag any citations in the answer not present in context."""
    cited_sections = set(re.findall(r"Section\s+([\d.]+)", state["answer"]))
    context_sections = {c["section_id"] for c in state["citations"]}
    unverified = cited_sections - context_sections
    if unverified:
        note = f"\n\n⚠️ Note: Section(s) {', '.join(sorted(unverified))} were cited but not retrieved. Please verify."
        return {"answer": state["answer"] + note}
    return {}


# ── Routing functions ─────────────────────────────────────────────────────────

def route_intent(state: QueryState) -> Literal["query_expansion", "direct_answer"]:
    return "query_expansion" if state["intent"] == "retrieve" else "direct_answer"


def route_refs(state: QueryState) -> Literal["augmentation_fetcher", "pass_through"]:
    has_any = any(
        c.get("has_image_ref") or c.get("has_table_ref") or c.get("has_cross_refs")
        for c in state["reranked_chunks"]
    )
    return "augmentation_fetcher" if has_any else "pass_through"


# ── Graph construction ────────────────────────────────────────────────────────

def build_query_graph() -> StateGraph:
    graph = StateGraph(QueryState)

    graph.add_node("intent_router", node_intent_router)
    graph.add_node("direct_answer", node_direct_answer)
    graph.add_node("query_expansion", node_query_expansion)
    graph.add_node("dense_retriever", node_dense_retriever)
    graph.add_node("sparse_retriever", node_sparse_retriever)
    graph.add_node("rrf_fusion", node_rrf_fusion)
    graph.add_node("reranker", node_reranker)
    graph.add_node("reference_detector", node_reference_detector)
    graph.add_node("augmentation_fetcher", node_augmentation_fetcher)
    graph.add_node("pass_through", node_pass_through)
    graph.add_node("context_assembler", node_context_assembler)
    graph.add_node("generator", node_generator)
    graph.add_node("citation_validator", node_citation_validator)

    graph.set_entry_point("intent_router")

    graph.add_conditional_edges(
        "intent_router",
        route_intent,
        {"query_expansion": "query_expansion", "direct_answer": "direct_answer"},
    )
    graph.add_edge("direct_answer", END)

    graph.add_edge("query_expansion", "dense_retriever")
    graph.add_edge("query_expansion", "sparse_retriever")

    graph.add_edge("dense_retriever", "rrf_fusion")
    graph.add_edge("sparse_retriever", "rrf_fusion")

    graph.add_edge("rrf_fusion", "reranker")
    graph.add_edge("reranker", "reference_detector")

    graph.add_conditional_edges(
        "reference_detector",
        route_refs,
        {"augmentation_fetcher": "augmentation_fetcher", "pass_through": "pass_through"},
    )

    graph.add_edge("augmentation_fetcher", "context_assembler")
    graph.add_edge("pass_through", "context_assembler")

    graph.add_edge("context_assembler", "generator")
    graph.add_edge("generator", "citation_validator")
    graph.add_edge("citation_validator", END)

    return graph


def get_compiled_graph():
    """Return the compiled (executable) LangGraph."""
    return build_query_graph().compile()


def run_query(query: str) -> dict:
    """Execute the full query pipeline and return the final state."""
    graph = get_compiled_graph()
    initial_state: QueryState = {
        "query": query,
        "expanded_query": "",
        "intent": "",
        "dense_results": [],
        "sparse_results": [],
        "fused_results": [],
        "reranked_chunks": [],
        "augmented_context": [],
        "final_context": "",
        "answer": "",
        "citations": [],
        "confidence_score": 0.0,
        "retry_count": 0,
    }
    final_state = graph.invoke(initial_state)
    return final_state