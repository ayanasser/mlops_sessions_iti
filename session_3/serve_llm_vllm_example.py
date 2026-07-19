"""
LLM SERVING — FastAPI as an ORCHESTRATION layer in front of vLLM.

Point of this file: compare it to serve_dl_resnet.py and notice what's MISSING.
There is no queue, no Batcher class, no Job/Future plumbing, no batch window.
vLLM's AsyncLLMEngine owns the GPU loop: continuous batching, PagedAttention,
prefix caching, preemption, scheduling. You cannot do that job better from
FastAPI, and trying to (batching prompts yourself) actively HURTS throughput.

What FastAPI is left with, and what it's actually good at:
  retrieval -> prompt assembly -> guardrails -> streaming -> tracing -> fallback

Run:  uvicorn serve_llm_vllm:app --host 0.0.0.0 --port 8000 --workers 1
"""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_MODEL_LEN = 8192


# ---------------------------------------------------------------------------
# SCHEMAS
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None
    max_tokens: int = Field(512, le=2048)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    top_k_docs: int = Field(4, ge=0, le=10)
    stream: bool = False


# ---------------------------------------------------------------------------
# ORCHESTRATION — the part that is genuinely yours
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a support assistant. Answer only from the CONTEXT below. "
    "If the context does not contain the answer, say you don't know. "
    "Cite sources as [1], [2]."
)


async def retrieve(query: str, k: int) -> list[dict]:
    """Stand-in for your vector store / hybrid search. Network-bound, so async."""
    if k == 0:
        return []
    await asyncio.sleep(0.02)
    return [
        {"id": f"doc-{i}", "title": f"Policy section {i}", "text": f"Relevant passage {i} about: {query}"}
        for i in range(1, k + 1)
    ]


def build_prompt(tokenizer, query: str, docs: list[dict]) -> str:
    """Prompt assembly.

    NOTE the ordering: SYSTEM_PROMPT first, then retrieved docs, then the query.
    That's deliberate — vLLM's automatic prefix caching only reuses KV for a
    SHARED PREFIX. Stable text up front = cache hit = lower TTFT. Put the
    user's query first and you throw that away on every request.
    """
    context = "\n\n".join(f"[{i}] {d['title']}\n{d['text']}" for i, d in enumerate(docs, 1))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}" if docs else query},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# LIFESPAN — engine construction is where the real serving config lives
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    engine_args = AsyncEngineArgs(
        model=MODEL,
        max_model_len=MAX_MODEL_LEN,
        dtype="bfloat16",
        # --- KV cache economics: these knobs decide your max concurrency ---
        gpu_memory_utilization=0.90,   # fraction of VRAM for weights + KV blocks
        kv_cache_dtype="fp8",          # halves KV footprint -> ~2x concurrent seqs
        enable_prefix_caching=True,    # reuse KV across shared system prompt / docs
        # --- scheduling ---
        max_num_seqs=256,              # concurrent sequences in the running batch
        max_num_batched_tokens=8192,   # chunked prefill budget per step
        enable_chunked_prefill=True,   # long prefills don't stall other users' decode
        # --- parallelism ---
        tensor_parallel_size=1,
    )
    app.state.engine = AsyncLLMEngine.from_engine_args(engine_args)
    app.state.tokenizer = await app.state.engine.get_tokenizer()
    yield


app = FastAPI(title="RAG endpoint (vLLM)", lifespan=lifespan)


# ---------------------------------------------------------------------------
# NON-STREAMING
# ---------------------------------------------------------------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    if req.stream:
        return await chat_stream(req)

    request_id = f"req-{uuid.uuid4().hex[:12]}"
    t_start = time.perf_counter()

    docs = await retrieve(req.query, req.top_k_docs)
    prompt = build_prompt(app.state.tokenizer, req.query, docs)

    params = SamplingParams(
        temperature=req.temperature,
        top_p=0.9,
        max_tokens=req.max_tokens,
        stop=["</s>"],
    )

    # generate() is an ASYNC GENERATOR yielding cumulative snapshots.
    # We just drain it. Meanwhile the engine is interleaving this sequence
    # with every other in-flight request, token by token — continuous batching.
    final = None
    ttft = None
    async for out in app.state.engine.generate(prompt, params, request_id):
        if ttft is None and out.outputs[0].token_ids:
            ttft = time.perf_counter() - t_start
        final = out

    if final is None:
        raise HTTPException(status_code=500, detail="generation failed")

    completion = final.outputs[0]
    total = time.perf_counter() - t_start
    n_out = len(completion.token_ids)

    return {
        "request_id": request_id,
        "answer": completion.text,
        "finish_reason": completion.finish_reason,
        "sources": [{"id": d["id"], "title": d["title"]} for d in docs],
        # LLM metrics: TTFT and TPOT, not a single "latency_ms"
        "usage": {
            "prompt_tokens": len(final.prompt_token_ids),
            "completion_tokens": n_out,
            "cached_prompt_tokens": final.num_cached_tokens,  # prefix cache hits
        },
        "timings_ms": {
            "ttft": round((ttft or total) * 1000, 1),
            "total": round(total * 1000, 1),
            "tpot": round((total - (ttft or 0)) / max(n_out - 1, 1) * 1000, 2),
        },
    }


# ---------------------------------------------------------------------------
# STREAMING — the reason TTFT is a first-class metric
# ---------------------------------------------------------------------------
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    request_id = f"req-{uuid.uuid4().hex[:12]}"

    async def sse():
        t_start = time.perf_counter()
        docs = await retrieve(req.query, req.top_k_docs)
        yield f"data: {json.dumps({'type': 'sources', 'sources': [d['id'] for d in docs]})}\n\n"

        prompt = build_prompt(app.state.tokenizer, req.query, docs)
        params = SamplingParams(
            temperature=req.temperature, top_p=0.9, max_tokens=req.max_tokens
        )

        emitted = 0
        try:
            async for out in app.state.engine.generate(prompt, params, request_id):
                text = out.outputs[0].text
                delta, emitted = text[emitted:], len(text)
                if delta:
                    yield f"data: {json.dumps({'type': 'token', 'text': delta})}\n\n"
        except asyncio.CancelledError:
            # client hung up -> free the KV blocks immediately, don't burn GPU
            await app.state.engine.abort(request_id)
            raise

        elapsed = round((time.perf_counter() - t_start) * 1000, 1)
        yield f"data: {json.dumps({'type': 'done', 'total_ms': elapsed})}\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/healthz")
async def healthz():
    await app.state.engine.check_health()
    return {"status": "ok", "model": MODEL}