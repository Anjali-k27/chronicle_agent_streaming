"""
╔══════════════════════════════════════════════════════════════════╗
║  CHRONICLE — api.py                                              ║
║  Session 12.2: Streaming Agentic Outputs                         ║
╚══════════════════════════════════════════════════════════════════╝

Changes in Session 12.2 (additions only — nothing removed):
  - POST /analyze/stream: replaces the 501 stub with a live SSE stream —
    StreamingResponse + text/event-stream, disconnect detection, keepalive
  - GET /health: updated to session "12.2", sse_streaming capability added
  - POST /analyze (S12.1 JSON endpoint) preserved unchanged

Changes in Session 12.1 (additions only — nothing removed):
  - asynccontextmanager lifespan: MCP pool built + LangGraph graph
    compiled once at startup, closed gracefully at shutdown
  - AnalysisResponse: response_model for /analyze
  - POST /analyze: now uses graph.ainvoke() instead of
    run_concurrent_analysis()
  - GET /health: updated to session "12.1", MCP connection status added
  - GET /health/live, GET /health/ready: k8s-style health probes
  - GET /calibration-stats: summary of the 30-sample calibration dataset
  - All S11.1/S11.2/S11.3 endpoints preserved unchanged
"""

# ── Imports (Session 12.1) ────────────────────────────────────────
import time
import uuid
import logging
import os
import asyncio   # S12.2: keepalive timeout on the SSE event stream
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import uvicorn

from agent import (
    # S11.1 — preserved
    calculate_chronicle_vram_budget,
    CHRONICLE_AGENTS,
    GPU_VRAM_GB,
    VRAM_BYTES_PER_PARAM,
    KV_CACHE_GB_PER_AGENT_4K,
    CUDA_OVERHEAD_GB,
    # S11.2/S11.3 — preserved
    calculate_tiered_vram_budget,
    calculate_monthly_gpu_cost,
    calculate_max_safe_concurrent,
    task_survivability_matrix,
    oom_prevention_check,
    vllm_config_per_agent,
    colocation_partitioner,
    kv_cache_growth_simulator,
    TASK_SURVIVABILITY_MATRIX,
    CHRONICLE_CALIBRATION_DATASET,
    # S12.1 — preserved
    AnalysisRequest,
    AnalysisResponse,
    build_chronicle_graph,
    build_mcp_client_pool,
    close_mcp_client_pool,
    build_initial_state,
    # S12.2 — new
    chronicle_stream_events,
    NODE_LABELS,
)
from stream_schemas import (
    ErrorStreamEvent,
    to_sse_frame,
    keepalive_frame,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("chronicle.api")


# ── Lifespan (Session 12.1) ───────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    What it does:   Startup: build MCP pool + compile LangGraph graph once.
                    Shutdown: close MCP sessions gracefully.
    Why async:      build_mcp_client_pool() awaits aiohttp session creation.
                    close_mcp_client_pool() awaits session close.
    Why here:       graph.compile() costs real time. Per-request compilation
                    adds that overhead to every Chronicle analysis. Startup-once
                    means zero compilation overhead at request time.
    Introduced:     Session 12.1. Permanent.
    """
    log.info("Chronicle gateway starting — building MCP pool and LangGraph graph...")

    # ── MCP client pool ───────────────────────────────────────────
    # One aiohttp session per Chronicle data source.
    # Reused across all requests. Created once, closed at shutdown.
    app.state.mcp_pool = await build_mcp_client_pool()
    log.info(f"MCP pool ready: {list(app.state.mcp_pool._sessions.keys())}")

    # ── LangGraph graph ───────────────────────────────────────────
    # Compiled once. Reused for every ainvoke() call.
    # The graph itself is stateless — state is passed per-request via ainvoke().
    # No shared mutable state between concurrent Chronicle analyses.
    app.state.graph = build_chronicle_graph(app.state.mcp_pool)
    log.info("LangGraph Chronicle swarm compiled. Gateway ready.")

    app.state.start_time = time.monotonic()

    yield  # Gateway accepts requests from here until shutdown signal

    # ── Shutdown ──────────────────────────────────────────────────
    # In-flight requests complete (not cancelled) before this runs.
    log.info("Chronicle gateway shutting down — closing MCP pool...")
    await close_mcp_client_pool(app.state.mcp_pool)
    log.info("Shutdown complete.")


# ── App Setup (Session 12.1) ──────────────────────────────────────

app = FastAPI(
    title="Chronicle API",
    description=(
        "Local-first personal AI analyst. "
        "Session 12.2 — Streaming Agentic Outputs."
    ),
    version="12.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Session 12.2 update: session → "12.2", sse_streaming and
    mid_stream_mcp_tool_calls capabilities added.
    """
    budget     = calculate_tiered_vram_budget()
    cost       = calculate_monthly_gpu_cost()
    oom        = oom_prevention_check()
    uptime_s   = round(time.monotonic() - getattr(app.state, "start_time", 0))
    mcp_status = (
        app.state.mcp_pool.connection_status()
        if hasattr(app.state, "mcp_pool")
        else {}
    )
    return {
        "status":    "ok",
        "session":   "12.2",
        "version":   "12.2.0",
        "uptime_s":  uptime_s,
        "oom_safe":  oom["all_safe"],
        "agents": {
            name: {
                "role":                   info["role"],
                "tier":                   info["tier"],
                "precision":              info["precision"],
                "gpu_tier":               info["gpu_tier"],
                "max_model_len":          info["max_model_len"],
                "gpu_memory_utilization": info["gpu_memory_utilization"],
            }
            for name, info in CHRONICLE_AGENTS.items()
        },
        "mcp_sources": {
            source: {"connected": connected}
            for source, connected in mcp_status.items()
        },
        "vram_summary": {
            "s11_3_calibrated_gb":    budget["s11_3_calibrated_gb"],
            "vram_saved_vs_s11_1_gb": budget["vram_saved_vs_s11_1_gb"],
            "recommended_gpu":        budget["recommended_gpu"],
        },
        "cost_summary": {
            "recommended_scenario":    cost["recommended_scenario"],
            "monthly_usd":             cost["scenarios"]["D_colocation_l4_a100"]["monthly_usd"],
            "annual_savings_vs_naive": cost["scenarios"]["D_colocation_l4_a100"]["annual_savings_vs_a"],
        },
        "capabilities": [
            "concurrent_5_agent_inference",      # S11.1
            "tiered_quantization_assignments",   # S11.2
            "task_survivability_matrix",         # S11.2
            "oom_prevention_check",              # S11.3
            "colocation_partitioning",           # S11.3
            "vllm_deployment_config",            # S11.3
            "pydantic_request_validation",       # S12.1
            "langgraph_swarm_via_ainvoke",       # S12.1
            "mcp_live_data_ingestion",           # S12.1
            "fastapi_lifespan_graph_compile",    # S12.1
            "sse_streaming",                     # S12.2
            "mid_stream_mcp_tool_calls",         # S12.2
            # "async_job_queue",                 # S12.3 — not yet
        ],
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """
    Session 12.1: Runs Chronicle's full 5-agent LangGraph swarm via ainvoke().

    WHY async def:      This function awaits graph.ainvoke() which makes LLM API calls.
                         Synchronous def would block the event loop for the full analysis
                         duration, preventing all other requests from making progress.

    WHY graph.ainvoke(): Routes through all 5 LangGraph nodes in sequence.
                         Replaces S11's direct Gemini REST calls per-agent.
                         State flows through ingestion → pattern → timeline →
                         brutality → synthesis.

    WHY response_model: Validates LangGraph's returned state against
                        AnalysisResponse before serialising. Catches agent
                        output bugs at the gateway.

    Session 12.2 adds POST /analyze/stream, a live SSE stream. This
    synchronous endpoint is preserved unchanged for clients that just
    want the final JSON.
    """
    if not hasattr(app.state, "graph"):
        raise HTTPException(
            status_code=503,
            detail="Chronicle graph not initialised. Check /health/ready.",
        )

    analysis_id = str(uuid.uuid4())
    wall_start  = time.monotonic()

    try:
        # ── Build initial state ────────────────────────────────────
        # Pydantic validation already passed — request is clean.
        # build_initial_state() maps validated fields to ChronicleState.
        initial_state = build_initial_state(request, analysis_id)

        # ── THE AWAIT POINT ────────────────────────────────────────
        # This coroutine suspends here.
        # Event loop runs other coroutines while Chronicle analyses.
        # When graph completes, coroutine resumes with final_state.
        final_state = await app.state.graph.ainvoke(initial_state)

        processing_ms = round((time.monotonic() - wall_start) * 1000)

        return AnalysisResponse(
            analysis_id=     analysis_id,
            question=        request.question,
            correlations=    final_state.get("correlations", []),
            honest_analysis= final_state.get("honest_analysis", ""),
            final_brief=     final_state.get("final_brief", ""),
            confidence=      float(final_state.get("confidence", 0.75)),
            sources_used=    list(final_state.get("raw_data", {}).keys()),
            sources_live=    final_state.get("sources_live", {}),
            processing_ms=   processing_ms,
            session=         "12.1",
            agent_trace=     final_state.get("agent_trace") if request.debug else None,
        )

    except Exception as e:
        log.exception(f"Analysis {analysis_id} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/stream")
async def analyze_stream(request: Request, body: AnalysisRequest):
    """
    Session 12.2: Live SSE stream of every Chronicle agent event.

    Replaces the 501 stub from Session 12.1.

    WHY POST not GET:
        Chronicle requires a question body.
        Browser EventSource only supports GET.
        We use fetch() with ReadableStream on the client instead.
        curl -N works fine for testing.

    WHY StreamingResponse:
        Keeps the HTTP connection open and writes frames incrementally.
        The response body is never closed until the generator exits.

    HEADERS:
        X-Accel-Buffering: no  → disables nginx proxy buffering
        Cache-Control: no-cache → prevents intermediate caching
        These two headers are MANDATORY. Without them: all events
        arrive simultaneously after the analysis completes (buffered).

    DISCONNECT DETECTION:
        await request.is_disconnected() checked before every yield.
        If True: generator returns → GeneratorExit propagates to
        graph.astream() → Chronicle analysis stops immediately.
        Without this: GPU runs to completion for a client who left.

    KEEPALIVE:
        If no event arrives within 15 seconds (a slow frontier-tier LLM
        call), a comment-line keepalive frame is sent so intermediate
        proxies (nginx, load balancers) don't kill the idle connection.
    """
    if not hasattr(app.state, "graph"):
        raise HTTPException(status_code=503, detail="Graph not ready.")

    analysis_id = str(uuid.uuid4())
    wall_start  = time.time()
    log.info(f"SSE stream started: {analysis_id}")

    async def sse_generator():
        """
        Async generator that drives the SSE response.
        Yields SSE-formatted strings — FastAPI writes each to the client.
        """
        try:
            initial_state = build_initial_state(body, analysis_id)
            event_gen     = chronicle_stream_events(
                app.state.graph, initial_state, analysis_id, wall_start
            ).__aiter__()

            while True:
                # ── Disconnect check BEFORE every yield ───────────
                # If client closed tab: stop immediately.
                # Without this check: GPU runs 60+ more seconds for nobody.
                if await request.is_disconnected():
                    log.info(f"Client disconnected mid-stream: {analysis_id}")
                    return

                # ── Keepalive: don't block forever waiting on a slow node ──
                try:
                    event = await asyncio.wait_for(event_gen.__anext__(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield keepalive_frame()
                    continue
                except StopAsyncIteration:
                    log.info(f"SSE stream complete: {analysis_id}")
                    return

                yield to_sse_frame(event)

                # Stop after final event — do not wait for client to close
                if getattr(event, "final", False):
                    log.info(f"SSE stream complete: {analysis_id}")
                    return

        except (BrokenPipeError, ConnectionResetError):
            # Client transport closed during a write. Normal — not an error.
            log.info(f"Client transport closed: {analysis_id}")
            return

        except Exception as exc:
            log.exception(f"Stream error for {analysis_id}: {exc}")
            err = ErrorStreamEvent(
                seq=999,
                error_code="STREAM_ERROR",
                message="Internal streaming error. Please retry.",
            )
            yield to_sse_frame(err, "error")

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",       # ← disables nginx buffer
            "Transfer-Encoding": "chunked",
        },
    )


# ── S11.1/S11.2/S11.3 endpoints preserved exactly ─────────────────

@app.get("/vram-budget")
async def vram_budget(precision: str = "fp16"):
    """Session 11.1: uniform VRAM budget at a given precision. Unchanged."""
    valid = {"fp32", "fp16", "int8", "int4"}
    if precision not in valid:
        raise HTTPException(status_code=422, detail=f"Invalid precision. Must be one of: {sorted(valid)}")
    return calculate_chronicle_vram_budget(precision)


@app.get("/vram-budget/tiered")
async def vram_budget_tiered():
    """Session 11.2/11.3: per-agent tiered budget. S11.3 uses max_model_len-aware KV calc."""
    return calculate_tiered_vram_budget()


@app.get("/cost-model")
async def cost_model():
    """Session 11.2/11.3: monthly GPU cost. S11.3 adds Scenario D co-location."""
    return calculate_monthly_gpu_cost()


@app.get("/survivability")
async def survivability(task_type: str = None):
    """Session 11.2: task survivability matrix. Unchanged."""
    return task_survivability_matrix(task_type)


@app.get("/calibration-stats")
async def calibration_stats():
    """Session 11.2: calibration dataset summary. Restored with S12.1's full dataset."""
    from collections import Counter
    sources = Counter(s["source"] for s in CHRONICLE_CALIBRATION_DATASET)
    tasks   = Counter(s["expected_task"] for s in CHRONICLE_CALIBRATION_DATASET)
    return {
        "total_samples":  len(CHRONICLE_CALIBRATION_DATASET),
        "sources":        dict(sources),
        "task_types":     dict(tasks),
        "unique_sources": len(sources),
    }


@app.get("/deployment-config")
async def deployment_config():
    """
    Session 11.3: vLLM launch configuration per agent.
    Returns the exact --flags to pass to `vllm serve` for each Chronicle agent.
    Includes model ID, port, tensor_parallel_size, max_model_len,
    gpu_memory_utilization, max_num_seqs, and the full launch command.
    """
    return {
        "agents":     vllm_config_per_agent(),
        "colocation": colocation_partitioner(),
        "session":    "11.3",
        "note": (
            "Model IDs are placeholders. Replace with your AWQ/FP16 "
            "HuggingFace model ID before production deploy. "
            "Gemini API is used for inference until local vLLM is configured."
        ),
    }


@app.get("/oom-check")
async def oom_check():
    """
    Session 11.3: OOM prevention check for all 5 Chronicle agents.
    Returns per-agent max_safe_concurrent calculation and overall pass/fail.
    Use this endpoint in your monitoring stack to verify deployment safety.
    Alert if any agent returns max_safe_concurrent == 0.
    """
    oom   = oom_prevention_check()
    coloc = colocation_partitioner()
    return {
        "oom_prevention": oom,
        "colocation":     coloc,
        "all_safe":       oom["all_safe"] and coloc["safe"],
        "session":        "11.3",
    }


@app.get("/concurrency-table")
async def concurrency_table():
    """
    Session 11.3: max_model_len vs max_concurrent_requests table.
    Shows the concurrency cost of each context window size for each agent's GPU.
    Use this to validate that Chronicle's locked max_model_len values
    provide enough concurrent capacity for expected traffic.
    """
    results     = {}
    mml_options = [1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072]

    for name, info in CHRONICLE_AGENTS.items():
        gpu_vram  = GPU_VRAM_GB.get(info["gpu_tier"], 24)
        bytes_pp  = VRAM_BYTES_PER_PARAM[info["precision"]]
        weight_gb = (info["model_size_b"] * 1e9 * bytes_pp) / (1024 ** 3)

        if info["tier"] == "utility":
            effective_vram = gpu_vram * info["gpu_memory_utilization"]
        else:
            effective_vram = gpu_vram

        table = []
        for mml in mml_options:
            kv_per_req = KV_CACHE_GB_PER_AGENT_4K * (mml / 4_096) * (info["model_size_b"] / 7.0)
            overhead   = CUDA_OVERHEAD_GB
            buffer     = effective_vram * 0.10
            available  = effective_vram - weight_gb - overhead - buffer
            max_conc   = max(0, int(available / kv_per_req)) if kv_per_req > 0 else 0
            table.append({
                "max_model_len":  mml,
                "kv_per_req_gb":  round(kv_per_req, 3),
                "max_concurrent": max_conc,
                "locked":         mml == info["max_model_len"],
            })

        results[name] = {
            "gpu_tier":             info["gpu_tier"],
            "effective_vram_gb":    round(effective_vram, 1),
            "locked_max_model_len": info["max_model_len"],
            "table":                table,
        }

    return {"per_agent": results, "session": "11.3"}


# ── Health probes (Session 12.1) ──────────────────────────────────

@app.get("/health/live")
async def health_live():
    """Kubernetes liveness probe: process is running."""
    return {"status": "alive"}


@app.get("/health/ready")
async def health_ready():
    """Kubernetes readiness probe: graph is compiled and MCP pool is ready."""
    graph_ready = hasattr(app.state, "graph") and app.state.graph is not None
    mcp_ready   = hasattr(app.state, "mcp_pool")
    ready       = graph_ready and mcp_ready
    return JSONResponse(
        content={"status": "ready" if ready else "not_ready",
                 "graph":  graph_ready,
                 "mcp":    mcp_ready},
        status_code=200 if ready else 503,
    )


# ── Server Entry Point ────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Chronicle API — Session 12.2")
    print("  Starting on http://localhost:8000")
    print("  Swagger UI: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
