# Chronicle — Personal AI Analyst

Chronicle connects to your real data — Spotify, GitHub, finances, fitness records, journal entries — and tells you what it says about you that you haven't admitted yet. Five specialised AI agents, each with a locked inference tier, deployment configuration, and OOM safety check, run as a compiled LangGraph swarm behind a validated FastAPI gateway. As of Session 12.2, every agent handoff, MCP tool call, and synthesised word streams to the browser live over Server-Sent Events instead of arriving as one blocking JSON response.

This is a multi-session build. Each session extends the previous one without removing anything.

---

## Quick Start

**Every command below is run from inside the `chronicle/` directory** — that's where `agent.py`, `api.py`, `requirements.txt`, and `.env` all live:

```bash
cd chronicle
```

**You need one thing before anything else: a Gemini API key.**

Get one free at [aistudio.google.com](https://aistudio.google.com) → "Get API key" → Create. It's free with generous limits.

A `.env.example` template is already there. Copy it to `.env` and drop in your real key:

```bash
cp .env.example .env
```

```
GEMINI_API_KEY=your_actual_key_here
```

`.env` is listed in `.gitignore` — it never gets committed, even by accident. Only `.env.example` (with the placeholder, no real key) is meant to be checked in.

That's the only external step. Everything else is handled below.

---

## Option A — Local Setup (Python)

**Requirements:** Python 3.11 or later. Check with `python3 --version`.

### Step 1 — Create a virtual environment

```bash
python3 -m venv .venv
```

### Step 2 — Activate it

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Your prompt will now show `(.venv)`.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs: FastAPI, uvicorn, aiohttp, pydantic, google-generativeai, langchain-google-genai, langgraph, python-dotenv, certifi.

### Step 4 — Confirm your API key is set

Already done if you followed **Quick Start** above. If not, copy the template and fill it in:

```bash
cp .env.example .env
```

```
GEMINI_API_KEY=your_actual_key_here
```

### Step 5 — Run the verification

```bash
python agent.py
```

Expected output:

```
╔══════════════════════════════════════════════════════╗
║  Chronicle — Session 12.2 Verification               ║
╚══════════════════════════════════════════════════════╝

  Verification: 5/5 checks passed in ~16000ms

  ✓ First event is StreamStartEvent
  ✓ 5 AgentHandoffEvents emitted (one per node)
  ✓ At least 1 ToolCallEvent + 1 ToolResultEvent emitted
  ✓ Last event is FinalAnswerEvent with final=True and non-empty brief
  ✓ to_sse_frame() produces correct SSE wire format

  ✓ Session 12.2 COMPLETE. Start the API: python api.py
  Test the stream: curl -N -X POST http://localhost:8000/analyze/stream \
    -H 'Content-Type: application/json' \
    -d '{"question": "What does my data say about me?"}'
```

If all 5 checks pass, proceed. (This check runs the full LangGraph swarm and makes real Gemini API calls, so it takes ~15-20 seconds — that's expected, not a hang.)

### Step 6 — Start the server

```bash
python api.py
```

### Step 7 — Open the UI

Go to: **http://localhost:8000**

The dashboard, agent cards, and chat interface will load. Type a question and click Analyse.

---

## Option B — Docker Setup

**Requirements:** Docker Desktop installed and running. Check with `docker --version`.

### Step 1 — Confirm your API key is set

Already done if you followed **Quick Start** above. If not, copy the template and fill it in:

```bash
cp .env.example .env
```

```
GEMINI_API_KEY=your_actual_key_here
```

### Step 2 — Build and start

```bash
docker compose up --build
```

Docker will pull the Python base image, install all dependencies, and start the server. First build takes ~60 seconds. Subsequent starts take ~3 seconds.

### Step 3 — Open the UI

Go to: **http://localhost:8000**

To stop:

```bash
docker compose down
```

To rebuild after code changes:

```bash
docker compose up --build
```

---

## Verifying Everything Works

Once the server is running, you can check each endpoint directly:

| URL | What it returns |
|-----|-----------------|
| `http://localhost:8000` | The Chronicle UI |
| `http://localhost:8000/health` | Session version, OOM status, MCP connector status, all agent configs |
| `http://localhost:8000/health/ready` | 200 once the LangGraph graph is compiled and the MCP pool is connected, else 503 |
| `http://localhost:8000/docs` | Swagger UI — interactive docs for all endpoints |
| `http://localhost:8000/vram-budget/tiered` | Per-agent VRAM breakdown across S11.1/11.2/11.3 |
| `http://localhost:8000/oom-check` | OOM prevention pass/fail per agent |
| `http://localhost:8000/deployment-config` | Full `vllm serve` launch command per agent |
| `http://localhost:8000/cost-model` | 4 GPU cost scenarios with annual savings |
| `http://localhost:8000/concurrency-table` | How context window size affects concurrent capacity |
| `http://localhost:8000/survivability` | Which tasks survive INT4 quantization |
| `http://localhost:8000/calibration-stats` | 30-sample calibration dataset summary across 5 sources |

---

## What Was Built — Session by Session

### Session 11.1 — Inference Foundation

**Goal:** Get all 5 Chronicle agents firing concurrently against a real AI API and measure the performance baseline.

**What was built:**

- `CHRONICLE_AGENTS` — the 5 permanent agents defined with their roles and tiers:
  - `ingestion` — parses and normalises raw data from all sources
  - `pattern` — finds cross-source correlations
  - `timeline` — sequences life events chronologically
  - `brutality` — delivers honest analysis without softening
  - `synthesis` — produces the final structured analyst brief

- `calculate_chronicle_vram_budget()` — calculates total VRAM needed for all 5 agents at a given precision (FP16, INT4, etc.). Establishes the S11.1 baseline: **90 GB** at uniform FP16.

- `chronicle_infer()` — fires a single async inference request against the Gemini REST API and measures Time to First Token (TTFT) and Time Per Output Token (TPOT).

- `run_concurrent_analysis()` — dispatches all 5 agents simultaneously using `asyncio` + `aiohttp`. All agents fire at the same moment. Wall clock time reflects true concurrent load.

- `BenchmarkResult` / `AnalysisRequest` — Pydantic schemas that remain permanent through all sessions.

- **API endpoints added:** `GET /health`, `POST /analyze`, `GET /vram-budget`

- **Dashboard:** Split layout with agent status card, inference metrics card, and VRAM budget card with precision selector.

**Key result:** 5 agents fire concurrently in a single wall-clock window. TTFT measured across all agents.

---

### Session 11.2 — Model Quantization

**Goal:** Assign the right precision to each agent based on whether its task survives quantization. Not every agent needs full FP16.

**What was built:**

- `CHRONICLE_AGENTS` extended with per-agent fields:
  - `precision` — `int4` for utility agents, `fp16` for frontier agents
  - `model_size_b` — 7B for utility, 13B for frontier
  - `gpu_tier` — `L4` for utility, `A100-40` for frontier
  - `monthly_gpu_cost_usd` — $450 (L4), $1,500 (A100-40)
  - `survivability_note` — why this precision is safe for this task

- `TASK_SURVIVABILITY_MATRIX` — 11 task types tested at INT4. Results:
  - **Survives INT4 (≥90% retention):** intent classification, NER, sentiment, summarisation, data parsing, temporal sequencing, cross-source correlation
  - **Requires FP16 (<90% retention):** structured generation, long-context coherence, multi-constraint reasoning, code generation

- `calculate_tiered_vram_budget()` — replaces the uniform budget with per-agent precision. Reduced from 90 GB to ~84 GB.

- `calculate_monthly_gpu_cost()` — 3 GPU deployment scenarios:
  - **Scenario A:** All A100-80, no tiering → $9,375/mo
  - **Scenario B:** 3× L4 (utility) + 2× A100-40 (frontier) → $4,350/mo, saves $60,300/yr
  - **Scenario C:** 3× A10G + 2× A100-40 → $4,650/mo

- `task_survivability_matrix()` — queryable by task type.

- `chronicle_infer()` updated with tier-aware prompts: utility agents get structured 2-sentence prompts, frontier agents get full analytical prompts.

- **API endpoints added:** `GET /vram-budget/tiered`, `GET /cost-model`, `GET /survivability`, `GET /calibration-stats`

- **Dashboard:** Precision badges on agent cards (INT4 green, FP16 purple), tiered VRAM card, cost model card with 3 scenarios.

**Key result:** VRAM dropped from 90 GB to ~84 GB. Monthly GPU cost halved vs naive all-A100 setup.

---

### Session 11.3 — GPU Resource Allocation

**Goal:** Lock the exact deployment configuration that prevents Chronicle from crashing at 2 AM. Every number calculated here goes into the actual `vllm serve` command.

**What was built:**

- `CHRONICLE_AGENTS` extended with:
  - `max_model_len` — 4,096 for utility agents, 8,192 for frontier agents. Without this lock, Llama-3 defaults to 128K context, consuming 64 GB KV cache per agent.
  - `gpu_memory_utilization` — 0.28 for utility (co-located on shared L4), 0.85 for frontier (dedicated A100-40 with 15% safety buffer)

- `GPU_VRAM_GB` — reference dict for all 6 GPU tiers (T4→H100-80).

- `calculate_max_safe_concurrent()` — the OOM prevention formula:
  ```
  Max Safe Concurrent = (Effective VRAM - Weights - Overhead - Buffer) / KV_per_request
  ```
  Results: utility agents handle 1 concurrent request each on their L4 partition. Frontier agents handle 5 concurrent requests each on their A100-40.

- `oom_prevention_check()` — runs the formula for all 5 agents at startup. If any agent returns 0 concurrent slots, Chronicle refuses to start. The crash is caught at deploy time, not at 2 AM.

- `vllm_config_per_agent()` — generates the exact `vllm serve` command for each agent, including `--max-model-len`, `--gpu-memory-utilization`, `--max-num-seqs`, `--tensor-parallel-size`, and port assignments (8100–8104).

- `colocation_partitioner()` — validates the 3 utility agents fit on one shared L4:
  - 3 × 0.28 = 0.84 model fraction + 0.08 system overhead = **0.92 total** (safe, ≤ 1.0)
  - Remaining 1.9 GB headroom

- `kv_cache_growth_simulator()` — simulates KV cache VRAM growth under a given requests-per-minute rate. Shows the exact minute OOM would occur without the concurrent request guard.

- `calculate_tiered_vram_budget()` updated — KV cache now calibrated to per-agent `max_model_len`. Utility agents locked at 4K (2.0 GB KV each) instead of the conservative 8K estimate from S11.2, saving 6 GB total.

- `calculate_monthly_gpu_cost()` updated — **Scenario D added (co-location):**
  - 1× L4 shared by 3 utility agents + 2× A100-40 for frontier → **$3,450/mo**
  - Saves $10,800/yr vs S11.2's separate-GPU approach
  - Saves $71,100/yr vs naive all-A100 setup

- `chronicle_infer()` updated — input length guard added. Requests longer than the agent's `max_model_len` are rejected before dispatch with a clear error message.

- **API endpoints added:** `GET /deployment-config`, `GET /oom-check`, `GET /concurrency-table`

- **Dashboard:** Deployment config card (per-agent mml / util / concurrent slots), OOM safety card (✓ ALL AGENTS SAFE), `mml:` badge on agent cards.

**Session 11.3 verification — 5/5 checks:**
1. All 5 agents have `max_model_len` and `gpu_memory_utilization` set
2. OOM prevention passes: all agents have `max_safe_concurrent > 0`
3. S11.3 calibrated VRAM (78.2 GB) < S11.2 conservative estimate (84.2 GB) — saves 6 GB
4. Co-location partition valid: grand total 0.92 ≤ 1.0
5. Scenario D ($3,450/mo) < Scenario B ($4,350/mo) — co-location wins

**VRAM journey across Week 11:**
```
S11.1 uniform FP16 (no tiering):       90.0 GB
S11.2 tiered precision (8K budget):    84.2 GB   saved  5.8 GB
S11.3 calibrated max_model_len:        78.2 GB   saved 11.8 GB total
```

---

### Session 12.1 — FastAPI Gateway + MCP Ingestion

**Goal:** Put a real HTTP front door in front of Chronicle. Replace the direct Gemini REST calls with a compiled LangGraph swarm, validate every request before the graph boots, and pull data through an MCP client pool instead of hardcoded prompts.

**What was built:**

- `MCP_SOURCE_CONFIG` — maps each of the 5 Chronicle data sources to an MCP server URL (`localhost:3001`–`3005`) and tool name.

- `MCPClientPool` — manages one `aiohttp` session per data source. `fetch_source()` calls the MCP server and falls back to `CHRONICLE_CALIBRATION_DATASET` (restored to its full 30 samples in this session — S11.3 had shipped it as an empty stub) when the server is unreachable. Every fetch reports a `live` flag so downstream code always knows whether it got real or fallback data. **No MCP servers actually exist yet in this exercise** — every source currently resolves via the calibration fallback, which is expected.

- `ChronicleState` — a LangGraph `TypedDict` shared across all 5 agent nodes, threading `raw_data`, `sources_live`, `correlations`, `timeline_events`, `honest_analysis`, `final_brief`, `confidence`, and a debug `agent_trace`.

- Five LangGraph node functions (`ingestion_node`, `pattern_node`, `timeline_node`, `brutality_node`, `synthesis_node`) — utility-tier nodes use a cheap/fast `ChatGoogleGenerativeAI` instance, frontier-tier nodes use a slower/higher-quality one, mirroring the S11.2 precision tiers.

- `build_chronicle_graph()` — compiles a linear `StateGraph`: `ingestion → pattern → timeline → brutality → synthesis → END`. Compiled **once** at FastAPI startup via `lifespan`, not per-request.

- `AnalysisRequest` replaced with a `Field`-validated Pydantic model: `question` (1–2000 chars), `data_sources` restricted to a `Literal` of the 5 known sources, `depth` restricted to `quick`/`standard`/`deep`. Invalid requests get a 422 in under a millisecond, before any Gemini call or graph work happens.

- `AnalysisResponse` — the new output contract: `correlations`, `honest_analysis`, `final_brief`, `confidence` (bounded 0–1), `sources_used`, `sources_live`, `processing_ms`, optional `agent_trace`.

- **The 3-level async chain is now real and verified end-to-end:** `POST /analyze` → `await graph.ainvoke()` → `await llm.ainvoke()` inside each node.

- **API endpoints added:** `GET /analyze/stream` (501 stub — real SSE lands in S12.2), `GET /health/live`, `GET /health/ready`. `/health` now reports live MCP connector status per source. `/calibration-stats` restored.

- **Dashboard:** MCP Data Connectors card (live/fallback badge per source), Gateway Status card (session, version, uptime, graph-compiled indicator).

**Session 12.1 verification — 5/5 checks (makes real Gemini calls, ~10-15s):**
1. `build_chronicle_graph()` compiles without error
2. Graph has all 5 Chronicle agent nodes
3. `MCPClientPool` instantiates correctly
4. Empty question correctly rejected by `AnalysisRequest` (`min_length=1`)
5. `graph.ainvoke()` returns a non-empty `final_brief` end-to-end

---

### Session 12.2 — Streaming Agentic Outputs (Current)

**Goal:** Stop making the user stare at a blank screen for 60–90 seconds. Every agent handoff, every MCP tool call, every synthesised word should reach the browser the moment it happens, not after the whole graph finishes.

**What was built:**

- `stream_schemas.py` (new file) — the SSE event contract, treated as a versioned public API. 7 Pydantic event types, all extending `BaseStreamEvent` (`event_type`, `seq`, `ts_ms`, `session`):
  - `StreamStartEvent` — fired before any graph work begins, so the client gets *something* within ~100ms
  - `AgentHandoffEvent` — fired every time a new LangGraph node becomes active
  - `ToolCallEvent` / `ToolResultEvent` — fired when Pattern or Brutality agents invoke MCP mid-reasoning
  - `TokenChunkEvent` — one per word as the Synthesis Agent's brief is produced
  - `FinalAnswerEvent` — the terminal event, `final=True`, carries the full structured result
  - `ErrorStreamEvent` — terminal error event if anything breaks mid-stream
  - `to_sse_frame()` — serialises an event to the wire format (`event: <type>\ndata: <json>\n\n` — the double newline is mandatory or browsers buffer forever)
  - `keepalive_frame()` — a `: keepalive <ts>` comment line so proxies don't kill an idle connection during a slow LLM call

- **Mid-stream MCP calls** — Pattern Agent and Brutality Agent no longer just read pre-ingested data. Pattern Agent calls `mcp_pool.fetch_source("spotify", ...)` mid-reasoning to verify its correlation with fresh data; Brutality Agent calls `mcp_pool.fetch_source("github", ...)` to back its honest analysis with live commit evidence. Both emit `tool_calls` / `tool_results` into `ChronicleState`, which the stream adapter turns into `tool_call` / `tool_result` events.

- `ChronicleState` extended with `tool_calls`, `tool_results` (accumulator fields, `operator.add`) and `token_chunk` (Synthesis Agent's output, picked up per-node by the stream adapter).

- `chronicle_stream_events()` — an async generator in `agent.py` that drives `graph.astream(state, stream_mode="updates")` and translates raw LangGraph chunks into the typed events above. This is the **only** place graph node names are known — `NODE_LABELS` / `AGENT_NODES` map `"pattern"` → `"Pattern Agent"` etc., so renaming a node later only touches this one file. It also tracks `correlations` / `honest_analysis` / `raw_data` locally as chunks arrive, since `graph.astream()` never mutates the state dict you passed in — reading those fields back off the original `initial_state` at the end would silently return empty values.

- `POST /analyze/stream` — replaces the Session 12.1 `501` stub. Returns a `StreamingResponse` with `media_type="text/event-stream"` and the two headers that actually matter: `Cache-Control: no-cache` and `X-Accel-Buffering: no` (without the second one, nginx buffers the whole response and every event arrives at once, right after the analysis finishes — silently defeating the entire point). Checks `await request.is_disconnected()` before every event so a closed browser tab stops the Chronicle swarm immediately instead of burning GPU time for nobody. Falls back to a 15-second keepalive comment frame if a frontier-tier LLM call runs long.

- `POST /analyze` (the Session 12.1 synchronous JSON endpoint) is untouched — both endpoints coexist for clients that just want a plain JSON response.

- **Dashboard:** new **Live Event Timeline** card — a scrolling, icon-coded log of every SSE event as it arrives (🔵 stream start, ⚡ agent handoff, 📡 tool call, ✓/↩/✗ tool result, ✅ final answer). The chat panel's `sendQuestion()` now opens the stream with `fetch()` + `ReadableStream` (browsers' built-in `EventSource` only supports `GET`, and Chronicle needs a `POST` body), splits incoming bytes on the double-newline SSE frame boundary, and appends `token_chunk` events to the analyst's reply word by word instead of dumping the whole brief at once. Agent Status badges flip `idle → running → done` live, driven by `agent_handoff` events rather than a single before/after toggle.

**Session 12.2 verification — 5/5 checks (drives the full LangGraph swarm via `astream()`, ~15-20s):**
1. First event emitted is `StreamStartEvent`
2. Exactly 5 `AgentHandoffEvent`s fire, one per Chronicle node
3. At least 1 `ToolCallEvent` + 1 `ToolResultEvent` fire (Pattern's Spotify check, Brutality's GitHub check)
4. Last event is `FinalAnswerEvent` with `final=True` and a non-empty `final_brief`
5. `to_sse_frame()` produces a wire-correct frame (starts with `event:`, ends with a double newline)

**Verified live in a browser** (Playwright, headless Chromium): the Live Event Timeline populates in real time, Agent Status badges transition correctly, the synthesis reply visibly streams in word by word, and the final message carries a `confidence / processing_ms / sources` footnote — with zero console errors.

---

## What's Coming — Upcoming Sessions

### Session 12.3 — Async Job Queue

Deep analyses that take longer than 30 seconds get queued properly.

- `POST /analyze` returns `202 Accepted` with a job ID immediately
- `GET /jobs/{id}` polls for result
- Background worker processes the queue
- No more HTTP timeouts on long analyses

### Session 13.1 — OpenTelemetry Tracing

Every agent request becomes a traceable span.

- OTel instrumentation on all 5 agents
- Distributed trace per analysis: one root span, 5 child spans (one per agent)
- Trace viewer card in the dashboard showing per-agent latency breakdown
- Export to any OTel-compatible backend (Jaeger, Grafana Tempo, etc.)

### Session 14.1 — Semantic Caching

Reduce inference cost by catching semantically similar questions.

- Embedding-based cache: if a new question is >90% similar to a cached one, return the cached result
- Cache hit rate tracked per agent
- Reduces effective GPU-hours by 30–60% in practice

### Session 14.2 — Per-Agent Spend Ledger

Know exactly what each agent costs per question, per day, per month.

- Token counting per agent per request
- Cost attribution: $X per question broken down by agent
- Monthly spend projection card in the dashboard
- Alert threshold: flag when spend exceeds a per-agent daily budget

---

## Project Structure

```
chronicle/
├── agent.py           # Inference core: LangGraph swarm, stream adapter, MCP pool, VRAM, OOM, vLLM config, cost model
├── api.py             # FastAPI server: lifespan + all HTTP endpoints (JSON + SSE)
├── stream_schemas.py  # SSE event contract — 7 Pydantic event types (Session 12.2)
├── index.html         # Dashboard UI: streaming chat + live metrics cards
├── requirements.txt   # Python dependencies
├── .env               # API key (never commit this — see .gitignore)
├── Dockerfile         # Container build
└── docker-compose.yml # Multi-service orchestration
```

`agent.py` is the source of truth. Every number in `api.py` and `index.html` comes from functions defined there. `stream_schemas.py` is the public wire contract between server and client — field renames there are breaking changes. Sessions extend these files — nothing is ever removed.

---

## Security — Protecting Your API Key

Chronicle's only secret is `GEMINI_API_KEY`, and it's protected at two levels:

- **`.env` is gitignored** — both `.gitignore` (repo root) and `chronicle/.gitignore` exclude `.env`, `.env.*`, `*.pem`, `*.key`, and anything matching `*credentials*.json` or `secrets.*`. Running `git status` should never show `.env` as a trackable file.
- **`.env.example` is the only file meant to be committed** — it ships with a placeholder (`your_actual_key_here`), never a real key. New setups start with `cp .env.example .env` and fill in the real value locally.

If you ever suspect a real key was committed to git history, don't just delete the file in a new commit — the key is still readable in history. Rotate the key at [aistudio.google.com](https://aistudio.google.com) and update your local `.env`; only then worry about scrubbing history (`git filter-repo` or BFG Repo-Cleaner).

---

## Endpoints Reference

| Method | Path | Session | Description |
|--------|------|---------|-------------|
| `GET` | `/` | 11.1 | Chronicle UI |
| `GET` | `/health` | 11.1, updated 12.1, 12.2 | Version, OOM status, MCP connector status, agent configs, capability flags |
| `GET` | `/health/live` | 12.1 | Liveness probe — process is running |
| `GET` | `/health/ready` | 12.1 | Readiness probe — 200 once graph + MCP pool are ready, else 503 |
| `POST` | `/analyze` | 11.1, replaced 12.1 | Runs the compiled LangGraph swarm via `graph.ainvoke()`, returns one JSON response |
| `POST` | `/analyze/stream` | 12.1 stub, live in 12.2 | Live SSE stream — `stream_start` → `agent_handoff` → `tool_call`/`tool_result` → `token_chunk` → `final_answer`. Test with `curl -N` |
| `GET` | `/vram-budget` | 11.1 | Uniform VRAM at a given precision |
| `GET` | `/vram-budget/tiered` | 11.2 | Per-agent tiered VRAM breakdown |
| `GET` | `/cost-model` | 11.2 | 4 GPU deployment cost scenarios |
| `GET` | `/survivability` | 11.2 | INT4 task survivability matrix |
| `GET` | `/calibration-stats` | 11.2, restored 12.1 | 30-sample calibration dataset summary |
| `GET` | `/deployment-config` | 11.3 | vLLM launch commands per agent |
| `GET` | `/oom-check` | 11.3 | OOM prevention check per agent |
| `GET` | `/concurrency-table` | 11.3 | Context window vs concurrent capacity |

---

## Troubleshooting

**`GEMINI_API_KEY environment variable is not set`**
Open `.env` and make sure the key is set with no quotes and no spaces around `=`:
```
GEMINI_API_KEY=AIza...your_key_here
```

**`address already in use` on port 8000**
Something is already running on port 8000. Kill it:
```bash
# macOS / Linux
lsof -ti :8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```
Then restart with `python api.py`.

**SSL certificate error on macOS**
This is handled automatically via `certifi`. If it still appears, run:
```bash
/Applications/Python\ 3.x/Install\ Certificates.command
```
Replace `3.x` with your Python version.

**Dashboard cards show "API offline"**
The UI is running but can't reach the API. Make sure `python api.py` (or `docker compose up`) is running, then refresh the page.

**MCP Connectors card shows "fallback" for every source**
This is expected in Session 12.1 — no MCP servers actually exist yet at `localhost:3001`–`3005`. `MCPClientPool.fetch_source()` tries each connection, fails, and falls back to `CHRONICLE_CALIBRATION_DATASET`. The client pool, live/fallback flag, and graceful degradation are all real and working; only the servers on the other end are stubs. Standing up real MCP servers for these 5 sources is out of scope for this session.

**Docker: `Cannot connect to the Docker daemon`**
Docker Desktop is not running. Open Docker Desktop from your Applications folder and wait for it to start (the whale icon in the menu bar stops animating when ready), then re-run `docker compose up --build`.

**`/analyze/stream` events all arrive at once instead of streaming**
This means something between the server and your client is buffering the response. If you're behind nginx or another reverse proxy, make sure it isn't stripping the `X-Accel-Buffering: no` header. If you're testing with `curl`, make sure you pass `-N` (`--no-buffer`) — without it, curl itself buffers the output:
```bash
curl -N -X POST http://localhost:8000/analyze/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What does my data say about me?"}'
```

**`ModuleNotFoundError: No module named 'langgraph'`**
Your virtual environment was created before `langgraph` was added to `requirements.txt`. Re-run `pip install -r requirements.txt` inside the activated `.venv`.
