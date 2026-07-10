"""
TamAGI — Main Application Entry Point

Initializes all subsystems and starts the FastAPI server.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from backend.auth import load_session_secret
from backend.config import get_config, load_config, set_config
from backend.core.llm import LLMClient
from backend.core.memory import create_memory_store
from backend.core.personality import PersonalityEngine
from backend.core.agent import TamAGIAgent
from backend.core.identity import IdentityManager
from backend.core.world_thread import WorldThread
from backend.core.self_model import SelfModel, seed_self_model
from backend.core.planning_engine import PlanningEngine
from backend.core.reflection import ReflectionEngine
from backend.skills.registry import SkillRegistry
from backend.skills.read_skill import ReadSkill
from backend.skills.write_skill import WriteSkill
from backend.skills.exec_skill import ExecSkill
from backend.skills.web_search_skill import WebSearchSkill
from backend.skills.express_skill import ExpressSkill
from backend.api.chat import router as chat_router, set_agent
from backend.api.skills import router as skills_router
from backend.api.onboarding import router as onboarding_router
from backend.skills.recall_memory_skill import RecallMemorySkill
from backend.skills.read_world_graph_skill import ReadWorldGraphSkill
from backend.api.auth import router as auth_router
from backend.api.self_model import router as self_model_router
from backend.api.monologue import router as monologue_router, set_monologue_log
from backend.api.sprites import router as sprites_router
from backend.api.secrets import router as secrets_router
from backend.api.mcp import router as mcp_router
from backend.api.world import router as world_router
from backend.core.monologue import MonologueLog

# ── Logging ───────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tamagi")

# ── ASCII Banner ──────────────────────────────────────────────

BANNER = r"""
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   ████████╗ █████╗ ███╗   ███╗ █████╗  ██████╗ ██╗    ║
║   ╚══██╔══╝██╔══██╗████╗ ████║██╔══██╗██╔════╝ ██║    ║
║      ██║   ███████║██╔████╔██║███████║██║  ███╗██║    ║
║      ██║   ██╔══██║██║╚██╔╝██║██╔══██║██║   ██║██║    ║
║      ██║   ██║  ██║██║ ╚═╝ ██║██║  ██║╚██████╔╝██║    ║
║      ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝    ║
║                                                       ║
║              Your Local-First AI Companion            ║
║                         v0.1                          ║
╚═══════════════════════════════════════════════════════╝
"""


# ── Application Lifecycle ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and teardown TamAGI subsystems."""
    print(BANNER)

    # Load config
    config = load_config()
    set_config(config)
    logger.info(f"Config loaded — LLM: {config.llm.base_url} model={config.llm.model}")

    # Ensure workspace exists
    Path(config.workspace.path).mkdir(parents=True, exist_ok=True)

    # Initialize LLM client
    llm = LLMClient(config.llm)
    logger.info("LLM client initialized")

    # Initialize memory store (chromadb or elasticsearch, per config.memory.backend)
    memory = create_memory_store(config.memory)
    await memory.initialize()

    # Initialize personality engine (loads from tamagi_state.json or uses defaults).
    # config.tamagi.name is used as the bootstrap name only on first run (no state
    # file). personality_traits are set by the onboarding workflow.
    personality = PersonalityEngine(name=config.tamagi.name)
    personality.state.decay()  # Apply any time-based decay since last run
    logger.info(f"Personality: {personality.state.summary()}")

    # Initialize skill registry
    skills = SkillRegistry()
    skills.register(ReadSkill())
    skills.register(WriteSkill())
    skills.register(ExecSkill())
    skills.register(WebSearchSkill(
        provider=config.web_search.provider,
        brave_api_key=config.web_search.brave_api_key,
        searxng_url=config.web_search.searxng_url,
    ))
    skills.register(ExpressSkill(personality.state))
    from backend.skills.task_skill import TaskSkill
    skills.register(TaskSkill(
        workspace_path=config.workspace.path,
        done_cap=config.task_board.done_cap,
    ))

    # Migrate custom skills from backend/skills/custom/ to workspace/skills/
    workspace_skills = Path(config.workspace.path) / "skills"
    old_custom = Path("backend/skills/custom")
    if old_custom.exists():
        for py_file in old_custom.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            dest = workspace_skills / py_file.name
            if not dest.exists():
                workspace_skills.mkdir(parents=True, exist_ok=True)
                dest.write_text(py_file.read_text())
                logger.info(f"Migrated custom skill {py_file.name} to workspace/skills/")

    skills.discover_custom_skills(workspace_skills)
    logger.info(f"Skills registered: {skills.skill_count}")

    # Connect MCP servers and register their tools
    mcp_manager = None
    if config.mcp.servers:
        from backend.skills.mcp_adapter import MCPManager, set_mcp_manager
        mcp_manager = MCPManager(config.mcp.servers)
        n = await mcp_manager.connect_all(skills)
        set_mcp_manager(mcp_manager)
        logger.info(f"MCP tools registered: {n} (across {len(config.mcp.servers)} server(s))")

    # Initialize identity manager
    identity = IdentityManager(
        data_dir="data",
        workspace_dir=config.workspace.path,
        done_cap=config.task_board.done_cap,
        file_char_limit=config.identity_files.file_char_limit,
    )
    if identity.needs_onboarding:
        logger.info("First run detected — onboarding required")
    else:
        logger.info(f"Identity loaded: {identity.get_identity()}")

    # ── Self-model: load from disk or seed from identity ──────
    self_model_path = config.self_model.data_path
    self_model = SelfModel(data_path=self_model_path)
    workspace_path = Path(config.workspace.path)
    if Path(self_model_path).exists():
        try:
            self_model.load()
            logger.info(
                f"Self-model loaded: {self_model.node_count} nodes, "
                f"{self_model.edge_count} edges"
            )
        except ValueError as exc:
            # Old v1 format (pre-world-native) — wipe and re-seed
            logger.warning("Self-model incompatible (%s) — deleting and re-seeding.", exc)
            Path(self_model_path).unlink(missing_ok=True)
            counts = seed_self_model(self_model, workspace_path=workspace_path)
            logger.info(f"Self-model re-seeded: {counts}")
    else:
        counts = seed_self_model(self_model, workspace_path=workspace_path)
        logger.info(f"Self-model seeded: {counts}")

    # Wire any orphaned nodes — catches seed nodes and nodes that predate auto-wiring
    wired = self_model.wire_orphaned_nodes()
    if wired:
        logger.info(f"Wired {wired} edge(s) for previously orphaned self-model nodes")
    self_model.save()

    # ── Brain engines ──────────────────────────────────────────
    planning_engine = PlanningEngine(model=self_model)
    planning_engine.set_llm(llm)
    reflection_engine = ReflectionEngine(model=self_model)
    logger.info("Brain engines initialized (planning, reflection)")

    # ── Monologue log ──────────────────────────────────────────
    monologue_log = MonologueLog(log_path="data/monologue.jsonl")
    set_monologue_log(monologue_log)

    # ── Q&A belief pipeline ────────────────────────────────────
    from backend.core.qa_pipeline import QAPipeline
    qa_pipeline = QAPipeline(
        llm=llm,
        self_model=self_model,
        monologue_log=monologue_log,
        data_path="data/qa_pending.json",
        entropy_threshold=config.agent.qa_entropy_threshold,
        enabled=config.agent.qa_enabled,
    )
    qa_pipeline.load()
    logger.info("Q&A belief pipeline initialized (threshold=%.2f)", config.agent.qa_entropy_threshold)

    # ── Create the agent ───────────────────────────────────────
    agent = TamAGIAgent(
        config=config,
        llm=llm,
        memory=memory,
        personality=personality,
        skills=skills,
        identity=identity,
        self_model=self_model,
        planning_engine=planning_engine,
        reflection_engine=reflection_engine,
        monologue_log=monologue_log,
        qa_pipeline=qa_pipeline,
    )
    set_agent(agent)

    # Initialize multi-agent orchestration
    orchestrator = None
    if config.orchestrator.enabled:
        from backend.core.orchestrator import Orchestrator
        from backend.skills.orchestration_skill import OrchestrationSkill
        orchestrator = Orchestrator(llm=llm, skills=skills, config=config.orchestrator)
        skills.register(OrchestrationSkill(orchestrator=orchestrator))
        logger.info("Orchestration skill registered")

    skills.register(RecallMemorySkill(agent=agent))
    skills.register(ReadWorldGraphSkill(agent=agent))
    from backend.skills.write_world_graph_skill import WriteWorldGraphSkill
    skills.register(WriteWorldGraphSkill(agent=agent))

    # Initialize the Living World thread (replaces dream + motivation engines)
    world_thread = WorldThread(
        agent=agent,
        config=config.world_thread,
        monologue_log=monologue_log,
        autonomy_enabled=config.autonomy.enabled,
        schedule=config.autonomy.schedule,
        active_hours=(config.autonomy.active_hours_start, config.autonomy.active_hours_end),
        resume_after_conversation=config.autonomy.resume_after_conversation,
    )
    agent.set_world_thread(world_thread)

    # Sleep-time consolidation: distills lived world-thread experience into the
    # agent's own SOUL.md / IDENTITY.md. Triggered after every Nth autonomous tick
    # (see ConsolidationConfig) and on demand via POST /api/world/consolidate.
    from backend.core.consolidation import ConsolidationEngine, RelationalConsolidator
    agent.consolidation = ConsolidationEngine(
        llm=llm,
        identity=identity,
        self_model=self_model,
        monologue_log=monologue_log,
        config=config.world_thread.consolidation,
    )
    logger.info(
        "Consolidation engine ready (enabled=%s, every_n_ticks=%d)",
        config.world_thread.consolidation.enabled,
        config.world_thread.consolidation.every_n_ticks,
    )

    # Relational consolidation: distills conversation history into USER.md + a named
    # supplemental relationship file. Triggered on conversation-end cadence and on demand.
    agent.relational_consolidator = RelationalConsolidator(
        llm=llm,
        identity=identity,
        dialogue_provider=agent.recent_dialogue_text,
        monologue_log=monologue_log,
        config=config.relational,
    )
    logger.info(
        "Relational consolidator ready (enabled=%s, every_n_conversations=%d → %s)",
        config.relational.enabled,
        config.relational.every_n_conversations,
        config.relational.supplemental_filename,
    )

    # First-run world seed: if no world state exists yet and onboarding is done,
    # auto-generate a seed from identity context so the world thread has a starting point.
    from backend.core.world_state import WorldStateStore
    _ws_store = WorldStateStore()
    if _ws_store.load() is None and not identity.needs_onboarding:
        logger.info("No world state found — generating first-run world seed from identity")
        try:
            from backend.core.world_seed import OnboardingInput, generate_world_seed
            from backend.core.world_state import parse_new_state, WorldState
            from datetime import datetime, timezone
            identity_ctx = identity.get_system_prompt_context()
            raw = await generate_world_seed(llm, OnboardingInput(), identity_ctx)
            ws = parse_new_state(raw)
            if ws is None:
                now = datetime.now(timezone.utc).isoformat()
                ws = WorldState(timestamp=now, last_tick=now, location="", mood="",
                                focus="", available_actions=[], raw_state_block=raw)
            _ws_store.save(ws)
            logger.info("World seed saved: location=%r", ws.location)
        except Exception as exc:
            logger.warning("Auto world seed failed: %s — frontend will prompt for seed", exc)

    world_thread.start()
    logger.info("World thread started — Living World is active")

    logger.info(f"═══ TamAGI is awake! ═══")
    logger.info(f"    Open http://localhost:{config.server.port} in your browser")

    yield

    # Shutdown
    logger.info(f"═══ {config.tamagi.name} is going to sleep... ═══")
    await world_thread.stop()
    personality.save_state()
    self_model.save()
    logger.info(
        f"Self-model saved: {self_model.node_count} nodes, "
        f"{self_model.edge_count} edges"
    )
    if mcp_manager is not None:
        await mcp_manager.close()
    if orchestrator is not None:
        await orchestrator.close()
    await llm.close()


# ── FastAPI App ───────────────────────────────────────────────

app = FastAPI(
    title="TamAGI",
    description="Your local-first AI companion",
    version="0.2.0",
    lifespan=lifespan,
)

# ── Auth middleware class ─────────────────────────────────────────────────────
# Using BaseHTTPMiddleware so it can be added via add_middleware() and therefore
# participates in the correct middleware stack order.
#
# Starlette stacks middleware so the LAST add_middleware() call is outermost
# (first to see the request). We need:
#   Request → CORS → Session (decodes cookie) → Auth (reads session) → app
# so we add: Auth first, Session second, CORS last (see below).

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        config = get_config()

        # Auth is opt-in — zero overhead when disabled
        if not config.auth.enabled:
            return await call_next(request)

        path = request.url.path

        # Always allow: auth endpoints and the login page itself
        public_prefixes = ("/api/auth",)
        public_exact = {"/login", "/manifest.json", "/sw.js"}
        if path.startswith(public_prefixes) or path in public_exact:
            return await call_next(request)

        # Allow authenticated sessions through
        if request.session.get("authenticated"):
            return await call_next(request)

        # API / WebSocket paths → 401 JSON (consumed by frontend JS)
        if path.startswith("/api") or path.startswith("/ws"):
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)

        # All other browser navigations → redirect to login page
        return RedirectResponse("/login")


# ── Middleware stack (added in reverse order of execution) ────────────────────
# add_middleware() wraps: last-added is outermost. Execution order:
#   CORS (outermost) → Session (decodes cookie) → Auth (checks session) → app

_config = get_config()
_data_dir = str(Path(_config.history.persist_path).parent)

# 1. Auth — added first so it runs innermost (after session decodes the cookie)
app.add_middleware(AuthMiddleware)

# 2. Session — decodes the signed session cookie before Auth can read it
app.add_middleware(
    SessionMiddleware,
    secret_key=load_session_secret(_data_dir),
    session_cookie="tamagi_session",
    https_only=False,   # allow HTTP for local/LAN use
    same_site="strict",
)

# 3. CORS — outermost, handles preflight before anything else
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API routes
app.include_router(auth_router)
app.include_router(secrets_router)
app.include_router(mcp_router)
app.include_router(chat_router)
app.include_router(skills_router)
app.include_router(onboarding_router)
app.include_router(self_model_router)
app.include_router(monologue_router)
app.include_router(sprites_router)
app.include_router(world_router)

# Serve user sprite PNGs — directory is created eagerly so the mount always succeeds.
_sprites_data_dir = Path("data/sprites")
_sprites_data_dir.mkdir(parents=True, exist_ok=True)
app.mount("/sprites", StaticFiles(directory=str(_sprites_data_dir)), name="sprites")

# Serve frontend static files
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_path / "assets")), name="assets")

    # Serve frontend files with explicit routes (PWA compatibility)
    @app.get("/manifest.json")
    async def manifest():
        return FileResponse(frontend_path / "manifest.json", media_type="application/manifest+json")

    @app.get("/sw.js")
    async def service_worker():
        return FileResponse(frontend_path / "sw.js", media_type="application/javascript")

    @app.get("/login")
    async def login_page():
        return FileResponse(frontend_path / "login.html")

    @app.get("/settings")
    async def settings_page():
        return FileResponse(frontend_path / "settings.html")

    @app.get("/rig-editor")
    async def rig_editor_page():
        return FileResponse(frontend_path / "rig-editor.html")

    @app.get("/self-model")
    async def self_model_page():
        return FileResponse(frontend_path / "self-model.html")

    @app.get("/{path:path}")
    async def serve_frontend(path: str):
        # Try to serve the exact file first
        file_path = frontend_path / path
        if file_path.is_file():
            return FileResponse(file_path)
        # Fall back to index.html (SPA routing)
        return FileResponse(frontend_path / "index.html")


# ── CLI Entry Point ───────────────────────────────────────────

def main():
    config = load_config()
    uvicorn.run(
        "backend.main:app",
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
