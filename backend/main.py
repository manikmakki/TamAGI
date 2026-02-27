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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.config import load_config, set_config
from backend.core.llm import LLMClient
from backend.core.memory import MemoryStore
from backend.core.personality import PersonalityEngine
from backend.core.agent import TamAGIAgent
from backend.core.identity import IdentityManager
from backend.core.dreamer import DreamEngine
from backend.skills.registry import SkillRegistry
from backend.skills.read_skill import ReadSkill
from backend.skills.write_skill import WriteSkill
from backend.skills.exec_skill import ExecSkill
from backend.skills.web_search_skill import WebSearchSkill
from backend.api.chat import router as chat_router, set_agent
from backend.api.skills import router as skills_router
from backend.api.onboarding import router as onboarding_router
from backend.api.dreams import router as dreams_router, set_dream_engine

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

    # Initialize memory store
    memory = MemoryStore(config.memory)
    await memory.initialize()

    # Initialize personality engine (loads from tamagi_state.json or uses defaults)
    personality = PersonalityEngine()
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

    # Initialize identity manager
    identity = IdentityManager(data_dir="data", workspace_dir=config.workspace.path)
    if identity.needs_onboarding:
        logger.info("First run detected — onboarding required")
    else:
        logger.info(f"Identity loaded: {identity.get_identity()}")

    # Create the agent
    agent = TamAGIAgent(
        config=config,
        llm=llm,
        memory=memory,
        personality=personality,
        skills=skills,
        identity=identity,
    )
    set_agent(agent)

    # Initialize dream engine (autonomous idle behavior)
    dream_engine = DreamEngine(
        agent=agent,
        enabled=config.autonomy.enabled and config.autonomy.interval_minutes > 0,
        interval_minutes=config.autonomy.interval_minutes,
        active_hours=(config.autonomy.active_hours_start, config.autonomy.active_hours_end),
        activities=config.autonomy.activities,
        weights=config.autonomy.weights,
    )
    set_dream_engine(dream_engine)
    dream_engine.start()

    logger.info(f"═══ TamAGI is awake! ═══")
    logger.info(f"    Open http://localhost:{config.server.port} in your browser")

    yield

    # Shutdown
    logger.info(f"═══ {config.tamagi.name} is going to sleep... ═══")
    await dream_engine.stop()
    personality.save_state()
    await llm.close()


# ── FastAPI App ───────────────────────────────────────────────

app = FastAPI(
    title="TamAGI",
    description="Your local-first AI companion",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS — permissive for local/PWA use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(chat_router)
app.include_router(skills_router)
app.include_router(onboarding_router)
app.include_router(dreams_router)

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
