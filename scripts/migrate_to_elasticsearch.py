#!/usr/bin/env python3
"""
Migrate TamAGI memory to Elasticsearch.

Sources
-------
1. ChromaDB — all stored memory entries (conversation archives, facts,
   knowledge, skills, preferences).
2. History flat files — data/history/*.json — each conversation is indexed
   as individual user/assistant message documents so they are independently
   searchable.
3. Dream files — workspace/dreams/**/*.md — the full dream text. Note that
   dream *summaries* are already written to the memory store at runtime, so
   these are already in ES for new dreams. This source migrates the rich
   markdown content of dreams written before the ES backend was enabled.

The Elasticsearch store uses SHA-256 content hashing as the document _id,
so running this script multiple times is safe — duplicates are silently
skipped via ES upsert semantics.

Usage
-----
    # Dry run — shows what would be migrated, writes nothing
    python scripts/migrate_to_elasticsearch.py --dry-run

    # Migrate everything
    python scripts/migrate_to_elasticsearch.py

    # Only one source
    python scripts/migrate_to_elasticsearch.py --source chromadb
    python scripts/migrate_to_elasticsearch.py --source history
    python scripts/migrate_to_elasticsearch.py --source dreams

    # Point at a non-default config
    python scripts/migrate_to_elasticsearch.py --config /path/to/config.yaml

Requires
--------
    pip install elasticsearch[async] sentence-transformers chromadb
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# ── Make sure the repo root is on sys.path so backend.* imports work ─────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# workspace/dreams subdirectory → data.type stored in ES
DREAM_SUBDIR_TO_TYPE: dict[str, str] = {
    "memories":    "dream_memory",
    "explorations": "dream_exploration",
    "experiments": "dream_experiment",
    "journals":    "dream_journal",
    "wanderings":  "dream_wandering",
}


async def migrate(
    config_path: str | None,
    sources: list[str],
    dry_run: bool,
) -> None:
    from backend.config import load_config
    from backend.core.memory import MemoryEntry, MemoryType
    from backend.core.memory_elasticsearch import ElasticsearchMemoryStore

    cfg = load_config(config_path)
    es_cfg = cfg.memory.elasticsearch

    print(f"Target : {es_cfg.url}  index={es_cfg.index}")
    print(f"Model  : {es_cfg.embedding_model}")
    print(f"Dry run: {dry_run}")
    print()

    store = ElasticsearchMemoryStore(es_cfg)
    if not dry_run:
        await store.initialize()

    total_written = 0
    total_skipped = 0

    # ── 1. ChromaDB ───────────────────────────────────────────────────────────

    if "chromadb" in sources:
        print("=== ChromaDB ===")
        chroma_cfg = cfg.memory.chromadb

        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=str(Path(chroma_cfg.persist_directory)),
                settings=Settings(anonymized_telemetry=False),
            )
            collection = client.get_collection(chroma_cfg.collection_name)
        except Exception as e:
            print(f"  [skip] Could not open ChromaDB: {e}")
        else:
            total = collection.count()
            print(f"  Found {total} documents in '{chroma_cfg.collection_name}'")

            if total > 0:
                # Fetch in one batch (ChromaDB handles this fine for typical sizes)
                results = collection.get(include=["documents", "metadatas"])

                for i, doc in enumerate(results["documents"]):
                    meta = results["metadatas"][i] if results["metadatas"] else {}
                    doc_id = results["ids"][i]

                    raw_type = meta.get("memory_type", "knowledge")
                    try:
                        mem_type = MemoryType(raw_type)
                    except ValueError:
                        mem_type = MemoryType.KNOWLEDGE

                    entry = MemoryEntry(
                        content=doc,
                        memory_type=mem_type,
                        metadata={
                            k: v for k, v in meta.items()
                            if k not in ("memory_type", "timestamp")
                        },
                        id=doc_id,
                        timestamp=float(meta.get("timestamp", 0)),
                    )

                    if dry_run:
                        print(f"  [dry] Would index: [{mem_type.value}] {doc[:80]!r}")
                        total_written += 1
                    else:
                        try:
                            await store.store(entry)
                            total_written += 1
                            if (i + 1) % 25 == 0:
                                print(f"  ... {i + 1}/{total}")
                        except Exception as e:
                            print(f"  [error] doc {doc_id}: {e}")
                            total_skipped += 1

            print(f"  ChromaDB done: {total} documents processed")

    # ── 2. History flat files ─────────────────────────────────────────────────

    if "history" in sources:
        print()
        print("=== Conversation history ===")
        history_dir = Path(cfg.history.persist_path)

        if not history_dir.exists():
            print(f"  [skip] History directory not found: {history_dir}")
        else:
            files = sorted(history_dir.glob("*.json"))
            print(f"  Found {len(files)} conversation files in {history_dir}")

            for file in files:
                try:
                    data = json.loads(file.read_text())
                except Exception as e:
                    print(f"  [error] {file.name}: {e}")
                    total_skipped += 1
                    continue

                conv_id = data.get("id", file.stem)
                title = data.get("title", "")
                messages: list[dict] = data.get("messages", [])

                # Index each individual message as its own document so it is
                # independently retrievable by semantic search.
                for msg in messages:
                    role: str = msg.get("role", "")
                    content: str = msg.get("content", "").strip()

                    if not content:
                        continue

                    # Prefix with role so the content is self-describing
                    full_content = f"[{role}] {content}"

                    entry = MemoryEntry(
                        content=full_content,
                        memory_type=MemoryType.CONVERSATION,
                        metadata={
                            "data_type": "conversation_history",
                            "references": [conv_id],
                            "source": "history_migration",
                            "conversation_title": title,
                            "role": role,
                        },
                        timestamp=float(msg.get("timestamp", 0)),
                    )

                    if dry_run:
                        preview = full_content[:80]
                        print(f"  [dry] Would index: {conv_id[:8]}… [{role}] {preview!r}")
                        total_written += 1
                    else:
                        try:
                            await store.store(entry)
                            total_written += 1
                        except Exception as e:
                            print(f"  [error] {conv_id} / {role}: {e}")
                            total_skipped += 1

            print(f"  History done: {len(files)} conversations processed")

    # ── 3. Dream markdown files ───────────────────────────────────────────────

    if "dreams" in sources:
        print()
        print("=== Dream files ===")
        dreams_root = Path("workspace") / "dreams"

        if not dreams_root.exists():
            print(f"  [skip] Dreams directory not found: {dreams_root}")
        else:
            dream_count = 0
            for subdir, data_type in DREAM_SUBDIR_TO_TYPE.items():
                subdir_path = dreams_root / subdir
                if not subdir_path.exists():
                    continue

                files = sorted(subdir_path.glob("*.md"))
                if not files:
                    continue

                print(f"  {subdir}/  ({len(files)} files → data.type={data_type})")

                for file in files:
                    content = file.read_text().strip()
                    if not content:
                        continue

                    # Parse timestamp from filename (YYYYMMDD_HHMMSS.md)
                    timestamp = _parse_dream_timestamp(file.stem)

                    entry = MemoryEntry(
                        content=content,
                        memory_type=MemoryType.KNOWLEDGE,
                        metadata={
                            "data_type": data_type,
                            "source": "dream_migration",
                            "filename": file.name,
                        },
                        timestamp=timestamp,
                    )

                    if dry_run:
                        print(f"    [dry] Would index: {file.name}  {content[:60]!r}")
                        total_written += 1
                    else:
                        try:
                            await store.store(entry)
                            total_written += 1
                            dream_count += 1
                        except Exception as e:
                            print(f"    [error] {file.name}: {e}")
                            total_skipped += 1

            print(f"  Dreams done: {dream_count} files processed")

    # ── Summary ───────────────────────────────────────────────────────────────

    print()
    if dry_run:
        print(f"Dry run complete — {total_written} documents would be written.")
    else:
        stats = await store.get_stats()
        print(
            f"Migration complete — {total_written} documents written, "
            f"{total_skipped} errors."
        )
        print(f"ES index now contains {stats['total_documents']} total documents.")


def _parse_dream_timestamp(stem: str) -> float:
    """Parse a YYYYMMDD_HHMMSS filename stem into a Unix timestamp.
    Falls back to current time if the format doesn't match."""
    from datetime import datetime
    try:
        return datetime.strptime(stem, "%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        import time
        return time.time()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate TamAGI memory to Elasticsearch."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (default: auto-detected)",
    )
    parser.add_argument(
        "--source",
        dest="sources",
        choices=["chromadb", "history", "dreams", "all"],
        default="all",
        help="Which source(s) to migrate (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be migrated without writing anything",
    )
    args = parser.parse_args()

    sources = ["chromadb", "history", "dreams"] if args.sources == "all" else [args.sources]

    asyncio.run(migrate(args.config, sources, args.dry_run))


if __name__ == "__main__":
    main()
