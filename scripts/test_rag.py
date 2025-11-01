import asyncio
import argparse
import json
import random
import re
import string
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence, Union

from py.memory.vector_store import VectorStore
from py.reason.rag import RAGService
from py.utils.embeddings import embed_text

WORDS_PER_MINUTE = 150
DEFAULT_CHUNK_SECONDS = 15
DEFAULT_LOG_DIR = Path("logs/rag_tests")

PRESET_TRANSCRIPTS = {
    "irl": Path("scripts/transcripts/irl_transcript.txt"),
    "gaming": Path("scripts/transcripts/gaming_transcript.txt"),
    "react": Path("scripts/transcripts/react_transcript.txt"),
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_to_project(path: Union[str, Path]) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _project_root() / candidate


def load_transcript_file(filepath: Union[str, Path]) -> str:
    resolved = _resolve_to_project(filepath)
    try:
        text = resolved.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Transcript file not found: {resolved}") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to read transcript file: {resolved}") from exc

    cleaned = text.strip()
    if not cleaned:
        raise ValueError(f"Transcript file is empty: {resolved}")
    return cleaned


def chunk_text_by_duration(
    text: str,
    chunk_seconds: int,
    *,
    words_per_minute: int = WORDS_PER_MINUTE,
) -> list[str]:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")

    normalized = text.strip()
    if not normalized:
        raise ValueError("Transcript text is empty")

    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", normalized)]
    sentences = [segment for segment in sentences if segment]

    if len(sentences) <= 1:
        fallback_segments = [
            segment.strip() for segment in re.split(r"\n+", normalized)
        ]
        sentences = [segment for segment in fallback_segments if segment]

    if not sentences:
        return [normalized]

    chunk_word_target = max(1, int(words_per_minute * chunk_seconds / 60))
    tolerance = max(chunk_word_target, int(chunk_word_target * 1.3))

    chunks: list[str] = []
    current_sentences: list[str] = []
    word_count = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)

        if current_sentences and word_count + sentence_word_count > tolerance:
            chunks.append(" ".join(current_sentences))
            current_sentences = []
            word_count = 0

        current_sentences.append(sentence)
        word_count += sentence_word_count

        if word_count >= chunk_word_target:
            chunks.append(" ".join(current_sentences))
            current_sentences = []
            word_count = 0

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    if not chunks:
        chunks = [normalized]

    return chunks


def _random_suffix() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=6))


async def seed_channel_from_chunks(
    store: VectorStore,
    channel_id: str,
    chunks: Sequence[str],
    *,
    chunk_seconds: int = DEFAULT_CHUNK_SECONDS,
    start_time: Optional[datetime] = None,
) -> None:
    if not chunks:
        raise ValueError("No transcript chunks provided")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")

    if start_time is None:
        total_seconds = chunk_seconds * len(chunks)
        start_time = datetime.utcnow() - timedelta(seconds=total_seconds)

    current_start = start_time
    for text in chunks:
        current_end = current_start + timedelta(seconds=chunk_seconds)
        embedding = await embed_text(text)
        await store.insert_transcript(
            channel_id=channel_id,
            text_value=text,
            start_time=current_start,
            end_time=current_end,
            embedding=embedding,
        )
        current_start = current_end


async def seed_channel_from_file(
    store: VectorStore,
    channel_id: str,
    transcript_path: Union[str, Path],
    *,
    chunk_seconds: int = DEFAULT_CHUNK_SECONDS,
) -> dict[str, Union[str, int]]:
    resolved_path = _resolve_to_project(transcript_path)
    transcript_text = load_transcript_file(resolved_path)
    chunks = chunk_text_by_duration(transcript_text, chunk_seconds)
    await seed_channel_from_chunks(
        store,
        channel_id,
        chunks,
        chunk_seconds=chunk_seconds,
    )
    return {
        "channel_id": channel_id,
        "chunk_count": len(chunks),
        "chunk_seconds": chunk_seconds,
        "transcript_path": str(resolved_path),
    }


async def insert_single_chunk(
    store: VectorStore,
    channel_id: str,
    text: str,
    *,
    duration_seconds: int = DEFAULT_CHUNK_SECONDS,
) -> None:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    end = datetime.utcnow()
    start = end - timedelta(seconds=duration_seconds)
    embedding = await embed_text(text)
    await store.insert_transcript(
        channel_id=channel_id,
        text_value=text,
        start_time=start,
        end_time=end,
        embedding=embedding,
    )


def get_preset_transcript_path(preset: str) -> Path:
    try:
        return _resolve_to_project(PRESET_TRANSCRIPTS[preset])
    except KeyError as exc:
        valid = ", ".join(sorted(PRESET_TRANSCRIPTS.keys()))
        raise ValueError(f"Unknown preset '{preset}'. Valid presets: {valid}") from exc


class TestLogger:
    def __init__(self, log_dir: Optional[Union[str, Path]] = None) -> None:
        directory = log_dir or DEFAULT_LOG_DIR
        resolved = _resolve_to_project(directory)
        resolved.mkdir(parents=True, exist_ok=True)
        self._base_dir = resolved

    def log_query(
        self,
        *,
        channel_id: str,
        question: str,
        result: dict,
        system_prompt: str,
        user_prompt: str,
        metadata: Optional[dict] = None,
        category: str,
        label: Optional[str] = None,
    ) -> None:
        timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        payload = {
            "timestamp": timestamp,
            "channel_id": channel_id,
            "question": question,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "answer": result.get("answer"),
            "citations": result.get("citations", []),
            "context": result.get("context", []),
            "chunks": result.get("chunks", []),
            "metadata": metadata or {},
        }

        base = self._timestamped_path(category, label or category)
        json_path = base.with_suffix(".json")
        text_path = base.with_suffix(".txt")

        json_path.write_text(
            json.dumps(payload, indent=2, default=self._json_default), encoding="utf-8"
        )
        text_path.write_text(
            self._format_query_text(payload),
            encoding="utf-8",
        )

    def log_seed(
        self,
        *,
        channel_id: str,
        transcript_path: Union[str, Path],
        chunk_count: int,
        chunk_seconds: int,
        label: Optional[str] = None,
        category: str = "seed",
    ) -> None:
        timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        resolved = _resolve_to_project(transcript_path)
        payload = {
            "timestamp": timestamp,
            "channel_id": channel_id,
            "transcript_path": str(resolved),
            "chunk_count": chunk_count,
            "chunk_seconds": chunk_seconds,
        }

        base = self._timestamped_path(category, label or category)
        json_path = base.with_suffix(".json")
        text_path = base.with_suffix(".txt")

        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        text_path.write_text(self._format_seed_text(payload), encoding="utf-8")

    def _timestamped_path(self, category: str, prefix: str) -> Path:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        category_dir = self._base_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        return category_dir / f"{prefix}_{stamp}"

    @staticmethod
    def _json_default(value):  # type: ignore[no-untyped-def]
        if isinstance(value, datetime):
            return value.isoformat() + "Z"
        if isinstance(value, timedelta):
            return value.total_seconds()
        return str(value)

    @staticmethod
    def _format_query_text(payload: dict) -> str:
        lines = [
            "=== RAG Query Result ===",
            f"Timestamp: {payload['timestamp']}",
            f"Channel: {payload['channel_id']}",
            "",
            "--- System Prompt ---",
            payload.get("system_prompt", ""),
            "",
            "--- User Prompt ---",
            payload.get("user_prompt", ""),
            "",
            "--- Question ---",
            payload.get("question", ""),
            "",
            "--- Answer ---",
            payload.get("answer", ""),
            "",
            "--- Citations ---",
        ]

        citations = payload.get("citations") or []
        if citations:
            for cite in citations:
                lines.append(f"  - {cite}")
        else:
            lines.append("  - (none)")

        lines.extend(["", "--- Context ---"])
        context_lines = payload.get("context") or []
        if context_lines:
            for line in context_lines:
                lines.append(f"  - {line}")
        else:
            lines.append("  - (none)")

        if payload.get("metadata"):
            lines.extend(["", "--- Metadata ---"])
            for key, value in payload["metadata"].items():
                lines.append(f"  - {key}: {value}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_seed_text(payload: dict) -> str:
        lines = [
            "=== RAG Seed Event ===",
            f"Timestamp: {payload['timestamp']}",
            f"Channel: {payload['channel_id']}",
            f"Transcript: {payload['transcript_path']}",
            f"Chunks Inserted: {payload['chunk_count']}",
            f"Chunk Duration (s): {payload['chunk_seconds']}",
        ]
        return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG test utilities")
    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_LOG_DIR),
        help="Directory where JSON and text logs will be written (default: %(default)s)",
    )
    parser.add_argument(
        "--demo-chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
        help="Chunk duration in seconds for default demo seeding (default: %(default)s)",
    )
    sub = parser.add_subparsers(dest="cmd")

    ask = sub.add_parser("ask", help="Ask a question for a given channel")
    ask.add_argument("channel", help="Channel ID")
    ask.add_argument("question", help="Question text")

    seed = sub.add_parser("seed", help="Seed transcript data for a channel")
    seed.add_argument("channel", help="Channel ID to seed")
    seed.add_argument(
        "transcript",
        nargs="?",
        help="Path to transcript file (defaults to preset specified by --type)",
    )
    seed.add_argument(
        "--type",
        choices=sorted(PRESET_TRANSCRIPTS.keys()),
        default="irl",
        help="Preset transcript to use when no transcript path is provided",
    )
    seed.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk duration in seconds for transcript seeding (defaults to root --chunk-size)",
    )

    addc = sub.add_parser("add-chunk", help="Insert a single transcript chunk")
    addc.add_argument("channel", help="Channel ID")
    addc.add_argument("text", help="Transcript text for the chunk")
    addc.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
        help="Duration in seconds represented by the chunk (default: %(default)s)",
    )

    return parser


async def run_default_demo(
    *,
    log_dir: Union[str, Path],
    chunk_seconds: int = DEFAULT_CHUNK_SECONDS,
) -> None:
    store = VectorStore()
    logger = TestLogger(log_dir)

    channels = {
        "IRL": f"test_irl_{_random_suffix()}",
        "GAMING": f"test_gaming_{_random_suffix()}",
        "REACT": f"test_react_{_random_suffix()}",
    }

    presets = {
        "IRL": get_preset_transcript_path("irl"),
        "GAMING": get_preset_transcript_path("gaming"),
        "REACT": get_preset_transcript_path("react"),
    }

    seed_results = await asyncio.gather(
        *[
            seed_channel_from_file(
                store,
                channel_id=channels[title],
                transcript_path=presets[title],
                chunk_seconds=chunk_seconds,
            )
            for title in channels
        ]
    )

    for title, seed_result in zip(channels, seed_results):
        logger.log_seed(
            channel_id=seed_result["channel_id"],
            transcript_path=seed_result["transcript_path"],
            chunk_count=seed_result["chunk_count"],
            chunk_seconds=seed_result["chunk_seconds"],
            label=f"seed_{title.lower()}",
        )

    rag = RAGService(vector_store=store)

    questions = {
        "IRL": "@bot what did we just do before entering the bookstore?",
        "GAMING": "@bot how did the last boss attempt finish?",
        "REACT": "@bot what was that clip showing and did we plan to try it?",
    }

    answers = await asyncio.gather(
        *[
            rag.answer(channel_id=channels[title], question=questions[title])
            for title in channels
        ]
    )

    def print_result(title: str, question: str, result: dict) -> None:
        print(f"\n=== {title} ===")
        chunk_channel = result["chunks"][0]["channel_id"] if result["chunks"] else "-"
        print("Channel:", chunk_channel)
        print("Question:", question)
        print("Answer:", result["answer"])
        print("Citations:")
        for cite in result.get("citations", []):
            print("  -", cite)
        print("Context lines:")
        for line in result.get("context", []):
            print("  -", line)

    for title, answer in zip(channels, answers):
        question = questions[title]
        print_result(title, question, answer)
        prompts = answer.get("prompts", {})
        metadata = {
            "mode": "demo",
            "title": title.lower(),
            "chunk_seconds": chunk_seconds,
        }
        logger.log_query(
            channel_id=channels[title],
            question=question,
            result=answer,
            system_prompt=prompts.get("system", ""),
            user_prompt=prompts.get("user", ""),
            metadata=metadata,
            category="demo",
            label=f"demo_{title.lower()}",
        )


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.cmd:
        await run_default_demo(
            log_dir=args.log_dir,
            chunk_seconds=args.demo_chunk_size,
        )
        return

    store = VectorStore()

    if args.cmd == "ask":
        rag = RAGService(vector_store=store)
        result = await rag.answer(channel_id=args.channel, question=args.question)
        print("Answer:", result["answer"])
        print("Citations:")
        for cite in result["citations"]:
            print("  -", cite)
        print("Context lines:")
        for line in result["context"]:
            print("  -", line)
        prompts = result.get("prompts", {})
        TestLogger(args.log_dir).log_query(
            channel_id=args.channel,
            question=args.question,
            result=result,
            system_prompt=prompts.get("system", ""),
            user_prompt=prompts.get("user", ""),
            metadata={"command": "ask"},
            category="ask",
            label="ask",
        )
        return

    if args.cmd == "seed":
        chunk_size = (
            args.chunk_size if args.chunk_size is not None else DEFAULT_CHUNK_SECONDS
        )
        transcript_path = (
            args.transcript
            if args.transcript is not None
            else get_preset_transcript_path(args.type)
        )
        seed_result = await seed_channel_from_file(
            store,
            args.channel,
            transcript_path,
            chunk_seconds=chunk_size,
        )
        print(
            f"Seeded {seed_result['chunk_count']} chunks from '{seed_result['transcript_path']}' "
            f"into channel '{args.channel}'"
        )
        TestLogger(args.log_dir).log_seed(
            channel_id=args.channel,
            transcript_path=seed_result["transcript_path"],
            chunk_count=seed_result["chunk_count"],
            chunk_seconds=seed_result["chunk_seconds"],
            label="seed_cli",
        )
        return

    if args.cmd == "add-chunk":
        await insert_single_chunk(
            store,
            args.channel,
            args.text,
            duration_seconds=args.duration,
        )
        print(f"Inserted single chunk into channel '{args.channel}'")
        TestLogger(args.log_dir).log_seed(
            channel_id=args.channel,
            transcript_path="<inline>",
            chunk_count=1,
            chunk_seconds=args.duration,
            label="add_chunk",
        )
        return


if __name__ == "__main__":
    asyncio.run(main())
