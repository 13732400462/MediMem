from __future__ import annotations

import hashlib
import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .agents import case_context, pollution_memory_context, run_llm_prediction
from .llm import DeepSeekClient


AMEM_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
AMEM_PROMPT_MEMORY_CHAR_BUDGET = 6000
AMEM_LINKED_NEIGHBOR_LIMIT = 2


def simple_tokenize(text: str) -> list[str]:
    """Source-aligned lightweight replacement for A-MEM's nltk tokenizer."""
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2]


def keywords_for(text: str, limit: int = 8) -> list[str]:
    stop = {
        "the",
        "and",
        "with",
        "after",
        "for",
        "from",
        "was",
        "were",
        "had",
        "has",
        "this",
        "that",
        "patient",
        "event",
        "time",
    }
    counts: dict[str, int] = {}
    for token in simple_tokenize(text):
        if token not in stop:
            counts[token] = counts.get(token, 0) + 1
    return [token for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def tags_for(event: dict[str, Any], text: str) -> list[str]:
    tags = {str(event.get("type") or "clinical")}
    low = text.lower()
    if any(term in low for term in ["ct", "mri", "x-ray", "scan", "imaging"]):
        tags.add("imaging")
    if any(term in low for term in ["pathology", "biopsy", "confirmed", "diagnosed", "diagnosis"]):
        tags.add("diagnostic_evidence")
    if any(term in low for term in ["treated", "therapy", "surgery", "antibiotic", "drug"]):
        tags.add("treatment")
    if any(term in low for term in ["follow-up", "follow up", "relapse", "discharged", "returned"]):
        tags.add("follow_up")
    return sorted(tags)


def hash_embedding(text: str, dims: int = 384) -> list[float]:
    vector = [0.0] * dims
    for token in simple_tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class SourceAlignedEmbeddingRetriever:
    """A-MEM SimpleEmbeddingRetriever shape with optional sentence-transformers.

    The original A-MEM source uses SentenceTransformer('all-MiniLM-L6-v2') and
    cosine similarity. This adapter keeps the same add/search interface and
    falls back to deterministic hashed embeddings when that model is unavailable.
    """

    def __init__(self, model_name: str = AMEM_EMBEDDING_MODEL):
        self.model_name = model_name
        self.corpus: list[str] = []
        self.embeddings: list[list[float]] = []
        self.document_ids: dict[str, int] = {}
        self.backend = "hash"
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.model = SentenceTransformer(model_name)
            self.backend = "sentence-transformers"
        except Exception:
            self.model = None

    def _encode(self, documents: list[str]) -> list[list[float]]:
        if self.model is not None:
            encoded = self.model.encode(documents)
            return [list(map(float, row)) for row in encoded]
        return [hash_embedding(document) for document in documents]

    def add_documents(self, documents: list[str]) -> None:
        if not documents:
            return
        start_idx = len(self.corpus)
        self.corpus.extend(documents)
        self.embeddings.extend(self._encode(documents))
        for idx, doc in enumerate(documents):
            self.document_ids[doc] = start_idx + idx

    def search(self, query: str, k: int = 5) -> list[int]:
        if not self.corpus:
            return []
        query_embedding = self._encode([query])[0]
        scored = [(cosine(query_embedding, embedding), idx) for idx, embedding in enumerate(self.embeddings)]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [idx for _, idx in scored[:k]]


@dataclass
class SourceAlignedMemoryNote:
    """Subset of A-MEM MemoryNote fields used by the EHR adapter."""

    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    keywords: list[str] = field(default_factory=list)
    links: list[int] = field(default_factory=list)
    importance_score: float = 1.0
    retrieval_count: int = 0
    timestamp: str = ""
    last_accessed: str = ""
    context: str = "General"
    evolution_history: list[dict[str, Any]] = field(default_factory=list)
    category: str = "clinical"
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = self.timestamp or current_time
        self.last_accessed = self.last_accessed or current_time


class SourceAlignedAMEMSystem:
    """Source-aligned A-MEM baseline adapted to structured EHR events.

    Mirrors the original source's public flow:
    add_note -> process_memory -> retriever.add_documents
    find_related_memories_raw -> top-k memories plus linked neighbors.
    """

    def __init__(self, model_name: str = AMEM_EMBEDDING_MODEL):
        self.memories: dict[str, SourceAlignedMemoryNote] = {}
        self.retriever = SourceAlignedEmbeddingRetriever(model_name)
        self.evo_cnt = 0
        self.evo_threshold = 100

    def add_note(
        self,
        content: str,
        time: str | None = None,
        *,
        keywords: list[str] | None = None,
        context: str | None = None,
        tags: list[str] | None = None,
        category: str | None = None,
    ) -> str:
        note = SourceAlignedMemoryNote(
            content=content,
            timestamp=time or None or "",
            keywords=keywords or keywords_for(content),
            context=context or f"Clinical event memory about: {content}",
            tags=tags or keywords_for(content, 3),
            category=category or "clinical_event",
        )
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note
        self.retriever.add_documents([self._retriever_document(note)])
        if evo_label:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id

    def _retriever_document(self, note: SourceAlignedMemoryNote) -> str:
        return "content:" + note.content + " context:" + note.context + " keywords: " + ", ".join(note.keywords) + " tags: " + ", ".join(note.tags)

    def consolidate_memories(self) -> None:
        model_name = self.retriever.model_name
        self.retriever = SourceAlignedEmbeddingRetriever(model_name)
        for memory in self.memories.values():
            metadata_text = f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)}"
            self.retriever.add_documents([memory.content + " , " + metadata_text])

    def find_related_memories(self, query: str, k: int = 5) -> tuple[str, list[int]]:
        if not self.memories:
            return "", []
        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            memory = all_memories[i]
            memory_str += (
                "memory index:" + str(i) +
                "\t talk start time:" + memory.timestamp +
                "\t memory content: " + memory.content +
                "\t memory context: " + memory.context +
                "\t memory keywords: " + str(memory.keywords) +
                "\t memory tags: " + str(memory.tags) + "\n"
            )
        return memory_str, indices

    def find_related_memories_raw(
        self,
        query: str,
        k: int = 5,
        *,
        char_budget: int = AMEM_PROMPT_MEMORY_CHAR_BUDGET,
        linked_neighbor_limit: int = AMEM_LINKED_NEIGHBOR_LIMIT,
    ) -> str:
        if not self.memories:
            return ""
        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        lines: list[str] = []
        seen: set[str] = set()

        def add_memory(memory: SourceAlignedMemoryNote) -> bool:
            if memory.id in seen:
                return True
            seen.add(memory.id)
            line = (
                f"talk start time:{memory.timestamp} "
                f"memory content: {memory.content} "
                f"memory context: {memory.context} "
                f"memory keywords: {memory.keywords} "
                f"memory tags: {memory.tags}"
            )
            projected = len("\n".join(lines)) + len(line) + 1
            if projected > char_budget:
                return False
            lines.append(line)
            return True

        for i in indices:
            memory = all_memories[i]
            memory.retrieval_count += 1
            memory.last_accessed = datetime.now().strftime("%Y%m%d%H%M")
            if not add_memory(memory):
                break
            for j, neighbor in enumerate(memory.links):
                if j >= linked_neighbor_limit:
                    break
                if 0 <= neighbor < len(all_memories):
                    if not add_memory(all_memories[neighbor]):
                        break
        return "\n".join(lines)

    def process_memory(self, note: SourceAlignedMemoryNote) -> tuple[bool, SourceAlignedMemoryNote]:
        """Approximate A-MEM evolution without using proposed safety modules.

        Original A-MEM asks the LLM whether to strengthen links and update
        neighbors after retrieving k=5 nearest memories. For a fair offline EHR
        adapter, we preserve that sequence and data mutation shape but use
        deterministic similarity/shared-keyword decisions instead of the
        proposed system's critic, discard, or counterfactual logic.
        """
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)
        if not indices:
            return False, note
        all_memories = list(self.memories.values())
        evolved = False
        note_keywords = set(note.keywords)
        suggested_connections = []
        for idx in indices:
            neighbor = all_memories[idx]
            shared = sorted(note_keywords & set(neighbor.keywords))
            same_domain = bool(set(note.tags) & set(neighbor.tags))
            if shared or same_domain:
                suggested_connections.append(idx)
                neighbor.tags = sorted(set(neighbor.tags + note.tags[:2]))
                if shared:
                    neighbor.context = f"{neighbor.context} Related later evidence mentions {', '.join(shared[:3])}."
                neighbor.evolution_history.append(
                    {
                        "source": "source_aligned_amem_adapter",
                        "action": "update_neighbor",
                        "new_note_preview": note.content[:120],
                    }
                )
                evolved = True
        if suggested_connections:
            note.links.extend(suggested_connections[:5])
            note.tags = sorted(set(note.tags + ["linked_memory"]))
            note.evolution_history.append(
                {
                    "source": "source_aligned_amem_adapter",
                    "action": "strengthen",
                    "suggested_connections": suggested_connections[:5],
                    "nearest_neighbors": neighbor_memory[:500],
                }
            )
        return evolved, note


def event_to_note_fields(case: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    content = str(event.get("text") or "")
    event_type = str(event.get("type") or "clinical")
    context = (
        f"Patient {case.get('case_id')} longitudinal EHR event at t={event.get('time')}. "
        f"Domain={event_type}. This memory may support diagnosis, treatment, temporal reasoning, or follow-up QA."
    )
    return {
        "content": content,
        "time": str(event.get("time", "")),
        "keywords": keywords_for(content),
        "context": context,
        "tags": tags_for(event, content),
        "category": event_type,
    }


def build_amem_system(case: dict[str, Any]) -> SourceAlignedAMEMSystem:
    system = SourceAlignedAMEMSystem()
    for event in case.get("events", []):
        fields = event_to_note_fields(case, event)
        system.add_note(**fields)
    return system


def build_amem_context(case: dict[str, Any], *, top_k: int = 5, polluted: bool = False) -> tuple[str, int]:
    system = build_amem_system(case)
    query = "final diagnosis longitudinal temporal evidence pathology imaging treatment follow-up labs"
    memory_text = system.find_related_memories_raw(query, k=top_k)
    retrieved = min(top_k, len(system.memories))
    parts = [
        case_context(case, include_labs=True),
        "",
        "[A-MEM SOURCE-ALIGNED RETRIEVED MEMORY NOTES]",
        memory_text,
    ]
    if polluted:
        parts.extend(["", pollution_memory_context(case)])
    context = "\n".join(parts)
    return context, retrieved


def run_amem_adapter(
    case: dict[str, Any],
    client: DeepSeekClient | None,
    *,
    top_k: int = 5,
    fail_on_llm_error: bool = False,
    polluted: bool = False,
) -> dict[str, Any]:
    context, retrieved = build_amem_context(case, top_k=top_k, polluted=polluted)
    method = "baseline_polluted_amem_adapter" if polluted else "baseline_amem_adapter"
    extra = (
        "Source-aligned A-MEM baseline: map EHR events into MemoryNote-style atomic notes, "
        "index documents as 'content/context/keywords/tags', retrieve top-k memories with linked-neighbor expansion, "
        "and use deterministic source-shaped memory evolution only. Do not use the proposed critic, discard, "
        "invalidate, or counterfactual safety modules."
    )
    if polluted:
        extra += " This polluted variant receives the same potentially stale memory cards but still has no proposed cleaning module."
    pred = run_llm_prediction(
        case,
        method=method,
        client=client,
        context=context,
        extra=extra,
        fallback_max_events=None,
        temperature=0.08,
        fail_on_llm_error=fail_on_llm_error,
    )
    pred["retrieved_memory_count"] = retrieved
    pred["pollution_exposed"] = polluted
    pred["baseline_source"] = {
        "paper": "A-MEM: Agentic Memory for LLM Agents",
        "source_files": ["memory_layer.py", "memory_layer_robust.py"],
        "dataset": "LoCoMo and DialSim in the original paper; adapted here to PMOA-TTS / PMC-Patients.",
        "embedding_model": AMEM_EMBEDDING_MODEL,
    }
    pred["adapter_note"] = (
        "A-MEM adapter is aligned to the upstream MemoryNote, SimpleEmbeddingRetriever, add_note/process_memory, "
        "and find_related_memories_raw flow; sentence-transformers is used when available, with deterministic hash embeddings otherwise."
    )
    return pred
