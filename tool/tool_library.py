"""Tool Library – explicit compositional DAG ``G = (V, E)`` with a
four-layer hierarchical index ``I`` (Algorithms 2 & 3 of the paper).

- ``V``: vertices are :class:`ToolEntry` records (primitive or composite).
- ``E``: directed edges go from a composite ``c`` to each of its dependencies
  ``d`` (i.e. "c uses d").
- ``I``: hierarchical index over
    L1 signatures (name, input_type -> output_type),
    L2 descriptions (tags, domain, summary tokens),
    L3 specifications (complexity, pre/post conditions),
    L4 examples (materialised I/O pairs).

All inserts go through :meth:`insert_tool` (Algorithm 3: validate the
signature, duplicate-check via hierarchical retrieval, classify as
primitive/composite, attach edges, update index).  ``add_primitive`` /
``add_composite`` are kept as thin wrappers so existing call sites still
work.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from tool.tool import Tool

log = logging.getLogger(__name__)


class ToolType(str, Enum):
    """Whether a tool is an atomic primitive or a composition."""

    PRIMITIVE = "primitive"
    COMPOSITE = "composite"


@dataclass
class ToolEntry:
    """A tool together with its type classification."""

    tool: Tool
    tool_type: ToolType

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool.to_dict(),
            "tool_type": self.tool_type.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolEntry:
        return cls(
            tool=Tool.from_dict(d["tool"]),
            tool_type=ToolType(d["tool_type"]),
        )


# ---------------------------------------------------------------------------
# Hierarchical index
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenise(text: str) -> set[str]:
    return {tok.lower() for tok in _WORD_RE.findall(text or "")}


@dataclass
class HierIndex:
    """Four-layer inverted index over tool metadata.

    Maintained incrementally by :meth:`ToolLibrary.insert_tool`.
    """

    # L1 – signature
    by_io: dict[tuple[str, str], set[str]]    # (input_type, output_type) -> names
    by_input: dict[str, set[str]]             # input_type -> names
    by_output: dict[str, set[str]]            # output_type -> names
    # L2 – description
    by_domain: dict[str, set[str]]
    by_tag: dict[str, set[str]]
    by_summary_token: dict[str, set[str]]     # token -> names (for cheap BM25-lite)
    # L3 – specification
    by_complexity: dict[str, set[str]]
    # L4 – examples
    example_count: dict[str, int]

    @classmethod
    def empty(cls) -> "HierIndex":
        return cls(
            by_io=defaultdict(set),
            by_input=defaultdict(set),
            by_output=defaultdict(set),
            by_domain=defaultdict(set),
            by_tag=defaultdict(set),
            by_summary_token=defaultdict(set),
            by_complexity=defaultdict(set),
            example_count={},
        )

    # -- maintenance ----------------------------------------------------
    def add(self, entry: ToolEntry) -> None:
        meta = entry.tool.metadata
        sig = meta.signature
        name = sig.name
        self.by_io[(sig.input_type, sig.output_type)].add(name)
        self.by_input[sig.input_type].add(name)
        self.by_output[sig.output_type].add(name)
        desc = meta.description
        if desc.domain:
            self.by_domain[desc.domain].add(name)
        for tag in desc.tags:
            self.by_tag[tag].add(name)
        for tok in _tokenise(desc.summary):
            self.by_summary_token[tok].add(name)
        if meta.specification.complexity:
            self.by_complexity[meta.specification.complexity].add(name)
        self.example_count[name] = len(meta.examples)

    def remove(self, entry: ToolEntry) -> None:
        meta = entry.tool.metadata
        sig = meta.signature
        name = sig.name

        def _discard(d: dict[Any, set[str]], key: Any) -> None:
            bucket = d.get(key)
            if bucket is not None:
                bucket.discard(name)
                if not bucket:
                    d.pop(key, None)

        _discard(self.by_io, (sig.input_type, sig.output_type))
        _discard(self.by_input, sig.input_type)
        _discard(self.by_output, sig.output_type)
        if meta.description.domain:
            _discard(self.by_domain, meta.description.domain)
        for tag in meta.description.tags:
            _discard(self.by_tag, tag)
        for tok in _tokenise(meta.description.summary):
            _discard(self.by_summary_token, tok)
        if meta.specification.complexity:
            _discard(self.by_complexity, meta.specification.complexity)
        self.example_count.pop(name, None)


# ---------------------------------------------------------------------------
# ToolLibrary
# ---------------------------------------------------------------------------
class ToolLibrary:
    """Registry over a compositional DAG ``G=(V,E)`` with hierarchical index
    ``I``.  Vertices are tools; edges are composite->dependency links.
    """

    def __init__(self) -> None:
        # V
        self._entries: dict[str, ToolEntry] = {}
        # E: adjacency in both directions
        self._out_edges: dict[str, set[str]] = defaultdict(set)  # composite -> deps
        self._in_edges: dict[str, set[str]] = defaultdict(set)   # dep -> composites
        # I
        self.index: HierIndex = HierIndex.empty()

    # -- queries -----------------------------------------------------------

    @property
    def tool_names(self) -> list[str]:
        return list(self._entries.keys())

    @property
    def primitive_names(self) -> list[str]:
        return [
            n for n, e in self._entries.items()
            if e.tool_type is ToolType.PRIMITIVE
        ]

    @property
    def composite_names(self) -> list[str]:
        return [
            n for n, e in self._entries.items()
            if e.tool_type is ToolType.COMPOSITE
        ]

    @property
    def vertices(self) -> list[str]:
        """DAG vertex set ``V``."""
        return list(self._entries.keys())

    @property
    def edges(self) -> list[tuple[str, str]]:
        """DAG edge set ``E`` as ``[(composite, dependency), ...]``."""
        return [(c, d) for c, deps in self._out_edges.items() for d in deps]

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    # -- access ------------------------------------------------------------

    def get_tool(self, name: str) -> Tool:
        return self._entries[name].tool

    def get_entry(self, name: str) -> ToolEntry:
        return self._entries[name]

    def get_type(self, name: str) -> ToolType:
        return self._entries[name].tool_type

    def get_dependencies(self, name: str) -> list[str]:
        """Return the dependency names for a tool (empty for primitives)."""
        return list(self._out_edges.get(name, ()))

    def get_dependents(self, name: str) -> list[str]:
        """Return names of composite tools that depend on *name*."""
        return list(self._in_edges.get(name, ()))

    # -- DAG invariants ----------------------------------------------------
    def _would_cycle(self, new_name: str, deps: Iterable[str]) -> bool:
        """Return True if adding edges new_name -> deps would create a cycle."""
        # DFS from every dep; if we can reach new_name it means new_name is
        # already an ancestor of that dep → cycle.
        for dep in deps:
            stack = [dep]
            seen: set[str] = set()
            while stack:
                cur = stack.pop()
                if cur == new_name:
                    return True
                if cur in seen:
                    continue
                seen.add(cur)
                stack.extend(self._out_edges.get(cur, ()))
        return False

    # -- hierarchical retrieval (Algorithm 2, index-only variant) ----------
    def hier_retrieve_candidate(
        self,
        *,
        name: str,
        input_type: str,
        output_type: str,
        summary: str = "",
        tags: Iterable[str] = (),
        domain: str = "",
        shortlist: int = 8,
    ) -> list[str]:
        """Return candidate *existing* tool names that look like duplicates of
        a tool with the given signature + description.

        This is the cascaded retrieval from Algorithm 2 applied to duplicate
        detection in Algorithm 3 (line 2: "Validate t+; if duplicate in I
        return ∅").  No LLM is required — we rank by overlap in the
        hierarchical index.  A separate, LLM-backed retriever in
        :mod:`tool.tool_retriever` is used for query-time retrieval; this
        method is the insert-time gate.
        """
        if name in self._entries:
            return [name]

        # L1 filter: same input/output signature first, then same I or same O
        l1 = set(self.index.by_io.get((input_type, output_type), ()))
        if not l1:
            l1 = (
                set(self.index.by_input.get(input_type, ()))
                | set(self.index.by_output.get(output_type, ()))
            )
        if not l1:
            return []

        # L2 scoring: domain match (+2), tag overlap (+1/tag), summary tokens (+1/tok)
        summary_tokens = _tokenise(summary)
        scores: dict[str, float] = {cand: 0.0 for cand in l1}
        for cand in l1:
            meta = self._entries[cand].tool.metadata
            if domain and meta.description.domain == domain:
                scores[cand] += 2.0
            tag_overlap = set(tags) & set(meta.description.tags)
            scores[cand] += float(len(tag_overlap))
            cand_tokens = _tokenise(meta.description.summary)
            scores[cand] += float(len(summary_tokens & cand_tokens))

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [n for n, _ in ranked[:shortlist]]

    # -- hierarchical retrieval at query time (Algorithm 2, index-only) ----
    def retrieve_for_query(
        self,
        problem: str,
        *,
        k: int = 8,
        shortlist: int = 16,
        include_leaves: bool = True,
    ) -> list[str]:
        """LLM-free variant of Algorithm 2 ``HIERRETRIEVE(q, I, G, k)``.

        Used inside the GRPO rollout loop where calling a teacher LLM per
        trajectory would be too expensive.  Scores every tool on three
        cascaded signals drawn from ``I``:

          * L2 description: token overlap between the problem and the
            tool's summary / tag set / domain.
          * L1 signature: tools whose input/output names appear in the
            problem text get a small bonus.
          * L4 examples: tools with at least one example get a tiny bonus.

        Then performs the Algorithm 2 "materialise leaves" step by adding
        the direct primitive dependencies of every selected composite so
        the student has the full call chain available.
        """
        if not self._entries:
            return []
        if len(self._entries) <= k:
            return self.tool_names

        q_tokens = _tokenise(problem)
        scores: dict[str, float] = {}
        for name, entry in self._entries.items():
            meta = entry.tool.metadata
            desc = meta.description
            sig = meta.signature
            score = 0.0
            # L2 description tokens
            score += float(len(q_tokens & _tokenise(desc.summary)))
            # L2 tags
            for tag in desc.tags:
                if tag.lower() in q_tokens:
                    score += 1.0
            # L2 domain
            if desc.domain and desc.domain.lower() in q_tokens:
                score += 2.0
            # L1 signature: reward tools whose I/O type appears in the query
            for typ in (sig.input_type, sig.output_type):
                if typ and typ.lower() in q_tokens:
                    score += 0.5
            # L4 examples bonus
            if self.index.example_count.get(name, 0) > 0:
                score += 0.25
            # Slight preference for composites (they encode reusable plans)
            if entry.tool_type is ToolType.COMPOSITE:
                score += 0.25
            scores[name] = score

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        shortlisted = [n for n, _ in ranked[:shortlist]]
        selected = shortlisted[:k]

        if include_leaves:
            # Algorithm 2 line 9 – "materialise leaves": for each composite
            # in the selection, ensure its direct primitive deps come along.
            picked: list[str] = list(selected)
            seen = set(picked)
            for name in selected:
                for dep in self._out_edges.get(name, ()):
                    if dep not in seen:
                        picked.append(dep)
                        seen.add(dep)
            return picked
        return selected

    # -- mutation (Algorithm 3 INSERTTOOL) ---------------------------------
    def insert_tool(
        self,
        tool: Tool,
        *,
        tool_type: ToolType | str | None = None,
        check_duplicates: bool = True,
    ) -> str | None:
        """Insert *tool* into ``V, E, I`` (Algorithm 3).

        Steps:
          1. Validate signature: name present, compile succeeds.
          2. Duplicate check via :meth:`hier_retrieve_candidate`; if a
             signature/description duplicate exists, reject and return ``None``.
          3. Classify: if no deps or ``tool_type == PRIMITIVE`` it's a
             primitive leaf; otherwise a composite whose deps must all be
             registered (Algorithm 3 line 7: "link to parents").
          4. Update V, E, I atomically; guarantee no cycles.

        Returns the inserted tool name, or ``None`` if the insertion was
        rejected as a duplicate.
        """
        sig = tool.metadata.signature
        name = sig.name
        if not name:
            raise ValueError("Tool is missing a name in its signature")

        # (1) Validate signature by compiling the code
        tool.compile()

        # (2) Duplicate check
        if check_duplicates:
            cands = self.hier_retrieve_candidate(
                name=name,
                input_type=sig.input_type,
                output_type=sig.output_type,
                summary=tool.metadata.description.summary,
                tags=tool.metadata.description.tags,
                domain=tool.metadata.description.domain,
                shortlist=4,
            )
            if name in self._entries:
                return None
            for cand in cands:
                existing = self._entries[cand].tool
                e_sig = existing.metadata.signature
                same_io = (
                    e_sig.input_type == sig.input_type
                    and e_sig.output_type == sig.output_type
                )
                same_name = e_sig.name == name
                if same_name or (
                    same_io
                    and existing.metadata.description.summary.strip()
                    == tool.metadata.description.summary.strip()
                    and existing.metadata.description.summary.strip() != ""
                ):
                    log.info(
                        "INSERTTOOL: rejecting '%s' as duplicate of '%s'",
                        name, cand,
                    )
                    return None

        # (3) Classify primitive vs composite
        declared_deps = list(sig.dependencies)
        if tool_type is None:
            t_type = ToolType.COMPOSITE if declared_deps else ToolType.PRIMITIVE
        else:
            t_type = ToolType(tool_type)

        if t_type is ToolType.PRIMITIVE:
            if declared_deps:
                # Primitives have no outgoing edges; strip any stray deps.
                sig.dependencies = []
                declared_deps = []
        else:  # COMPOSITE
            if not declared_deps:
                raise ValueError(
                    f"Composite tool '{name}' must declare at least one dependency"
                )
            missing = [d for d in declared_deps if d not in self._entries]
            if missing:
                raise KeyError(
                    f"Dependencies {missing} of composite '{name}' are not registered"
                )
            if self._would_cycle(name, declared_deps):
                raise RuntimeError(
                    f"Inserting '{name}' with deps {declared_deps} would create a cycle"
                )

        # (4) Commit V, E, I updates
        entry = ToolEntry(tool=tool, tool_type=t_type)
        self._entries[name] = entry
        if t_type is ToolType.COMPOSITE:
            self._out_edges[name] = set(declared_deps)
            for dep in declared_deps:
                self._in_edges[dep].add(name)
        else:
            self._out_edges[name] = set()
        self.index.add(entry)
        log.info("INSERTTOOL: added %s tool '%s'", t_type.value, name)
        return name

    # -- back-compat wrappers ---------------------------------------------
    def add_primitive(self, tool: Tool) -> None:
        """Register an atomic primitive tool (delegates to :meth:`insert_tool`)."""
        self.insert_tool(tool, tool_type=ToolType.PRIMITIVE)

    def add_composite(self, tool: Tool) -> None:
        """Register a composite tool (delegates to :meth:`insert_tool`)."""
        self.insert_tool(tool, tool_type=ToolType.COMPOSITE)

    def add_entry_dict(self, name: str, entry_d: dict[str, Any]) -> None:
        """Insert a raw serialized entry (used by online co-evolution to apply
        a tool produced on rank 0 to every other rank after broadcast).

        Uses :meth:`insert_tool` so the DAG + index stay in sync.  Silently
        no-ops if *name* is already present so repeated broadcasts are
        idempotent.
        """
        if name in self._entries:
            return
        entry = ToolEntry.from_dict(entry_d)
        try:
            self.insert_tool(
                entry.tool,
                tool_type=entry.tool_type,
                # Rank 0 already duplicate-checked; re-running on other ranks
                # would just reject the broadcast.
                check_duplicates=False,
            )
        except Exception as exc:
            log.warning("add_entry_dict failed for %s: %s", name, exc)

    def remove(self, name: str) -> None:
        """Remove a tool from V, E, I.  Raises if anything still depends on it."""
        dependents = self.get_dependents(name)
        if dependents:
            raise RuntimeError(
                f"Cannot remove '{name}': still depended on by {dependents}"
            )
        entry = self._entries.pop(name)
        # Drop outgoing edges; reverse-index cleanup
        for dep in self._out_edges.pop(name, ()):
            self._in_edges.get(dep, set()).discard(name)
        self._in_edges.pop(name, None)
        self.index.remove(entry)

    # -- execution ---------------------------------------------------------

    def execute(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a tool by name, compiling it first if needed."""
        return self.get_tool(name).execute(*args, **kwargs)

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": {n: e.to_dict() for n, e in self._entries.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolLibrary:
        lib = cls()
        # Rebuild V/E/I from entries.  Primitives first so composite deps
        # always resolve (topological insert).
        raw = {n: ToolEntry.from_dict(ed) for n, ed in d["entries"].items()}
        for name, entry in raw.items():
            if entry.tool_type is ToolType.PRIMITIVE:
                lib._commit(name, entry)
        # Composites in dependency order (Kahn-style)
        pending = {n: e for n, e in raw.items() if e.tool_type is ToolType.COMPOSITE}
        while pending:
            progressed = False
            for name, entry in list(pending.items()):
                deps = entry.tool.metadata.signature.dependencies
                if all(d in lib._entries for d in deps):
                    lib._commit(name, entry)
                    pending.pop(name)
                    progressed = True
            if not progressed:
                # Cyclic or dangling deps — fall back to inserting what's left
                # in arbitrary order so load() remains total.
                for name, entry in pending.items():
                    lib._commit(name, entry, enforce_deps=False)
                break
        return lib

    def _commit(self, name: str, entry: ToolEntry, enforce_deps: bool = True) -> None:
        """Low-level insert used by :meth:`from_dict`; bypasses duplicate
        checks and compilation but maintains V/E/I consistency.
        """
        self._entries[name] = entry
        deps = list(entry.tool.metadata.signature.dependencies)
        if entry.tool_type is ToolType.COMPOSITE:
            if enforce_deps:
                deps = [d for d in deps if d in self._entries]
            self._out_edges[name] = set(deps)
            for dep in deps:
                self._in_edges[dep].add(name)
        else:
            self._out_edges[name] = set()
        self.index.add(entry)

    def save(self, path: str | Path) -> None:
        """Persist the library to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ToolLibrary:
        """Load a library from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    # -- display -----------------------------------------------------------

    def __repr__(self) -> str:
        n_prim = len(self.primitive_names)
        n_comp = len(self.composite_names)
        n_edges = sum(len(s) for s in self._out_edges.values())
        return f"ToolLibrary(V={n_prim + n_comp}, primitives={n_prim}, composites={n_comp}, E={n_edges})"

    def describe(self) -> str:
        """Return a human-readable summary of the library contents."""
        lines = [repr(self), ""]
        if self.primitive_names:
            lines.append("Primitives:")
            for n in sorted(self.primitive_names):
                lines.append(f"  - {n}")
        if self.composite_names:
            lines.append("Composites:")
            for n in sorted(self.composite_names):
                deps = self.get_dependencies(n)
                lines.append(f"  - {n}  ->  [{', '.join(deps)}]")
        return "\n".join(lines)
