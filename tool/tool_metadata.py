"""Tool metadata record with 4-layer structured representation.

L1: Signature  – name, input/output types, dependencies
L2: Description – natural language summary, tags, domain
L3: Specification – pre/post conditions, I/O description, complexity
L4: Examples – I/O pairs, test cases, usage snippets
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# L1: Signature
# ---------------------------------------------------------------------------
@dataclass
class ToolSignature:
    """Layer 1 – the type-level identity of a tool."""

    name: str
    input_type: str  # e.g. "str", "List[int]", a free-form type descriptor
    output_type: str
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "dependencies": self.dependencies,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolSignature:
        return cls(
            name=d["name"],
            input_type=d["input_type"],
            output_type=d["output_type"],
            dependencies=d.get("dependencies", []),
        )


# ---------------------------------------------------------------------------
# L2: Description
# ---------------------------------------------------------------------------
@dataclass
class ToolDescription:
    """Layer 2 – human-readable metadata."""

    summary: str  # natural language summary
    tags: list[str] = field(default_factory=list)
    domain: str = ""  # e.g. "math", "string", "data_processing"

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "tags": self.tags,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolDescription:
        return cls(
            summary=d["summary"],
            tags=d.get("tags", []),
            domain=d.get("domain", ""),
        )


# ---------------------------------------------------------------------------
# L3: Specification
# ---------------------------------------------------------------------------
@dataclass
class ToolSpecification:
    """Layer 3 – formal specification."""

    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    input_description: str = ""
    output_description: str = ""
    complexity: str = ""  # e.g. "O(n log n)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "input_description": self.input_description,
            "output_description": self.output_description,
            "complexity": self.complexity,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolSpecification:
        return cls(
            preconditions=d.get("preconditions", []),
            postconditions=d.get("postconditions", []),
            input_description=d.get("input_description", ""),
            output_description=d.get("output_description", ""),
            complexity=d.get("complexity", ""),
        )


# ---------------------------------------------------------------------------
# L4: Examples
# ---------------------------------------------------------------------------
@dataclass
class ToolExample:
    """A single input/output example with optional metadata."""

    input: Any
    output: Any
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"input": self.input, "output": self.output}
        if self.explanation:
            d["explanation"] = self.explanation
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolExample:
        return cls(
            input=d["input"],
            output=d["output"],
            explanation=d.get("explanation", ""),
        )


# ---------------------------------------------------------------------------
# Top-level: Tool Metadata Record
# ---------------------------------------------------------------------------
@dataclass
class ToolMetadata:
    """Complete per-tool structured entry combining all four layers."""

    signature: ToolSignature          # L1
    description: ToolDescription      # L2
    specification: ToolSpecification  # L3
    examples: list[ToolExample] = field(default_factory=list)  # L4

    # -- serialisation helpers ------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature.to_dict(),
            "description": self.description.to_dict(),
            "specification": self.specification.to_dict(),
            "examples": [e.to_dict() for e in self.examples],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolMetadata:
        return cls(
            signature=ToolSignature.from_dict(d["signature"]),
            description=ToolDescription.from_dict(d["description"]),
            specification=ToolSpecification.from_dict(d["specification"]),
            examples=[ToolExample.from_dict(e) for e in d.get("examples", [])],
        )
