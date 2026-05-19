"""Tool – metadata + executable code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from tool.tool_metadata import ToolMetadata


@dataclass
class Tool:
    """A tool pairs its metadata record with executable code."""

    metadata: ToolMetadata
    code: str  # the source code of the tool (e.g. a Python function body)
    _fn: Callable[..., Any] | None = field(default=None, repr=False)

    # -- execution ---------------------------------------------------------
    def compile(self) -> None:
        """Compile ``self.code`` and bind the callable to ``_fn``."""
        namespace: dict[str, Any] = {}
        exec(self.code, namespace)  # noqa: S102
        fn_name = self.metadata.signature.name
        if fn_name not in namespace:
            raise RuntimeError(
                f"Compiled code does not define a function named '{fn_name}'"
            )
        self._fn = namespace[fn_name]

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Run the tool. Compiles on first call if needed."""
        if self._fn is None:
            self.compile()
        assert self._fn is not None
        return self._fn(*args, **kwargs)

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "code": self.code,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Tool:
        return cls(
            metadata=ToolMetadata.from_dict(d["metadata"]),
            code=d["code"],
        )
