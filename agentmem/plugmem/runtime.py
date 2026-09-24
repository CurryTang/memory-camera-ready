from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping, Optional

from agentmem.plugmem.bridge import PlugMemBridge
from agentmem.plugmem.config import PlugMemConfig
from agentmem.plugmem.session import PlugMemQuestion, PlugMemSession, PlugMemStep

class _NoopBackend:
    _ERROR = "PlugMemRuntimeEngine requires a real backend_factory."

    def _raise(self) -> None:
        raise RuntimeError(self._ERROR)

    def build(self, session: PlugMemSession, sample_dir: Path, bridge: Any) -> Any:
        self._raise()

    def ask(
        self,
        memory: Any,
        question: PlugMemQuestion,
        session: PlugMemSession,
        sample_dir: Path,
        bridge: Any,
    ) -> Any:
        self._raise()

    def shutdown(self) -> None:
        return None

def _coerce_backend_result(result: Any) -> tuple[str, str]:
    if result is None:
        return "", ""
    if isinstance(result, Mapping):
        return str(result.get("answer", "") or ""), str(result.get("context", "") or "")
    answer = getattr(result, "answer", None)
    context = getattr(result, "context", None)
    if answer is not None or context is not None:
        return str(answer or ""), str(context or "")
    return str(result), ""

class PlugMemRuntimeEngine:
    """Small runtime wrapper around PlugMem lifecycle/state management."""

    def __init__(
        self,
        config: PlugMemConfig,
        *,
        bridge: Optional[Any] = None,
        artifact_root: Optional[Path | str] = None,
        backend_factory: Optional[
            Callable[[PlugMemConfig, Any, Path], Any]
        ] = None,
    ) -> None:
        self.config = config
        self.bridge = bridge or PlugMemBridge(config)
        self.artifact_root = Path(
            artifact_root
            or (getattr(self.bridge, "source_root", Path.cwd()) / "plugmem_artifacts")
        ).expanduser().resolve()
        self.backend_factory = backend_factory or (lambda *_args, **_kwargs: _NoopBackend())
        self.sample_id: Optional[str] = None
        self.sample_dir: Optional[Path] = None
        self.last_retrieval_context: str = ""
        self._session: Optional[PlugMemSession] = None
        self._memory: Any = None
        self._backend: Any = None
        self._scope: Optional[AbstractContextManager[None]] = None

    @property
    def session(self) -> Optional[PlugMemSession]:
        return self._session

    def reset(self, sample_id: str, goal: str = "", metadata: Optional[dict[str, Any]] = None) -> None:
        self.shutdown()
        self.sample_id = str(sample_id)
        self.sample_dir = self.artifact_root / self.sample_id
        self._prepare_sample_dir(self.sample_dir)

        self._scope = self.bridge.scoped_env({"DIR_PATH": str(self.sample_dir)})
        self._scope.__enter__()

        try:
            self._backend = self.backend_factory(self.config, self.bridge, self.sample_dir)
            self._session = PlugMemSession(
                session_id=self.sample_id,
                goal=str(goal or ""),
                metadata=dict(metadata or {}),
            )
            self._memory = None
            self.last_retrieval_context = ""
        except Exception:
            self._reset_runtime_state()
            raise

    def observe_step(self, step: PlugMemStep) -> None:
        if self._session is None:
            raise RuntimeError("Call reset() before observe_step().")
        if not isinstance(step, PlugMemStep):
            raise TypeError("observe_step() expects a PlugMemStep.")
        self._session.steps.append(step)

    def finalize(self, session: Optional[PlugMemSession] = None) -> Any:
        if session is not None:
            self._session = session
        if self._session is None or self.sample_dir is None or self._backend is None:
            raise RuntimeError("Call reset() before finalize().")
        if self._memory is not None:
            return self._memory

        self._memory = self._backend.build(self._session, self.sample_dir, self.bridge)
        return self._memory

    def ask(self, question: PlugMemQuestion | str) -> str:
        if self._session is None or self.sample_dir is None or self._backend is None:
            raise RuntimeError("Call reset() before ask().")
        if self._memory is None:
            self.finalize()

        if isinstance(question, PlugMemQuestion):
            question_obj = question
        else:
            question_obj = PlugMemQuestion(
                index=len(self._session.questions),
                question=str(question),
                answer=None,
            )
            self._session.questions.append(question_obj)

        result = self._backend.ask(
            self._memory,
            question_obj,
            self._session,
            self.sample_dir,
            self.bridge,
        )
        answer, context = _coerce_backend_result(result)
        self.last_retrieval_context = context
        return answer

    def shutdown(self) -> None:
        try:
            if self._backend is not None and hasattr(self._backend, "shutdown"):
                self._backend.shutdown()
        finally:
            self._reset_runtime_state()

    @staticmethod
    def _prepare_sample_dir(sample_dir: Path) -> None:
        if sample_dir.exists():
            if sample_dir.is_dir():
                shutil.rmtree(sample_dir)
            else:
                sample_dir.unlink()
        sample_dir.mkdir(parents=True, exist_ok=True)
        for rel in (
            "episodic_memory",
            "semantic_memory",
            "procedural_memory",
            "tag",
            "subgoal",
        ):
            (sample_dir / rel).mkdir(parents=True, exist_ok=True)

    def _reset_runtime_state(self) -> None:
        if self._scope is not None:
            self._scope.__exit__(None, None, None)
        self._backend = None
        self._scope = None
        self._memory = None
        self._session = None
        self.sample_id = None
        self.sample_dir = None
        self.last_retrieval_context = ""
