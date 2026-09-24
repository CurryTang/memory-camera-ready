"""
agentmem — Agentic Context Engineering Library

Expose the public API lazily so importing a narrow evaluation module does not
pull in every agent/runtime stack and their heavyweight dependencies.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BaseLLMProvider": ("agentmem.providers", "BaseLLMProvider"),
    "PlugMemRuntimeEngine": ("agentmem.plugmem.runtime", "PlugMemRuntimeEngine"),
    "MemRLConfig": ("agentmem.memrl", "MemRLConfig"),
    "MemRLRuntimeEngine": ("agentmem.memrl", "MemRLRuntimeEngine"),
}

__all__ = list(_EXPORTS)

def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'agentmem' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
