"""POMDP interface after Kaelbling, Littman & Cassandra (1998).

A POMDP is ⟨S, A, T, R, Ω, O⟩ with T(s, a, s') = P(s' | s, a),
O(s', a, o) = P(o | s', a), and R(s, a) = E[r | s, a] for r ~ r(· | s, a, s').
"""

import importlib
from collections.abc import Callable
from typing import Any, Protocol, overload, runtime_checkable

import jax

from jax_pomdps._types import Key
from jax_pomdps.spaces import Space


@runtime_checkable
class POMDP[State = Any, Action = Any, Obs = Any](Protocol):
    """Methods are pure functions of pytrees, safe under `jit` and `vmap`."""

    @property
    def action_space(self) -> Space[Action]: ...

    @property
    def observation_space(self) -> Space[Obs]: ...

    def reset(self, key: Key) -> State:
        """Sample an initial state s₀ ~ b₀."""
        ...

    def step(self, key: Key, state: State, action: Action) -> State:
        """Sample s' ~ T(s, a, ·)."""
        ...

    def reward(
        self, key: Key, state: State, action: Action, next_state: State
    ) -> jax.Array:
        """Sample r ~ r(· | s, a, s')."""
        ...

    def observe(self, key: Key, next_state: State, action: Action) -> Obs:
        """Sample o ~ O(s', a, ·)."""
        ...

    def done(self, state: State) -> jax.Array:
        """Return whether `state` is terminal."""
        ...


@runtime_checkable
class Renderable[State](Protocol):
    def render(self, state: State) -> object: ...


_REGISTRY: dict[str, Callable[..., object]] = {}


@overload
def register[F: Callable[..., POMDP]](name: str, factory: F) -> F: ...
@overload
def register[F: Callable[..., POMDP]](name: str) -> Callable[[F], F]: ...
def register[F: Callable[..., POMDP]](
    name: str, factory: F | None = None
) -> F | Callable[[F], F]:
    """Usable as `@register(name)`."""
    if factory is None:
        return lambda f: register(name, f)
    if name in _REGISTRY:
        raise ValueError(f"{name!r} is already registered")
    _REGISTRY[name] = factory
    return factory


def make(name: str, **kwargs: object) -> POMDP:
    """Build the POMDP registered as `name`; `"module:name"` imports `module` first."""
    module, _, name = name.rpartition(":")
    if module:
        importlib.import_module(module)
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no POMDP registered as {name!r}; registered: {registered()}"
        ) from None
    env = factory(**kwargs)
    if not isinstance(env, POMDP):
        raise TypeError(
            f"factory for {name!r} returned {type(env).__name__}, which is not a POMDP"
        )
    return env


def registered() -> list[str]:
    return sorted(_REGISTRY)
