import pathlib
import sys
from collections.abc import Iterator
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

import jax_pomdps
from jax_pomdps import Key, pomdp
from jax_pomdps.spaces import Box, Discrete


@dataclass(frozen=True, slots=True)
class Counter:
    """Adds `increment` to the state on action 1."""

    increment: float = 1.0

    @property
    def action_space(self) -> Discrete:
        return Discrete(2)

    @property
    def observation_space(self) -> Box:
        return Box(0.0, jnp.inf)

    def reset(self, key: Key) -> jax.Array:
        return jnp.zeros(())

    def step(self, key: Key, state: jax.Array, action: jax.Array) -> jax.Array:
        return state + self.increment * action

    def reward(
        self, key: Key, state: jax.Array, action: jax.Array, next_state: jax.Array
    ) -> jax.Array:
        return next_state - state

    def observe(self, key: Key, next_state: jax.Array, action: jax.Array) -> jax.Array:
        return next_state

    def done(self, state: jax.Array) -> jax.Array:
        return jnp.zeros((), bool)

    def render(self, state: jax.Array) -> str:
        return str(float(state))


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register `counter` in a registry private to the test."""
    monkeypatch.setattr(pomdp, "_REGISTRY", {})
    jax_pomdps.register("counter", Counter)


@pytest.fixture
def extra_envs(
    registry: None, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Write an unimported module `extra_envs` that registers `extra_counter`."""
    (tmp_path / "extra_envs.py").write_text(
        "import jax_pomdps\nfrom test_pomdp import Counter\n"
        "jax_pomdps.register('extra_counter', Counter)\n"
    )
    monkeypatch.syspath_prepend(tmp_path)
    yield
    sys.modules.pop("extra_envs", None)


def test_counter_satisfies_protocols():
    assert isinstance(Counter(), jax_pomdps.POMDP)
    assert isinstance(Counter(), jax_pomdps.Renderable)


def test_make_returns_registered_pomdp_with_kwargs(registry: None):
    assert jax_pomdps.make("counter", increment=2.0) == Counter(increment=2.0)
    assert jax_pomdps.registered() == ["counter"]


def test_make_unknown_name_raises(registry: None):
    with pytest.raises(KeyError, match="no-such-env"):
        jax_pomdps.make("no-such-env")


def test_register_duplicate_name_raises(registry: None):
    with pytest.raises(ValueError, match="counter"):
        jax_pomdps.register("counter", Counter)


def test_make_rejects_factory_that_is_not_a_pomdp(registry: None):
    jax_pomdps.register("not-a-pomdp", object)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="not-a-pomdp"):
        jax_pomdps.make("not-a-pomdp")


def test_make_imports_module_before_lookup(extra_envs: None):
    env = jax_pomdps.make("extra_envs:extra_counter", increment=2.0)
    assert env == Counter(increment=2.0)


def test_make_with_module_still_requires_registration(extra_envs: None):
    with pytest.raises(KeyError, match="not_registered"):
        jax_pomdps.make("extra_envs:not_registered")


def test_make_with_missing_module_raises():
    with pytest.raises(ModuleNotFoundError, match="no_such_module"):
        jax_pomdps.make("no_such_module:counter")
