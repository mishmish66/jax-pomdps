from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, Self, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax.typing import ArrayLike

from jax_pomdps._types import Key


@runtime_checkable
class Space[T = jax.Array](Protocol):
    """A set of pytrees."""

    def sample(self, key: Key) -> T: ...

    def contains(self, x: T) -> jax.Array: ...


@dataclass(frozen=True, slots=True)
class Discrete:
    """The integers {0, …, n - 1}."""

    n: int

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be positive, got {self.n}")

    @property
    def shape(self) -> tuple[()]:
        return ()

    @property
    def dtype(self) -> np.dtype[np.int32]:
        return np.dtype(np.int32)

    def sample(self, key: Key) -> jax.Array:
        return jax.random.randint(key, (), 0, self.n, dtype=self.dtype)

    def contains(self, x: ArrayLike) -> jax.Array:
        x = jnp.asarray(x)
        if x.shape != () or not jnp.issubdtype(x.dtype, jnp.integer):
            return jnp.asarray(False)
        return (x >= 0) & (x < self.n)


@dataclass(frozen=True, slots=True, init=False)
class Box:
    """The float32 arrays x with low ≤ x ≤ high elementwise; bounds may be infinite."""

    low: npt.NDArray[np.float32]
    high: npt.NDArray[np.float32]

    def __init__(
        self,
        low: npt.ArrayLike,
        high: npt.ArrayLike,
        shape: tuple[int, ...] | None = None,
    ) -> None:
        if shape is None:
            shape = np.broadcast_shapes(np.shape(low), np.shape(high))
        low = np.broadcast_to(np.array(low, np.float32), shape)
        high = np.broadcast_to(np.array(high, np.float32), shape)
        if not np.all(low <= high):
            raise ValueError(f"low must not exceed high, got low={low}, high={high}")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def __repr__(self) -> str:
        return f"Box(low={self.low.tolist()}, high={self.high.tolist()})"

    def __reduce__(
        self,
    ) -> tuple[type[Self], tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]:
        return type(self), (self.low, self.high)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Box):
            return NotImplemented
        return (
            self.shape == other.shape
            and np.array_equal(self.low, other.low)
            and np.array_equal(self.high, other.high)
        )

    def __hash__(self) -> int:
        return hash((self.shape, tuple(self.low.flat), tuple(self.high.flat)))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.low.shape

    @property
    def dtype(self) -> np.dtype[np.float32]:
        return np.dtype(np.float32)

    def sample(self, key: Key) -> jax.Array:
        """Uniform on bounded axes, bound ± exponential on half-bounded, else normal."""
        uniform_key, exponential_key, normal_key = jax.random.split(key, 3)
        has_low, has_high = np.isfinite(self.low), np.isfinite(self.high)
        bounded = has_low & has_high
        uniform = jax.random.uniform(
            uniform_key,
            self.shape,
            self.dtype,
            minval=np.where(bounded, self.low, 0),
            maxval=np.where(bounded, self.high, 1),
        )
        exponential = jax.random.exponential(exponential_key, self.shape, self.dtype)
        normal = jax.random.normal(normal_key, self.shape, self.dtype)
        return jnp.select(
            [bounded, has_low, has_high],
            [uniform, self.low + exponential, self.high - exponential],
            normal,
        )

    def contains(self, x: ArrayLike) -> jax.Array:
        x = jnp.asarray(x)
        if x.shape != self.shape:
            return jnp.asarray(False)
        return jnp.all((self.low <= x) & (x <= self.high))


@dataclass(frozen=True, slots=True)
class Image:
    """uint8 images of shape (height, width, channels)."""

    height: int
    width: int
    channels: int = 3

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.channels)

    @property
    def dtype(self) -> np.dtype[np.uint8]:
        return np.dtype(np.uint8)

    def sample(self, key: Key) -> jax.Array:
        return jax.random.bits(key, self.shape, jnp.uint8)

    def contains(self, x: ArrayLike) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.asarray(x.shape == self.shape and x.dtype == self.dtype)


@dataclass(frozen=True, slots=True)
class Dict:
    """Dicts whose entries are elements of the same-named `spaces`."""

    spaces: Mapping[str, Space[Any]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "spaces", MappingProxyType(dict(self.spaces)))

    def __repr__(self) -> str:
        return f"Dict(spaces={dict(self.spaces)})"

    def __reduce__(self) -> tuple[type[Self], tuple[dict[str, Space[Any]]]]:
        return type(self), (dict(self.spaces),)

    def __eq__(self, other: object) -> bool:
        """Entry order counts, since it determines `sample`."""
        if not isinstance(other, Dict):
            return NotImplemented
        return tuple(self.spaces.items()) == tuple(other.spaces.items())

    def __hash__(self) -> int:
        return hash(tuple(self.spaces.items()))

    def sample(self, key: Key) -> dict[str, Any]:
        keys = jax.random.split(key, len(self.spaces))
        return {
            name: space.sample(k)
            for (name, space), k in zip(self.spaces.items(), keys, strict=True)
        }

    def contains(self, x: Mapping[str, Any]) -> jax.Array:
        if x.keys() != self.spaces.keys():
            return jnp.asarray(False)
        return jnp.array(
            [space.contains(x[name]) for name, space in self.spaces.items()], bool
        ).all()
