import copy
import dataclasses
import pickle
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from jax.typing import ArrayLike

from jax_pomdps import Space
from jax_pomdps.spaces import Box, Dict, Discrete, Image

KEYS = jax.random.split(jax.random.key(0), 1000)


def test_spaces_satisfy_protocol():
    assert isinstance(Discrete(3), Space)
    assert isinstance(Box(0.0, 1.0), Space)
    assert isinstance(Image(2, 2), Space)
    assert isinstance(Dict({}), Space)


@pytest.mark.parametrize(
    "space", [Discrete(3), Box(0.0, 1.0), Image(2, 2), Dict({"a": Discrete(2)})]
)
def test_spaces_are_frozen_slotted_dataclasses(space: Space[Any]):
    assert dataclasses.is_dataclass(space)
    assert not hasattr(space, "__dict__")
    field = dataclasses.fields(space)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(space, field, getattr(space, field))


def test_discrete_samples_cover_range_and_are_contained():
    space = Discrete(4)
    samples = jax.vmap(space.sample)(KEYS)
    assert samples.dtype == jnp.int32
    assert set(samples.tolist()) == {0, 1, 2, 3}
    assert jax.vmap(space.contains)(samples).all()


@pytest.mark.parametrize("x", [-1, 4, 1.0, jnp.array([1])])
def test_discrete_rejects_non_elements(x: ArrayLike):
    assert not Discrete(4).contains(x)


def test_discrete_requires_positive_n():
    with pytest.raises(ValueError):
        Discrete(0)


def test_box_broadcasts_bounds_to_shape():
    space = Box(-1.0, [1.0, 2.0])
    assert space.shape == (2,)
    assert Box(-2.0, 2.0, (3, 1)).shape == (3, 1)


def test_box_samples_respect_finite_and_infinite_bounds():
    space = Box([-1.0, 0.0, -np.inf, -np.inf], [1.0, np.inf, 5.0, np.inf])
    samples = jax.vmap(space.sample)(KEYS)
    assert samples.shape == (len(KEYS), 4)
    assert jnp.isfinite(samples).all()
    assert jax.vmap(space.contains)(samples).all()
    assert samples[:, 0].min() < -0.9 and samples[:, 0].max() > 0.9


def test_box_samples_have_box_dtype_with_x64_enabled():
    space = Box([-1.0, 0.0, -np.inf, -np.inf], [1.0, np.inf, 5.0, np.inf])
    with jax.enable_x64(True):
        assert space.sample(KEYS[0]).dtype == space.dtype


@pytest.mark.parametrize(
    "x", [jnp.array([0.0, 2.0]), jnp.array([0.0]), jnp.array([[0.0, 0.0]])]
)
def test_box_rejects_non_elements(x: jax.Array):
    assert not Box(-1.0, [1.0, 1.0]).contains(x)


def test_box_requires_low_at_most_high():
    with pytest.raises(ValueError):
        Box(1.0, 0.0)


@pytest.mark.parametrize("low", [np.nan, [0.0, np.nan]])
def test_box_rejects_nan_bounds(low: npt.ArrayLike):
    with pytest.raises(ValueError):
        Box(low, 1.0)


def test_box_bounds_are_read_only_copies():
    low = np.zeros(2, np.float32)
    space = Box(low, 1.0)
    low[0] = -1.0
    assert space == Box(0.0, 1.0, (2,))
    for box in space, pickle.loads(pickle.dumps(space)), copy.deepcopy(space):
        with pytest.raises(ValueError):
            box.low[0] = -1.0
        with pytest.raises(ValueError):
            box.high[0] = 2.0


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (Box(0.0, 1.0, (2,)), Box([0.0, 0.0], 1.0)),
        (Box(-np.inf, np.inf), Box(-np.inf, np.inf)),
        (Box(-0.0, 1.0), Box(0.0, 1.0)),
    ],
)
def test_equal_boxes_compare_and_hash_equal(a: Box, b: Box):
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
    assert {a: "value"}[b] == "value"


@pytest.mark.parametrize(
    "other",
    [
        Box(0.0, 2.0, (2,)),
        Box(-1.0, 1.0, (2,)),
        Box([0.0, 0.5], 1.0),
        Box(0.0, 1.0, (3,)),
        Box(0.0, 1.0, (2, 1)),
        Box(0.0, 1.0),
        Discrete(2),
    ],
)
def test_different_boxes_compare_unequal(other: Space[Any]):
    assert Box(0.0, 1.0, (2,)) != other
    assert len({Box(0.0, 1.0, (2,)), other}) == 2


def test_box_repr_shows_bounds():
    assert repr(Box(-1.0, [1.0, 2.0])) == "Box(low=[-1.0, -1.0], high=[1.0, 2.0])"


def test_image_samples_cover_uint8_range_and_are_contained():
    space = Image(4, 5, channels=2)
    samples = jax.vmap(space.sample)(KEYS)
    assert samples.shape == (len(KEYS), 4, 5, 2)
    assert samples.dtype == space.dtype
    assert samples.min() == 0 and samples.max() == 255
    assert jax.vmap(space.contains)(samples).all()


@pytest.mark.parametrize(
    "x",
    [
        jnp.zeros((4, 5, 2), jnp.float32),
        jnp.zeros((4, 5, 3), jnp.uint8),
        jnp.zeros((5, 4, 2), jnp.uint8),
    ],
)
def test_image_rejects_non_elements(x: ArrayLike):
    assert not Image(4, 5, channels=2).contains(x)


NESTED = Dict({"pixels": Image(3, 3), "proprio": Dict({"joint": Box(-1.0, 1.0, (2,))})})


def test_dict_samples_are_contained_under_vmap():
    samples = jax.vmap(NESTED.sample)(KEYS[:10])
    assert samples["pixels"].shape == (10, 3, 3, 3)
    assert samples["proprio"]["joint"].shape == (10, 2)
    assert jax.vmap(NESTED.contains)(samples).all()


def test_dict_entries_are_sampled_independently():
    space = Dict({"a": Box(0.0, 1.0, (8,)), "b": Box(0.0, 1.0, (8,))})
    sample = space.sample(KEYS[0])
    assert not jnp.array_equal(sample["a"], sample["b"])


@pytest.mark.parametrize(
    "x",
    [
        {"pixels": jnp.zeros((3, 3, 3), jnp.uint8)},
        {
            "pixels": jnp.zeros((3, 3, 3), jnp.uint8),
            "proprio": {"joint": jnp.zeros(2)},
            "extra": jnp.zeros(()),
        },
        {
            "pixels": jnp.zeros((3, 3, 3), jnp.uint8),
            "proprio": {"joint": jnp.full(2, 2.0)},
        },
    ],
)
def test_dict_rejects_non_elements(x: dict[str, object]):
    assert not NESTED.contains(x)


def test_dict_copies_its_entries_into_a_read_only_mapping():
    entries: dict[str, Space[Any]] = {"a": Discrete(2)}
    space = Dict(entries)
    entries["b"] = Discrete(3)
    assert space == Dict({"a": Discrete(2)})
    with pytest.raises(TypeError):
        space.spaces["b"] = Discrete(3)  # pyright: ignore[reportIndexIssue]


def test_equal_dicts_compare_and_hash_equal():
    copy = Dict(
        {"pixels": Image(3, 3), "proprio": Dict({"joint": Box([-1.0, -1.0], 1.0)})}
    )
    assert copy == NESTED
    assert hash(copy) == hash(NESTED)
    assert len({copy, NESTED}) == 1


@pytest.mark.parametrize(
    "other",
    [
        Dict({"pixels": Image(3, 3)}),
        Dict({"proprio": NESTED.spaces["proprio"], "pixels": Image(3, 3)}),
        Dict({"pixels": Image(3, 4), "proprio": NESTED.spaces["proprio"]}),
    ],
)
def test_dicts_with_different_or_reordered_entries_compare_unequal(other: Dict):
    assert other != NESTED
    assert len({other, NESTED}) == 2


@pytest.mark.parametrize(
    "space", [Discrete(3), Box(-1.0, [1.0, 2.0]), Image(2, 2), NESTED]
)
def test_spaces_survive_pickling_and_deep_copying(space: Space[Any]):
    assert pickle.loads(pickle.dumps(space)) == space
    assert copy.deepcopy(space) == space


def test_dict_repr_shows_entries():
    assert repr(Dict({"a": Discrete(2)})) == "Dict(spaces={'a': Discrete(n=2)})"


@pytest.mark.parametrize(
    ("space", "equal_space", "other_space"),
    [
        (Discrete(3), Discrete(3), Discrete(4)),
        (Box(0.0, 1.0, (2,)), Box([0.0, 0.0], 1.0), Box(0.0, 2.0, (2,))),
        (Image(2, 2), Image(2, 2), Image(2, 3)),
        (
            NESTED,
            Dict({"pixels": Image(3, 3), "proprio": NESTED.spaces["proprio"]}),
            Dict({"pixels": Image(3, 3)}),
        ),
    ],
)
def test_equal_spaces_share_a_jit_trace_as_static_arguments(
    space: Space[Any], equal_space: Space[Any], other_space: Space[Any]
):
    traces = 0

    def sample(space: Space[Any], key: jax.Array) -> Any:
        nonlocal traces
        traces += 1
        return space.sample(key)

    jitted = jax.jit(sample, static_argnums=0)
    jitted(space, KEYS[0])
    jitted(equal_space, KEYS[0])
    assert traces == 1
    jitted(other_space, KEYS[0])
    assert traces == 2
