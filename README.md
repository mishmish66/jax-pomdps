# jax-pomdps

POMDPs in JAX, after Kaelbling, Littman & Cassandra (1998): every method is a pure
function of pytrees, safe under `jit` and `vmap`. There is no initial observation;
observations follow actions.

```python
import jax
import jax_pomdps

env = jax_pomdps.make("jax_gym:lunar-lander")  # imports jax_gym first

keys = jax.random.split(jax.random.key(42), 5)
state = env.reset(keys[0])  # s₀ ~ b₀
action = env.action_space.sample(keys[1])
next_state = env.step(keys[2], state, action)  # s' ~ T(s, a, ·)
obs = env.observe(keys[3], next_state, action)  # o ~ O(s', a, ·)
reward = env.reward(keys[4], state, action, next_state)  # r ~ r(· | s, a, s')
done = env.done(next_state)
```

Spaces are `Discrete`, `Box`, `Image` and `Dict`, each with `sample(key)` and
`contains(x)`. Environments register with `jax_pomdps.register(name, factory)`.

## Rendering

Environments that implement `jax_pomdps.Renderable` draw a state:

```python
image = env.render(state)
```

## References

- Leslie Pack Kaelbling, Michael L. Littman and Anthony R. Cassandra, "Planning and
  acting in partially observable stochastic domains", *Artificial Intelligence*
  101(1–2):99–134, 1998. [doi:10.1016/S0004-3702(98)00023-X](https://doi.org/10.1016/S0004-3702(98)00023-X)
- Mark Towers et al., "Gymnasium: A Standard Interface for Reinforcement Learning
  Environments", 2024. [arXiv:2407.17032](https://arxiv.org/abs/2407.17032). The space
  types and `make("module:name")` follow Gymnasium's interface.
- Greg Brockman et al., "OpenAI Gym", 2016. [arXiv:1606.01540](https://arxiv.org/abs/1606.01540)
