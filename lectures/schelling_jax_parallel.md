---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Schelling Model: Performance Comparison

## Overview

In the previous lectures, we implemented the Schelling segregation model using:

1. {doc}`NumPy arrays and functions <schelling_numpy>`
2. {doc}`JAX with JIT compilation <schelling_jax>`

In this lecture, we compare these implementations and introduce a **parallel
algorithm** that fully leverages JAX's ability to perform vectorized operations
across all agents simultaneously.

We'll run a "horse race" to see how each approach performs.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from numpy.random import uniform
import time
```

## Parameters

We use the same parameters across all implementations:

```{code-cell} ipython3
num_of_type_0 = 1500
num_of_type_1 = 1500
n = num_of_type_0 + num_of_type_1
k = 10                  # number of neighbors
require_same_type = 5   # threshold for happiness
```

## Shared Plotting Function

```{code-cell} ipython3
def plot_distribution(locations, types, title):
    " Plot the distribution of agents. "
    # Convert to NumPy if needed (for JAX arrays)
    locations_np = np.asarray(locations)
    types_np = np.asarray(types)

    fig, ax = plt.subplots()
    plot_args = {'markersize': 6, 'alpha': 0.8, 'markeredgecolor': 'black', 'markeredgewidth': 0.5}
    colors = 'darkorange', 'green'
    for agent_type, color in zip((0, 1), colors):
        idx = (types_np == agent_type)
        ax.plot(locations_np[idx, 0],
                locations_np[idx, 1],
                'o',
                markerfacecolor=color,
                **plot_args)
    ax.set_title(title)
    plt.show()
```

## NumPy Implementation

First, let's define the NumPy version from {doc}`schelling_numpy`:

```{code-cell} ipython3
def np_initialize_state():
    locations = uniform(size=(n, 2))
    types = np.array([0] * num_of_type_0 + [1] * num_of_type_1)
    return locations, types


def np_get_distances(loc, locations):
    return np.linalg.norm(loc - locations, axis=1)


def np_get_neighbors(i, locations):
    loc = locations[i, :]
    distances = np_get_distances(loc, locations)
    distances[i] = np.inf
    indices = np.argsort(distances)
    return indices[:k]


def np_is_happy(i, locations, types):
    agent_type = types[i]
    neighbors = np_get_neighbors(i, locations)
    neighbor_types = types[neighbors]
    num_same = np.sum(neighbor_types == agent_type)
    return num_same >= require_same_type


def np_update_agent(i, locations, types, max_attempts=10_000):
    attempts = 0
    while not np_is_happy(i, locations, types):
        locations[i, :] = uniform(), uniform()
        attempts += 1
        if attempts >= max_attempts:
            break


def run_numpy_simulation(max_iter=100_000, seed=42):
    np.random.seed(seed)
    locations, types = np_initialize_state()

    plot_distribution(locations, types, 'NumPy: Initial distribution')

    start_time = time.time()
    someone_moved = True
    iteration = 0
    while someone_moved and iteration < max_iter:
        print(f'Entering iteration {iteration + 1}')
        iteration += 1
        someone_moved = False
        for i in range(n):
            if not np_is_happy(i, locations, types):
                np_update_agent(i, locations, types)
                someone_moved = True
    elapsed = time.time() - start_time

    plot_distribution(locations, types, f'NumPy: Iteration {iteration}')

    if not someone_moved:
        print(f'Converged in {elapsed:.2f} seconds after {iteration} iterations.')
    else:
        print('Hit iteration bound and terminated.')

    return locations, types
```

## JAX Sequential Implementation

Next, the JAX version from {doc}`schelling_jax`:

```{code-cell} ipython3
def jax_initialize_state(key):
    locations = random.uniform(key, shape=(n, 2))
    types = jnp.array([0] * num_of_type_0 + [1] * num_of_type_1)
    return locations, types


@jit
def jax_get_distances(loc, locations):
    diff = loc - locations
    return jnp.sum(diff**2, axis=1)


@jit
def jax_get_neighbors(loc, agent_idx, locations):
    distances = jax_get_distances(loc, locations)
    distances = distances.at[agent_idx].set(jnp.inf)
    _, indices = jax.lax.top_k(-distances, k)
    return indices


@jit
def jax_is_unhappy(loc, agent_type, agent_idx, locations, types):
    neighbors = jax_get_neighbors(loc, agent_idx, locations)
    neighbor_types = types[neighbors]
    num_same = jnp.sum(neighbor_types == agent_type)
    return num_same < require_same_type


@jit
def jax_update_agent(i, locations, types, key, max_attempts=10_000):
    loc = locations[i, :]
    agent_type = types[i]

    def cond_fn(state):
        loc, key, attempts = state
        return (attempts < max_attempts) & jax_is_unhappy(loc, agent_type, i, locations, types)

    def body_fn(state):
        _, key, attempts = state
        key, subkey = random.split(key)
        new_loc = random.uniform(subkey, shape=(2,))
        return new_loc, key, attempts + 1

    final_loc, key, _ = jax.lax.while_loop(cond_fn, body_fn, (loc, key, 0))
    return final_loc, key


@jit
def jax_get_unhappy_agents(locations, types):
    def check_agent(i):
        return jax_is_unhappy(locations[i], types[i], i, locations, types)

    all_unhappy = vmap(check_agent)(jnp.arange(n))
    indices = jnp.where(all_unhappy, size=n, fill_value=-1)[0]
    count = jnp.sum(all_unhappy)
    return indices, count


def jax_simulation_loop(locations, types, key, max_iter):
    iteration = 0
    while iteration < max_iter:
        print(f'Entering iteration {iteration + 1}')
        iteration += 1

        unhappy, num_unhappy = jax_get_unhappy_agents(locations, types)

        if num_unhappy == 0:
            break

        for j in range(int(num_unhappy)):
            i = int(unhappy[j])
            new_loc, key = jax_update_agent(i, locations, types, key)
            locations = locations.at[i, :].set(new_loc)

    return locations, iteration, key


def run_jax_simulation(max_iter=100_000, seed=42):
    key = random.PRNGKey(seed)
    key, init_key = random.split(key)
    locations, types = jax_initialize_state(init_key)

    plot_distribution(locations, types, 'JAX Sequential: Initial distribution')

    start_time = time.time()
    locations, iteration, key = jax_simulation_loop(locations, types, key, max_iter)
    elapsed = time.time() - start_time

    plot_distribution(locations, types, f'JAX Sequential: Iteration {iteration}')

    if iteration < max_iter:
        print(f'Converged in {elapsed:.2f} seconds after {iteration} iterations.')
    else:
        print('Hit iteration bound and terminated.')

    return locations, types
```

## JAX Parallel Implementation

Now we introduce the parallel algorithm. The key insight is that instead of
updating agents one at a time, we can:

1. **Identify all unhappy agents** in parallel
2. **Generate candidate locations** for all unhappy agents in parallel
3. **Test happiness** at all candidate locations in parallel
4. **Update all agents** simultaneously

This approach is well-suited to GPU execution, where thousands of operations
can run concurrently.

### The Trade-off

The sequential algorithm guarantees that each agent finds a happy location
before moving on. The parallel algorithm instead proposes a fixed number of
candidate locations per agent per iteration. If none of the candidates make
the agent happy, the agent stays put and tries again next iteration.

This means the parallel algorithm may need more iterations, but each iteration
is much faster because all work is done in parallel.

```{code-cell} ipython3
num_candidates = 3  # number of candidate locations to try per agent per iteration
```

### Core Parallel Functions

```{code-cell} ipython3
@jit
def find_happy_candidate(i, locations, types, key):
    """
    Propose num_candidates random locations for agent i.
    Return the first one where agent is happy, or current location if none work.
    """
    current_loc = locations[i, :]
    agent_type = types[i]

    # Generate num_candidates random locations
    keys = random.split(key, num_candidates)
    candidates = vmap(lambda k: random.uniform(k, shape=(2,)))(keys)

    # Check happiness at each candidate location (in parallel)
    def check_candidate(loc):
        return ~jax_is_unhappy(loc, agent_type, i, locations, types)

    happy_at_candidates = vmap(check_candidate)(candidates)

    # Find first happy candidate (or -1 if none)
    first_happy_idx = jnp.argmax(happy_at_candidates)
    any_happy = jnp.any(happy_at_candidates)

    # Return first happy candidate, or current location if none are happy
    new_loc = jnp.where(any_happy, candidates[first_happy_idx], current_loc)
    return new_loc


@jit
def parallel_update_step(locations, types, key):
    """
    One step of the parallel algorithm:
    1. Generate keys for all agents
    2. For each agent, find a happy candidate location (in parallel)
    3. Only update unhappy agents
    """
    # Generate keys for all agents
    keys = random.split(key, n + 1)
    key = keys[0]
    agent_keys = keys[1:]

    # For each agent, find a happy candidate location (in parallel)
    def try_move(i):
        return find_happy_candidate(i, locations, types, agent_keys[i])

    new_locations = vmap(try_move)(jnp.arange(n))

    # Only update unhappy agents
    def check_agent(i):
        return jax_is_unhappy(locations[i], types[i], i, locations, types)

    is_unhappy_mask = vmap(check_agent)(jnp.arange(n))

    # Keep old location for happy agents, use new for unhappy
    final_locations = jnp.where(is_unhappy_mask[:, None], new_locations, locations)

    return final_locations, key
```

### Parallel Simulation Loop

```{code-cell} ipython3
def parallel_simulation_loop(locations, types, key, max_iter):
    iteration = 0
    while iteration < max_iter:
        print(f'Entering iteration {iteration + 1}')
        iteration += 1

        _, num_unhappy = jax_get_unhappy_agents(locations, types)

        if num_unhappy == 0:
            break

        locations, key = parallel_update_step(locations, types, key)

    return locations, iteration, key


def run_parallel_simulation(max_iter=100_000, seed=42):
    key = random.PRNGKey(seed)
    key, init_key = random.split(key)
    locations, types = jax_initialize_state(init_key)

    plot_distribution(locations, types, 'JAX Parallel: Initial distribution')

    start_time = time.time()
    locations, iteration, key = parallel_simulation_loop(locations, types, key, max_iter)
    elapsed = time.time() - start_time

    plot_distribution(locations, types, f'JAX Parallel: Iteration {iteration}')

    if iteration < max_iter:
        print(f'Converged in {elapsed:.2f} seconds after {iteration} iterations.')
    else:
        print('Hit iteration bound and terminated.')

    return locations, types
```

## Warming Up JAX

Before timing, we compile all JAX functions:

```{code-cell} ipython3
key = random.PRNGKey(0)
key, init_key = random.split(key)
test_locations, test_types = jax_initialize_state(init_key)

# Warm up JAX sequential functions
_ = jax_get_distances(test_locations[0], test_locations)
_ = jax_get_neighbors(test_locations[0], 0, test_locations)
_ = jax_is_unhappy(test_locations[0], test_types[0], 0, test_locations, test_types)
_, _ = jax_get_unhappy_agents(test_locations, test_types)
key, subkey = random.split(key)
_, _ = jax_update_agent(0, test_locations, test_types, subkey)

# Warm up JAX parallel functions
key, subkey = random.split(key)
_ = find_happy_candidate(0, test_locations, test_types, subkey)
key, subkey = random.split(key)
_, _ = parallel_update_step(test_locations, test_types, subkey)

print("JAX functions compiled and ready!")
```

## The Horse Race

Now let's run all three implementations and compare their performance.

### NumPy

```{code-cell} ipython3
print("=" * 50)
print("NUMPY")
print("=" * 50)
locations_np, types_np = run_numpy_simulation()
```

### JAX Sequential

```{code-cell} ipython3
print("=" * 50)
print("JAX SEQUENTIAL")
print("=" * 50)
locations_jax, types_jax = run_jax_simulation()
```

### JAX Parallel

```{code-cell} ipython3
print("=" * 50)
print("JAX PARALLEL")
print("=" * 50)
locations_par, types_par = run_parallel_simulation()
```

## Discussion

The results reveal interesting trade-offs:

1. **NumPy** provides a straightforward implementation but runs entirely on
   the CPU with Python loops.

2. **JAX Sequential** uses JIT compilation for individual operations, but the
   outer loop still processes agents one at a time.

3. **JAX Parallel** processes all agents simultaneously each iteration. While
   it may require more iterations (since agents might not find a happy location
   in their limited candidates), each iteration leverages massive parallelism.

The parallel approach shines on GPUs, where thousands of threads can evaluate
candidate locations concurrently. On CPUs, the benefits are more modest, but
the parallel structure still allows JAX to optimize memory access patterns and
use SIMD instructions effectively.

## Key Takeaways

1. **Algorithm structure matters**: Simply porting code to JAX doesn't
   automatically make it faster. To fully benefit from JAX's capabilities,
   algorithms often need to be restructured for parallelism.

2. **Trade iteration count for parallelism**: The parallel algorithm may need
   more iterations, but each iteration does more work in parallel. This
   trade-off often favors parallelism on modern hardware.

3. **GPU acceleration**: The parallel algorithm is particularly well-suited
   for GPUs, where the speedup can be dramatic. On CPU-only systems, the
   difference is smaller.
