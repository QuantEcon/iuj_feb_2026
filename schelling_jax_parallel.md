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

# Schelling Model with JAX: Parallel Algorithm

## Overview

In the {doc}`previous lecture <schelling_jax>`, we translated the Schelling model
to JAX, keeping the same sequential algorithm. While that demonstrated JAX syntax
and concepts, it didn't leverage JAX's main strength: **parallel computation**.

The original algorithm updates agents one at a time, with each agent potentially
making many moves until they find a happy location. This sequential structure
doesn't map well to parallel hardware like GPUs.

In this lecture, we redesign the algorithm to be **fully parallelizable**. The
key insight is to update all unhappy agents simultaneously rather than one at a
time.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import time
```

## The Parallel Algorithm

Our modified algorithm works as follows:

```{prf:algorithm} Parallel Schelling Update
:label: parallel_schelling

**Input:** Agent locations, types, random key

**Output:** Updated locations

1. Compute happiness for all agents in parallel
2. Generate one random candidate location for each unhappy agent
3. For each unhappy agent, check if the candidate location would make them happy
   (using current locations of all other agents)
4. Update locations: agents who found a happy spot move; others stay
5. Repeat until no one moves or max iterations reached

```

The critical difference from the original algorithm:

- **Original**: Agents move sequentially. When agent $i$ moves, agent $i+1$ sees
  the updated location.
- **Parallel**: All agents "explore" simultaneously using the same snapshot of
  locations. Then all successful moves happen at once.

This changes the dynamics slightly but makes the algorithm embarrassingly
parallel — perfect for GPUs.

## Parameters

```{code-cell} ipython3
num_of_type_0 = 1000    # number of agents of type 0 (orange)
num_of_type_1 = 1000    # number of agents of type 1 (green)
n = num_of_type_0 + num_of_type_1  # total number of agents
k = 10                  # number of agents regarded as neighbors
require_same_type = 5   # want >= require_same_type neighbors of the same type
```

## Initialization

```{code-cell} ipython3
def initialize_state(key):
    """Initialize agent locations and types."""
    locations = random.uniform(key, shape=(n, 2))
    types = jnp.array([0] * num_of_type_0 + [1] * num_of_type_1)
    return locations, types
```

## Vectorized Helper Functions

The key to parallelization is writing functions that operate on entire arrays
at once, rather than looping through elements.

### Computing All Pairwise Distances

First, we compute a full distance matrix between all pairs of agents:

```{code-cell} ipython3
@jit
def compute_all_distances(locations):
    """
    Compute the distance matrix between all pairs of agents.

    Returns an (n, n) matrix where entry (i, j) is the distance
    between agent i and agent j.
    """
    # locations has shape (n, 2)
    # We want distances[i, j] = ||locations[i] - locations[j]||

    # Expand dimensions for broadcasting:
    # locations[:, None, :] has shape (n, 1, 2)
    # locations[None, :, :] has shape (1, n, 2)
    # Subtraction broadcasts to shape (n, n, 2)
    diff = locations[:, None, :] - locations[None, :, :]
    return jnp.sqrt(jnp.sum(diff ** 2, axis=2))
```

This is a classic JAX pattern: instead of nested loops, we use broadcasting to
compute all pairwise operations at once.

### Finding All Neighbors

```{code-cell} ipython3
@jit
def compute_all_neighbors(distances):
    """
    For each agent, find the indices of their k nearest neighbors.

    Returns an (n, k) array where row i contains the indices of
    agent i's k nearest neighbors (excluding self).
    """
    num_agents = distances.shape[0]
    # Set diagonal to infinity so agents don't count as their own neighbor
    distances_no_self = distances + jnp.diag(jnp.full(num_agents, jnp.inf))
    # argsort each row and take first k
    sorted_indices = jnp.argsort(distances_no_self, axis=1)
    return sorted_indices[:, :k]
```

### Computing Happiness for All Agents

```{code-cell} ipython3
@jit
def compute_all_happiness(neighbors, types):
    """
    Compute happiness for all agents in parallel.

    Returns a boolean array of length n, where entry i is True if
    agent i is happy.
    """
    # neighbors has shape (n, k)
    # types has shape (n,)

    # Get types of all neighbors: shape (n, k)
    neighbor_types = types[neighbors]

    # Compare to each agent's own type: shape (n, k)
    # types[:, None] broadcasts to match neighbor_types
    same_type = neighbor_types == types[:, None]

    # Count same-type neighbors for each agent: shape (n,)
    num_same = jnp.sum(same_type, axis=1)

    return num_same >= require_same_type
```

## Computing Happiness at a New Location

Here's where it gets interesting. We need to check if an agent *would be* happy
at a candidate location, without actually moving them there yet.

```{code-cell} ipython3
@jit
def would_be_happy(agent_idx, candidate_loc, locations, types):
    """
    Check if agent would be happy at candidate_loc.

    This computes distances from candidate_loc to all OTHER agents
    (using their current locations), finds the k nearest, and checks
    if enough are the same type.
    """
    agent_type = types[agent_idx]

    # Distance from candidate location to all agents
    diff = candidate_loc - locations
    distances = jnp.sqrt(jnp.sum(diff ** 2, axis=1))

    # Exclude self (set distance to self to infinity)
    distances = distances.at[agent_idx].set(jnp.inf)

    # Find k nearest neighbors at the candidate location
    neighbor_indices = jnp.argsort(distances)[:k]
    neighbor_types = types[neighbor_indices]

    # Count same-type neighbors
    num_same = jnp.sum(neighbor_types == agent_type)

    return num_same >= require_same_type
```

Now we vectorize this to check all unhappy agents at once:

```{code-cell} ipython3
# vmap over the agent index and candidate location
would_be_happy_batch = vmap(would_be_happy, in_axes=(0, 0, None, None))
```

The `vmap` function is JAX's way of automatically vectorizing a function. Here
we're saying: "run `would_be_happy` for multiple agents in parallel, where
`agent_idx` and `candidate_loc` vary but `locations` and `types` are shared."

## The Parallel Update Step

Now we can write the main update function:

```{code-cell} ipython3
@jit
def parallel_update(locations, types, key):
    """
    Perform one parallel update step.

    Returns:
        new_locations: Updated location array
        num_moved: Number of agents who moved
        key: New random key
    """
    num_agents = locations.shape[0]

    # Step 1: Compute current happiness for all agents
    distances = compute_all_distances(locations)
    neighbors = compute_all_neighbors(distances)
    happy = compute_all_happiness(neighbors, types)

    # Step 2: Generate candidate locations for ALL agents
    # (We'll only use them for unhappy agents, but generating for all
    # is simpler and JAX will optimize away unused computation)
    key, subkey = random.split(key)
    candidates = random.uniform(subkey, shape=(num_agents, 2))

    # Step 3: For efficiency, we check all agents but only unhappy ones matter
    all_indices = jnp.arange(num_agents)
    would_be_happy_at_candidate = would_be_happy_batch(
        all_indices, candidates, locations, types
    )

    # Step 4: Determine who moves
    # An agent moves if: (1) they are unhappy, and (2) candidate makes them happy
    should_move = (~happy) & would_be_happy_at_candidate

    # Step 5: Update locations
    # Use jnp.where to select new or old location for each agent
    new_locations = jnp.where(
        should_move[:, None],  # condition, broadcast to (num_agents, 2)
        candidates,             # if True: use candidate
        locations               # if False: keep current
    )

    num_moved = jnp.sum(should_move)

    return new_locations, num_moved, key
```

## Visualization

```{code-cell} ipython3
def plot_distribution(locations, types, title):
    """Plot the distribution of agents."""
    locations_np = np.asarray(locations)
    types_np = np.asarray(types)

    fig, ax = plt.subplots()
    plot_args = {'markersize': 6, 'alpha': 0.8}
    ax.set_facecolor('azure')
    colors = 'orange', 'green'
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

## The Simulation

```{code-cell} ipython3
def run_simulation(max_iter=1000, seed=1234):
    """
    Run the parallel Schelling simulation.
    """
    key = random.PRNGKey(seed)
    key, init_key = random.split(key)
    locations, types = initialize_state(init_key)

    plot_distribution(locations, types, 'Initial distribution')

    # Run until convergence
    start_time = time.time()
    for iteration in range(max_iter):
        locations, num_moved, key = parallel_update(locations, types, key)
        print(f'Iteration {iteration + 1}: {num_moved} agents moved')

        if num_moved == 0:
            break

    elapsed = time.time() - start_time

    plot_distribution(locations, types, f'After {iteration + 1} iterations')

    if num_moved == 0:
        print(f'Converged in {elapsed:.2f} seconds after {iteration + 1} iterations.')
    else:
        print(f'Did not converge after {max_iter} iterations.')

    return locations, types
```

## Warming Up JAX

```{code-cell} ipython3
# Warm up with a smaller problem
key = random.PRNGKey(42)
test_locations = random.uniform(key, shape=(100, 2))
test_types = jnp.array([0] * 50 + [1] * 50)

# Compile the main functions
distances = compute_all_distances(test_locations)
neighbors = compute_all_neighbors(distances)
happy = compute_all_happiness(neighbors, test_types)

key, subkey = random.split(key)
_ = parallel_update(test_locations, test_types, subkey)

print("JAX functions compiled and ready!")
```

## Results

```{code-cell} ipython3
locations, types = run_simulation()
```

## Performance Comparison

Let's time the parallel update:

```{code-cell} ipython3
%%time
# Set up
key = random.PRNGKey(1234)
key, init_key = random.split(key)
locations, types = initialize_state(init_key)

# Time 10 iterations
for _ in range(10):
    locations, num_moved, key = parallel_update(locations, types, key)
```

The parallel algorithm processes all 2000 agents simultaneously in each
iteration. On a GPU, this would be significantly faster than the sequential
version.

## Trade-offs

The parallel algorithm has different dynamics from the original:

**Advantages:**
- Fully parallelizable — can leverage GPUs effectively
- Predictable iteration time (no variable-length while loops)
- Each iteration does a fixed amount of work

**Differences in dynamics:**
- Agents only get one chance to move per iteration (vs. moving until happy)
- All agents see the same "snapshot" of locations during exploration
- May take more iterations to converge
- Agents who can't find a happy spot in one try must wait for the next iteration

**When this matters:**
- In the original algorithm, early movers can "claim" good spots before later
  agents explore
- In the parallel algorithm, multiple agents might simultaneously try to move
  to similar areas, creating new conflicts

Despite these differences, both algorithms demonstrate the same fundamental
phenomenon: mild preferences lead to strong segregation.

## Summary

By restructuring the Schelling model for parallel execution, we can effectively
leverage JAX's strengths:

- **Vectorized distance computation** using broadcasting
- **Parallel happiness checking** for all agents simultaneously
- **Batched candidate evaluation** using `vmap`
- **Simultaneous location updates** using `jnp.where`

This pattern — restructuring algorithms to operate on entire arrays rather than
individual elements — is the key to getting good performance from JAX and GPUs
in general.
