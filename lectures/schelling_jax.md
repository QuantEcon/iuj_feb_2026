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

# Schelling Model with JAX

## Overview

In the {doc}`previous lecture <schelling_numpy>`, we rewrote our Schelling model
using NumPy arrays and functions.

In this lecture, we explore [JAX](https://github.com/google/jax), a library
developed by Google for high-performance numerical computing.

JAX offers several powerful features:

1. **GPU/TPU acceleration** — JAX can run your code on GPUs and TPUs with
   minimal changes
2. **Automatic differentiation** — JAX can compute gradients automatically
   (useful for machine learning)
3. **Functional programming style** — JAX encourages pure functions without
   side effects
4. **Just-in-time compilation** — JAX can compile functions for faster execution

Let's start with some imports:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import time
```

## How JAX Differs from NumPy

Before diving into the code, let's understand what makes JAX special.

### Immutable Arrays

In NumPy, we often modify arrays in place:

```python
# NumPy style (mutable)
locations[i, :] = new_location  # modifies the array
```

JAX arrays are **immutable** — they cannot be modified after creation. Instead,
you create new arrays:

```python
# JAX style (immutable)
locations = locations.at[i, :].set(new_location)  # returns a new array
```

This might seem inefficient, but JAX's compiler can optimize these operations,
often avoiding unnecessary copies.

### Functional Programming

JAX works best with **pure functions** — functions that:

1. Always return the same output for the same input
2. Don't modify any external state (no side effects)

This style makes code easier to reason about and enables JAX's powerful
optimizations.

### Random Numbers

NumPy's random number generator maintains hidden internal state. JAX takes a
different approach: you explicitly manage random "keys":

```python
# NumPy style
np.random.seed(42)
x = np.random.uniform()  # uses hidden state

# JAX style
key = random.PRNGKey(42)       # create a key
x = random.uniform(key)        # pass key explicitly
key, subkey = random.split(key)  # get new keys for future use
```

This explicit handling makes JAX programs reproducible and parallelizable.

## Parameters

We use the same parameters as before:

```{code-cell} ipython3
num_of_type_0 = 1000    # number of agents of type 0 (orange)
num_of_type_1 = 1000    # number of agents of type 1 (green)
n = num_of_type_0 + num_of_type_1  # total number of agents
k = 10                  # number of agents regarded as neighbors
require_same_type = 5   # want >= require_same_type neighbors of the same type
```

## Initialization

Here's our initialization function. Note that we use `jax.random` instead of
`numpy.random`:

```{code-cell} ipython3
def initialize_state(key):
    """
    Initialize agent locations and types.

    """
    locations = random.uniform(key, shape=(n, 2))
    types = jnp.array([0] * num_of_type_0 + [1] * num_of_type_1)
    return locations, types
```

The key difference from NumPy is that we pass a `key` argument to
`random.uniform`. This makes the random generation deterministic and
reproducible.

## JAX-Compiled Functions

Now let's rewrite our core functions for JAX. We add the `@jit` decorator
to compile functions for faster execution.

### Computing Distances

```{code-cell} ipython3
@jit
def get_distances(loc, locations):
    """
    Compute squared Euclidean distance from one location to all agent locations.

    """
    diff = loc - locations  # broadcasting: (2,) - (n, 2) -> (n, 2)
    return jnp.sum(diff**2, axis=1)
```

Notice that we use vectorized operations like in NumPy. JAX compiles these
vectorized operations very efficiently, especially when running on GPUs.

We use `jnp` (JAX NumPy) instead of `np` (NumPy). The functions are similar,
but `jnp` operations return JAX arrays and can be compiled by JAX's JIT
compiler.

### Finding Neighbors

```{code-cell} ipython3
@jit
def get_neighbors(loc, agent_idx, locations):
    """
    Get indices of the k nearest neighbors to a location (excluding agent_idx).

    Parameters
    ----------
    loc : array of shape (2,)
        The location to find neighbors for.
    agent_idx : int
        The index of the agent (excluded from neighbors).
    locations : array of shape (n, 2)
        All agent locations.
    """
    distances = get_distances(loc, locations)
    # Set self-distance to infinity so agent doesn't count as own neighbor
    distances = distances.at[agent_idx].set(jnp.inf)
    # Use top_k on negated distances to find k smallest in O(n) instead of O(n log n)
    _, indices = jax.lax.top_k(-distances, k)
    return indices
```

Note that we use `distances.at[i].set(jnp.inf)` instead of `distances[i] = jnp.inf`
because JAX arrays are immutable. This returns a new array with the value at
index `i` set to infinity.

### Checking Unhappiness

```{code-cell} ipython3
@jit
def is_unhappy(loc, agent_type, agent_idx, locations, types):
    """
    True if an agent at loc would NOT have enough same-type neighbors.

    Parameters
    ----------
    loc : array of shape (2,)
        The location to test.
    agent_type : int
        The type of the agent (0 or 1).
    agent_idx : int
        The index of the agent (excluded from neighbor calculation).
    locations : array of shape (n, 2)
        All agent locations.
    types : array of shape (n,)
        All agent types.
    """
    neighbors = get_neighbors(loc, agent_idx, locations)
    neighbor_types = types[neighbors]
    num_same = jnp.sum(neighbor_types == agent_type)
    return num_same < require_same_type
```

This function takes the location and type as explicit arguments, rather than
looking them up from the arrays. This design allows us to test hypothetical
locations without modifying the `locations` array — useful when an agent is
searching for a new location.


### Moving Unhappy Agents

This function finds a location where the agent would be happy. Rather than
updating the `locations` array on each iteration, it tests candidate locations
directly and returns only the final location.

```{code-cell} ipython3
@jit
def update_agent(i, locations, types, key, max_attempts=10_000):
    """
    Find a location where agent i is happy.

    Returns the new location and updated random key. The calling code
    is responsible for updating the locations array if the agent moved.
    """
    loc = locations[i, :]
    agent_type = types[i]

    def cond_fn(state):
        loc, key, attempts = state
        return (attempts < max_attempts) & is_unhappy(loc, agent_type, i, locations, types)

    def body_fn(state):
        _, key, attempts = state
        key, subkey = random.split(key)
        new_loc = random.uniform(subkey, shape=(2,))
        return new_loc, key, attempts + 1

    final_loc, key, _ = jax.lax.while_loop(cond_fn, body_fn, (loc, key, 0))
    return final_loc, key
```

Let's break down the key JAX concepts here:

1. **`jax.lax.while_loop`**: Takes three arguments:
   - `cond_fn(state)` — returns True to continue looping, False to stop
   - `body_fn(state)` — executes one iteration, returns new state
   - `(loc, key)` — initial state (a tuple containing location and random key)

2. **`random.split(key)`**: Since JAX random numbers are deterministic, we
   need to "split" the key to get new randomness. Each split produces two new
   keys: one to use now, one to save for later.

3. **Testing without updating**: By passing `loc` directly to `is_unhappy`, we
   can test candidate locations without modifying the `locations` array. This
   avoids creating new arrays inside the loop, improving efficiency.

## Visualization

Plotting uses Matplotlib, which works with regular NumPy arrays. We convert
JAX arrays to NumPy arrays using `np.asarray()`:

```{code-cell} ipython3
def plot_distribution(locations, types, title):
    """
    Plot the distribution of agents.
    """
    # Convert JAX arrays to NumPy for matplotlib
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

## The Simulation

We separate the core simulation loop from the setup and plotting code. This
makes it easier to optimize or JIT-compile the loop independently.

```{code-cell} ipython3
@jit
def get_unhappy_agents(locations, types):
    """
    Find indices and count of all unhappy agents using vectorized computation.
    """
    def check_agent(i):
        return is_unhappy(locations[i], types[i], i, locations, types)

    all_unhappy = vmap(check_agent)(jnp.arange(n))
    indices = jnp.where(all_unhappy, size=n, fill_value=-1)[0]
    count = jnp.sum(all_unhappy)
    return indices, count


def simulation_loop(locations, types, key, max_iter):
    """
    Run the simulation loop until convergence or max iterations.

    Returns
    -------
    locations : array
        Final agent locations.
    iteration : int
        Number of iterations completed.
    key : PRNGKey
        Updated random key.
    """
    iteration = 0
    while iteration < max_iter:
        print(f'Entering iteration {iteration + 1}')
        iteration += 1

        # Find unhappy agents using vectorized computation
        unhappy, num_unhappy = get_unhappy_agents(locations, types)

        # Check if everyone is happy
        if num_unhappy == 0:
            break

        # Update only the unhappy agents
        for j in range(int(num_unhappy)):
            i = int(unhappy[j])
            new_loc, key = update_agent(i, locations, types, key)
            locations = locations.at[i, :].set(new_loc)

    return locations, iteration, key
```

```{code-cell} ipython3
def run_simulation(max_iter=100_000, seed=42):
    """
    Run the Schelling simulation using JAX.
    """
    key = random.PRNGKey(seed)
    key, init_key = random.split(key)
    locations, types = initialize_state(init_key)

    plot_distribution(locations, types, 'Initial distribution')

    start_time = time.time()
    locations, iteration, key = simulation_loop(locations, types, key, max_iter)
    elapsed = time.time() - start_time

    plot_distribution(locations, types, f'Iteration {iteration}')

    if iteration < max_iter:
        print(f'Converged in {elapsed:.2f} seconds after {iteration} iterations.')
    else:
        print('Hit iteration bound and terminated.')

    return locations, types
```

The simulation loop differs from the NumPy version in several ways:

1. **Vectorized unhappiness check**: We use `get_unhappy_agents` to identify all
   unhappy agents in parallel, then only process those agents
2. We pass and receive the random `key` in each call to `update_agent`
3. `update_agent` returns the new location, not the whole array
4. We only update `locations` when an agent actually moves

This hybrid approach uses vectorized computation to identify unhappy agents,
then processes them sequentially. As the simulation progresses and more agents
become happy, fewer agents need processing each iteration.

## Warming Up JAX

JAX compiles functions the first time they're called. Let's warm up the
functions:

```{code-cell} ipython3
# Warm up: use actual problem size to trigger compilation
# (JAX recompiles when array shapes change)
key = random.PRNGKey(42)
key, init_key = random.split(key)
test_locations, test_types = initialize_state(init_key)

# Call each function once to compile it
_ = get_distances(test_locations[0], test_locations)
_ = get_neighbors(test_locations[0], 0, test_locations)
_ = is_unhappy(test_locations[0], test_types[0], 0, test_locations, test_types)
_, _ = get_unhappy_agents(test_locations, test_types)
key, subkey = random.split(key)
_, _ = update_agent(0, test_locations, test_types, subkey)

print("JAX functions compiled and ready!")
```

## Results

Now let's run the simulation:

```{code-cell} ipython3
locations, types = run_simulation()
```

## Tips for Using JAX

1. **Think functionally**: Write pure functions that don't modify external
   state. This makes your code easier to JIT-compile and parallelize.

2. **Use `jnp` instead of `np`**: Replace NumPy operations with their JAX
   equivalents. Most functions have the same names.

3. **Manage random keys explicitly**: Always split keys before generating
   random numbers. Never reuse the same key.

4. **Use JAX's loop constructs**: Replace Python `for` and `while` loops with
   `jax.lax.fori_loop` and `jax.lax.while_loop` inside JIT-compiled functions.

5. **Remember immutability**: Use `.at[].set()` to "update" arrays. The
   original array is never modified.

6. **Warm up before timing**: Always call your functions once before measuring
   performance to exclude compilation time.

## Limitations of This Approach

While this lecture demonstrated JAX syntax and concepts, the algorithm itself
doesn't fully leverage JAX's parallel capabilities. The original Schelling
algorithm has inherent sequential dependencies:

- Agents update one at a time
- Each agent's move changes the state for subsequent agents
- The "move until happy" while loop has unpredictable length

These characteristics don't map well to parallel hardware like GPUs, which
excel at performing the same operation on many data points simultaneously.


## Summary

JAX provides a powerful framework for accelerating Python code. Its key features
are:

- **Immutable arrays** that encourage functional programming
- **Explicit random key management** for reproducibility
- **JIT compilation** via the `@jit` decorator
- **GPU/TPU support** for hardware acceleration

JAX's functional style and unique capabilities like automatic differentiation
and seamless GPU acceleration make it particularly valuable for machine learning
and large-scale numerical computing.

To fully benefit from these capabilities, algorithms often need to be
restructured for parallelism — as we'll see in the next lecture.
