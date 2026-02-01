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

In the {doc}`previous lecture <schelling_numba>`, we used Numba to accelerate our
Schelling model by compiling Python code to machine code.

In this lecture, we explore [JAX](https://github.com/google/jax), a library
developed by Google that takes a different approach to high-performance
computing.

JAX offers several unique features:

1. **GPU/TPU acceleration** — JAX can run your code on GPUs and TPUs with
   minimal changes
2. **Automatic differentiation** — JAX can compute gradients automatically
   (useful for machine learning)
3. **Functional programming style** — JAX encourages pure functions without
   side effects
4. **Just-in-time compilation** — Like Numba, JAX can compile functions for
   faster execution

Let's start with some imports:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit
import time
```

## How JAX Differs from NumPy and Numba

Before diving into the code, let's understand what makes JAX special.

### Immutable Arrays

In NumPy and Numba, we often modify arrays in place:

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

    Parameters
    ----------
    key : JAX PRNGKey
        Random key for reproducibility.

    Returns
    -------
    locations : jnp.ndarray of shape (n, 2)
        Initial (x, y) coordinates of all agents.
    types : jnp.ndarray of length n
        Type (0 or 1) of each agent.
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
(similar to Numba's `@njit`) to compile functions for faster execution.

### Computing Distances

```{code-cell} ipython3
@jit
def get_distances(loc, locations):
    """
    Compute the Euclidean distance from one location to all agent locations.

    Unlike the Numba version, we use vectorized operations here because JAX
    excels at compiling array operations, especially on GPUs.
    """
    diff = loc - locations  # broadcasting: (2,) - (n, 2) -> (n, 2)
    return jnp.sqrt(jnp.sum(diff ** 2, axis=1))
```

Notice that we use vectorized operations like in NumPy, rather than explicit
loops like in Numba. JAX compiles these vectorized operations very efficiently,
especially when running on GPUs.

We use `jnp` (JAX NumPy) instead of `np` (NumPy). The functions are similar,
but `jnp` operations return JAX arrays and can be compiled by JAX's JIT
compiler.

### Finding Neighbors

```{code-cell} ipython3
@jit
def get_neighbors(i, locations):
    """
    Get indices of the k nearest neighbors to agent i (excluding self).
    """
    loc = locations[i, :]
    distances = get_distances(loc, locations)
    # Set self-distance to infinity so we don't count ourselves as a neighbor
    distances = distances.at[i].set(jnp.inf)
    # jnp.argsort works just like np.argsort
    indices = jnp.argsort(distances)
    return indices[:k]
```

Note that we use `distances.at[i].set(jnp.inf)` instead of `distances[i] = jnp.inf`
because JAX arrays are immutable. This returns a new array with the value at
index `i` set to infinity.

### Checking Happiness

```{code-cell} ipython3
@jit
def is_happy(i, locations, types):
    """
    True if agent i has at least require_same_type neighbors of the same type.
    """
    agent_type = types[i]
    neighbors = get_neighbors(i, locations)
    neighbor_types = types[neighbors]
    num_same = jnp.sum(neighbor_types == agent_type)
    return num_same >= require_same_type
```

This looks almost identical to the NumPy version. The main difference is using
`jnp` instead of `np`.

### Counting Happy Agents

```{code-cell} ipython3
@jit
def count_happy(locations, types):
    """
    Count the number of happy agents.

    We use jax.lax.fori_loop instead of a Python for loop. This is necessary
    because JAX's JIT compiler needs to know the loop structure at compile time.
    """
    def body_fn(i, count):
        return count + is_happy(i, locations, types)

    return jax.lax.fori_loop(0, n, body_fn, 0)
```

Here we encounter something new: `jax.lax.fori_loop`. In JAX, regular Python
`for` loops don't work well inside JIT-compiled functions. Instead, JAX
provides special loop constructs:

- `jax.lax.fori_loop(start, stop, body_fn, init_val)` — like `for i in
  range(start, stop)`
- `jax.lax.while_loop(cond_fn, body_fn, init_val)` — like `while condition:`

These constructs can be compiled and optimized by JAX.

### Moving Unhappy Agents

This function is more complex because it needs to:
1. Use a while loop (agent keeps moving until happy)
2. Handle random number generation
3. Update the locations array (which is immutable in JAX)

```{code-cell} ipython3
@jit
def update_agent(i, locations, types, key):
    """
    Move agent i to a new location where they are happy.

    Returns the updated locations array and a new random key.
    """
    def cond_fn(state):
        locations, key = state
        return ~is_happy(i, locations, types)  # ~ is logical NOT

    def body_fn(state):
        locations, key = state
        key, subkey = random.split(key)
        new_loc = random.uniform(subkey, shape=(2,))
        # Create new array with updated location (JAX arrays are immutable)
        locations = locations.at[i, :].set(new_loc)
        return locations, key

    locations, key = jax.lax.while_loop(cond_fn, body_fn, (locations, key))
    return locations, key
```

Let's break down the key JAX concepts here:

1. **`jax.lax.while_loop`**: Takes three arguments:
   - `cond_fn(state)` — returns True to continue looping, False to stop
   - `body_fn(state)` — executes one iteration, returns new state
   - `(locations, key)` — initial state (a tuple)

2. **`random.split(key)`**: Since JAX random numbers are deterministic, we
   need to "split" the key to get new randomness. Each split produces two new
   keys: one to use now, one to save for later.

3. **`locations.at[i, :].set(new_loc)`**: This is JAX's way of "updating" an
   immutable array. It returns a *new* array with the value at position
   `[i, :]` changed to `new_loc`. The original array is unchanged.

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
def run_simulation(max_iter=100_000, seed=1234):
    """
    Run the Schelling simulation using JAX.
    """
    key = random.PRNGKey(seed)
    key, init_key = random.split(key)
    locations, types = initialize_state(init_key)

    plot_distribution(locations, types, 'Initial distribution')

    # Loop until no agent wishes to move
    start_time = time.time()
    someone_moved = True
    iteration = 0
    while someone_moved and iteration < max_iter:
        print(f'Entering iteration {iteration + 1}')
        iteration += 1
        someone_moved = False
        for i in range(n):
            old_loc = locations[i, :]
            locations, key = update_agent(i, locations, types, key)
            if not jnp.array_equal(old_loc, locations[i, :]):
                someone_moved = True
    elapsed = time.time() - start_time

    plot_distribution(locations, types, f'Iteration {iteration}')

    if not someone_moved:
        print(f'Converged in {elapsed:.2f} seconds after {iteration} iterations.')
    else:
        print('Hit iteration bound and terminated.')

    return locations, types
```

The main simulation loop is similar to the NumPy version, but with two key
differences:

1. We pass and receive the random `key` in each call to `update_agent`
2. The `locations` array is replaced (not modified) in each iteration

## Warming Up JAX

Like Numba, JAX compiles functions the first time they're called. Let's warm
up the functions:

```{code-cell} ipython3
# Warm up: run with a small example to trigger compilation
key = random.PRNGKey(42)
test_locations = random.uniform(key, shape=(100, 2))
test_types = jnp.array([0] * 50 + [1] * 50)

# Call each function once to compile it
_ = get_distances(test_locations[0], test_locations)
_ = get_neighbors(0, test_locations)
_ = is_happy(0, test_locations, test_types)
_ = count_happy(test_locations, test_types)
key, subkey = random.split(key)
_, _ = update_agent(0, test_locations, test_types, subkey)

print("JAX functions compiled and ready!")
```

## Results

Now let's run the simulation:

```{code-cell} ipython3
locations, types = run_simulation()
```

## Performance Comparison

Let's time one iteration:

```{code-cell} ipython3
%%time
# Set up the initial state
key = random.PRNGKey(1234)
key, init_key = random.split(key)
locations, types = initialize_state(init_key)

# Time one iteration (one pass through all agents)
for i in range(n):
    locations, key = update_agent(i, locations, types, key)
```

On a CPU, JAX's performance is often similar to Numba. The real advantage of
JAX comes when running on GPUs, where the parallel nature of array operations
can provide significant speedups for larger problems.

## When to Use JAX vs Numba

Both JAX and Numba can accelerate Python code, but they have different
strengths:

**Use Numba when:**
- You need to accelerate existing NumPy code with minimal changes
- Your code has complex control flow (nested loops, conditionals)
- You want to stay close to standard Python/NumPy patterns

**Use JAX when:**
- You want GPU/TPU acceleration
- You need automatic differentiation (for machine learning)
- You're comfortable with functional programming patterns
- You're working on problems that benefit from parallelization

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

In the {doc}`next lecture <schelling_jax_parallel>`, we redesign the algorithm
to be fully parallelizable, demonstrating how to get the most out of JAX.

## Summary

JAX provides a powerful alternative to Numba for accelerating Python code. Its
key features are:

- **Immutable arrays** that encourage functional programming
- **Explicit random key management** for reproducibility
- **JIT compilation** via the `@jit` decorator
- **GPU/TPU support** for hardware acceleration

While JAX requires more significant code changes than Numba (due to its
functional style), it offers unique capabilities like automatic differentiation
and seamless GPU acceleration that make it particularly valuable for machine
learning and large-scale numerical computing.

To fully benefit from these capabilities, algorithms often need to be
restructured for parallelism — as we'll see in the next lecture.
