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

# Schelling Model with Numba

## Overview

In the {doc}`previous lecture <schelling_numpy>`, we implemented the Schelling
model using NumPy arrays and functions. While NumPy provides significant
speedups over pure Python through vectorization, we can do even better.

In this lecture, we use [Numba](https://numba.pydata.org/) to accelerate our
code further. Numba is a *just-in-time (JIT) compiler* that translates Python
functions into optimized machine code at runtime.

The key idea is simple: we add a decorator `@njit` to our functions, and Numba
compiles them to fast machine code the first time they're called.

Let's start with some imports:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
import time
```

## Why is Numba Faster?

Before diving into the code, let's understand why Numba can speed things up.

**Regular Python** is an *interpreted* language. When you run Python code, the
interpreter reads each line, figures out what to do, and executes it. This
flexibility comes at a cost: the interpreter has to do a lot of work at
runtime, checking types, looking up methods, and so on.

**NumPy** speeds things up by moving the heavy lifting to pre-compiled C code.
When you call `np.sum(arr)`, Python hands the array to optimized C code that
runs much faster than a Python loop. However, there's still overhead in
calling NumPy functions and moving data around.

**Numba** takes a different approach. When you decorate a function with `@njit`,
Numba:

1. Analyzes your Python code
2. Infers the types of all variables (integers, floats, arrays, etc.)
3. Compiles the function to machine code optimized for those types
4. Caches the compiled code so future calls are fast

The first call to a Numba function is slower (because of compilation), but
subsequent calls run at speeds comparable to C or Fortran.

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

The initialization function doesn't need Numba — it only runs once at the
start:

```{code-cell} ipython3
def initialize_state():
    locations = np.random.uniform(size=(n, 2))
    types = np.array([0] * num_of_type_0 + [1] * num_of_type_1)
    return locations, types
```

## Numba-Compiled Functions

Now let's rewrite our core functions with Numba. The key change is adding the
`@njit` decorator before each function.

### Computing Distances

```{code-cell} ipython3
@njit
def get_distances(loc, locations):
    """
    Compute the Euclidean distance from one location to all agent locations.
    """
    n = locations.shape[0]
    distances = np.empty(n)
    for i in range(n):
        dx = loc[0] - locations[i, 0]
        dy = loc[1] - locations[i, 1]
        distances[i] = np.sqrt(dx * dx + dy * dy)
    return distances
```

Notice that we've replaced `np.linalg.norm(loc - locations, axis=1)` with an
explicit loop. This might seem like a step backward, but here's the key
insight:

**In regular Python, loops are slow because of interpreter overhead. But Numba
compiles the loop to machine code, making it as fast as a C loop.**

Numba works best with simple operations inside loops. Complex NumPy operations
like `np.linalg.norm` with the `axis` argument aren't always supported, and
even when they are, explicit loops often perform just as well or better after
compilation.

### Finding Neighbors

```{code-cell} ipython3
@njit
def get_neighbors(i, locations):
    """
    Get indices of the k nearest neighbors to agent i (excluding self).
    """
    loc = locations[i, :]
    distances = get_distances(loc, locations)
    # Set self-distance to infinity so we don't count ourselves as a neighbor
    distances[i] = np.inf
    # np.argsort works in Numba
    indices = np.argsort(distances)
    neighbors = indices[:k]
    return neighbors
```

This function calls `get_distances`, which is also Numba-compiled. When one
Numba function calls another, Numba can optimize across the function boundary,
potentially inlining the code for even better performance.

Note that we set `distances[i] = np.inf` to exclude the agent from their own
neighbor list, matching the behavior of the class-based implementation.

### Checking Happiness

```{code-cell} ipython3
@njit
def is_happy(i, locations, types):
    """
    True if agent i has at least require_same_type neighbors of the same type.
    """
    agent_type = types[i]
    neighbors = get_neighbors(i, locations)

    # Count neighbors of the same type
    num_same = 0
    for j in range(k):
        if types[neighbors[j]] == agent_type:
            num_same += 1

    return num_same >= require_same_type
```

Again, we use an explicit loop instead of `np.sum(neighbor_types == agent_type)`.
The loop is clear and, after Numba compilation, very fast.

### Counting Happy Agents

```{code-cell} ipython3
@njit
def count_happy(locations, types):
    """
    Count the number of happy agents.
    """
    num_agents = locations.shape[0]
    happy_count = 0
    for i in range(num_agents):
        if is_happy(i, locations, types):
            happy_count += 1
    return happy_count
```

### Moving Unhappy Agents

```{code-cell} ipython3
@njit
def update_agent(i, locations, types):
    """
    Move agent i to a new location where they are happy.
    """
    while not is_happy(i, locations, types):
        locations[i, 0] = np.random.uniform(0, 1)
        locations[i, 1] = np.random.uniform(0, 1)
```

Note that we use `np.random.uniform(0, 1)` instead of
`numpy.random.uniform()` — Numba has its own random number generator that
works inside compiled functions.

## Visualization

The plotting function uses Matplotlib, which Numba doesn't support. That's
fine — plotting isn't the bottleneck, so we leave it as regular Python:

```{code-cell} ipython3
def plot_distribution(locations, types, title):
    """
    Plot the distribution of agents.
    """
    fig, ax = plt.subplots()
    plot_args = {'markersize': 6, 'alpha': 0.8}
    ax.set_facecolor('azure')
    colors = 'orange', 'green'
    for agent_type, color in zip((0, 1), colors):
        idx = (types == agent_type)
        ax.plot(locations[idx, 0],
                locations[idx, 1],
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
    Run the Schelling simulation.
    """
    np.random.seed(seed)
    locations, types = initialize_state()

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
            old_loc = locations[i, :].copy()
            update_agent(i, locations, types)
            if not np.array_equal(old_loc, locations[i, :]):
                someone_moved = True
    elapsed = time.time() - start_time

    plot_distribution(locations, types, f'Iteration {iteration}')

    if not someone_moved:
        print(f'Converged in {elapsed:.2f} seconds after {iteration} iterations.')
    else:
        print('Hit iteration bound and terminated.')

    return locations, types
```

## Warming Up Numba

The first time a Numba function is called, it gets compiled. This compilation
takes some time. To get accurate timing, we should "warm up" the functions
first:

```{code-cell} ipython3
# Warm up: run with a small example to trigger compilation
np.random.seed(42)
test_locations = np.random.uniform(size=(100, 2))
test_types = np.array([0] * 50 + [1] * 50)

# Call each function once to compile it
_ = get_distances(test_locations[0], test_locations)
_ = get_neighbors(0, test_locations)
_ = is_happy(0, test_locations, test_types)
_ = count_happy(test_locations, test_types)
update_agent(0, test_locations, test_types)

print("Numba functions compiled and ready!")
```

## Results

Now let's run the simulation:

```{code-cell} ipython3
locations, types = run_simulation()
```

## Performance Comparison

Let's time one iteration to compare with the NumPy version:

```{code-cell} ipython3
%%time
# Set up the initial state
np.random.seed(1234)
locations, types = initialize_state()

# Time one iteration (one pass through all agents)
for i in range(n):
    update_agent(i, locations, types)
```

Compare this to the NumPy version from the previous lecture. Numba should be
significantly faster because:

1. **No Python interpreter overhead**: The compiled code runs directly on the
   CPU without the Python interpreter getting in the way.

2. **No NumPy function call overhead**: Each NumPy function call has some
   overhead. With Numba, everything is compiled together.

3. **Better memory access patterns**: Numba can optimize how data is accessed
   in memory, reducing cache misses.

4. **Loop optimization**: Numba can unroll loops, use SIMD instructions, and
   apply other optimizations that aren't possible in interpreted Python.

## Tips for Using Numba

Here are some guidelines for getting the most out of Numba:

1. **Use `@njit` (not `@jit`)**: The `@njit` decorator is short for
   `@jit(nopython=True)`. It forces Numba to compile everything to machine
   code. If Numba can't compile something, you'll get an error (which is
   good — it tells you what to fix).

2. **Write simple loops**: Numba excels at compiling simple loops over arrays.
   Don't be afraid of explicit loops — they'll be fast after compilation.

3. **Avoid unsupported features**: Numba supports a subset of Python and NumPy.
   Things like dictionaries, sets, and some NumPy functions aren't supported.
   Check the [Numba documentation](https://numba.pydata.org/numba-doc/latest/reference/pysupported.html)
   for details.

4. **Warm up before timing**: Always call your functions once before measuring
   performance to exclude compilation time.

5. **Use NumPy arrays**: Numba works best with NumPy arrays. Avoid Python lists
   inside Numba functions.

## Summary

By adding the `@njit` decorator to our core functions, we achieved significant
speedups with minimal code changes. The key insight is that Numba compiles
Python to machine code, eliminating interpreter overhead and enabling
low-level optimizations.

In the next lecture, we'll explore JAX, which takes a different approach by
enabling GPU acceleration and automatic differentiation.
