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

# Schelling Model with NumPy

## Overview

In the {doc}`previous lecture <schelling>`, we implemented the Schelling
segregation model using Python classes.

In this lecture, we rewrite the model using NumPy arrays and functions.

This approach has several advantages:

1. **Simpler code** — no need to understand object-oriented programming
2. **Faster execution** — NumPy operations are optimized for numerical computation
3. **Easier to accelerate** — this style of code can be readily sped up with Numba or JAX

Let's start with some imports:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform, randint
import time
```

## Data Representation

In the class-based version, each agent was an object storing its own type and location.

Here we take a different approach: we store all agent data in NumPy arrays.

* `locations` — an $n \times 2$ array where row $i$ holds the $(x, y)$ coordinates of agent $i$
* `types` — an array of length $n$ where entry $i$ is 0 or 1, indicating agent $i$'s type

This is sometimes called "struct of arrays" (SoA) style, as opposed to "array of structs" (AoS).

Let's set up the parameters:

```{code-cell} ipython3
num_of_type_0 = 1000    # number of agents of type 0 (orange)
num_of_type_1 = 1000    # number of agents of type 1 (green)
n = num_of_type_0 + num_of_type_1  # total number of agents
k = 10                  # number of agents regarded as neighbors
require_same_type = 5   # want >= require_same_type neighbors of the same type
```

Here's a function to initialize the state with random locations and types:

```{code-cell} ipython3
def initialize_state():
    locations = uniform(size=(n, 2))
    types = np.array([0] * num_of_type_0 + [1] * num_of_type_1)
    return locations, types
```

Let's see what this looks like:

```{code-cell} ipython3
np.random.seed(1234)
locations, types = initialize_state()

print(f"locations shape: {locations.shape}")
print(f"First 5 locations:\n{locations[:5]}")
print(f"\ntypes shape: {types.shape}")
print(f"First 20 types: {types[:20]}")
```

## Helper Functions

Instead of methods on a class, we write standalone functions that operate on
the arrays.

### Computing Distances

To find an agent's neighbors, we need to compute distances.

```{code-cell} ipython3
def get_distances(loc, locations):
    """
    Compute the Euclidean distance from one location to all agent locations.

    Parameters
    ----------
    loc : array of length 2
        The (x, y) coordinates of the reference point.
    locations : array of shape (n, 2)
        The (x, y) coordinates of all agents.

    Returns
    -------
    array of length n
        The distance from loc to each agent.
    """
    return np.linalg.norm(loc - locations, axis=1)
```

Let's break down how this function works:

1. `loc - locations` subtracts the reference point `loc` from every row of
   `locations`. NumPy "broadcasts" the subtraction, so if `loc = [0.5, 0.3]`
   and `locations` has 2000 rows, the result is a 2000 × 2 array where each
   row is the difference vector from `loc` to that agent.

2. `np.linalg.norm(..., axis=1)` computes the length (Euclidean norm) of each
   row. The `axis=1` argument tells NumPy to compute the norm across columns
   (i.e., for each row separately), giving us 2000 distances.

This vectorized approach is much faster than looping through agents one by one.

Let's test it:

```{code-cell} ipython3
# Distance from agent 0 to all agents
distances = get_distances(locations[0], locations)
print(f"Distances from agent 0: {distances[:10].round(3)}...")
print(f"Distance to self: {distances[0]}")  # Should be 0
```

### Finding Neighbors

Now we can find the $k$ nearest neighbors:

```{code-cell} ipython3
def get_neighbors(i, locations):
    " Get indices of the k nearest neighbors to agent i (excluding self). "
    loc = locations[i, :]
    distances = get_distances(loc, locations)
    # Set self-distance to infinity so we don't count ourselves as a neighbor
    distances[i] = np.inf
    indices = np.argsort(distances)   # sort agents by distance
    neighbors = indices[:k]            # keep the k closest
    return neighbors
```

Here's how this function works:

1. First we call `get_distances` to get an array of 2000 distances (one for
   each agent).

2. We set `distances[i] = np.inf` so that agent $i$ doesn't count as their own
   neighbor. This matches the class-based implementation from the previous
   lecture, where an agent's neighbors are other agents, not themselves.

3. `np.argsort(distances)` returns the *indices* that would sort the distances
   from smallest to largest. For example, if the smallest distance is at
   position 42, then `indices[0]` equals 42. This is different from
   `np.sort()`, which returns the sorted values themselves.

4. `indices[:k]` uses slicing to keep only the first $k$ indices — these
   correspond to the $k$ agents with the smallest distances (the nearest
   neighbors).

```{code-cell} ipython3
# Find neighbors of agent 0
neighbors = get_neighbors(0, locations)
print(f"Agent 0's nearest neighbors: {neighbors}")
print(f"Agent 0 is NOT included: {0 not in neighbors}")
```

### Checking Happiness

An agent is happy if enough of their neighbors share their type:

```{code-cell} ipython3
def is_happy(i, locations, types):
    " True if agent i has at least require_same_type neighbors of the same type. "
    agent_type = types[i]
    neighbors = get_neighbors(i, locations)
    neighbor_types = types[neighbors]
    num_same = np.sum(neighbor_types == agent_type)
    return num_same >= require_same_type
```

Let's walk through this function step by step:

1. `types[i]` gets the type (0 or 1) of agent $i$.

2. `get_neighbors(i, locations)` returns an array of indices for the $k$
   nearest neighbors (excluding agent $i$ themselves).

3. `types[neighbors]` uses these indices to look up the types of the
   neighbors. This is called "fancy indexing" — when you pass an array of
   indices to another array, NumPy returns the elements at those positions.
   For example, if `neighbors = [42, 7, 15, ...]`, then `types[neighbors]`
   returns `[types[42], types[7], types[15], ...]`.

4. `neighbor_types == agent_type` compares each neighbor's type to the agent's
   type, producing an array of `True`/`False` values (e.g.,
   `[True, False, True, ...]`).

5. `np.sum(...)` counts how many `True` values there are. In NumPy, `True`
   is treated as 1 and `False` as 0, so summing a boolean array counts the
   `True` entries.

6. Finally, we check if this count meets the threshold `require_same_type`.

```{code-cell} ipython3
# Check if agent 0 is happy
print(f"Agent 0 type: {types[0]}")
print(f"Agent 0 happy: {is_happy(0, locations, types)}")
```

### Counting Happy Agents

```{code-cell} ipython3
def count_happy(locations, types):
    " Count the number of happy agents. "
    happy_count = 0
    for i in range(n):
        happy_count += is_happy(i, locations, types)
    return happy_count
```

This function uses a simple loop to check each agent and count how many are
happy. Since `is_happy` returns `True` or `False`, and Python treats `True`
as 1 when adding, we can accumulate the count directly.

```{code-cell} ipython3
print(f"Initially happy agents: {count_happy(locations, types)} out of {n}")
```

### Moving Unhappy Agents

When an agent is unhappy, they keep trying new random locations until they find
one where they're happy:

```{code-cell} ipython3
def update_agent(i, locations, types):
    " Move agent i to a new location where they are happy. "
    while not is_happy(i, locations, types):
        locations[i, :] = uniform(), uniform()
```

Here's how this works:

1. The `while` loop keeps running as long as the agent is unhappy.

2. `locations[i, :] = uniform(), uniform()` assigns a new random $(x, y)$
   location to agent $i$. The left side `locations[i, :]` selects row $i$
   (all columns), and the right side creates a tuple of two random numbers
   between 0 and 1.

3. Importantly, this modifies the `locations` array *in place*. We don't need
   to return anything because the original array is changed directly. This is
   a key feature of NumPy arrays — when you modify a slice, you modify the
   underlying data.

## Visualization

```{code-cell} ipython3
def plot_distribution(locations, types, title):
    " Plot the distribution of agents. "
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

The key NumPy technique here is **boolean indexing**:

1. `types == agent_type` creates a boolean array of length 2000, with `True`
   for agents of the current type and `False` otherwise. For example, when
   `agent_type = 0`, this might produce `[True, False, True, True, False, ...]`.

2. `locations[idx, 0]` uses this boolean array to select rows. Only the rows
   where `idx` is `True` are kept. The `, 0` selects the first column (x
   coordinates). Similarly, `locations[idx, 1]` gets the y coordinates.

3. This means we can plot all agents of one type in a single call to
   `ax.plot()`, without needing to loop through agents individually.

4. The `zip((0, 1), colors)` pairs each type (0 or 1) with its color
   (orange or green), so the loop runs twice — once for each type.

Let's visualize the initial random distribution:

```{code-cell} ipython3
np.random.seed(1234)
locations, types = initialize_state()
plot_distribution(locations, types, 'Initial random distribution')
```

## The Simulation

Now we put it all together.

As in the first lecture, each iteration cycles through all agents in order,
giving each the opportunity to move:

```{code-cell} ipython3
def run_simulation(max_iter=100_000, seed=1234):
    """
    Run the Schelling simulation.

    Each iteration cycles through all agents, giving each a chance to move.
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
            old_location = locations[i, :].copy()
            update_agent(i, locations, types)
            if not np.array_equal(old_location, locations[i, :]):
                someone_moved = True
    elapsed = time.time() - start_time

    plot_distribution(locations, types, f'Iteration {iteration}')

    if not someone_moved:
        print(f'Converged in {elapsed:.2f} seconds after {iteration} iterations.')
    else:
        print('Hit iteration bound and terminated.')

    return locations, types
```

## Results

Let's run the simulation:

```{code-cell} ipython3
locations, types = run_simulation()
```

We see the same phenomenon as in the class-based version: starting from a
random mixed distribution, agents self-organize into segregated clusters.

## Performance Comparison

One advantage of this NumPy-based approach is speed.

Let's time how long it takes to run one iteration (one pass through all agents):

```{code-cell} ipython3
%%time
# Set up the initial state
np.random.seed(1234)
locations, types = initialize_state()

# Time one iteration (one pass through all agents)
for i in range(n):
    update_agent(i, locations, types)
```

Compare this to the class-based version — the NumPy approach is faster because
it uses vectorized distance calculations.

This functional style with NumPy arrays is also a good foundation for further
optimization with Numba or JAX, which we'll explore in subsequent lectures.

## Exercises

```{exercise-start}
:label: schelling_numpy_ex1
```

Modify `run_simulation` to add a `flip_prob` parameter.

After each agent update, flip the agent's type with probability `flip_prob`.

We can imagine that, every so often, an agent moves to a different city and,
with small probability, is replaced by an agent of the other type.

Run the simulation with `flip_prob=0.01` and observe how this additional
randomness affects the dynamics. Does the system still converge? What happens
to the segregation patterns?

```{exercise-end}
```

```{solution-start} schelling_numpy_ex1
:class: dropdown
```

```{code-cell} ipython3
def run_simulation_with_flip(max_iter=20, flip_prob=0.01, seed=1234):
    """
    Run the Schelling simulation with type flipping.

    After each agent update, flip the agent's type with probability flip_prob.
    """
    np.random.seed(seed)
    locations, types = initialize_state()

    plot_distribution(locations, types, 'Initial distribution')

    for iteration in range(max_iter):
        print(f'Entering iteration {iteration + 1}')
        for i in range(n):
            update_agent(i, locations, types)
            # Flip agent's type with probability flip_prob
            if uniform() < flip_prob:
                types[i] = 1 - types[i]

    plot_distribution(locations, types, f'After {max_iter} iterations')
    print(f"Happy agents: {count_happy(locations, types)} / {n}")
    return locations, types
```

```{code-cell} ipython3
locations, types = run_simulation_with_flip(max_iter=20, flip_prob=0.01)
```

With type flipping, the system no longer fully converges because the random
flips continuously introduce unhappy agents. However, we still observe
segregation patterns emerging and persisting, demonstrating the robustness of
the segregation phenomenon.

```{solution-end}
```
