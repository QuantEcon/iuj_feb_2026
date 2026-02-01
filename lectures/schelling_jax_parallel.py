import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import time

num_of_type_0 = 1000
num_of_type_1 = 1000
n = num_of_type_0 + num_of_type_1
num_neighbors = 10
require_same_type = 5
num_candidates = 3  # number of candidate locations to try per unhappy agent


def initialize_state(key):
    locations = random.uniform(key, shape=(n, 2))
    types = jnp.array([0] * num_of_type_0 + [1] * num_of_type_1)
    return locations, types


@jit
def get_distances(loc, locations):
    """
    Get distances between loc and all other locations.

    """
    return jnp.sum((loc - locations)**2, axis=1)


@jit
def get_neighbors(loc, agent_idx, locations):
    """
    Compute the indices of neighbors to an agent with index agent_idx and
    location loc.

    """
    distances = get_distances(loc, locations)
    # Insert infinity at the agent's own location
    distances = distances.at[agent_idx].set(jnp.inf)
    # Use top_k to get the indices of closest agents
    _, indices = jax.lax.top_k(-distances, num_neighbors)
    return indices


@jit
def is_unhappy(loc, agent_type, agent_idx, locations, types):
    """
    Test if an agent with index agent_idx, type agent_type, and location loc
    is happy or not.

    """
    neighbors = get_neighbors(loc, agent_idx, locations)
    neighbor_types = types[neighbors]
    num_of_same_type = jnp.sum(neighbor_types == agent_type)
    return num_of_same_type < require_same_type


@jit
def update_agent(i, locations, types, key):
    """
    Update agent i by offering new locations until happy.

    """
    loc = locations[i, :]
    agent_type = types[i]

    def cond_fn(state):
        loc, key = state
        return is_unhappy(loc, agent_type, i, locations, types)

    def body_fn(state):
        _, key = state
        key, subkey = random.split(key)
        new_loc = random.uniform(subkey, shape=(2,))
        return new_loc, key

    final_loc, key = jax.lax.while_loop(cond_fn, body_fn, (loc, key))
    return final_loc, key


def plot_distribution(locations, types, title):
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


@jit
def get_unhappy_agents(locations, types):
    """
    Get the indices and total number of unhappy agents.

    """
    def check_agent(i):
        return is_unhappy(locations[i], types[i], i, locations, types)

    all_unhappy = vmap(check_agent)(jnp.arange(n))
    indices = jnp.where(all_unhappy, size=n, fill_value=-1)[0]
    count = jnp.sum(all_unhappy)
    return indices, count


# --- Parallel algorithm ---

@jit
def find_happy_candidate(i, locations, types, key):
    """
    Propose num_candidates random locations for agent i. Return the first one
    where agent is happy, or current location if none work.

    """
    current_loc = locations[i, :]
    agent_type = types[i]

    # Generate num_candidates random locations
    keys = random.split(key, num_candidates)
    candidates = vmap(lambda k: random.uniform(k, shape=(2,)))(keys)

    # Check happiness at each candidate location (in parallel)
    def check_candidate(loc):
        return ~is_unhappy(loc, agent_type, i, locations, types)

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
    1. Find all unhappy agents
    2. Propose num_candidates locations for each (in parallel)
    3. Move each to first happy candidate (if any)
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
        return is_unhappy(locations[i], types[i], i, locations, types)

    is_unhappy_mask = vmap(check_agent)(jnp.arange(n))

    # Keep old location for happy agents, use new for unhappy
    final_locations = jnp.where(is_unhappy_mask[:, None], new_locations, locations)

    return final_locations, key


def parallel_simulation_loop(locations, types, key, max_iter):
    """Parallel version: propose k candidates for all unhappy agents simultaneously."""
    iteration = 0
    while iteration < max_iter:
        print(f'Entering iteration {iteration + 1}')
        iteration += 1

        _, num_unhappy = get_unhappy_agents(locations, types)

        if num_unhappy == 0:
            break

        locations, key = parallel_update_step(locations, types, key)

    return locations, iteration, key


def run_parallel_simulation(max_iter=100_000, seed=42):
    """Run the parallel Schelling simulation."""
    key = random.PRNGKey(seed)
    key, init_key = random.split(key)
    locations, types = initialize_state(init_key)

    plot_distribution(locations, types, 'Initial distribution')

    start_time = time.time()
    locations, iteration, key = parallel_simulation_loop(locations, types, key, max_iter)
    elapsed = time.time() - start_time

    plot_distribution(locations, types, f'Iteration {iteration}')

    if iteration < max_iter:
        print(f'Converged in {elapsed:.2f} seconds after {iteration} iterations.')
    else:
        print('Hit iteration bound and terminated.')

    return locations, types


# --- Original sequential algorithm ---

def simulation_loop(locations, types, key, max_iter):
    iteration = 0
    while iteration < max_iter:
        print(f'Entering iteration {iteration + 1}')
        iteration += 1

        unhappy, num_unhappy = get_unhappy_agents(locations, types)

        if num_unhappy == 0:
            break

        for j in range(int(num_unhappy)):
            i = int(unhappy[j])
            new_loc, key = update_agent(i, locations, types, key)
            locations = locations.at[i, :].set(new_loc)

    return locations, iteration, key


def run_simulation(max_iter=100_000, seed=42):
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


if __name__ == '__main__':
    # Warm up JAX
    key = random.PRNGKey(42)
    key, init_key = random.split(key)
    test_locations, test_types = initialize_state(init_key)

    # Warm up shared functions
    _ = get_distances(test_locations[0], test_locations)
    _ = get_neighbors(test_locations[0], 0, test_locations)
    _ = is_unhappy(test_locations[0], test_types[0], 0, test_locations, test_types)
    _, _ = get_unhappy_agents(test_locations, test_types)

    # Warm up sequential algorithm
    key, subkey = random.split(key)
    _, _ = update_agent(0, test_locations, test_types, subkey)

    # Warm up parallel algorithm
    key, subkey = random.split(key)
    _ = find_happy_candidate(0, test_locations, test_types, subkey)
    key, subkey = random.split(key)
    _, _ = parallel_update_step(test_locations, test_types, subkey)

    print("JAX functions compiled and ready!")

    # Run both simulations for comparison
    print("\n" + "="*50)
    print("SEQUENTIAL ALGORITHM")
    print("="*50)
    locations, types = run_simulation()

    print("\n" + "="*50)
    print("PARALLEL ALGORITHM")
    print("="*50)
    locations, types = run_parallel_simulation()
