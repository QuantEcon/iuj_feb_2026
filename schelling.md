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


# Racial Segregation


## Outline

In 1969, Thomas C. Schelling developed a simple but striking model of racial
segregation {cite}`Schelling1969`.

His model studies the dynamics of racially mixed neighborhoods.

Like much of Schelling's work, the model shows how local interactions can lead
to surprising aggregate outcomes.

It studies a setting where agents (think of households) have relatively mild
preference for neighbors of the same race.

For example, these agents might be comfortable with a mixed race neighborhood
but uncomfortable when they feel "surrounded" by people from a different race.

Schelling illustrated the follow surprising result: in such a setting, mixed
race neighborhoods are likely to be unstable, tending to collapse over time.

In fact the model predicts strongly divided neighborhoods, with high levels of
segregation.

In other words, extreme segregation outcomes arise even though people's
preferences are not particularly extreme.

These extreme outcomes happen because of *interactions* between agents in the
model (e.g., households in a city) that drive self-reinforcing dynamics in the
model.

These ideas will become clearer as the lecture unfolds.

In recognition of his work on segregation and other research, Schelling was
awarded the 2005 Nobel Prize in Economic Sciences.


Let's start with some imports:

```{code-cell} ipython3
import matplotlib.pyplot as plt
from random import uniform, seed
from math import sqrt
import numpy as np
import time
```

## The model

In this section we will build a version of Schelling's model.

### Set-Up

We will cover a variation of Schelling's model that is different from the
original but also easy to program and, at the same time, captures his main
idea.

Suppose we have two types of people: orange people and green people.

Assume there are $n$ of each type.

These agents all live on a single unit square.

Thus, the location (e.g, address) of an agent is just a point $(x, y)$,  where
$0 < x, y < 1$.

* The set of all points $(x,y)$ satisfying $0 < x, y < 1$ is called the **unit square**
* Below we denote the unit square by $S$

+++

### Preferences

We will say that an agent is *happy* if 5 or more of her 10 nearest neighbors are of the same type.

An agent who is not happy is called *unhappy*.

For example,

*  if an agent is orange and 5 of her 10 nearest neighbors are green, then she is happy.
*  if an agent is green and 8 of her 10 nearest neighbors are orange, then she is unhappy.

'Nearest' is in terms of [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance).

An important point to note is that agents are **not** averse to living in mixed areas.

They are perfectly happy if half of their neighbors are of the other color.

Let's set up the parameters for our simulation:

```{code-cell} ipython3
num_of_type_0 = 1000    # number of agents of type 0 (orange)
num_of_type_1 = 1000    # number of agents of type 1 (green)
num_neighbors = 10      # number of agents viewed as neighbors
require_same_type = 5   # number of neighbors that must be same type for happiness
```

+++

### Behavior

Initially, agents are mixed together (integrated).

In particular, we assume that the initial location of each agent is an
independent draw from a bivariate uniform distribution on the unit square $S$.

* First their $x$ coordinate is drawn from a uniform distribution on $(0,1)$
* Then, independently, their $y$ coordinate is drawn from the same distribution.

Now, cycling through the set of all agents, each agent is now given the chance to stay or move.

Each agent stays if they are happy and moves if they are unhappy.

The algorithm for moving is as follows

```{prf:algorithm} Relocation Algorithm
:label: move_algo

1. Draw a random location in $S$
2. If happy at new location, move there
3. Else go to step 1

```

We cycle continuously through the agents, each time allowing an unhappy agent
to move.

We continue to cycle until no one wishes to move.

+++

## Results

Let's now implement and run this simulation.

In what follows, agents are modeled as [objects](https://python-programming.quantecon.org/python_oop.html).

Here's an indication of their structure:

```{code-block} none
* Data:

    * type (green or orange)
    * location

* Methods:

    * determine whether happy or not given locations of other agents
    * If not happy, move
        * find a new location where happy
```

Let's build them.

```{code-cell} ipython3
class Agent:

    def __init__(self, type):
        self.type = type
        self.draw_location()

    def draw_location(self):
        self.location = uniform(0, 1), uniform(0, 1)

    def get_distance(self, other):
        "Computes the euclidean distance between self and other agent."
        a = (self.location[0] - other.location[0])**2
        b = (self.location[1] - other.location[1])**2
        return sqrt(a + b)

    def happy(self,
                agents,                                    # List of other agents
                num_neighbors=num_neighbors,               # No. of agents viewed as neighbors
                require_same_type=require_same_type):      # How many neighbors must be same type
        """
            True if a sufficient number of nearest neighbors are of the same
            type.
        """

        distances = []

        # Distances is a list of pairs (d, agent), where d is distance from
        # agent to self
        for agent in agents:
            if self != agent:
                distance = self.get_distance(agent)
                distances.append((distance, agent))

        # Sort from smallest to largest, according to distance
        distances.sort()

        # Extract the neighboring agents
        neighbors = [agent for d, agent in distances[:num_neighbors]]

        # Count how many neighbors have the same type as self
        num_same_type = sum(self.type == agent.type for agent in neighbors)
        return num_same_type >= require_same_type

    def update(self, agents):
        "If not happy, then randomly choose new locations until happy."
        while not self.happy(agents):
            self.draw_location()
```

Here's some code that takes a list of agents and produces a plot showing their
locations on the unit square.

Orange agents are represented by orange dots and green ones are represented by
green dots.

```{code-cell} ipython3
def plot_distribution(agents, cycle_num):
    "Plot the distribution of agents after cycle_num rounds of the loop."
    x_values_0, y_values_0 = [], []
    x_values_1, y_values_1 = [], []
    # == Obtain locations of each type == #
    for agent in agents:
        x, y = agent.location
        if agent.type == 0:
            x_values_0.append(x)
            y_values_0.append(y)
        else:
            x_values_1.append(x)
            y_values_1.append(y)
    fig, ax = plt.subplots()
    plot_args = {'markersize': 6, 'alpha': 0.8}
    ax.set_facecolor('azure')
    ax.plot(x_values_0, y_values_0,
        'o', markerfacecolor='orange', **plot_args)
    ax.plot(x_values_1, y_values_1,
        'o', markerfacecolor='green', **plot_args)
    ax.set_title(f'Cycle {cycle_num-1}')
    plt.show()
```

The main loop cycles through all agents until no one wishes to move.

```{prf:algorithm} Main Simulation Loop
:label: main_loop_algo

**Input:** Set of agents with initial random locations

**Output:** Final distribution of agents

1. Plot initial distribution
2. Set `someone_moved` $\leftarrow$ True
3. While `someone_moved` is True:
    1. Set `someone_moved` $\leftarrow$ False
    2. For each agent:
        1. If agent is unhappy, relocate using {prf:ref}`move_algo`
        2. Set `someone_moved` $\leftarrow$ True
4. Plot final distribution

```

The code is below

```{code-cell} ipython3
def run_simulation(num_of_type_0=num_of_type_0,
                   num_of_type_1=num_of_type_1,
                   max_iter=100_000,
                   set_seed=1234):

    # Set the seed for reproducibility
    seed(set_seed)

    # Create a list of agents of type 0
    agents = [Agent(0) for i in range(num_of_type_0)]
    # Append a list of agents of type 1
    agents.extend(Agent(1) for i in range(num_of_type_1))

    # Initialize a counter
    count = 1

    # Plot the initial distribution
    plot_distribution(agents, count)

    # Loop until no agent wishes to move
    start_time = time.time()
    someone_moved = True
    while someone_moved and count < max_iter:
        print('Entering loop ', count)
        count += 1
        someone_moved = False
        for agent in agents:
            old_location = agent.location
            agent.update(agents)
            if agent.location != old_location:
                someone_moved = True
    elapsed = time.time() - start_time

    # Plot final distribution
    plot_distribution(agents, count)

    if count < max_iter:
        print(f'Converged in {elapsed:.2f} seconds after {count} iterations.')
    else:
        print('Hit iteration bound and terminated.')

```

Let's have a look at the results.

```{code-cell} ipython3
run_simulation()
```

As discussed above, agents are initially mixed randomly together.

But after several cycles, they become segregated into distinct regions.

In this instance, the program terminated after a small number of cycles
through the set of agents, indicating that all agents had reached a state of
happiness.

What is striking about the pictures is how rapidly racial integration breaks down.

This is despite the fact that people in the model don't actually mind living mixed with the other type.

Even with these preferences, the outcome is a high degree of segregation.

## Performance

Let's measure how long it takes to run one iteration (one pass through all agents):

```{code-cell} ipython3
%%time
# Set up agents
seed(1234)
agents = [Agent(0) for i in range(num_of_type_0)]
agents.extend(Agent(1) for i in range(num_of_type_1))

# Time one iteration (one pass through all agents)
for agent in agents:
    agent.update(agents)
```

This gives us a baseline to compare against when we optimize the code in later lectures.

## References

```{bibliography}
:filter: docname in docnames
```
