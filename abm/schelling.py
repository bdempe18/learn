import numpy as np
from random import shuffle
from itertools import product as CartesianProduct
import plotly.express as px
from plotly.offline import plot


class Agent(object):
    """
    Citizens of some underlying World object.

    Schelling's agents have knowledge of the world in which they live, their
    discrete type, and their preference for living near like-typed agents.

    Agents have the ability to move and to determine if they are happy living
    in their current location.
    """
    def __init__(self, world, agent_type, preference=0.5):
        self.world = world
        self.preference = preference
        self.x, self.y = -1, -1
        self.agent_type = agent_type

    def __repr__(self):
        return str(self.agent_type)

    def move(self, x, y):
        self.world.relocate(self, x, y)
        self.x, self.y = (x, y)

    def is_happy(self):
        neighbors = self.world.neighbors(self.x, self.y)
        n_neighbors = len(neighbors)
        n_same_type = sum(neighbor.agent_type == self.agent_type
                          for neighbor in neighbors)
        proportion = 0 if n_neighbors == 0 else (n_same_type/n_neighbors)
        return proportion >= self.preference

    def update(self):
        if not self.is_happy():
            new_x, new_y = self.world.draw_location()
            self.move(new_x, new_y)


class World(object):
    """
    City in which agents operate.

    The World is an n-by-n dimension grid composed of discrete blocks. Agents
    can occupy one block at a time. The world has knowledge of the agents which
    live in the city and the free squares to which agents may move.

    The World controls the simulation by randomly giving agents the opportunity
    to move.  Simulations last a pre-determined length of time.
    """
    def __init__(self, dim=20):
        self.dim = dim
        self.board = np.zeros((dim, dim), dtype="object")
        self.agents = []
        self.free_squares = []
        self.find_free_squares()

    def occupied(self, x, y):
        return isinstance(self.board[x, y], Agent)

    def find_free_squares(self):
        sqs = CartesianProduct(range(0, self.dim), range(0, self.dim))
        self.free_squares = [(x, y) for x, y in sqs if not self.occupied(x, y)]

    def run(self):
        for i in range(50):
            self.step()

    def step(self):
        np.random.shuffle(self.agents)
        for agent in self.agents:
            agent.update()

    def populate(self, agents):
        for agent in agents:
            x, y = self.draw_location()
            agent.move(x, y)

        self.agents = agents
        self.find_free_squares()

    def neighbors(self, x, y):
        center = lambda a: return (a[0] + x, a[1] + y)
        tups = CartesianProduct([-1,0,1], [-1,0,1])
        neighs = [center(loc) for loc in tups]

        # xs = [x + i for i in [-1, 0, 1]]
        # ys = [y + i for i in [-1, 0, 1]]
        # neighs = list(CartesianProduct(xs, ys))
        neighs.remove((x, y))
        return [self.board[a, b] for a, b in neighs if
                a < self.dim and b < self.dim and
                a >= 0 and b >= 0 and
                isinstance(self.board[a, b], Agent)]

    def draw_location(self):
        shuffle(self.free_squares)
        self.free_squares.pop()

    def relocate(self, agent, new_x, new_y):
        self.board[new_x, new_y] = agent

        if agent.x is not None and agent.y is not None:
            self.board[agent.x, agent.y] = 0
            self.free_squares.append((agent.x, agent.y))


    def show(self):
        b = self.board
        sqrs = CartesianProduct(range(0, self.dim), range(0, self.dim))
        grid = np.array([b[x,y].agent_type if isinstance(b[x, y], Agent)
                         else 0 for x,y in sqrs])

        color_scale = ['#7f7f7f', '#2ca02c', '#1f77b4']
        grid = grid.reshape((self.dim, self.dim))
        fig = px.imshow(grid, color_continuous_scale = color_scale)
        fig.update_layout(coloraxis_showscale = False)
        fig.update_xaxes(showticklabels = False)
        fig.update_yaxes(showticklabels = False)
        plot(fig)

# ============================================================================
# SIMULATION
# ============================================================================

n_a = n_b = 4500
board = World(100)

group_a = [Agent(board, 1) for _ in range(n_a)]
group_b = [Agent(board, 2) for _ in range(n_b)]
agents = np.concatenate((group_a, group_b))
board.populate(agents)
board.run()
board.show()
