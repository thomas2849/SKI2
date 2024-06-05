from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
import numpy as np
import random

class PathFinder(Agent):
    def __init__(self, unique_id, model, found=False):
        super().__init__(unique_id, model)
        self.found = found

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def step(self):
        self.move()
        if self.pos == self.model.destination:
            self.found = True
            self.model.running = False

class Blockage(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class Labyrinth(Model):
    def __init__(self, width, height, dim):
        super().__init__()
        self.width = width
        self.height = height
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(self.width, self.height, torus=False)
        self.maze = self.create_maze(dim)
        self.start = (1, 0)
        self.destination = (2*dim-1, 2*dim)
        self.place_agents()

    def step(self):
        self.schedule.step()

    def create_maze(self, dim):
        # Create a grid filled with walls
        maze = np.ones((dim * 2 + 1, dim * 2 + 1))

        # Define the starting point
        x, y = (0, 0)
        maze[2 * x + 1, 2 * y + 1] = 0

        # Initialize the stack with the starting point
        stack = [(x, y)]
        while len(stack) > 0:
            x, y = stack[-1]

            # Define possible directions
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if nx >= 0 and ny >= 0 and nx < dim and ny < dim and maze[2 * nx + 1, 2 * ny + 1] == 1:
                    maze[2 * nx + 1, 2 * ny + 1] = 0
                    maze[2 * x + 1 + dx, 2 * y + 1 + dy] = 0
                    stack.append((nx, ny))
                    break
            else:
                stack.pop()

        # Create an entrance and an exit
        maze[1, 0] = 0
        maze[2 * dim - 1, 2 * dim] = 0
        return maze

    def place_agents(self):
        for (x, y), value in np.ndenumerate(self.maze):
            if value == 1:
                blockage = Blockage((x, y), self)
                self.grid.place_agent(blockage, (x, y))
            elif (x, y) == self.start:
                pathfinder = PathFinder((x, y), self)
                self.grid.place_agent(pathfinder, (x, y))
                self.schedule.add(pathfinder)
            elif (x, y) == self.destination:
                # Just to mark the destination
                blockage = Blockage((x, y), self)
                self.grid.place_agent(blockage, (x, y))

def agent_portrayal(agent):
    if isinstance(agent, PathFinder):
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 0,
                     "Color": "red",
                     "r": 0.5}
    elif isinstance(agent, Blockage):
        portrayal = {"Shape": "rect",
                     "Filled": "true",
                     "Layer": 0,
                     "Color": "black",
                     "w": 1,
                     "h": 1}
    return portrayal

canvas_element = CanvasGrid(agent_portrayal, 21, 21, 500, 500)
server = ModularServer(Labyrinth, [canvas_element], "Maze Pathfinder", {"width": 21, "height": 21, "dim": 10})
server.launch()
