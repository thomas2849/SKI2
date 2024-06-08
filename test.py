import mesa.visualization
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
import numpy as np
import random
from queue import Queue, PriorityQueue

class PathFinder(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.path = []
        self.path_index = 0
        self.cost = 0
    def move(self):
        if self.path_index < len(self.path):
            new_position = self.path[self.path_index]
            self.model.grid.move_agent(self, new_position)
            self.path_index += 1
            self.cost +=1

    def step(self):
        self.move()

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
        self.destination = (2 * dim - 1, 2 * dim)
        self.pathfinder = PathFinder(self.next_id(), self)
        self.place_agents()
        self.move_agents()

        self.datacollector = DataCollector(
            model_reporters={
                "Cost": lambda m: m.pathfinder.cost},
        )

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

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
                blockage = Blockage(self.next_id(), self)
                self.grid.place_agent(blockage, (x, y))
            elif (x, y) == self.start:
                self.grid.place_agent(self.pathfinder, (x, y))
                self.schedule.add(self.pathfinder)

    def move_agents(self):
        path, cost = self.A_star(self.start, self.destination)
        self.pathfinder.path = path
        self.pathfinder.cost = 0


    def heuristic(self, start, end):
        (x1, y1) = start
        (x2, y2) = end
        return abs(x1 - x2) + abs(y1 - y2)

    def A_star(self, start, end):
        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

        while not frontier.empty():
            _, current = frontier.get()
            if current == end:
                break
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next = (current[0] + dx, current[1] + dy)
                if (0 <= next[0] < self.maze.shape[0] and
                    0 <= next[1] < self.maze.shape[1] and
                    self.maze[next] == 0):
                    new_cost = cost_so_far[current] + 1
                    if next not in cost_so_far or new_cost < cost_so_far[next]:
                        cost_so_far[next] = new_cost
                        priority = new_cost + self.heuristic(next, end)
                        frontier.put((priority, next))
                        came_from[next] = current

        current = end
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path, cost_so_far[end]



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

canvas_element = CanvasGrid(agent_portrayal, 31, 31, 500, 500)
chart_element = ChartModule([{"Label": "Cost", "Color": "Red"}], data_collector_name="datacollector")
server = ModularServer(Labyrinth, [canvas_element, chart_element], "Maze Pathfinder", {"width": 31, "height": 31, "dim": 15})
server.launch()
