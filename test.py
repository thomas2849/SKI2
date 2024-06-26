import mesa.visualization
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
import numpy as np
import random
from queue import PriorityQueue
import matplotlib.pyplot as plt

class PathFinder(Agent):
    def __init__(self, unique_id, model, mazemodus):
        super().__init__(unique_id, model)
        self.path = []
        self.path_index = 0
        self.cost = 0
        self.stepcount = 0
        self.muenze = 0
        self.mazemodus = mazemodus

    def move(self):
        if self.path_index < len(self.path):
            new_position = self.path[self.path_index]  #Iterates through the path from the A* function
            self.model.grid.move_agent(self, new_position)
            self.path_index += 1
            self.cost += 1

    def move2(self):   # move function for the mazemodus == False
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def give_money(self):         # Give money function for pathfinder
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        for i in range(len(possible_steps)):
            cellmates = self.model.grid.get_cell_list_contents([possible_steps[i]])
            for other in cellmates:
                if other.muenze < 3:
                    other.muenze += 1
                    self.muenze -= 1

    def step(self):
        if self.mazemodus:
            self.move()
        else:
            self.stepcount += 1
            if self.stepcount >= 20:  #checks 20 steps
                return
            self.move2()
            if self.muenze > 5 and self.stepcount < 20:
                self.give_money()


class Blockage(Agent):     # Wall Agents
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)


class Neighbours(Agent):  # Neighbour Agent in the second phase
    def __init__(self, unique_id, model, muenze):
        super().__init__(unique_id, model)
        self.muenze = muenze
        self.lebendig = True
        self.count = 0

    def give_money(self):  #give money function for the Neighbour Agents
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        for i in range(len(possible_steps)):
            cellmates = self.model.grid.get_cell_list_contents([possible_steps[i]])
            for other in cellmates:
                if other.muenze < 3:
                    other.muenze += 1
                    self.muenze -= 1

    def move(self):
        self.count += 1
        if self.count < 20:  # Checks 20 steps

            if self.lebendig:
                possible_steps = self.model.grid.get_neighborhood(
                    self.pos, moore=True, include_center=False
                )
                new_position = self.random.choice(possible_steps)
                self.model.grid.move_agent(self, new_position)

                if self.muenze == 0:
                    self.count += 1
                    if self.count == 5 and self.muenze == 0:
                        self.lebendig = False

        elif self.count == 20:

            exit()  # STOPS THE PROGRAM

    def step(self):
        self.move()
        if self.muenze > 5:
            self.give_money()


class Labyrinth(Model):  # Umgebung / Enviroment
    def __init__(self, width, height, dim, mazemodus):
        super().__init__()
        self.width = width
        self.height = height
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(self.width, self.height, torus=False)
        self.mazemodus = mazemodus
        self.maze = self.create_maze(dim)  # dim = dimensions of the maze, increase for higher complexity
        self.start = (1, 0)  # starting point
        self.destination = (2 * dim - 1, 2 * dim)  # destination point
        self.pathfinder = PathFinder(self.next_id(), self, mazemodus)
        self.place_agents()
        self.move_agents()

    def reset_maze(self, L):  # Second phase
        self.grid = MultiGrid(self.width, self.height, torus=False)
        self.pathfinder.muenze = 10
        self.schedule = mesa.time.RandomActivation(self)
        self.schedule.add(self.pathfinder)
        x = self.random.randrange(self.grid.width)
        y = self.random.randrange(self.grid.height)
        self.grid.place_agent(self.pathfinder, (x, y))

        for i in range(L - 1):  # Length of the path L for the number of agents
            a = Neighbours(i, self, 10)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        for i in range(L):  # Arme Agents
            a = Neighbours(i, self, 0)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

    def step(self):
        self.schedule.step()
        path, cost = self.A_star(self.start, self.destination)
        if not self.grid.is_cell_empty(self.destination) and self.pathfinder.mazemodus:
            self.grid.remove_agent(self.pathfinder)
            self.pathfinder.mazemodus = False
            self.reset_maze(len(path))

        if self.pathfinder.mazemodus == False:
            self.plot_muenze_distribution() # Call the plotting function

    def create_maze(self, dim):
        maze = np.ones((dim * 2 + 1, dim * 2 + 1))
        x, y = (0, 0)
        maze[2 * x + 1, 2 * y + 1] = 0
        stack = [(x, y)]
        while len(stack) > 0:
            x, y = stack[-1]
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
        cx, cy = zip(*path)
        plt.scatter([1], [0], color='green', marker='o', s=80)
        plt.scatter([9], [10], color='red', marker='o', s=80)
        for (x, y), value in np.ndenumerate(self.maze):
            if value == 0:
                plt.plot(x, y, marker='o', color='black')
        plt.plot(cx, cy, marker='o', color='blue')
        plt.show()

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
        path = [(1, 0)] + path
        return path, cost_so_far[end]

    def plot_muenze_distribution(self):
        muenze_counts = [agent.muenze for agent in self.schedule.agents if isinstance(agent, (PathFinder, Neighbours))]
        muenze_distribution = {}

        for muenze in muenze_counts:
            if muenze in muenze_distribution:
                muenze_distribution[muenze] += 1
            else:
                muenze_distribution[muenze] = 1

        alive_agents = sum(1 for agent in self.schedule.agents if isinstance(agent, Neighbours) and agent.lebendig)
        dead_agents = sum(1 for agent in self.schedule.agents if isinstance(agent, Neighbours) and not agent.lebendig)

        x = list(muenze_distribution.keys())
        y = list(muenze_distribution.values())

        plt.figure(figsize=(10, 6))
        plt.bar(x, y, color='blue')
        plt.xlabel('Number of Muenze')
        plt.ylabel('Number of Agents')
        plt.title(f'Distribution of Muenze Among Agents\nAlive Agents: {alive_agents}, Dead Agents: {dead_agents}')
        plt.show()


def agent_portrayal(agent):   # Visualation Code
    if isinstance(agent, PathFinder):
        if agent.mazemodus == True:
            portrayal = {"Shape": "circle",
                         "Filled": "true",
                         "Layer": 0,
                         "Color": "red",
                         "r": 0.5}
        elif agent.mazemodus == False and agent.muenze >= 5:
            portrayal = {"Shape": "circle",
                         "Filled": "true",
                         "Layer": 0,
                         "Color": "grey",
                         "r": 1}
        elif agent.mazemodus == False and 0 < agent.muenze < 5:
            portrayal = {"Shape": "circle",
                         "Filled": "true",
                         "Layer": 0,
                         "Color": "orange",
                         "r": 1}
    elif isinstance(agent, Blockage):
        portrayal = {"Shape": "rect",
                     "Filled": "true",
                     "Layer": 0,
                     "Color": "black",
                     "w": 1,
                     "h": 1}
    elif isinstance(agent, Neighbours):
        if agent.lebendig == True:
            if agent.muenze == 0:
                portrayal = {"Shape": "circle",
                             "Filled": "true",
                             "Layer": 0,
                             "Color": "blue",
                             "r": 1}
            elif 5 > agent.muenze > 0:
                portrayal = {"Shape": "circle",
                             "Filled": "true",
                             "Layer": 0,
                             "Color": "yellow",
                             "r": 1}
            elif agent.muenze >= 5:  # Reiche
                portrayal = {"Shape": "circle",
                             "Filled": "true",
                             "Layer": 0,
                             "Color": "green",
                             "r": 1}
        else:  # self.lebendig = False
            portrayal = {"Shape": "circle",
                         "Filled": "true",
                         "Layer": 0,
                         "Color": "black",
                         "r": 1}
    return portrayal

canvas_element = CanvasGrid(agent_portrayal, 11, 11, 500, 500)
server = ModularServer(Labyrinth, [canvas_element], "Maze Pathfinder", {"width": 11, "height": 11, "dim": 5, "mazemodus": True})
server.launch()
