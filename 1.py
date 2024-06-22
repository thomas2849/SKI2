from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import random
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
import numpy as np

class PathFinder(Agent):

    def __init__(self, unique_id, model, found = False):
        super().__init__(unique_id,model)
        self.found = found

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_self=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def step(self):
        self.move()


class WallAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id,model)

    def step(self):
        pass
class Labyrinth(Model):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.maze = self.create_maze(dim)
        maze_height, maze_width = self.maze.shape

        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(maze_width, maze_height, torus=True)

        self.countx = 0
        for x, row in enumerate(self.maze):
            for y, cell in enumerate(row):
                if cell == 1:  # Wall cell
                    agent = WallAgent(self.next_id(), self)
                    self.grid.place_agent(agent, (x, y))
                    self.schedule.add(agent)

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
        maze[-2, -1] = 0
        return maze

def agent_portrayal(agent):
    portrayal = {"Shape": "rect",
                 "Filled": "true",
                 "Layer": 0,
                 "Color": "black",
                 "w": 1,
                 "h": 1}
    return portrayal


'''
loop = []
            xcellloop = 0
            for i in range(12):
                ycellloop = 0
                for j in range(12):
                    loop.append ((xcellloop, ycellloop))
                    ycellloop +=1

                xcellloop +=1

            for ii in loop:
                countingcoincell = self.model.grid.get_cell_list_contents([ii])
                if len(countingcoincell) > 0:
                    for all in countingcoincell:
                        print("Muenze:",ii," :", all.muenze)
'''
canvas_element = CanvasGrid(agent_portrayal, 21, 21, 500, 500)


server = ModularServer(Labyrinth, [canvas_element], "Labyrinth", {"dim": 10})
server.launch()
