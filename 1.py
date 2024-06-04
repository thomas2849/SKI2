from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import random
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule

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
class Labyrinth(Model):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.width = width
        self.height = height
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(self.width, self.height, torus=True)

        a = self.random.randrange(self.width)
        b = self.random.randrange(self.height)
        ma = PathFinder(self.next_id(), self)
        self.grid.place_agent(ma, (a, b))
        self.schedule.add(ma)

    def step(self):
        self.schedule.step()




def Sim_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "Layer": 0,
                 "Color": "red",
                 "r": 0.5}
    return portrayal



canvas_element = CanvasGrid(Sim_portrayal, 20, 20, 500, 500)


server = ModularServer(Labyrinth, [canvas_element], "Simulation Visualization", {"width":10,"height":10})
server.launch()
