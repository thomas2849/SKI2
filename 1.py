from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import random
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule


class Labyrinth(Model):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.width = width
        self.height = height
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(self.width, self.height, torus=True)




def Sim_portrayal(agent):
    if agent is None:
        return
    portrayal = {}


canvas_element = CanvasGrid(Sim_portrayal, 20, 20, 500, 500)


server = ModularServer(Labyrinth, [canvas_element], "Simulation Visualization", {"width":10,"height":10})
server.launch()
