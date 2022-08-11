import networkx as nx
import matplotlib.pyplot as plt

n = 20                     # number of cities
G = nx.complete_graph(20)  # graph with a vertex for each city

# for convenience, pick the city (x,y)-coordinates at random
import random

my_pos = { i : ( random.random()*20, random.random()*20) for i in G.nodes } # pos[i] = (x_i, y_i)
print(G.edges())
nx.draw(G, pos=my_pos)
plt.show()