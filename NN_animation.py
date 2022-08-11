from ast import While
from pickle import TRUE
from manim import *
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random


class NearestNeighbors(Scene):
    def construct(self):
        # number of cities
        n = 10
        # graph with a vertex for each city
        # Initially there is an edge between each vertex
        # This graph will continue to be updated as we remove and add edges
        currG = nx.complete_graph(n)

        # Random node positions
        # chose those values because the nodes will not go out of frame
        node_pos = { i : [random.uniform(-6,6), random.uniform(-3.5,3.5), 0] for i in range(0,10) } # pos[i] = (x_i, y_i)

        # adding a length attribute to the edges
        for i,j in currG.edges:
            loc1 = tuple(node_pos[i])
            loc2 = tuple(node_pos[j])
            currG.edges[i,j]['length'] = distance.euclidean(loc1, loc2)

        # This graph is a complete graph and will not change through the code  
        compG = currG.copy()
        
        # manim graph used for the animations
        manimG = Graph.from_networkx(currG, layout=node_pos, labels=True)

        self.play(Create(manimG))
        self.wait()

        self.animate_NN(manimG, currG)

    def animate_NN(self, manimG, currG):

        # remove all the edges
        self.play(manimG.animate.remove_edges(*manimG.edges))

        # List of vertices left to visit
        vert_left = list(currG.nodes).copy()
        vert_left.remove(0)

        # make origin red
        self.play(manimG[0].animate.set_color(RED, 0.6))
        self.play(manimG[0].animate.set_stroke(BLACK, width=4))
        startV = 0

        # make the nearest neighbor yellow and then draw an edge to it
        while len(vert_left) > 0:

            # find nearest neighbor
            nn, minEdge = self.find_NN(currG, startV, vert_left)
            
            # remove visited vertex from list
            vert_left.remove(nn)

            # animations
            self.play(manimG[nn].animate.set_color(YELLOW, 0.6))
            self.play(manimG[nn].animate.set_color(WHITE, 1))
            self.play(manimG[nn].animate.set_stroke(BLACK, width=4))

            self.play(manimG.animate.add_edges(minEdge))
            startV = nn
        
        # edge back to origin
        self.play(manimG.animate.add_edges((startV, 0)))


    def find_NN(self, currG, currV, vert_left):
        
        # loop through all the vertices not yet visited
        # find the closest one
        minEdge = (currV, vert_left[0])
        minLen = currG.edges[currV, vert_left[0]]['length']
        nn = vert_left[0]
        for v in vert_left:
            if minLen > currG.edges[currV, v]['length']:
                minLen = currG.edges[currV, v]['length']
                nn = v

                minEdge = (currV, v)
        
        # return nearest neighbor and edge to be drawn
        return nn, minEdge



    




