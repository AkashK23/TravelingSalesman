from ast import While
from pickle import TRUE
from manim import *
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random


class MovingVertices(Scene):
    def construct(self):
        edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]

        n = 10                     # number of cities
        currG = nx.complete_graph(n)  # graph with a vertex for each city
        compG = currG.copy()
        # for convenience, pick the city (x,y)-coordinates at random

        node_pos = { i : [random.uniform(-6,6), random.uniform(-3.5,3.5), 0] for i in range(0,10) } # pos[i] = (x_i, y_i)
        g = Graph.from_networkx(currG, layout=node_pos, labels=True)

        self.play(Create(g))
        self.wait()

        edges_to_remove = self.minSpanTree(compG, node_pos)
        currG.remove_edges_from(edges_to_remove)

        self.play(g.animate.remove_edges(*edges_to_remove))
        self.wait()

        perf_matching = self.OddNodesPerfMatch(g, currG, compG)

        self.eulerian_circuit(g, currG, perf_matching, n)

    def minSpanTree(self, currG, node_pos):
        for i,j in currG.edges:
            loc1 = tuple(node_pos[i])
            loc2 = tuple(node_pos[j])
            currG.edges[i,j]['length'] = distance.euclidean(loc1, loc2)
        
        mst_graph = nx.minimum_spanning_tree(currG, weight='length')
        edges_to_remove = set(currG.edges).difference(set(mst_graph.edges))

        return edges_to_remove
    
    def OddNodesPerfMatch(self, graph, currG, compG):

        odd_degree_nodes = [ i for i in currG.nodes if currG.degree(i) % 2 ]
        print(graph[odd_degree_nodes[0]].color, graph[odd_degree_nodes[0]].stroke_color)
        # print(graph[odd_degree_nodes[0]].get_label())
        # fill = [graph[v].animate.set_fill(RED, 0.8) for v in odd_degree_nodes]
        # stroke = [graph[v].animate.set_stroke(BLACK, width=3) for v in odd_degree_nodes]
        # odd_deg_animations = fill + stroke

        # self.play(*[graph[v].animate.set_fill(RED, 0.8) for v in odd_degree_nodes])
        self.play(*[graph[v].animate.set_stroke(RED, width=4) for v in odd_degree_nodes])
        self.wait()

        for i,j in compG.edges:
            compG.edges[i,j]['neg_length'] = - compG.edges[i,j]['length']
            
        matching = nx.max_weight_matching( compG.subgraph(odd_degree_nodes), maxcardinality=True, weight='neg_length')
        perf_match_edges = list(matching)

        edge_config = { edge : {"stroke_color": RED} for edge in perf_match_edges }
        self.play(graph.animate.add_edges(*perf_match_edges, edge_config=edge_config))
        # print(currG.edges)
        # currG.add_edges_from(matching)
        # print(currG.edges)

        self.wait()
        graph[odd_degree_nodes[0]].set_default()
        print(graph[odd_degree_nodes[0]].color, graph[odd_degree_nodes[0]].stroke_color)
        # self.play(*[graph[v].animate.set_fill(WHITE, 1) for v in odd_degree_nodes])
        self.play(*[graph[v].animate.set_stroke(WHITE, width=0) for v in odd_degree_nodes])

        self.play(*[graph.edges[edge].animate.set_stroke(WHITE) for edge in perf_match_edges])
        self.wait()

        return matching

    def eulerian_circuit(self, graph, currG, matching, n):
        M = nx.MultiGraph()

        M.add_nodes_from(range(n))

        M.add_edges_from(currG.edges())
        M.add_edges_from(matching)

        initial_tour = list ( nx.eulerian_circuit(M,source=0) )
        fix_init_tour = []
        for edge in initial_tour:
            if edge[0] > edge[1]:
                edge = (edge[1], edge[0])
                fix_init_tour.append(edge)
            else:
                fix_init_tour.append(edge)
        print(fix_init_tour)
        print(M.edges)
        print(graph.edges)

        for edge in initial_tour:
            if not edge in graph.edges:
                edge = (edge[1], edge[0])
            self.play(graph.edges[edge].animate.set_stroke(RED, width=8))
            self.play(graph.edges[edge].animate.set_stroke(WHITE, width=4))
        self.wait()
        tour = [ 0 ]
        for (i,j) in initial_tour:
            if j not in tour:
                tour.append(j)
        
        for v in tour:
            self.play(graph[v].animate.set_stroke(YELLOW, width=4))
            self.play(graph[v].animate.set_stroke(WHITE, width=0))

        print(tour)
        tour_edges = [ (tour[i-1],tour[i]) for i in range(n) ]
        self.wait()
        self.play(graph.animate.remove_edges(*graph.edges))
        self.wait()
        self.play(graph.animate.add_edges(*tour_edges))




