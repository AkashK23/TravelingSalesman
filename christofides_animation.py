from ast import While
from pickle import TRUE
from manim import *
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random


class MovingVertices(Scene):
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

        # minSpanTree function determines the MST of the graph and changes graphs to create an MST
        self.minSpanTree(manimG, currG, node_pos)

        # OddNodesPerfMatch function determines the odd degree nodes and performs perfect matching
        # perf_matching is an array of the edges added in the perf matching process
        perf_matching = self.OddNodesPerfMatch(manimG, currG, compG)

        # eulerian_circuit does a eulerian circuit around the graph
        # does a circuit around the graph that is the MST graph + perfect matching
        self.eulerian_circuit(manimG, currG, perf_matching, n)

    def minSpanTree(self, manimG, currG, node_pos):
        
        # mst_graph is the mst of the input complete graph
        mst_graph = nx.minimum_spanning_tree(currG, weight='length')

        # determine edges to remove from complete graph to create MST
        edges_to_remove = set(currG.edges).difference(set(mst_graph.edges))

        # update currG to now be the MST graph
        currG.remove_edges_from(edges_to_remove)

        # remove edges from manim graph to create MST and animate
        self.play(manimG.animate.remove_edges(*edges_to_remove))
        self.wait()
    
    def OddNodesPerfMatch(self, graph, currG, compG):

        # list of the odd degree vertices
        odd_degree_verts = [ i for i in currG.nodes if currG.degree(i) % 2 ]

        # make odd-deg vertices red
        self.play(*[graph[v].animate.set_stroke(RED, width=4) for v in odd_degree_verts])
        self.wait()

        # need to convert the lengths to negative value
        # this is because there a max_weight_matching function, and we need min weight
        for i,j in compG.edges:
            compG.edges[i,j]['neg_length'] = - compG.edges[i,j]['length']
            
        # perfect matching function
        matching = nx.max_weight_matching( compG.subgraph(odd_degree_verts), maxcardinality=True, weight='neg_length')
        
        # edges added for perfect matching
        perf_match_edges = list(matching)

        # add perfect matching edges, make them red
        edge_config = { edge : {"stroke_color": RED} for edge in perf_match_edges }
        self.play(graph.animate.add_edges(*perf_match_edges, edge_config=edge_config))
        self.wait()
        
        # make red vertices white
        self.play(*[graph[v].animate.set_stroke(WHITE, width=0) for v in odd_degree_verts])

        # make the perfect match edges white
        self.play(*[graph.edges[edge].animate.set_stroke(WHITE) for edge in perf_match_edges])
        self.wait()

        return matching

    def eulerian_circuit(self, graph, currG, matching, n):
        # need a multigraph because perfect edges adds duplicate edges sometimes
        multi_graph = nx.MultiGraph()

        # add the vertices into multigraph
        multi_graph.add_nodes_from(range(n))

        # add the MST edges and the perfect matching edges
        multi_graph.add_edges_from(currG.edges())
        multi_graph.add_edges_from(matching)

        # determine eulerian circuit
        initial_tour = list ( nx.eulerian_circuit(multi_graph,source=0) )

        # go step by step and turn the edges red as we go through them in the eulerian circuit
        for edge in initial_tour:
            if not edge in graph.edges:
                edge = (edge[1], edge[0])
            self.play(graph.edges[edge].animate.set_stroke(RED, width=8))
            self.play(graph.edges[edge].animate.set_stroke(WHITE, width=4))
        self.wait()

        # remove duplicate visited vertices in the eulerian circuit
        tour = [ 0 ]
        for (i,j) in initial_tour:
            if j not in tour:
                tour.append(j)
        
        # visit the nodes and make them turn yellow
        # tour is the order of vertices visited in our final circuit
        for v in tour:
            self.play(graph[v].animate.set_stroke(YELLOW, width=4))
            self.play(graph[v].animate.set_stroke(WHITE, width=0))

        # finally remove all the edges and only keep the edges used in the final circuit
        tour_edges = [ (tour[i-1],tour[i]) for i in range(n) ]
        self.wait()
        self.play(graph.animate.remove_edges(*graph.edges))
        self.wait()
        self.play(graph.animate.add_edges(*tour_edges))




