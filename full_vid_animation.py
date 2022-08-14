from ast import While
from pickle import TRUE
from platform import node
from manim import *
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random
import numpy as np
import math


class MovingVertices(Scene):
    def construct(self):
        # number of cities
        n = 10
        # graph with a vertex for each city
        # Initially there is an edge between each vertex
        # This graph will continue to be updated as we remove and add edges
        currG = nx.complete_graph(n)

        # chose node locations
        node_pos = {0: [2.4139401310977497, -0.41508508922315634, 0], 1: [5.602522921758945, -2.531384029713401, 0], 2: [-0.6024190810625818, 0.09632583643947257, 0], 3: [-1.554469632585299, -0.18289114089004954, 0], 4: [4.6143445734724455, -2.9587361786239383, 0], 5: [-4.9106267737919875, 3.2443526403143164, 0], 6: [-3.1977437369282953, 3.3592881500462184, 0], 7: [-2.956389301602393, 0.051788982891457724, 0], 8: [1.6371935498777033, 2.7058680916616096, 0], 9: [-5.9805241354102625, -1.066043246033015, 0]}
        # adding a length attribute to the edges
        for i,j in currG.edges:
            loc1 = tuple(node_pos[i])
            loc2 = tuple(node_pos[j])
            currG.edges[i,j]['length'] = distance.euclidean(loc1, loc2)

        # This graph is a complete graph and will not change through the code  
        compG = currG.copy()
        
        # manim graph used for the animations
        manimG = Graph.from_networkx(currG, layout=node_pos, labels=True)

        # self.play(Create(manimG))
        # self.wait()

        # grid = NumberPlane().set_opacity(0.5).set_z_index(-2)
        # self.play(FadeIn(grid))
                
        # # minSpanTree function determines the MST of the graph and changes graphs to create an MST
        # self.minSpanTree(manimG, currG, node_pos, grid)

        # OddNodesPerfMatch function determines the odd degree nodes and performs perfect matching
        # perf_matching is an array of the edges added in the perf matching process
        num_match = 6
        pMatchG = nx.complete_graph(num_match)
        node_pos_match = {0: [2.4139401310977497, -0.41508508922315634, 0], 1: [-3.1977437369282953, 3.3592881500462184, 0], 2: [-2.956389301602393, 0.051788982891457724, 0], 3: [1.6371935498777033, 2.7058680916616096, 0], 4: [0, 2, 0], 5: [0, -1, 0]}
        for i,j in pMatchG.edges:
            loc1 = tuple(node_pos_match[i])
            loc2 = tuple(node_pos_match[j])
            pMatchG.edges[i,j]['length'] = distance.euclidean(loc1, loc2)
        compMatch = pMatchG.copy()
        manimMatch = Graph.from_networkx(pMatchG, layout=node_pos_match, labels=True)
        manimMatch.remove_edges(*manimMatch.edges)

        grid = NumberPlane().set_opacity(0.5).set_z_index(-2)
        self.play(Create(manimMatch), FadeIn(grid))
        self.wait()

       
        # self.play(FadeIn(grid))

        perf_matching = self.OddNodesPerfMatch(manimMatch, pMatchG, compMatch, node_pos_match, grid)

        # eulerian_circuit does a eulerian circuit around the graph
        # does a circuit around the graph that is the MST graph + perfect matching
        # self.eulerian_circuit(manimG, currG, perf_matching, n)

    def minSpanTree(self, manimG, currG, node_pos, grid):
        
        spanning_trees = []
        # mst_graph is the mst of the input complete graph
        mst_graph = nx.minimum_spanning_tree(currG, weight='length')
        span_tree1 = mst_graph.edges

        # determine edges to remove from complete graph to create MST
        edges_to_remove = set(currG.edges).difference(set(mst_graph.edges))
        edges_to_remove = list(edges_to_remove)

        # update currG to now be the MST graph
        currG.remove_edges_from(edges_to_remove)
        
        # remove edges from manim graph to create MST and animate
        self.play(manimG.animate.remove_edges(*edges_to_remove))

        span_tree2 = [(2, 5), (2, 6), (7, 9), (3, 7), (0, 1), (2, 8), (2, 3), (2, 4), (2,0)]
        self.play(manimG.animate.remove_edges(*manimG.edges))
        self.play(manimG.animate.add_edges(*span_tree2))

        span_tree3 = [(1, 4), (4, 0), (0, 2), (2, 3), (3, 7), (7, 9), (9, 5), (5, 6), (6,8)]
        self.play(manimG.animate.remove_edges(*manimG.edges))
        # self.play(manimG.animate.add_edges(*span_tree3))

        spanning_trees.append(span_tree3)
        spanning_trees.append(span_tree2)
        spanning_trees.append(span_tree1)

        iter = 0

        for span_tree in spanning_trees:
            self.play(manimG.animate.add_edges(*span_tree))
            add_decimals = []
            decimals = []
            edge_sum = 0
            for edge in span_tree:
                xAvg = (node_pos[edge[0]][0] + node_pos[edge[1]][0])/2
                yAvg = (node_pos[edge[0]][1] + node_pos[edge[1]][1])/2
                xDiff = abs(node_pos[edge[0]][0] - node_pos[edge[1]][0])
                yDiff = abs(node_pos[edge[0]][1] - node_pos[edge[1]][1])
                decimal = DecimalNumber(number=math.hypot(xDiff, yDiff), num_decimal_places=1, font_size=30, stroke_width=1, color=WHITE)
                edge_sum += math.hypot(xDiff, yDiff)
                if xDiff > yDiff:
                    add_decimals.append(decimal.animate.move_to(np.array([xAvg, yAvg+0.35,0])))
                else:
                    add_decimals.append(decimal.animate.move_to(np.array([xAvg+0.35, yAvg,0])))
                decimals.append(decimal)
            
            
            edge_sum = round(edge_sum, 1)
            tc_val = Text(str(edge_sum)).set_color(WHITE)
            tc_val.scale(0.7)

            tc_val.to_corner(UR)

            self.play(*add_decimals)
            if iter == 0:
                tc_txt = Text("Total Cost: ").set_color(WHITE)
                tc_txt.scale(0.7)
                tc_txt.next_to(tc_val, LEFT)
                self.play(FadeIn(tc_txt, tc_val))
            else:
                self.play(FadeIn(tc_val))
            self.wait()
            if iter != 2:
                self.remove(*decimals)
                self.remove(tc_val)
                self.play(manimG.animate.remove_edges(*manimG.edges))

            iter += 1
        # self.remove(tc_txt)

        self.play(VGroup(grid, manimG, *decimals, tc_val, tc_txt).animate.scale(0.75).shift(2.5*LEFT))

        txt1 = Text("Notation:")

        txtM = MathTex("M").scale(0.65)
        txt2 = Text(" = Minimum Spanning Tree").scale(0.45)
        mstTxtGroup = VGroup(txtM, txt2).arrange(direction=RIGHT, buff=0.1)

        txtCost = MathTex("c(M)").scale(0.65)
        txt3 = Text(" = sum of edge weights in M").scale(0.42)
        costTxtGroup = VGroup(txtCost, txt3).arrange(direction=RIGHT, buff=0.1)

        txt4 = MathTex("c(M) = 21.4").scale(0.65)

        txt1.shift(5*RIGHT + 2*UP)
        puzzletxt2 = VGroup(mstTxtGroup, costTxtGroup, txt4).arrange(direction=DOWN, buff=0.4).shift(5*RIGHT + 0.5*DOWN)
        self.play(FadeIn(txt1, puzzletxt2))
        self.wait()

        self.play(FadeOut(puzzletxt2))

        txt5 = MathTex("T_G").scale(0.65)
        txt6 = Text(" = Optimal Tour of Graph").scale(0.5)
        tourTxt = VGroup(txt5, txt6).arrange(direction=RIGHT, buff=0.1).shift(5*RIGHT + 0.3*DOWN)
        self.play(FadeIn(tourTxt))

        self.remove(tc_txt, tc_val, *decimals)
        self.play(manimG.animate.remove_edges(*manimG.edges))
        self.wait()

        tsp_edges = [(0,4), (4,1), (1,8), (8,6), (6,5), (5,9), (9,7), (7,3), (3,2), (2,0)]
        self.play(manimG.animate.add_edges(*tsp_edges))

        self.wait()

        self.play(FadeOut(txt1, tourTxt))

        txt1 = Text("NOTE:").shift(5*RIGHT + 2*UP)
        self.play(FadeIn(txt1))

        costs = MathTex("c(M) \le c(T_G)").scale(0.65).next_to(txt1, DOWN, buff = 0.5)
        self.play(FadeIn(costs))

        bullet_tex = ["c(T_G - e) \le c(T_G)",
                      "c(M) \le c(T_G - e)",
                      "c(M) \le c(T_G)"]

        blist = BulletedList(*bullet_tex, tex_environment='align*').scale(0.65).next_to(costs, DOWN, buff = 1.2)

        self.play(FadeIn(blist))
        self.play(blist.animate.fade_all_but(0, opacity=0.2))
        self.wait()

        self.play(manimG.animate.remove_edges((6,8)))
        self.wait()

        self.play(manimG.animate.add_edges((6,8)))
        self.play(manimG.animate.remove_edges((5,9)))
        self.wait()

        self.play(manimG.animate.add_edges((5,9)))
        self.play(manimG.animate.remove_edges((8,1)))

        self.play(blist.animate.fade_all_but(1, opacity=0.2))
        self.wait()

        self.play(manimG.animate.add_edges((8,1)))
        self.play(manimG.animate.remove_edges((6,8)))
        self.wait()

        self.play(manimG.animate.add_edges((6,8)))
        self.play(manimG.animate.remove_edges((5,9)))
        self.wait()


        self.play(manimG.animate.remove_edges(*manimG.edges))
        self.play(manimG.animate.add_edges(*span_tree1))

        self.wait()

        self.play(blist.animate.fade_all_but(2, opacity=0.2))

        self.wait()

        self.play(FadeOut(blist, txt1, costs))
        self.play(VGroup(grid, manimG).animate.scale(4/3).shift(2.5*RIGHT))

        self.wait()
        self.play(FadeOut(grid, manimG))

    def OddNodesPerfMatch(self, manimG, currG, compG, node_pos, grid):

        # # list of the odd degree vertices
        # odd_degree_verts = [ i for i in currG.nodes if currG.degree(i) % 2 ]

        # # make odd-deg vertices red
        # self.play(*[graph[v].animate.set_stroke(RED, width=4) for v in odd_degree_verts])
        # self.wait()

        # need to convert the lengths to negative value
        # this is because there a max_weight_matching function, and we need min weight
        for i,j in compG.edges:
            compG.edges[i,j]['neg_length'] = - compG.edges[i,j]['length']
            
        # perfect matching function
        matching = nx.max_weight_matching( compG.subgraph(manimG.vertices), maxcardinality=True, weight='neg_length')
        
        # edges added for perfect matching
        perf_match_edges = list(matching)

        # add perfect matching edges, make them red
        # edge_config = { edge : {"stroke_color": RED} for edge in perf_match_edges }
        self.play(manimG.animate.add_edges(*perf_match_edges))
        self.wait()
        
        # make red vertices white
        # self.play(*[graph[v].animate.set_stroke(WHITE, width=0) for v in graph.vertices])

        # make the perfect match edges white
        # self.play(*[graph.edges[edge].animate.set_stroke(WHITE) for edge in perf_match_edges])
        # self.wait()

        self.play(manimG.animate.remove_edges(*manimG.edges))
        match2 = [(1,2), (4,5), (3, 0)]
        self.play(manimG.animate.add_edges(*match2))
        self.wait()

        self.play(manimG.animate.remove_edges(*manimG.edges))
        match3 = [(1,3), (4,0), (2, 5)]
        # self.play(graph.animate.add_edges(*match3))
        # self.wait()

        matches = [match3, match2, perf_match_edges]
        iter = 0
        for match in matches:
            self.play(manimG.animate.add_edges(*match))
            add_decimals = []
            decimals = []
            edge_sum = 0
            for edge in match:
                xAvg = (node_pos[edge[0]][0] + node_pos[edge[1]][0])/2
                yAvg = (node_pos[edge[0]][1] + node_pos[edge[1]][1])/2
                xDiff = abs(node_pos[edge[0]][0] - node_pos[edge[1]][0])
                yDiff = abs(node_pos[edge[0]][1] - node_pos[edge[1]][1])
                decimal = DecimalNumber(number=math.hypot(xDiff, yDiff), num_decimal_places=1, font_size=30, stroke_width=1, color=WHITE)
                edge_sum += math.hypot(xDiff, yDiff)
                if xDiff > yDiff:
                    add_decimals.append(decimal.animate.move_to(np.array([xAvg, yAvg+0.35,0])))
                else:
                    add_decimals.append(decimal.animate.move_to(np.array([xAvg+0.35, yAvg,0])))
                decimals.append(decimal)
            
            
            edge_sum = round(edge_sum, 1)
            tc_val = Text(str(edge_sum)).set_color(WHITE)
            tc_val.scale(0.7)

            tc_val.to_corner(UR)

            self.play(*add_decimals)
            if iter == 0:
                tc_txt = Text("Total Cost: ").set_color(WHITE)
                tc_txt.scale(0.7)
                tc_txt.next_to(tc_val, LEFT)
                self.play(FadeIn(tc_txt, tc_val))
            else:
                self.play(FadeIn(tc_val))
            self.wait()
            if iter != 2:
                self.remove(*decimals)
                self.remove(tc_val)
                self.play(manimG.animate.remove_edges(*manimG.edges))

            iter += 1
        
        self.play(VGroup(grid, manimG, *decimals, tc_val, tc_txt).animate.scale(0.75).shift(2.5*LEFT))

        txt1 = Text("Notation:")

        txtM = MathTex("P").scale(0.65)
        txt2 = Text(" = Min-Cost Perfect Match Edges").scale(0.4)
        mstTxtGroup = VGroup(txtM, txt2).arrange(direction=RIGHT, buff=0.1)

        txtCost = MathTex("S").scale(0.65)
        txt3 = Text(" = Collection of Vertices").scale(0.42)
        costTxtGroup = VGroup(txtCost, txt3).arrange(direction=RIGHT, buff=0.1)

        txtCost2 = MathTex("T_S").scale(0.65)
        txt4 = Text(" = Optimal Tour of S").scale(0.42)
        costTxtGroup2 = VGroup(txtCost2, txt4).arrange(direction=RIGHT, buff=0.1)

        txtCost3 = MathTex("P_b").scale(0.65)
        txt5 = Text(" = Blue Edges").scale(0.42)
        costTxtGroup3 = VGroup(txtCost3, txt5).arrange(direction=RIGHT, buff=0.1)

        txtCost4 = MathTex("P_g").scale(0.65)
        txt6 = Text(" = Green Edges").scale(0.42)
        costTxtGroup4 = VGroup(txtCost4, txt6).arrange(direction=RIGHT, buff=0.1)

        txt1.shift(5*RIGHT + 2*UP)
        puzzletxt2 = VGroup(mstTxtGroup, costTxtGroup, costTxtGroup2, costTxtGroup3, costTxtGroup4).arrange(direction=DOWN, buff=0.2, aligned_edge=LEFT).shift(5*RIGHT + 0.5*DOWN)
        self.play(FadeIn(txt1, puzzletxt2))
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
