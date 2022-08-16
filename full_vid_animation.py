from ast import While
import graphlib
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

        self.introScene()
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

        grid = NumberPlane().set_opacity(0.5).set_z_index(-2)
        # self.play(FadeIn(grid))
                
        # minSpanTree function determines the MST of the graph and changes graphs to create an MST
        remem_group, remem_list = self.minSpanTree(manimG, currG, node_pos, grid)
        
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
        # self.play(Create(manimMatch), FadeIn(grid))
        # self.wait()

        remem_group, remem_list = self.OddNodesPerfMatch(manimMatch, pMatchG, compMatch, node_pos_match, grid, remem_group, remem_list)


        num_match = 6
        eulerianG = nx.complete_graph(num_match)
        node_pos_eul = {0: [-3, -2, 0], 1: [-3, 0, 0], 2: [0, -2, 0], 3: [0, 2, 0], 4: [3, -2, 0], 5: [3, 1, 0]}
        eul_edges = [(0, 1), (0, 2), (1, 2), (1, 3), (1, 5), (2, 5), (2, 4), (3, 5), (4, 5)]
        for i,j in eulerianG.edges:
            loc1 = tuple(node_pos_eul[i])
            loc2 = tuple(node_pos_eul[j])
            eulerianG.edges[i,j]['length'] = distance.euclidean(loc1, loc2)
        compEul = eulerianG.copy()
        manimEul = Graph.from_networkx(eulerianG, layout=node_pos_eul, labels=True)
        manimEul.remove_edges(*manimEul.edges)
        manimEul.add_edges(*eul_edges)

        grid = NumberPlane().set_opacity(0.5).set_z_index(-2)
        

        self.traceEulerian(manimEul, eulerianG, node_pos_eul, grid)
        # # eulerian_circuit does a eulerian circuit around the graph
        # # does a circuit around the graph that is the MST graph + perfect matching
        grid = NumberPlane().set_opacity(0.5).set_z_index(-2)

        manimG = Graph.from_networkx(currG, layout=node_pos, labels=True)
        self.algorithm(manimG, currG, compG, node_pos, grid, remem_group, remem_list)

    def introScene(self):
        mst_title = Text("Christofides Algorithm").to_edge(UP)
        ul = Underline(mst_title)
        self.add(mst_title, ul)

        appTxt = Text("3/2 Approximation Algorithm").next_to(mst_title, DOWN).scale(0.75)

        self.play(FadeIn(appTxt))

        img = ImageMobject("./images/christofides_pic.jpeg")
        img.next_to(appTxt, DOWN, buff=0.3)
        
        self.play(FadeIn(img))
        self.wait()

        self.play(FadeOut(img))
        self.remove(mst_title, ul, appTxt)

        self.wait()
        mst_title = Text("Preliminary Concepts").to_edge(UP)
        self.play(Write(mst_title))
        self.wait()
        self.play(Unwrite(mst_title))
        


    def minSpanTree(self, manimG, currG, node_pos, grid):
        
        grid_graph = VGroup(grid, manimG).scale(0.73)
        self.play(Create(grid_graph))

        mst_title = Text("Minimum Spanning Trees").to_edge(UP).scale(0.8)
        self.play(Write(mst_title))
        
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

        span_tree2 = [(2, 5), (2, 6), (7, 9), (3, 7), (0, 1), (0, 8), (2, 3), (2, 4), (2,0)]
        self.play(manimG.animate.remove_edges(*manimG.edges))
        self.play(manimG.animate.add_edges(*span_tree2))

        span_tree3 = [(1, 4), (4, 0), (0, 2), (2, 3), (3, 7), (7, 9), (9, 5), (5, 6), (6,8)]
        self.play(manimG.animate.remove_edges(*manimG.edges))
        # self.play(manimG.animate.add_edges(*span_tree3))

        spanning_trees.append(span_tree3)
        spanning_trees.append(span_tree2)
        spanning_trees.append(span_tree1)

        iter = 0

        # dec_group = VGroup()

        for span_tree in spanning_trees:
            self.play(manimG.animate.add_edges(*span_tree))
            add_decimals = []
            dec_group = VGroup()

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
                    decimal.move_to(np.array([xAvg, yAvg+0.3,0]))
                    dec_group += decimal
                else:
                    decimal.move_to(np.array([xAvg+0.3, yAvg,0]))
                    dec_group += decimal
                decimals.append(decimal)
            
            
            edge_sum = round(edge_sum, 1)
            tc_val = Text(str(edge_sum)).set_color(WHITE)
            tc_val.scale(0.7)

            
            
            

            if iter == 0:
                tc_val.to_corner(UR)
                dec_group += tc_val
                dec_group.scale(0.73)

                tc_txt = Text("Total Cost: ").set_color(WHITE)
                tc_txt.scale(0.5)
                tc_txt.next_to(tc_val, LEFT)

                
                # dec_group += tc_txt
                
                self.play(FadeIn(dec_group, tc_txt))
            else:
                tc_val.next_to(tc_txt, RIGHT)
                tc_val.scale(0.73)


                dec_group.scale(0.73)
                self.play(FadeIn(tc_val, dec_group))
            self.wait()
            if iter != 2:
                dec_group -= tc_txt
                self.play(FadeOut(dec_group, tc_val))
                dec_group -= tc_val
                # self.play(FadeOut(tc_val))
                self.play(manimG.animate.remove_edges(*manimG.edges))

            iter += 1
        # self.remove(tc_txt)

        self.play(VGroup(grid, manimG, *decimals, tc_val, tc_txt).animate.shift(2.5*LEFT))

        txt1 = Text("Notation:")

        txtM = MathTex("M").scale(0.65)
        txt2 = Text(" = Minimum Spanning Tree").scale(0.45)
        mstTxtGroup = VGroup(txtM, txt2).arrange(direction=RIGHT, buff=0.1)

        txtCost = MathTex("c(M)").scale(0.65)
        txt3 = Text(" = sum of edge weights in M").scale(0.42)
        costTxtGroup = VGroup(txtCost, txt3).arrange(direction=RIGHT, buff=0.1)

        txt4 = MathTex("c(M) = 21.4").scale(0.65)

        txt5 = MathTex("G").scale(0.65)
        txt6 = Text(" = Current Graph").scale(0.5)
        gTxt = VGroup(txt5, txt6).arrange(direction=RIGHT, buff=0.1)

        txt7 = MathTex("T_G").scale(0.65)
        txt8 = Text(" = Optimal Tour of Graph").scale(0.5)
        tourTxt = VGroup(txt7, txt8).arrange(direction=RIGHT, buff=0.1)
        

        txt1.shift(5*RIGHT + 2*UP)
       
        puzzletxt2 = VGroup(mstTxtGroup, costTxtGroup, txt4, gTxt, tourTxt).arrange(direction=DOWN, buff=0.2,  aligned_edge=LEFT).shift(5*RIGHT + 0.5*DOWN)

        self.play(FadeIn(txt1, puzzletxt2))
        puzzletxt2.set_opacity(0.3)
        mstTxtGroup.set_opacity(1)

        # self.play(txtM.animate.set_color_by_tex_to_color_map({"M": PURPLE}))
        # self.play(txtCost.animate.set_color_by_tex_to_color_map({"M": PURPLE}))
        # self.play(txt4.animate.set_color_by_tex_to_color_map({"M": PURPLE}))

        # self.play(*[manimG.edges[e].animate.set_stroke(color=PURPLE, width=6) for e in manimG.edges])
        self.wait()

        mstTxtGroup.set_opacity(0.3)
        costTxtGroup.set_opacity(1)
        txt4.set_opacity(1)
        self.play(Circumscribe(tc_val, Circle))
        self.wait()

        costTxtGroup.set_opacity(0.3)
        txt4.set_opacity(0.3)
        gTxt.set_opacity(1)
        self.play(AnimationGroup(*[Flash(manimG[v]) for v in manimG.vertices]))
        self.wait()

        gTxt.set_opacity(0.3)
        tourTxt.set_opacity(1)
        self.remove(tc_txt, tc_val, *decimals)
        self.play(manimG.animate.remove_edges(*manimG.edges))
        self.wait()

        tsp_edges = [(0,4), (4,1), (1,8), (8,6), (6,5), (5,9), (9,7), (7,3), (3,2), (2,0)]
        self.play(manimG.animate.add_edges(*tsp_edges))

        self.wait()




        # self.play(FadeOut(puzzletxt2))

        # txt5 = MathTex("T_G").scale(0.65)
        # txt6 = Text(" = Optimal Tour of Graph").scale(0.5)
        # tourTxt = VGroup(txt5, txt6).arrange(direction=RIGHT, buff=0.1).shift(5*RIGHT + 0.3*DOWN)
        # self.play(FadeIn(tourTxt))

        # self.remove(tc_txt, tc_val, *decimals)
        # self.play(manimG.animate.remove_edges(*manimG.edges))
        # self.wait()

        # tsp_edges = [(0,4), (4,1), (1,8), (8,6), (6,5), (5,9), (9,7), (7,3), (3,2), (2,0)]
        # self.play(manimG.animate.add_edges(*tsp_edges))

        # self.wait()

        self.play(FadeOut(puzzletxt2, txt1))

        txt1 = Text("NOTE:").shift(5*RIGHT + 2*UP)
        self.play(FadeIn(txt1))

        costs = MathTex("c(M) \le c(T_G)").scale(0.65).next_to(txt1, DOWN, buff = 0.5)
        self.play(FadeIn(costs))

        bullet_tex = ["c(T_G - e) \le c(T_G)",
                      "c(M) \le c(T_G - e)",
                      "c(M) \le c(T_G)"]

        blist = BulletedList(*bullet_tex, tex_environment='align*').scale(0.65).next_to(costs, DOWN, buff = 1.2)
        # blist.set_color_by_tex_to_color_map({"M": PURPLE})

        self.play(FadeIn(blist))
        self.play(blist.animate.fade_all_but(0, opacity=0.3))
        self.wait()

        self.play(manimG.animate.remove_edges((6,8)))
        self.wait()

        self.play(manimG.animate.add_edges((6,8)))
        self.play(manimG.animate.remove_edges((5,9)))
        self.wait()

        self.play(manimG.animate.add_edges((5,9)))
        self.play(manimG.animate.remove_edges((8,1)))

        self.play(blist.animate.fade_all_but(1, opacity=0.3))
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

        self.play(blist.animate.fade_all_but(2, opacity=0.3))
        self.wait()

        remem = Text("Remember: ").to_corner(DL).scale(0.8)
        self.play(Write(remem))
        costs.scale(1.1)
        self.play(costs.animate.next_to(remem, RIGHT, buff = 0.4))
        surr_rec = SurroundingRectangle(costs)
        self.play(Create(surr_rec))
        

        remem_group = VGroup(remem, costs)
        remem_list = [remem, costs, surr_rec]

        self.play(FadeOut(blist, txt1))
        self.play(Unwrite(mst_title))
        self.play(VGroup(grid, manimG).animate.shift(2.5*RIGHT))

        self.wait()
        self.play(FadeOut(grid, manimG))

        return remem_group, remem_list

    def OddNodesPerfMatch(self, manimG, currG, compG, node_pos, grid, remem_group, remem_list):

        # # list of the odd degree vertices
        # odd_degree_verts = [ i for i in currG.nodes if currG.degree(i) % 2 ]

        # # make odd-deg vertices red
        # self.play(*[graph[v].animate.set_stroke(RED, width=4) for v in odd_degree_verts])
        # self.wait()

        # need to convert the lengths to negative value
        # this is because there a max_weight_matching function, and we need min weight
        graph_grid = VGroup(grid, manimG).scale(0.73)
        self.play(Create(graph_grid))
        mst_title = Text("Minimum Cost Perfect Matching").to_edge(UP).scale(0.8)
        self.play(Write(mst_title))

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
            dc_group = VGroup()
            grid3 = NumberPlane().set_opacity(0).set_z_index(-2)
            dc_group += grid3
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
                    # add_decimals.append(decimal.animate.move_to(np.array([xAvg, yAvg+0.25,0])))
                    decimal.move_to(np.array([xAvg, yAvg+0.3,0]))
                    dc_group += decimal
                else:
                    # add_decimals.append(decimal.animate.move_to(np.array([xAvg+0.25, yAvg,0])))
                    decimal.move_to(np.array([xAvg+0.3, yAvg,0]))
                    dc_group += decimal
                decimals.append(decimal)
            
            
            edge_sum = round(edge_sum, 1)
            tc_val = Text(str(edge_sum)).set_color(WHITE)
            tc_val.scale(0.7)

            

            
            if iter == 0:
                tc_val.to_corner(UR)
                dc_group += tc_val
                dc_group.scale(0.73)
                tc_txt = Text("Total Cost: ").set_color(WHITE)
                tc_txt.scale(0.52)
                tc_txt.next_to(tc_val, LEFT)

                self.play(FadeIn(tc_txt))
                self.play(FadeIn(dc_group))
                
            else:
                
                tc_val.next_to(tc_txt, RIGHT)
                tc_val.scale(0.73)

                dc_group.scale(0.73)
                self.play(FadeIn(dc_group))
            self.wait()
            if iter != 2:
                # self.remove(*decimals)
                # self.remove(tc_val)
                self.play(FadeOut(dc_group, tc_val))
                self.play(manimG.animate.remove_edges(*manimG.edges))

            iter += 1
        
        self.play(VGroup(grid, manimG, *decimals, tc_val, tc_txt).animate.shift(2.5*LEFT))

        txt1 = Text("Notation:")

        txtM = MathTex("P").scale(0.65)
        txt2 = Text(" = Min-Cost Perfect Match Edges").scale(0.4)
        mstTxtGroup = VGroup(txtM, txt2).arrange(direction=RIGHT, buff=0.1)

        txtCost = MathTex("S").scale(0.65)
        txt3 = Text(" = Collection of Nodes").scale(0.42)
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
        puzzletxt2.set_opacity(0.3)
        mstTxtGroup.set_opacity(1)
        self.play(FadeIn(txt1, puzzletxt2))
        self.wait()

        mstTxtGroup.set_opacity(0.3)
        costTxtGroup.set_opacity(1)
        self.play(FadeOut(*decimals, tc_txt, tc_val))
        self.play(manimG.animate.remove_edges(*manimG.edges))
        self.play(AnimationGroup(*[Flash(manimG[v]) for v in manimG.vertices]))
        self.wait()

        costTxtGroup.set_opacity(0.3)
        costTxtGroup2.set_opacity(1)
        opTour_S = [(0,3), (3,4), (4,1), (1,2), (2,5), (5, 0)]
        self.play(manimG.animate.add_edges(*opTour_S))
        self.wait()

        costTxtGroup2.set_opacity(0.3)
        costTxtGroup3.set_opacity(1)
        costTxtGroup4.set_opacity(1)
        blue_edges = [(0,3), (4,1), (2,5)]
        green_edges = [(3,4), (1,2), (5, 0)]

        edge_coloring = []
        for edge in opTour_S:
            if edge in blue_edges:
                edge_coloring.append(manimG.edges[edge].animate.set_stroke(DARK_BLUE, width=8))
            else:
                edge_coloring.append(manimG.edges[edge].animate.set_stroke(GREEN_C, width=8))
        self.play(*edge_coloring)
        self.wait()

        self.play(FadeOut(puzzletxt2, txt1))


        txt1 = Text("NOTE:").shift(5*RIGHT + 2*UP)
        self.play(FadeIn(txt1))

        costs = MathTex("c(P) \le \\frac{1}{2} c(T_S)").scale(0.75).next_to(txt1, DOWN, buff = 0.5)
        self.play(FadeIn(costs))

        bullet_tex = ["c(P) \le c(P_b)",
                      "c(P) \le c(P_g)",
                      "c(P) \le \\frac{1}{2} [c(P_b)+c(P_g)])",
                      "c(P) \le \\frac{1}{2} c(T_S))"]

        blist = BulletedList(*bullet_tex, tex_environment='align*').scale(0.65).next_to(costs, DOWN, buff = 1)

        self.play(FadeIn(blist))

        
        self.play(blist.animate.fade_all_but(0, opacity=0.3))
        self.play(*[manimG.edges[edge].animate.set_opacity(0) for edge in green_edges])
        self.wait()

        self.play(blist.animate.fade_all_but(1, opacity=0.3))
        self.play(*[manimG.edges[edge].animate.set_opacity(0) for edge in blue_edges])
        self.play(*[manimG.edges[edge].animate.set_opacity(1) for edge in green_edges])
        self.wait()

        self.play(blist.animate.fade_all_but(2, opacity=0.3))
        self.play(*[manimG.edges[edge].animate.set_opacity(1) for edge in blue_edges])
        self.wait()

        
        self.play(blist.animate.fade_all_but(3, opacity=0.3))
        self.wait()

        self.play(costs.animate.next_to(remem_group, RIGHT, buff = 0.4))
        surr_rec = SurroundingRectangle(costs)
        self.play(Create(surr_rec))

        remem_group += costs
        remem_list.append(costs)
        remem_list.append(surr_rec)

        self.play(FadeOut(grid, manimG, blist, txt1))
        self.play(Unwrite(mst_title))
        return remem_group, remem_list

    def traceEulerian(self, manimG, currG, node_pos, grid):
        mst_title = Text("Eulerian Circuit").to_edge(UP).scale(0.8)
        self.play(Write(mst_title))

        labels = []
        arrows = []

        euler_circ = [(0, 2), (2, 4), (4, 5), (5, 3), (3, 1), (1, 5), (5, 2), (2, 1), (1, 0)]
        eul_label = Text("Eulerian Circuit: ").set_color(WHITE)
        eul_label.to_corner(UL)

        grid_graph = VGroup(grid, manimG, eul_label).scale(0.73)
        self.play(Create(grid_graph))
        self.wait()

        labels.append(eul_label)

        label_old = Text("0").set_color(WHITE).scale(0.6)
        labels.append(label_old)
        label_old.next_to(eul_label, RIGHT, buff=0.1)
        # self.play(FadeIn(eul_label))
        self.play(FadeIn(label_old))
        for edge in euler_circ:
            arrow = Line(manimG[edge[0]], manimG[edge[1]]).add_tip().set_color(RED)
            arrows.append(arrow)
            self.play(FadeIn(arrow))
            label_new = Text(str(edge[1])).set_color(WHITE).scale(0.6)
            labels.append(label_new)
            label_new.next_to(label_old, RIGHT, buff=0.4)
            self.play(FadeIn(label_new))
            label_old = label_new

        self.play(FadeOut(*arrows, *labels))

        self.wait()

        self.play(manimG.animate.add_edges((3,4)))

        self.wait()

        odd_arrows = []
        odd_arrows.append(Line(manimG[0], manimG[2]).add_tip().set_color(RED))
        odd_arrows.append(Line(manimG[2], manimG[4]).add_tip().set_color(RED))
        odd_arrows.append(Line(manimG[4], manimG[5]).add_tip().set_color(RED))
        odd_arrows.append(Line(manimG[5], manimG[3]).add_tip().set_color(RED))
        odd_arrows.append(Line(manimG[3], manimG[4]).add_tip().set_color(RED))

        for arrow in odd_arrows:
            self.play(FadeIn(arrow))
        
        self.wait()

        self.play(FadeOut(*odd_arrows))

        odd_arrows = []
        odd_arrows.append(Line(manimG[0], manimG[2]).add_tip().set_color(RED))
        odd_arrows.append(Line(manimG[2], manimG[5]).add_tip().set_color(RED))
        odd_arrows.append(Line(manimG[5], manimG[3]).add_tip().set_color(RED))
        odd_arrows.append(Line(manimG[3], manimG[1]).add_tip().set_color(RED))
        odd_arrows.append(Line(manimG[1], manimG[2]).add_tip().set_color(RED))
        odd_arrows.append(Line(manimG[2], manimG[4]).add_tip().set_color(RED))
        odd_arrows.append(Line(manimG[4], manimG[3]).add_tip().set_color(RED))

        for arrow in odd_arrows:
            self.play(FadeIn(arrow))
        
        self.wait()

        self.play(FadeOut(*odd_arrows))

        self.play(VGroup(grid, manimG).animate.shift(2.5*LEFT))

        txt1 = Text("NOTE:").shift(5*RIGHT + 2*UP)
        self.play(FadeIn(txt1))

        txtExp = Text("All Nodes must have")
        txtExp2 = Text("an even degree")

        expGroup = VGroup(txtExp, txtExp2).arrange(direction=DOWN, aligned_edge=LEFT, buff=0).next_to(txt1, DOWN, buff = 1.2).scale(0.6)

        # txtExp.next_to(txt1, DOWN, buff = 1.2).scale(0.5)

        self.play(FadeIn(expGroup))

        self.wait()

        everything = VGroup(grid, manimG, expGroup, txt1, mst_title)
        self.play(FadeOut(everything))
        self.wait()

    def algorithm(self, manimG, currG, compG, node_pos, grid, remem_group, remem_list):

        mst_title = Text("Christofides Algorithm").to_edge(UP, buff=0.25).scale(0.8)
        self.play(Write(mst_title))

        
        grid_graph = VGroup(grid, manimG).scale(0.73).shift(2.5*LEFT)
        self.play(Create(grid_graph))

        txt1 = Text("Steps:")

        step1 = Text("1. Build MST").scale(0.5)
        step2 = Text("2. Find Odd-Degree Nodes").scale(0.5)
        step3 = Text("3. Min-Cost Perfect Matching").scale(0.5)
        step4 = Text("4. Eulerian Circuit").scale(0.5)
        step5 = Text("5. Remove Duplicate Nodes").scale(0.5)


        txt1.shift(5*RIGHT + 2*UP)
        allSteps = VGroup(step1, step2, step3, step4, step5).arrange(direction=DOWN, buff=0.2, aligned_edge=LEFT).shift(5*RIGHT + 0.5*DOWN)
        allSteps.set_opacity(0)
        step1.set_opacity(1)

        rightSide = VGroup(txt1, allSteps)

        self.play(FadeIn(txt1, allSteps))

        mst_graph = nx.minimum_spanning_tree(currG, weight='length')
        # determine edges to remove from complete graph to create MST
        edges_to_remove = set(currG.edges).difference(set(mst_graph.edges))

        # update currG to now be the MST graph
        currG.remove_edges_from(edges_to_remove)

        # remove edges from manim graph to create MST and animate
        self.play(manimG.animate.remove_edges(*edges_to_remove))
        self.wait()

        step1.set_opacity(0.3)
        step2.set_opacity(1)

        odd_degree_verts = [ i for i in currG.nodes if currG.degree(i) % 2 ]

        # make odd-deg vertices red
        self.play(*[manimG[v].animate.set_stroke(RED, width=4) for v in odd_degree_verts])
        self.wait()

        step2.set_opacity(0.3)
        step3.set_opacity(1)

        for i,j in compG.edges:
            compG.edges[i,j]['neg_length'] = - compG.edges[i,j]['length']
            
        # perfect matching function
        matching = nx.max_weight_matching( compG.subgraph(odd_degree_verts), maxcardinality=True, weight='neg_length')
        
        # edges added for perfect matching
        perf_match_edges = list(matching)

        # add perfect matching edges, make them red
        edge_config = { edge : {"stroke_color": RED} for edge in perf_match_edges }
        self.play(manimG.animate.add_edges(*perf_match_edges, edge_config=edge_config))
        self.wait()
        
        # make red vertices white
        self.play(*[manimG[v].animate.set_stroke(WHITE, width=0) for v in odd_degree_verts])

        # make the perfect match edges white
        self.play(*[manimG.edges[edge].animate.set_stroke(WHITE) for edge in perf_match_edges])
        self.wait()

        step3.set_opacity(0.3)
        step4.set_opacity(1)

        multi_graph = nx.MultiGraph()

        # add the vertices into multigraph
        multi_graph.add_nodes_from(range(10))

        # add the MST edges and the perfect matching edges
        multi_graph.add_edges_from(currG.edges())
        multi_graph.add_edges_from(matching)

        # determine eulerian circuit
        initial_tour = list ( nx.eulerian_circuit(multi_graph,source=0) )

        labels = []
        eul_tour = []

        eul_label = Text("Eulerian Circuit: ").set_color(WHITE).scale(0.6)
        eul_label.to_edge(UP)
        eul_label.shift(0.65*DOWN)
        eul_label.shift(2.85*LEFT)
        self.play(FadeIn(eul_label))
        labels.append(eul_label)

        label_old = Text("0").set_color(WHITE).scale(0.5)
        labels.append(label_old)
        label_old.next_to(eul_label, RIGHT, buff=0.15)
        self.play(FadeIn(label_old))

        eul_tour.append(0)
        # go step by step and turn the edges red as we go through them in the eulerian circuit
        lines = []
        for edge in initial_tour:
            line = Line(manimG[edge[0]], manimG[edge[1]]).add_tip().set_color(RED)
            tip = line.get_tip()
            tip.set(width=0.28)
            lines.append(line)

            label_num = edge[1]
            eul_tour.append(label_num)

            if not edge in manimG.edges:
                edge = (edge[1], edge[0])
            self.play(manimG.animate.remove_edges(edge))
            self.play(FadeIn(line))

            label_new = Text(str(label_num)).set_color(WHITE).scale(0.5)
            labels.append(label_new)
            label_new.next_to(label_old, RIGHT, buff=0.15)
            self.play(FadeIn(label_new))
            label_old = label_new

            
            # self.play(manimG.edges[edge].animate.set_stroke(WHITE, width=4))
        self.wait()

        self.play(FadeOut(*lines))
        self.play(manimG.animate.add_edges(*perf_match_edges, *currG.edges))


        self.play(FadeOut(rightSide))

        txt1 = Text("NOTE:")

        tex1 = MathTex("E =").scale(0.5)
        text1 = Text("Eulerian Circuit").scale(0.5)
        step1 = VGroup(tex1, text1).arrange(direction=RIGHT, buff=0.1)
        step2 = MathTex("c(E) = c(M) + c(P)").scale(0.5)
        step3 = MathTex("c(M) \le c(T_G)").scale(0.5)
        step4 = MathTex("c(P) \le \\frac{1}{2} c(T_S))").scale(0.5)

        text5 = Text("Tour of subset").scale(0.5)
        tex5 = MathTex("\le").scale(0.5)
        text52 = Text(" Tour of Graph").scale(0.5)
        step5 = VGroup(text5, tex5, text52).arrange(direction=RIGHT, buff=0.1)

        step6 = MathTex("c(P) \le \\frac{1}{2} c(T_G))").scale(0.5)
        step7 = MathTex("c(E) \le c(T_G) + \\frac{1}{2} c(T_G))").scale(0.5)
        step8 = MathTex("c(E) \le \\frac{3}{2} c(T_G))").scale(0.5)

        txt1.shift(5*RIGHT + 2.5*UP)
        allSteps = VGroup(step1, step2, step3, step4, step5, step6, step7, step8).arrange(direction=DOWN, buff=0.2, aligned_edge=LEFT).shift(5*RIGHT + 0.5*DOWN)
        allSteps.set_opacity(0.3)
        step1.set_opacity(1)
        
        self.play(FadeIn(txt1, allSteps))
        self.wait()

        step1.set_opacity(0.3)
        step2.set_opacity(1)

        self.play(manimG.animate.remove_edges(*manimG.edges))

        edge_config = { edge : {"stroke_color": DARK_GREY} for edge in currG.edges }
        self.play(manimG.animate.add_edges(*currG.edges, edge_config=edge_config))
        self.wait()
        edge_config = { edge : {"stroke_color": YELLOW} for edge in matching }
        self.play(manimG.animate.add_edges(*matching, edge_config=edge_config))
        self.wait()
        self.play(*[manimG.edges[edge].animate.set_stroke(WHITE) for edge in manimG.edges])

        step2.set_opacity(0.3)
        step3.set_opacity(1)
        self.play(Wiggle(remem_list[1]))
        self.wait()

        step3.set_opacity(0.3)
        step4.set_opacity(1)
        self.play(Wiggle(remem_list[3]))
        self.wait()

        step4.set_opacity(0.3)
        step5.set_opacity(1)
        # AnimationGroup(*[Flash(manimG[v]) for v in manimG.vertices])
        self.play(AnimationGroup(*[Circumscribe(manimG[v], Circle) for v in odd_degree_verts]))

        save_edges = list(manimG.edges)
        subset_edges = [(0,1), (1,8), (8,5), (5,9), (9,7), (7,0)]
        edge_config = { edge : {"stroke_color": BLUE} for edge in subset_edges }
        lines = []
        for edge in subset_edges:
            line = Line(manimG[edge[0]], manimG[edge[1]], color=BLUE).set_stroke(width=8)
            lines.append(line)

        self.play(manimG.animate.remove_edges(*manimG.edges))
        tsp_edges = [(0,4), (4,1), (1,8), (8,6), (6,5), (5,9), (9,7), (7,3), (3,2), (2,0)]
        self.play(manimG.animate.add_edges(*tsp_edges))
        self.play(FadeIn(*lines))
        self.wait()

        self.play(FadeOut(*lines))
        self.play(manimG.animate.remove_edges(*manimG.edges))
        self.play(manimG.animate.add_edges(*save_edges))


        step5.set_opacity(0.3)
        step6.set_opacity(1)
        self.wait()

        step6.set_opacity(0.3)
        step7.set_opacity(1)
        self.wait()

        step7.set_opacity(0.3)
        step8.set_opacity(1)
        self.wait()

        visited = set()
        for i, v in enumerate(eul_tour):
            if i == len(eul_tour)-1:
                break
            if v in visited:
                self.play(Unwrite(labels[i+1]))
                print(labels[i+1:])
                rLabs = VGroup(*labels[i+1:])
                self.play(rLabs.animate.next_to(labels[i-1], RIGHT, buff=0.3))

                rem_edges = [(eul_tour[i-1], v), (v, eul_tour[i+1])]
                self.play(manimG.animate.remove_edges(rem_edges[0]))
                self.play(manimG.animate.remove_edges(rem_edges[1]))
                self.play(manimG.animate.add_edges((eul_tour[i-1], eul_tour[i+1])))
            else:
                visited.add(v)
        
        self.play(FadeOut(eul_label))
        fin_label = Text("Christofides Solution: ").set_color(WHITE).scale(0.5)
        fin_label.next_to(labels[1], LEFT, buff=0.2)
        self.play(FadeIn(fin_label))


        

        dec_group = VGroup()

        decimals = []
        edge_sum = 0
        for edge in manimG.edges:
            xAvg = (node_pos[edge[0]][0] + node_pos[edge[1]][0])/2
            yAvg = (node_pos[edge[0]][1] + node_pos[edge[1]][1])/2
            xDiff = abs(node_pos[edge[0]][0] - node_pos[edge[1]][0])
            yDiff = abs(node_pos[edge[0]][1] - node_pos[edge[1]][1])
            decimal = DecimalNumber(number=math.hypot(xDiff, yDiff), num_decimal_places=1, font_size=30, stroke_width=1, color=WHITE)
            edge_sum += math.hypot(xDiff, yDiff)
            if xDiff > yDiff:
                decimal.move_to(np.array([xAvg, yAvg+0.3,0]))
                dec_group += decimal
            else:
                decimal.move_to(np.array([xAvg+0.3, yAvg,0]))
                dec_group += decimal
            decimals.append(decimal)
        
        dec_group.scale(0.73).shift(2.5*LEFT)
        self.play(FadeIn(dec_group))
        edge_sum = round(edge_sum, 1)

        tc_val = Text(str(edge_sum)).set_color(WHITE)
        tc_val.scale(0.5)

        tc_txt = Text("Total Cost: ").set_color(WHITE)
        tc_txt.scale(0.525)
        tc_txt.next_to(tc_val, LEFT)

        tcGroup = VGroup(tc_val, tc_txt)
        tcGroup.next_to(eul_label, DOWN).shift(4*RIGHT)
        self.play(FadeIn(tcGroup))

        self.play(FadeOut(dec_group))
        self.play(FadeOut(tc_val))
        print(edge_sum, "chris")

        self.play(manimG.animate.remove_edges(*manimG.edges))
        tsp_edges = [(0,4), (4,1), (1,8), (8,6), (6,5), (5,9), (9,7), (7,3), (3,2), (2,0)]
        self.play(manimG.animate.add_edges(*tsp_edges))

        dec_group = VGroup()
        decimals = []
        edge_sum = 0
        for edge in manimG.edges:
            xAvg = (node_pos[edge[0]][0] + node_pos[edge[1]][0])/2
            yAvg = (node_pos[edge[0]][1] + node_pos[edge[1]][1])/2
            xDiff = abs(node_pos[edge[0]][0] - node_pos[edge[1]][0])
            yDiff = abs(node_pos[edge[0]][1] - node_pos[edge[1]][1])
            decimal = DecimalNumber(number=math.hypot(xDiff, yDiff), num_decimal_places=1, font_size=30, stroke_width=1, color=WHITE)
            edge_sum += math.hypot(xDiff, yDiff)
            if xDiff > yDiff:
                decimal.move_to(np.array([xAvg, yAvg-0.3,0]))
                dec_group += decimal
            else:
                decimal.move_to(np.array([xAvg+0.3, yAvg,0]))
                dec_group += decimal
            decimals.append(decimal)
        
        dec_group.scale(0.73).shift(2.5*LEFT)
        self.play(FadeIn(dec_group))
        edge_sum = round(edge_sum, 1)

        tc_val = Text(str(edge_sum)).set_color(WHITE)
        tc_val.scale(0.5)
        tc_val.next_to(tc_txt, RIGHT, buff=0.2)

        self.play(FadeIn(tc_val))
        self.wait()
        self.play(FadeOut(dec_group))
        self.play(FadeOut(tc_val))
        print(edge_sum, "optimal")




                




        self.play(Unwrite(mst_title))