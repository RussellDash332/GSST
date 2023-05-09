import numpy as np
import networkx as nx
from copy import deepcopy
from Graph import Graph
import os
import imageio

WALL_TIME = 50
class GSST:
    def __init__(self, graph: "Graph"=None, tree: "Graph"=None, filename='test_run') -> None:
        if tree == None:
            self.graph = graph
            self.spanning_tree, self.B = self.graph.get_spanning_tree()
        elif graph == None:
            self.graph = tree
            self.spanning_tree, self.B = tree, []
        else:
            raise ValueError("Either graph or tree should not be None")
        
        self.num_searcher = self.spanning_tree.mu
        self.t = 0
        self.N = self.graph.g.number_of_nodes()
        
        #Visitation status
        self.to_visit = {i for i in range(0, self.N - 1)}
        self.visited = {i: False for i in range(0, self.N - 1)}
        self.visited['sta'] = True
        
        #Searcher locations
        self.searcher_locations = ['sta' for _ in range(self.num_searcher)]
        self.searcher_per_locations ={i: 0 for i in range(self.N - 1)}
        self.searcher_per_locations['sta'] = self.num_searcher
        
        self.set_node_attributes()
        self.history = [deepcopy(self.spanning_tree)]
        
        self.fn = filename
    
    def set_node_attributes(self):
        nx.set_node_attributes(self.spanning_tree.g, self.searcher_per_locations, 'searcher_number')
        nx.set_node_attributes(self.spanning_tree.g, self.visited, 'visited')
        
    def search(self, visualize=False):
        print('Search started with {} searchers'.format(self.num_searcher))
        if visualize:
            self.png_saved = True
            self.history[-1].visualize(save=True, filename=f'{self.fn}_{self.t}.png')
            
        ## Might not be correct, think about labels?
        def can_move_searcher(node):
            if self.searcher_per_locations[node] > 1:
                return True
            
            if self.searcher_per_locations[node] == 0:
                raise ValueError("No searcher at node {}".format(node))
            
            unvisited = 0
            for neighbor in self.spanning_tree.g[node]:
                if not self.visited[neighbor]:
                    unvisited += 1
                    
            # print(f'Node {node} has {unvisited} unvisited neighbors among {self.spanning_tree.g[node]}')
            return unvisited <= 1
        
        def move_searcher(num, node, positive_edge=True):
            prev_node = self.searcher_locations[num]
            self.searcher_per_locations[prev_node] -= 1
            
            self.searcher_locations[num] = node
            self.searcher_per_locations[node] += 1
            self.visited[node] = True
            if node in self.to_visit:
                self.to_visit.remove(node)
            
            # print(f'Searcher {num} moves from {prev_node} to {node}')
            if positive_edge:
                self.spanning_tree.g[prev_node][node]['label'] -= 1
            else:
                self.spanning_tree.g[prev_node][node]['label'] += 1
            
        while len(self.to_visit) != 0:
            # print(f'At time {self.t}, {(self.to_visit)} nodes left to visit')
            if self.t > WALL_TIME:
                print(f'INTERRUPTED!\nTime: {self.t}, Number of searchers: {self.num_searcher}, unvisited area: {self.to_visit}')
                exit()
                
            for i in range(self.num_searcher):
                node = self.searcher_locations[i]
                can_move = can_move_searcher(node)
                
                # print(f'Searcher {i} at {node} & t={self.t} can move: {can_move}')
                if not can_move:
                    continue
                
                adj = list(self.spanning_tree.g[node])
                edge_labels = np.array([self.spanning_tree.g.edges[(node, neighbor)]['label'] for neighbor in adj])      
                positive = np.where(edge_labels > 0)[0]
                negative = np.where(edge_labels < 0)[0]
                
                if len(positive) > 0:
                    idx = np.argmin(edge_labels[positive])
                    # print(edge_labels)
                    # print(positive)
                    # print(positive[idx])
                    # print(adj)
                    next_node = adj[positive[idx]]
                    move_searcher(i, next_node)
                elif len(negative) > 0:
                    next_node_idx = negative[0]
                    next_node = adj[next_node_idx]
                    move_searcher(i, next_node, positive_edge=False)
                else:
                    continue
                
            self.set_node_attributes()      
            self.history.append(deepcopy(self.spanning_tree))
            self.t += 1

            if visualize:
                self.history[-1].visualize(save=True,filename=f'{self.fn}_{self.t}.png')
    
    def visualize(self):
        fns = []
        for i in range(self.t + 1):
            if not self.png_saved:
                self.history[i].visualize(save=True, filename=f'{self.fn}_{i}.png')
            fns.append(f'{self.fn}_{i}.png')
        
        imageio.mimsave(f'{self.fn}.gif', [imageio.imread(filename) for filename in fns], duration=1)
                
        for filename in set(fns):
            os.remove(filename)