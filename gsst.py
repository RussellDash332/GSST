import numpy as np
import networkx as nx
from copy import deepcopy
from graph import Graph
from random import choice, seed
import os
import imageio

WALL_TIME = 5

class Searcher:
    sid = -1
    def __init__(self, guard=False):
        Searcher.sid += 1
        self.id = Searcher.sid
        seed(self.id)
        self.color = choice(['red', 'green', 'cyan', 'yellow', 'limegreen', 'blue', 'purple', 'orange'])
        if guard: self.color = 'black'

class GSST:
    def __init__(self, graph:"Graph"=None, tree:"Graph"=None, filename='test_run') -> None:
        if tree == None:
            self.graph = graph
            self.spanning_tree, self.B = self.graph.get_spanning_tree()
        elif graph == None:
            self.graph = tree
            if self.graph.g.is_directed:
                self.graph.g = self.graph.g.to_undirected()
            self.spanning_tree, self.B = tree, []
        elif graph == None and tree == None:
            raise ValueError("Either graph or tree should not be None")

        self.num_searcher = self.spanning_tree.mu
        self.t = 0
        self.N = self.graph.g.number_of_nodes()

        # Visitation status
        self.to_visit = {i for i in self.graph.g.nodes()}
        self.to_visit.remove('sta')
        self.visited = {i: False for i in self.graph.g.nodes()}
        self.visited['sta'] = True
        self.unvisited_g = {n: self.graph.g.degree[n]
            for n in self.graph.g.nodes()}
        self.unvisited_g[graph.start] -= 1
        self.unvisited_t = {n: self.spanning_tree.g.out_degree[n]
            for n in self.spanning_tree.g.nodes()}
        self.unvisited_t[graph.start] -= 1

        # Searcher locations
        self.searcher_locations = ['sta' for _ in range(self.num_searcher)]
        self.searcher_per_locations = {i: 0 for i in self.graph.g.nodes()}
        self.searcher_per_locations['sta'] = self.num_searcher
        self.searcher_per_locations_viz = {i: [] for i in self.graph.g.nodes()}
        self.searcher_per_locations_viz['sta'] = [Searcher() for _ in range(self.num_searcher)]

        self.set_node_attributes()
        self.history = []
        self.save_history()
        self.fn = filename

    def set_node_attributes(self) -> None:
        nx.set_node_attributes(self.graph.g, self.searcher_per_locations, 'searcher_number')
        nx.set_node_attributes(self.graph.g, self.visited, 'visited')
        nx.set_node_attributes(self.graph.g, self.searcher_per_locations_viz, 'searcher_viz')

    ## Might not be correct, think about labels?
    def can_move_searcher(self, node) -> bool:
        '''
        Checks if we can move a searcher from node.
        '''
        if self.searcher_per_locations[node] > 1:
            return True
        if self.searcher_per_locations[node] == 0:
            raise ValueError("No searcher at node {}".format(node))
        return self.unvisited_t[node] <= 1

    def searcher_to_new_node(self, node) -> None:
        self.visited[node] = True
        for n in self.graph.g[node]:
            self.unvisited_g[n] -= 1
        for n in self.spanning_tree.g[node]:
            self.unvisited_t[n] -= 1

    def move_searcher(self, num, node, positive_edge=True) -> None:
        '''
        Moves the num-th searcher to node.
        '''
        prev_node = self.searcher_locations[num]
        self.searcher_per_locations[prev_node] -= 1
        prev_s = self.searcher_per_locations_viz[prev_node].pop()

        self.searcher_locations[num] = node
        self.searcher_per_locations[node] += 1
        self.searcher_per_locations_viz[node].append(prev_s)

        if self.visited[node] == False:
            self.searcher_to_new_node(node)

        if node in self.to_visit:
            self.to_visit.remove(node)

        if positive_edge:
            self.spanning_tree.g[prev_node][node]['label'] -= 1
        else:
            self.spanning_tree.g[prev_node][node]['label'] += 1

    def save_history(self) -> None:
        self.history.append(deepcopy(self.graph))

    def search_step(self) -> None:
        '''
        Performs search for a single step.
        '''
        for i in range(self.num_searcher):
            node = self.searcher_locations[i]
            can_move = self.can_move_searcher(node)

            if not can_move: continue

            adj = list(self.spanning_tree.g[node])
            edge_labels = np.array([self.spanning_tree.g.edges[(node, neighbor)]['label'] for neighbor in adj])      
            positive = np.where(edge_labels > 0)[0]
            negative = np.where(edge_labels < 0)[0]

            if len(positive) > 0:
                idx = np.argmin(edge_labels[positive])
                next_node = adj[positive[idx]]
                self.move_searcher(i, next_node)
            elif len(negative) > 0:
                next_node_idx = negative[0]
                next_node = adj[next_node_idx]
                self.move_searcher(i, next_node, positive_edge=False)
            else:
                continue

            self.after_search_step()

    def after_search_step(self) -> None:
        pass

    def search(self, visualize=False) -> None:
        '''
        Performs Algorithm 3.
        '''
        print('Search started with {} searchers'.format(self.num_searcher))
        self.png_saved = visualize
        if visualize:
            self.history[-1].visualize(save=True, filename=f'{self.fn}_{self.t}.png')
            self.history[-1].visualize(save=True, filename=f'{self.fn}_{self.t}_robot.png', robot=True)
        while len(self.to_visit) != 0:
            if self.t > WALL_TIME * self.N:
                print(f'INTERRUPTED!\nTime: {self.t}, Number of searchers: {self.num_searcher}, unvisited area: {self.to_visit}')
                exit()
            self.search_step()
            self.set_node_attributes()      
            self.save_history()
            self.t += 1
            if visualize:
                self.visualize_step(self.t)

    def visualize(self) -> None:
        fns = []
        for i in range(self.t + 1):
            if not self.png_saved:
                self.visualize_step(i)
            fns.append(f'{self.fn}_{i}.png')
        imageio.mimsave(f'{self.fn}.mp4', [imageio.imread(filename) for filename in fns], fps=2)
        fns = []
        for i in range(self.t + 1):
            if not self.png_saved:
                self.visualize_step(i)
            fns.append(f'{self.fn}_{i}_robot.png')
        imageio.mimsave(f'{self.fn}_robot.mp4', [imageio.imread(filename) for filename in fns], fps=2)

    def visualize_step(self, step: int) -> None:
        self.history[step].visualize(save=True, filename=f'{self.fn}_{step}.png', step=step)
        self.history[step].visualize(save=True, filename=f'{self.fn}_{step}_robot.png', robot=True, step=step)

class GSST_L(GSST):
    def __init__(self, graph: Graph=None, filename='test_run') -> None:
        '''
        Variant of GSST as shown in Algorithm 5.
        '''
        self.number_of_guards = 0
        self.N = graph.g.number_of_nodes()
        # Initially no guard needed at all
        self.guard_per_locations = {i: 0 for i in graph.g.nodes()}
        self.guard_per_locations['sta'] = self.number_of_guards
        self.guard_per_locations_viz = {i: [] for i in graph.g.nodes()}
        self.guard_per_locations_viz['sta'] = [Searcher(guard=True) for _ in range(self.number_of_guards)]
        self.to_guard = None

        super().__init__(graph, filename=filename)
        self.guard_locations = []
        self.guard_degree = {0: set(), 1: set()}

        self.non_tree_edge_nodes = {i: False for i in graph.g.nodes()}
        self.non_tree_edge_nodes['sta'] = False
        for edge in self.B:
            a, b = edge
            self.non_tree_edge_nodes[a] = True
            self.non_tree_edge_nodes[b] = True

    def call_guard(self, node):
        assert node != 'sta' or self.print_guard_info('Should not call guard at starting node')

        guard = None
        if len(self.guard_degree[0]) > 0:
            guard = self.guard_degree[0].pop()

        elif self.visited[node] == False:
            for g in self.guard_degree[1]:
                loc = self.guard_locations[g]
                for n in self.graph.g[loc]:
                    if node == n:
                        guard = g
                        break

        if guard == None:          
            guard = self.add_guard()

        self.move_guard(guard, node)

    def print_guard_info(self, txt):
        print()
        print('~'*50)
        print(txt)
        print(f'Number of guards: {self.number_of_guards}')
        print(f'Guard locations: {self.guard_locations}')
        print(f'Guard per locations: {self.guard_per_locations}')
        print(f'Guard degree: {self.guard_degree}')
        print('~'*50)
        return False
    
    def add_guard(self) -> None:
        assert self.guard_per_locations['sta'] == 0 or self.print_guard_info('Have existing guards available')
        self.guard_locations.append('sta')
        self.guard_per_locations['sta'] += 1
        self.guard_per_locations_viz['sta'].append(Searcher(guard=True))
        self.number_of_guards += 1
        return self.number_of_guards - 1

    def free_guard(self, guard) -> None:
        prev_loc = self.guard_locations[guard]
        if prev_loc != None:
            self.guard_per_locations[prev_loc] -= 1
            self.guard_per_locations_viz[prev_loc].pop()
        self.guard_locations[guard] = 'sta'
        self.guard_per_locations['sta'] += 1
        self.guard_per_locations_viz['sta'].append(Searcher(guard=True))

    def add_guard_in_degree(self, guard, deg):
        if deg in self.guard_degree:
            self.guard_degree[deg].add(guard)

    def check_guard_in_degree(self, guard, deg=None):
        if deg == None:
            return guard in self.guard_degree[0], 0 or guard in self.guard_degree[1], 1

        assert deg in [0,1] or self.print_guard_info(f'Wrong degree for {guard, deg}')
        return guard in self.guard_degree[deg], deg

    def remove_guard_from_degree(self, guard, deg=None):
        if deg == None:
            if guard in self.guard_degree[0]:
                self.guard_degree[0].remove(guard)
            elif guard in self.guard_degree[1]:
                self.guard_degree[1].remove(guard)
        else:
            if deg in self.guard_degree:
                self.guard_degree[deg].remove(guard)

    def move_guard(self, guard, node) -> None:
        self.free_guard(guard)    
        self.remove_guard_from_degree(guard)

        deg = self.unvisited_g[node]
        if deg == 0:
            raise ValueError("Guard should not be at a node with no unvisited neighbors")
        self.add_guard_in_degree(guard, deg)

        self.guard_locations[guard] = node
        self.guard_per_locations[node] += 1
        self.guard_per_locations['sta'] -= 1
        self.guard_per_locations_viz[node].append(self.guard_per_locations_viz['sta'].pop())

    def set_node_attributes(self) -> None:
        super().set_node_attributes()
        nx.set_node_attributes(self.graph.g, self.guard_per_locations, 'guard_number')
        nx.set_node_attributes(self.graph.g, self.guard_per_locations_viz, 'guard_viz')

    def searcher_to_new_node(self, node) -> None:
        super().searcher_to_new_node(node)
        for neighbor in self.graph.g[node]:
            deg = self.unvisited_g[neighbor]
            if deg >= 2: continue

            guards_to_update = [g for g, loc in enumerate(self.guard_locations) if loc == neighbor]
            for g in guards_to_update:
                self.remove_guard_from_degree(g)
                self.add_guard_in_degree(g, deg)
                if deg == 0:
                    self.free_guard(g)

    def after_search_step(self) -> None:
        if self.to_guard == None:
            return
        node = self.to_guard
        self.to_guard = None
        if self.unvisited_g[node] == 0 or\
            self.guard_per_locations[node] > 0 or\
            self.searcher_per_locations[node] > 0:
            return 
        self.call_guard(node)

    def can_move_searcher(self, node) -> bool:
        tree_can_move = super().can_move_searcher(node)

        #print(f'Node {node}: tree_can_move: {tree_can_move}')
        #print(f'unvisited_g: {self.unvisited_g[node]}, unvisited_t: {self.unvisited_t[node]}')
        #print(f'self.guard_per_locations[node]: {self.guard_per_locations[node]}')
        #print(f'self.searcher_per_locations[node]: {self.searcher_per_locations[node]}')

        if self.unvisited_g[node] == 0 and node != 'sta':
            assert self.guard_per_locations[node] == 0 or self.print_guard_info(f'Guard at node {node} should have been cleared')

        if tree_can_move == False:
            return False

        if self.non_tree_edge_nodes[node] == False or \
            self.unvisited_g[node] == 0 or\
            self.guard_per_locations[node] > 0 or\
            self.searcher_per_locations[node] > 1:
            return True

        self.to_guard = node
        return True

class GSST_R(GSST):
    def __init__(self, graph: Graph=None, filename='test_run') -> None:
        '''
        Variant of GSST as shown in Algorithm 6.
        '''
        self.number_of_guards = 0 # no guard needed
        self.N = graph.g.number_of_nodes()

        super().__init__(graph, filename=filename)
        self.searcher_locations = ['sta']
        self.searcher_per_locations['sta'] = 1
        self.num_searcher = 1
        self.history.clear()
        self.save_history()

    def can_move_searcher(self, node) -> bool:
        '''
        Checks if we can move a searcher from node.
        '''
        raise NotImplementedError

    def search(self) -> None:
        '''
        Performs Algorithm 6, ignores tree labelings.
        '''
        print('Search started with {} searchers'.format(self.num_searcher))
        N_c = {'sta'}
        while len(self.to_visit) != 0:
            if self.t > WALL_TIME * self.N:
                print(f'INTERRUPTED!\nTime: {self.t}, Number of searchers: {self.num_searcher}, unvisited area: {self.to_visit}')
                exit()
            self.search_step(N_c)
            self.set_node_attributes()      
            self.save_history()
            self.t += 1

    def search_step(self, N_c) -> None:
        '''
        Performs search for a single step.
        '''
        edges = [(a, b) for a, b in self.spanning_tree.g.edges() if a in N_c]
        src, nxt = choice(edges)
        if self.can_move_searcher(src):
            for idx in range(self.num_searcher):
                if self.searcher_locations[idx] == src:
                    self.move_searcher(idx, nxt)
                    break
            N_c.add(nxt)
        else:
            # generate a new searcher at the root
            self.searcher_locations.append('sta')
            self.searcher_per_locations['sta'] += 1
            self.num_searcher += 1
        print(self.to_visit, self.searcher_locations)
