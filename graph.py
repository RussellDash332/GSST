import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from collections import Counter
from random import random, seed

class Graph:
    def __init__(self, arg=None, directed=False, pos=None) -> None:
        '''
        Initializes a Graph object which will store the networkx graph and other informations.
        Attributes:
        - is_directed:  Is the graph directed or undirected?
        - start:        Starting node if the graph
        - g:            The nx graph object
        - pos:          Position of the nodes
        - t:            The (labeled) tree version of the graph, this attribute DNE if the graph is a tree
        '''
        self.is_directed = directed
        self.start = None

        if arg is None: # create a random graph if initial layout is not provided
            edge_list = self.random_graph()
            edge_attrs = -1
        elif isinstance(arg, nx.Graph):
            edge_list = arg.edges()
            edge_attrs = nx.get_edge_attributes(arg, 'label')
            print('edge attr', edge_attrs)
        elif isinstance(arg, list):
            edge_list = arg
            edge_attrs = -1
        else:
            raise NotImplementedError

        self.g = [nx.Graph, nx.DiGraph][directed](edge_list)
        nx.set_edge_attributes(self.g, edge_attrs, 'label')
        self.pos = pos

    def add_sta(self, sta=0) -> None:
        '''
        Adds a starting node,
        '''

        self.start = sta
        self.g.add_edge('sta', sta)

    def random_graph(self) -> list[list[int]]:
        '''
        Generates an edge list of a graph with N vertices and branching factor k,
        where N is randomized between 10 to 20 and k is randomized between 2 to 5.
        '''
        k = np.random.randint(2, 5)
        print(f'Random branching factor k = {k}')
        N = np.random.randint(10, 20)
        print(f'Random number of vertices N = {N}')

        edge_list = []
        visited = [0]
        to_visit = list(range(1, N))

        for _ in range(N-1):
            sta = np.random.choice(visited)
            end = np.random.choice(to_visit)
            to_visit.remove(end)
            visited.append(end)
            a,b = min(sta, end), max(sta, end)
            edge_list.append((a,b))

        edges = {}
        for i in range(N):
            for j in range(i+1, N):
                edges[(i, j)] = 1

        for edge in edge_list:
            edges.pop(edge)

        edges = list(edges.keys())
        length = k * N // 2 - (N-1)
        idxs = np.random.choice(len(edges), length, replace=False)
        for i in idxs:
            edge = edges[i]
            edge_list.append(edge)
        return edge_list

    def is_tree(self) -> bool:
        '''
        Checks if a graph is a tree or not.
        '''
        if self.g.is_directed():
            return nx.is_tree(self.g.to_undirected())
        return nx.is_tree(self.g)

    def generate_random_spanning_tree(self) -> None:
        '''
        Generates a random spanning tree.
        If the initial graph is not a tree, use Algorithm 4.
        '''
        if self.is_tree():
            return self
        else:
            # Use Algorithm 4
            edges = []
            sta = 'sta'
            visited = {node: False for node in self.g.nodes()}
            parents = dict()

            while len(edges) != self.g.number_of_nodes() - 1:
                visited[sta] = True
                neighbors = []
                for v in self.g[sta]:
                    if not visited[v]:
                        neighbors.append(v)
                if len(neighbors) == 0:
                    sta = parents[sta]
                    continue
                end = np.random.choice(neighbors)
                parents[end] = sta
                edges.append((sta, end))
                sta = end
            non_tree_edges = []
            for a,b in self.g.edges():
                if (a,b) in edges or (b,a) in edges:
                    continue
                non_tree_edges.append((min(a,b), max(a,b)))

            # Create the (undirected) tree version of it
            self.t = Graph(edges, directed=False)
            self.t.start = self.start
            self.t.pos = self.pos
            self.B = non_tree_edges
            self.t.label() # Label the edges as per Algorithm 2

    def get_spanning_tree(self) -> tuple["Graph", list[list[int]]]:
        '''
        Algorithm 4, returns the tree edges and the non-tree edges.
        '''
        if not hasattr(self, 't'):
            self.generate_random_spanning_tree()
        return self.t, self.B

    def label_reverse(self, parents: list[int]) -> None:
        '''
        Reverse labelling after Algorithm 2 is performed.
        '''
        for i in range(0, self.g.number_of_nodes()):
            if i in parents:
                self.g.edges[(i, parents[i])]['label'] = -self.g.edges[(parents[i], i)]['label']

    def label(self) -> None:
        '''
        Labels the spanning tree as per Algorithm 2.
        '''
        assert self.is_tree(), "Method only applicable to trees"

        # BFS traversal in a parent-child structure
        parents = dict(nx.bfs_predecessors(self.g, 'sta'))

        def is_leaf(node):
            if self.is_directed:
                return self.g.in_degree(node) == 1 and node != 'sta'
            else:
                return self.g.degree(node) == 1 and node != 'sta'

        def get_edge_info(node):
            adj = list(self.g[node])
            edge_labels = [self.g.edges[(node, neighbor)]['label'] for neighbor in adj]      
            label_counts = Counter(edge_labels) # to check for majority label
            return adj, edge_labels, label_counts

        def get_parent(node):
            return parents[node]

        # Algorithm 2
        buffer = [x for x in self.g.nodes() if is_leaf(x)]
        while buffer:
            node = buffer.pop()
            if is_leaf(node):
                # the only edge, so assign label to 1
                parent = get_parent(node)
                self.g.edges[(parent, node)]['label'] = 1
            else:
                adj, edge_labels, label_counts = get_edge_info(node)
                # if it has exactly one unlabeled edge
                if label_counts[-1] == 1:
                    # find that unlabeled edge
                    child = adj[edge_labels.index(-1)]
                    l_max = max(edge_labels)
                    if l_max == -1: continue
                    max_cout = label_counts[l_max]
                    if max_cout == 1:
                        self.g.edges[(node, child)]['label'] = l_max
                    else:
                        self.g.edges[(node, child)]['label'] = l_max + 1
            if node != 'sta':
                # if current node != start and its parent has exactly one unlabeled edge
                # add the parent node to buffer
                parent = get_parent(node)
                adj, edge_labels, label_counts = get_edge_info(parent)
                if label_counts[-1] == 1:
                    buffer.append(parent)

        # Set number of searchers for Algorithm 5
        self.mu = self.g.edges[('sta', self.start)]['label']
        self.g = self.g.to_directed()
        self.label_reverse(parents)

    def visualize(self, save=True, filename='testrun', ax=None, step=None, robot=False):  
        def get_nudge(node='sta', searcher=None, jitter=0.2):
            if node == 'sta': return (0, 0)
            if searcher: seed(searcher.id)
            return ((random()*2-1)*jitter, (random()*2-1)*jitter)

        if hasattr(self, 'fig_size'):
            fig_size = self.fig_size
        else:
            fig_size = (10, 10)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.set_xticks(np.arange(0, fig_size[0], 1))
        ax.set_yticks(np.arange(0, fig_size[1], 1))

        if hasattr(self, 'node_size'):
            node_size = self.node_size
        else:
            node_size = 300

        if hasattr(self, 'bg'):
            ax.imshow(self.bg, extent=[0, fig_size[0], 0, fig_size[1]])
        elif robot:
            robot = False

        if not self.is_tree():
            if not hasattr(self, 't'):
                if self.pos is None:
                    pos = nx.spring_layout(self.g)
                else:
                    pos = self.pos
                nx.draw_networkx(self.g, pos=pos, with_labels=True, node_color='c', ax=ax, node_size=node_size, width=3)
            else:
                if self.pos is None:
                    pos = nx.spring_layout(self.t.g, k=3, seed=1)
                else:
                    pos = self.pos
                try:
                    visited_nodes = {node for node in self.g.nodes() if self.g.nodes[node]['visited']}
                except KeyError:
                    visited_nodes = set('sta')
                searcher_per_node = nx.get_node_attributes(self.g, 'searcher_number')
                guard_per_node = nx.get_node_attributes(self.g, 'guard_number')
                searcher_per_node_viz = nx.get_node_attributes(self.g, 'searcher_viz')
                guard_per_node_viz = nx.get_node_attributes(self.g, 'guard_viz')
                robot_per_node = {k: searcher_per_node[k] + guard_per_node[k] for k in searcher_per_node.keys()}
                robot_per_node_viz = {k: searcher_per_node_viz[k] + guard_per_node_viz[k] for k in searcher_per_node.keys()}
                node_type = {}
                for n in self.g.nodes():
                    if n in visited_nodes:
                        node_type[n] = 'visited'
                        if n in robot_per_node and robot_per_node[n] > 0:
                            node_type[n] = 'current'
                    else:
                        node_type[n] = 'unvisited'
                node_colors = []
                for node in self.g.nodes():
                    if node == 'sta':
                        node_colors.append('red')
                    elif node_type[node] == 'unvisited':
                        node_colors.append('grey')
                    elif node_type[node] in 'visited':
                        node_colors.append('green')
                    elif not robot:
                        node_colors.append('cyan')
                    else:
                        node_colors.append(['grey', 'green'][robot])
                if not robot: ax.set_title(f'Time: {step if step != None else 0}\nRed: starting point     Green: cleared     Gray: may contain target     Cyan: has searcher', fontsize=17)
                else: ax.set_title(f'Time: {step if step != None else 0}\nRed: starting point     Green: cleared     Gray: may contain target', fontsize=17)
                node_label = {n: robot_per_node[n] if node_type[n]=='current' else '' for n in self.g.nodes()}
                nx.draw_networkx_nodes(self.g, pos=pos, node_color=node_colors, node_size=node_size, ax=ax)
                if not robot:
                    nx.draw_networkx_labels(self.g, labels=node_label, pos=pos, ax=ax)
                else:
                    for n in self.g.nodes():
                        for s in robot_per_node_viz[n]:
                            x, y = pos[n]
                            plt.gca().add_patch(plt.Circle((x+get_nudge(n, searcher=s)[0], y+get_nudge(n, searcher=s)[1]), 0.15, facecolor=to_rgba(s.color, alpha=0.5), edgecolor=s.color, zorder=10))
                tree_edges = self.t.g.edges()
                non_tree_edges = self.B
                edge_colors = []
                edge_styles = []
                for edge in self.g.edges():
                    a,b = edge
                    if (a,b) in tree_edges or (b,a) in tree_edges:
                        edge_styles.append('-')
                    else:
                        edge_styles.append('--')
                    if node_type[a] == 'unvisited' and node_type[b] == 'unvisited':
                        edge_colors.append('black')
                    elif node_type[a] == 'unvisited' or node_type[b] == 'unvisited':
                        edge_colors.append('cyan')
                    else:
                        edge_colors.append('green')
                edge_styles = np.array(edge_styles)
                if not robot: nx.draw_networkx_edges(self.g, pos=pos, edge_color=edge_colors, style=edge_styles, ax=ax, width=3)
        else:
            if self.g.is_directed():
                pos = nx.nx_agraph.graphviz_layout(self.t.g, prog='circo',args="-Grankdir=LR", root='sta')
                try:
                    visited_nodes = {node for node in self.g.nodes() if self.g.nodes[node]['visited']}
                except KeyError:
                    visited_nodes = set('sta')
                searcher_per_node = nx.get_node_attributes(self.g, 'searcher_number')
                guard_per_node = nx.get_node_attributes(self.g, 'guard_number')
                searcher_per_node_viz = nx.get_node_attributes(self.g, 'searcher_viz')
                guard_per_node_viz = nx.get_node_attributes(self.g, 'guard_viz')
                robot_per_node = {k: searcher_per_node[k] + guard_per_node[k] for k in searcher_per_node.keys()}
                robot_per_node_viz = {k: searcher_per_node_viz[k] + guard_per_node_viz[k] for k in searcher_per_node.keys()}
                node_type = {}
                for n in self.g.nodes():
                    if n in visited_nodes:
                        node_type[n] = 'visited'
                        if n in robot_per_node and robot_per_node[n] > 0:
                            node_type[n] = 'current'
                    else:
                        node_type[n] = 'unvisited'
                node_colors = []
                for node in self.g.nodes():
                    if node == 'sta':
                        node_colors.append('red')
                    elif node_type[node] == 'unvisited':
                        node_colors.append('grey')
                    elif node_type[node] == 'visited':
                        node_colors.append('green')
                    elif not robot:
                        node_colors.append('cyan')
                    else:
                        node_colors.append(['grey', 'green'][robot])
                if not robot: ax.set_title(f'Time: {step if step != None else 0}\nRed: starting point     Green: cleared     Gray: may contain target     Cyan: has searcher', fontsize=17)
                else: ax.set_title(f'Time: {step if step != None else 0}\nRed: starting point     Green: cleared     Gray: may contain target', fontsize=17)
                node_label = {n: robot_per_node[n] if node_type[n]=='current' else '' for n in self.g.nodes()}
                nx.draw_networkx_nodes(self.g, pos=pos, node_color=node_colors, node_size=node_size, ax=ax)
                if not robot:
                    nx.draw_networkx_labels(self.g, labels=node_label, pos=pos, ax=ax)
                else:
                    for n in self.g.nodes():
                        for s in robot_per_node_viz[n]:
                            x, y = pos[n]
                            plt.gca().add_patch(plt.Circle((x+get_nudge(n, searcher=s)[0], y+get_nudge(n, searcher=s)[1]), 0.15, facecolor=to_rgba(s.color, alpha=0.5), edgecolor=s.color, zorder=10))
                G = self.g
                curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
                straight_edges = list(set(G.edges()) - set(curved_edges))
                if not robot: nx.draw_networkx_edges(G, pos, ax=ax)
            else:
                pos = nx.nx_agraph.graphviz_layout(self.g, prog='dot')
                nx.draw(self.g, pos=pos, with_labels=True, node_color='c', ax=ax, node_size=node_size)
        if save:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

if __name__ == "__main__":
    g = Graph()
    g.add_sta()
    print('Vertex number:', g.g.number_of_nodes())
    print('Edge number:', len(g.g.edges()))
    print('Actual Branching Factor', np.average([tup[1] for tup in g.g.degree()]))
    g.visualize()
    print()
    print("Random spanning tree")
    g.generate_random_spanning_tree()
    g.visualize(filename='testrun_tree.png')
