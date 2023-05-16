import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

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

    def visualize(self, save=False, filename='testrun', ax=None):    
        if hasattr(self, 'fig_size'):
            fig_size = self.fig_size
        else:
            fig_size = (10, 10)
        
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=fig_size)
        ax.set_xticks(np.arange(0, fig_size[0], 1))
        ax.set_yticks(np.arange(0, fig_size[1], 1))
        
        if hasattr(self, 'node_size'):
            node_size = self.node_size
        else:
            node_size = 300
            
        if hasattr(self, 'offset'):
            offset = self.offset
        else:
            offset = 0.5

        # plt.grid('on')
        # plt.axis('on')
        ax.set_xlim(0, fig_size[0])
        ax.set_ylim(0, fig_size[1])

        if hasattr(self, 'bg'):
            ax.imshow(self.bg, extent=[0, fig_size[0], 0, fig_size[1]])     
        if not self.is_tree():
            if not hasattr(self, 't'):
                if self.pos is None:
                    pos = nx.planar_layout(self.g)
                else:
                    pos = self.pos
                    # print(f'Use existing pos {pos}')
                nx.draw_networkx(self.g, pos=pos, with_labels=True, node_color='c', ax=ax, node_size=node_size, width=3)
                # ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

                # nx.draw_networkx_edge_labels(self.g, pos=pos, edge_labels=nx.get_edge_attributes(self.g, 'label'), label_pos=0.6, ax=ax)
            else:
                # pos = nx.nx_agraph.graphviz_layout(self.t.g, prog='dot',args="-Grankdir=LR")
                if self.pos is None:
                    pos = nx.spring_layout(self.t.g, k=3, seed = 1)
                else:
                    pos = self.pos
                    
                try:
                    visited_nodes = {node for node in self.g.nodes() if self.g.nodes[node]['visited']}
                except KeyError:
                    visited_nodes = set('sta')
                    
                searcher_per_node = nx.get_node_attributes(self.g, 'searcher_number')
                guard_per_node = nx.get_node_attributes(self.g, 'guard_number')
                robot_per_node = {k: searcher_per_node[k] + guard_per_node[k] for k in searcher_per_node.keys()}
                
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
                    else:
                        node_colors.append('cyan')
                node_label = {n: robot_per_node[n] if node_type[n]=='current' else '' for n in self.g.nodes()}
        
                nx.draw_networkx_nodes(self.g, pos=pos, node_color=node_colors, node_size=node_size, ax=ax)
                nx.draw_networkx_labels(self.g, labels=node_label, pos=pos, ax=ax)

                tree_edges = self.t.g.edges()
                non_tree_edges = self.B
                # visited_edges = set()
                
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
                        
                    # a,b = edge
                    # if a in visited_nodes and b in visited_nodes:
                    #     visited_edges.add((a,b))
                    #     visited_edges.add((b,a))
                        
                # tree_edge_colors= ['g' if edge in visited_edges else 'r' for edge in tree_edges]
                # non_tree_edge_colors = ['g' if edge in visited_edges else 'b' for edge in non_tree_edges]
                # nx.draw_networkx_edges(self.g, pos=pos, edge_list=tree_edges, edge_color=tree_edge_colors, ax=ax)
                # nx.draw_networkx_edges(self.g, pos=pos, edge_list=non_tree_edges, style='dashed', edge_color=non_tree_edge_colors, ax=ax)
                edge_styles = np.array(edge_styles)
                nx.draw_networkx_edges(self.g, pos=pos, edge_color=edge_colors, style=edge_styles, ax=ax, width=3)
        else:
            if self.g.is_directed():
                pos = nx.nx_agraph.graphviz_layout(self.t.g, prog='circo',args="-Grankdir=LR", root='sta')
                try:
                    visited_nodes = {node for node in self.g.nodes() if self.g.nodes[node]['visited']}
                except KeyError:
                    visited_nodes = set('sta')
                
                searcher_per_node = nx.get_node_attributes(self.g, 'searcher_number')
                guard_per_node = nx.get_node_attributes(self.g, 'guard_number')
                robot_per_node = {k: searcher_per_node[k] + guard_per_node[k] for k in searcher_per_node.keys()}
                
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
                    else:
                        node_colors.append('cyan')

                node_label = {n: robot_per_node[n] if node_type[n]=='current' else '' for n in self.g.nodes()}
                nx.draw_networkx_nodes(self.g, pos=pos, node_color=node_colors, node_size=node_size, ax=ax)
                nx.draw_networkx_labels(self.g, labels=node_label, pos=pos, ax=ax)
                # nx.draw_networkx_labels(self.g, pos={k: (x, y + 4) for k,(x,y) in pos.items()}, labels=nx.get_node_attributes(self.g, 'searcher_number'), font_color='r', ax=ax)
                
                G = self.g
                curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
                straight_edges = list(set(G.edges()) - set(curved_edges))
                nx.draw_networkx_edges(G, pos, ax=ax)
                # arc_rad = 0.15
                # nx.draw_networkx_edges(G, pos, edge_list=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', ax=ax)
                # nx.draw_networkx_edge_labels(self.g, pos=pos, edge_labels=nx.get_edge_attributes(self.g, 'label'), label_pos=0.6. ax=ax)
                
            else:
                pos = nx.nx_agraph.graphviz_layout(self.g, prog='dot')
                nx.draw(self.g, pos=pos, with_labels=True, node_color='c', ax=ax, node_size=node_size)
                # nx.draw_networkx_edge_labels(self.g, pos=pos, edge_labels=nx.get_edge_attributes(self.g, 'label'), label_pos=0.6)
        
        if save:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

if __name__ == "__main__":
    g = Graph()
    g.add_sta()
    print('Vertex number: ',g.g.number_of_nodes())
    print('Edge number: ', len(g.g.edges()))
    print('Actual Branching Factor', np.average([tup[1] for tup in g.g.degree()]))
    print(g.g[0])
    g.visualize()
    
    print()
    print("Random spanning tree")
    g.generate_random_spanning_tree()
    g.visualize()
