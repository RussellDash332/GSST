import os
from graph import Graph
from gsst import GSST_L
import sys

test_graph_only = False
idx = int(sys.argv[1])
files = {
    1: 'demo/art_gallery',
    2: 'demo/simon_hall',
    3: 'demo/hallway',
    4: 'demo/gates'
}

if idx == 1:
    from graphs.gallery_of_art import *
elif idx == 2:
    from graphs.simon_hall import *
elif idx == 3:
    from graphs.hallway import *
elif idx == 4:
    from graphs.gates import *
else:
    raise ValueError("Invalid index")

fn = files[idx]
if not os.path.exists('demo'):
    os.mkdir('demo')
if not os.path.exists(fn+'/'):
    os.mkdir(fn+'/')

print(max([i[0] for i in locations.values()]))
print(max([i[1] for i in locations.values()]))
g = Graph(edges, pos=locations)
g.bg = bg
g.fig_size = fig_size
g.node_size = 1000

g.add_sta(sta=1)
g.pos['sta'] = sta
g.visualize(save=True, filename=fn+'_graph.png')

if test_graph_only:
    exit()

g.generate_random_spanning_tree()
g.offset = 0.4
g.visualize(save=True, filename=fn+'_tree.png')

gsst_l = GSST_L(graph=g, filename=fn+'/run')
gsst_l.search(visualize=True)
print(
    f'COMPLETED!\nTime: {gsst_l.t}, Number of searchers: {gsst_l.num_searcher}, Number of guards: {gsst_l.number_of_guards}')
gsst_l.visualize()
