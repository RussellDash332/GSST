from PIL import Image
import numpy as np

locations = [(48.0000, 56.0000),    (153.0000, 55.0000),    (257.0000, 56.0000),    (348.0000, 56.0000),    (438.0000, 56.0000),    (518.0000, 55.0000),    (576.0000, 38.0000),    (575.0000, 117.0000),    (576.0000, 212.0000),
             (523.0000, 213.0000),    (437.0000, 213.0000),    (359.0000, 211.0000),    (284.0000, 212.0000),    (213.0000, 212.0000),    (158.0000, 162.0000),    (157.0000, 99.0000),    (120.0000, 209.0000),    (48.0000, 214.0000)]

fig_size = (12, 5.6)
locations = {i+1: (l[0]/50, fig_size[1] - l[1]/50) for i,l in enumerate(locations)}

edges = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8),
            (8,9), (9,10), (10,11), (11,12), (12,13), (13,14),
            (14,17), (15,16), (2,16), (15,17), (17,18)
]

bg = 'graphs/hallway.png'
bg = np.array(Image.open(bg).convert('RGB'))
print(bg.shape)

sta = (0.5, 5.2)