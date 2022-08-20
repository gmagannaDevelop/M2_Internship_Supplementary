""" Helper functions for enhanced visualisation """

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import copy

def colorFader(c1,c2,mix=0):
    """ fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    taken from:
    https://stackoverflow.com/questions/25668828/how-to-create-colour-gradient-in-python
    """
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


def color_influence_graph(ig, mode = "influence", cold="lightblue", hot="yellow"):
    """
    available modes are:
    [
      in_degree, out_degree, influence, 
      total_degree, norm_total_degree, square_total_degree
    ]

    `mode` can be a lambda function, using the influence graph attibutes:
    The parameter will be the node name.
    For example, "influence" is defined as
        lambda node: (ig.out_degree[node]) / (max(ig.in_degree[node],1))

    another option could be:
        mode=lambda xx: (ig.out_degree[xx]+1)**2 - ig.in_degree[xx]
    """

    ig = copy.deepcopy(ig)

    scoring_modes = {
        "in_degree": lambda node: ig.in_degree[node],
        "out_degree": lambda node: ig.out_degree[node],
        "influence": lambda node: (ig.out_degree[node]) / (max(ig.in_degree[node],1)),
        "total_degree": lambda node: ig.in_degree[node] + ig.out_degree[node],
        "norm_total_degree": lambda node: (ig.out_degree[node] + ig.in_degree[node]) / max(ig.in_degree[node],1),
        "square_total_degree": lambda node: (ig.out_degree[node]**2 + ig.in_degree[node]) / max(ig.in_degree[node],1),
    }
    _f_score = scoring_modes.get(mode)
    _f_score = _f_score or mode # if mode is a callable

    node_scores = {_node: _f_score(_node) for _node in ig.nodes}
    highest_rank = max(node_scores.values())
    node_colour_map = {
        _node: colorFader(cold, hot, _score/highest_rank)
        for _node, _score
        in sorted(node_scores.items(), key=lambda _tuple: _tuple[1])
    }

    for node, data in ig.nodes(data=True):
        data["fillcolor"] = node_colour_map[node]
        data["style"] = "filled"

    return ig
