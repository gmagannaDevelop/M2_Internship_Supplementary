""" 
Theorems to verify properties of Boolean Network Influence Graphs.

Theorems taken from the lecture notes :
    "Analyse de la Dynamique des Systèmes Biologiques"
    Université Paris-Saclay
        Loïc Paulevé
        Stefan Haar
"""

from functools import reduce

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.utils import Bunch

import bonesis


def has_positive_circuit(domain: bonesis.domains.InfluenceGraph) -> bool:
    """Check for (Remy, Ruet et Thieffry, 2008)'s Theorem:

    ** If G(f) does not have any positive circuit,
    then f contains AT MOST a single fixed point **
    """
    raise ValueError("")

    return True


def get_cycles_and_signs(graph: bonesis.domains.InfluenceGraph):
    """Given an influence graph, compute all its cycles as
    well as their sign, in order to assess the possible"""
    _graph = nx.MultiDiGraph(graph).copy()
    _graph.remove_edges_from(nx.selfloop_edges(graph))
    signs = nx.get_edge_attributes(_graph, "sign")
    cycles = [_cycle for _cycle in nx.simple_cycles(_graph)]
    cycle_signs = []
    for cycle in cycles:
        cycle.append(cycle[0])  # needed because of nx convention
        # cycle yields the nodes but we need the edges in order to compute the sign
        sign = np.prod([signs[(i, j, 0)] for i, j in zip(cycle, cycle[1:])])
        cycle_signs.append(sign)
        _ = cycle.pop()

    # To check the Theorem(Aracena, 2008 ; Richard, 2009) (max # of attractors 2^|l|)
    # where l = {node for node in G.nodes if all(node in cycle for cycle in nx.simple_cycles(G))}
    # we convert the list of nodes in the cycle to a set
    cycles = [set(cycle) for cycle in cycles]

    return cycles, cycle_signs


def get_positive_cycles(cycles, signs):
    """Filter out negative cycles, return only the positive cycles"""
    return [cycle for cycle, sign in zip(cycles, signs) if sign > 0]


def generate_scale_free_ig(
    n_nodes,
    alpha=(1, 1, 1),
    size=10,
    p_pos=0.6,
    return_params_first=False,
    seed=None,
    **kwargs
):
    """
    This function returns a generator
    alpha parameter is a tuple containing the weights of the
    (alpha, beta, gamma) parameters needed by networkx.scale_free_graph()
    """
    rng = (
        np.random.default_rng(seed)
        if not isinstance(seed, np.random.Generator)
        else seed
    )
    f_sf_ig = bonesis.domains.InfluenceGraph.scale_free_

    random_alphas = rng.dirichlet(alpha=alpha, size=size)
    if return_params_first:
        yield random_alphas
    for i in np.split(random_alphas, random_alphas.shape[0], axis=0):
        seed = int(rng.integers(1, 20221231))
        _i_dirichlet = i.flatten()
        alpha, beta, gamma = _i_dirichlet
        _sf_kwargs = dict(
            n=n_nodes,
            p_pos=p_pos,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            seed=seed,
            **kwargs
        )
        yield f_sf_ig(**_sf_kwargs)
