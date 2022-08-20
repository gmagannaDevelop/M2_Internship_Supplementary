""" Library of random influence graphs """

from typing import List, Optional, Tuple

import itertools
import numpy as np
import networkx as nx
from sklearn.utils import Bunch

import bonesis

# TODO: add arr-like type alias

_DEFAULT_BIFURCATION_NAMES = ["init", "FP1", "FP2"]


def add_anonymous_bifurcation_constraints(
    bo: bonesis.BoNesis,
    configuration_names: List[str] = _DEFAULT_BIFURCATION_NAMES,
) -> Bunch:
    """
    caveats:
        This function assumes that the first name of the
        `configuration_names` list is the starting point.
    """
    # Create anonymous configurations
    configurations = {
        cfg_name: bo.cfg(name=cfg_name) for cfg_name in configuration_names
    }
    # inequality constraints, we require all states to be pairwise different
    inequality_constraints = {
        f"{lcfg}_neq_{rcfg}": configurations[lcfg] != configurations[rcfg]
        for lcfg, rcfg in itertools.combinations(configuration_names, 2)
    }
    # declare fixed points
    fixed_points = {
        cfg_name: bo.fixed(configurations[cfg_name])
        for cfg_name in configuration_names[1:]
    }
    # reachability constraints:
    # here, all fixed points accessible from the initial state
    _start = configuration_names[0]
    paths = {
        f"{_start}_to_{_name}": configurations[_start] >= _fp
        for _name, _fp in fixed_points.items()
    }
    _cfg = Bunch(
        configurations=configurations,
        inequality_constraints=inequality_constraints,
        fixed_points=fixed_points,
        reachability_constraints=paths,
    )

    return _cfg


def specification_from_anonymous_paths(
    bo: bonesis.BoNesis,
    paths: List[List[str]],
    fixed_points: Optional[List[str]] = None,
    enforce_fp_in_paths: bool = True,
) -> Bunch:
    """This function will add the necessary constraints to capture the dynamic
    behaviour specified by a set of paths between distinct configurations
    (inequality constraints are automatically added).

    CAVEATS:
    If fixed points are not specified, the last element of each trace is
    assumed to be a fixed point.

    Therefore it is better to specify your paths as:
    >>> paths = [['a', 'b', 'c'], ['a', 'b', 'd']]
    instead of:
    >>> paths = [['a', 'b'], ['b', 'c'], ['b', 'd']]

    If you use the second spec, and you don't specify that only 'c'
    and 'd' are fixed points, bonesis will find no solution as you
    will have specified a contradiction:
        * b is a fixed point
        * there exists a path going from b to a different state
        * it follows that b is not a fixed point, contradiction.
    """
    _are_lists = all(isinstance(_path, list) for _path in paths)
    _are_strings = all(all(isinstance(_cfg, str) for _cfg in _path) for _path in paths)
    if not (_are_lists and _are_strings):
        raise TypeError("`paths` argument should be a list of lists (of strings)")

    # Fixed point declaration, if they are not specified they are
    # assumed to be the last configuration in each path
    fixed_points = fixed_points or [_path[-1] for _path in paths]

    if enforce_fp_in_paths:
        if not all(any(_fp == _path[-1] for _path in paths) for _fp in fixed_points):
            raise ValueError(
                "At least one specified fixed point is absent from the paths."
            )

    _config_names = set(itertools.chain(*paths, fixed_points))
    # if fixed_points:
    #    _config_names = itertools.chain(_config_names, fixed_points)
    # _config_names = set(_config_names)

    configurations = {cfg_name: bo.cfg(name=cfg_name) for cfg_name in _config_names}
    # inequality constraints, we require all states to be pairwise different
    inequality_constraints = {
        f"{lcfg}_neq_{rcfg}": configurations[lcfg] != configurations[rcfg]
        for lcfg, rcfg in itertools.combinations(_config_names, 2)
    }
    # declare fixed points
    fixed_points_dict = {
        cfg_name: bo.fixed(configurations[cfg_name]) for cfg_name in fixed_points
    }
    # reachability constraints
    reach_list = []  # The actual reachability constraints
    reach_name_set = set()  # use a set for avoiding redundancy
    for path in paths:
        for _from, _to in zip(path, path[1:]):
            _reach_name = f"{_from}_to_{_to}"
            if _reach_name not in reach_name_set:
                reach_list.append(configurations[_from] >= configurations[_to])
                reach_name_set.update({_reach_name})

    _cfg = Bunch(
        configurations=configurations,
        inequality_constraints=inequality_constraints,
        fixed_points=fixed_points_dict,
        reachability_constraints=reach_list,
    )

    return _cfg


def enforce_distance(
    bo: bonesis.BoNesis,
    constraints: Bunch,
    distances: Optional[List[Tuple[str, str, int]]] = None,
    global_distance: Optional[int] = None,
) -> Bunch:
    """constraints is assumed to be the result of calling
    bn_synthesis.inference.constraints_from_anonymous_paths()

    returns constraints after updating it (the original object should reflect
    the changes as well)
    """
    _has_distances = distances is None
    _has_global_distance = global_distance is None
    if not _has_distances ^ _has_global_distance:
        raise ValueError("Specify one and only one of: `distances`, `global_distances`")

    if global_distance:
        distances = [
            (lcfg, rcfg, global_distance)
            for lcfg, rcfg in itertools.combinations(constraints.configurations, 2)
        ]
    else:
        if not set(
            itertools.chain.from_iterable((_[0], _[1]) for _ in distances)
        ).issubset(constraints.configurations):
            raise ValueError("param `distances` contains undeclared configurations!")

    # TODO: decide if list is appropriate
    constraints.update({"distances": []})
    for lcfg, rcfg, dist in distances:
        constraints.distances.append(
            bo.custom(
                f':- #count {{ N,V: cfg("{lcfg}",N,V),cfg("{rcfg}",N,-V) }} {dist}.'
            )
        )

    return constraints
