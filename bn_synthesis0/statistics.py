""" Descriptive statistics for Boolean Network Synthesis """

import itertools
import functools

from math import comb as n_comb
from typing import Union, List, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import distance

import mpbn

from sklearn import metrics

from .utils.formatting import rearrange_cols_first

# TODO: export types to a separate module if they can be useful elsewhere.
_Array = Union[np.ndarray, pd.Series]
_FuncSpec = Union[str, Callable]
_Numeric = Union[int, float, bool, np.int_, np.float_, np.bool_]
_Boolean = Union[int, bool, np.int_, np.bool_]
_ConfDict = Dict[str, _Boolean]
_ConfsDict = Dict[str, _ConfDict]
_Configuration = Union[_ConfsDict, pd.DataFrame]
_EnumeratedBonesisSolutions = List[Tuple[int, mpbn.MPBooleanNetwork, pd.DataFrame]]
_MetricTupleList = List[Tuple[str, _FuncSpec]]

_SCIPY_BINARY_METRICS = (
    "dice",
    "hamming",
    "jaccard",
    "kulsinski",
    "rogerstanimoto",
    "russellrao",
    "sokalmichener",
    "sokalsneath",
    "yule",
)

# The keys of this dictionary will not change, but the values
# may do so, if better names (more explicit or comprehensible)
# are found.
_STD_COL_NAMES: Dict[str, str] = {
    "cfg1": "cfg1",  # First configuration in pairwise comparisons
    "cfg2": "cfg2",  # Second configuration "    "         "
    "bn": "bn",  # Boolean network
}
_PAIRWISE_MULTI_INDEX: List[str] = [
    _STD_COL_NAMES[_col] for _col in ("bn", "cfg1", "cfg2")  # These will not change.
]

DEFAULT_METRICS = [
    ("jaccard_distance", "jaccard"),
    ("phi_coeff", metrics.matthews_corrcoef),
]


def pairwise_cfg_distances(
    cfgs: _Configuration, metric: _FuncSpec = "jaccard", name: Optional[str] = None
) -> pd.DataFrame:
    """Compute pairwise distances between configurations:
    >>> # note that here `bo` is an instance of bonesis.BoNesis
    >>> distances = []
    >>> for (f, cfg) in bo.boolean_networks(limit=20, extra="configurations"):
    >>>     distances.append(jaccard_distances(cfg))
    """
    _is_scipy = metric in _SCIPY_BINARY_METRICS
    if not (_is_scipy or name):
        raise ValueError(
            f"Specify one of {_SCIPY_BINARY_METRICS} or provide a name for custom callable"
        )
    name = name if name else f"{metric}_distance"

    cfgs = cfgs if isinstance(cfgs, pd.DataFrame) else pd.DataFrame(cfgs)
    config_pairs = [(i, j) for i, j in itertools.combinations(cfgs.columns, 2)]
    # TODO: test if config_pairs respects the order for bigger configuration sets
    _cfgs_np = cfgs.T.values.astype(np.bool_)
    # ^ slightly faster with prior type conversion
    _distances = distance.pdist(_cfgs_np, metric=metric)
    distance_df = pd.DataFrame(
        {
            _STD_COL_NAMES["cfg1"]: [i for i, j in config_pairs],
            _STD_COL_NAMES["cfg2"]: [j for i, j in config_pairs],
            name: _distances,
        }
    )
    return distance_df


# TODO: change _ls suffixes for bulk_ prefixes ?
def pairwise_cfg_distances_ls(
    n_bn_cfg: _EnumeratedBonesisSolutions, multi_index: bool = True, **kwargs
) -> pd.DataFrame:
    """
    1.- Apply `statistics.pairwise_cfg_distances` to the configurations
        contained in the `n_bn_cfg` parameter.
    2.- Concatenate all the resulting frames, adding a column which tells
        to which boolean network each state combination belongs.

    kwargs are directly passed to `statistics.pairwise_cfg_distances`
    see `help(statistics.pairwise_cfg_distances)` for further info.
    """
    _all_distances = pd.concat(
        (pairwise_cfg_distances(cfg, **kwargs) for i, bn, cfg in n_bn_cfg),
        ignore_index=True,
    )
    _all_distances[_STD_COL_NAMES["bn"]] = list(
        itertools.chain.from_iterable(
            (n_comb(len(cfg), 2) * [i] for i, bn, cfg in n_bn_cfg)
        )
    )
    _all_distances = rearrange_cols_first(_all_distances, _PAIRWISE_MULTI_INDEX)
    if multi_index:
        _all_distances = _all_distances.set_index(_PAIRWISE_MULTI_INDEX)
    return _all_distances


def compute_multiple_cfg_metrics(
    n_bn_cfg: _EnumeratedBonesisSolutions, metrics_ls: Optional[_MetricTupleList] = None
) -> pd.DataFrame:
    """Perform multiple calls to pairwise_cfg_distances_ls(),
    joining (cumulative left join from the first to the last frame)
    the resulting frames to obtain a single metric/distances frame.

    `metrics_ls` should be formatted as follows: (name, metric)
    i.e. :

    metrics_ls = [
        ("jaccard_distance", "jaccard"),
        ("phi_coeff", sklearn.metrics.matthews_corrcoef),
    ]
    """
    metrics_ls = metrics_ls or DEFAULT_METRICS
    if len(metrics_ls) < 2:
        raise ValueError(
            " ".join(
                (
                    "Calling compute_multiple_cfg_metrics() with a single metric is not supported.",
                    "To compute a single distance/metric, use pairwise_cfg_distances_ls() instead.",
                )
            )
        )

    # TODO: replace this with a comprehension?
    _distance_frames = []
    for _name, _metric in metrics_ls:
        _distance_frames.append(
            pairwise_cfg_distances_ls(
                n_bn_cfg, multi_index=True, name=_name, metric=_metric
            )
        )

    return functools.reduce(lambda acc, new: acc.join(new), _distance_frames)


def metrics_summary_statistics(
    metrics_df: pd.DataFrame, stats: Optional[List[str]] = None
) -> pd.DataFrame:
    """Resume a call to compute_multiple_cfg_metrics(),
    reducing boilerplate on analysis notebooks and development scripts.

    Caveats:
        All `stats` should be an attribute of a
        pandas.DataFrame.groupby object, otherwise an AttributeError
        will be raised.

    @param stats : defaults to ["mean", "median", "min", "max"]
    """
    _stats = stats or ["mean", "median", "min", "max"]
    grouper = metrics_df.reset_index().groupby(_STD_COL_NAMES["bn"])
    cols = metrics_df.columns
    _resumed = (
        grouper.std().sum(axis=1).to_frame().rename(columns={0: "sum_of_deviations"})
    )

    for _stat in _stats:
        _new_stat = getattr(grouper, _stat)().rename(
            columns={_col: f"{_stat}_{_col}" for _col in cols}
        )
        if _stat in ("max", "min"):  # these methods save the config name
            _new_stat = _new_stat.drop(  # which is unnecessary
                columns=[val for key, val in _STD_COL_NAMES.items() if key != "bn"]
            )
        _resumed = _resumed.join(_new_stat)

    return _resumed


def compute_attractor_statistics(
    bn: mpbn.MPBooleanNetwork,
    cfgs: _Configuration,
    root: str,
    index: Optional[int] = None,
) -> pd.DataFrame:
    """Compute basic attractor statistics.
    :param `root` is the name of the initial configuration from which the
                  attractor reachability should be computed."""
    cfgs = cfgs.to_dict() if isinstance(cfgs, pd.DataFrame) else cfgs
    index = index or 0

    def count(_iterable):
        return sum(1 for _ in _iterable)

    _data = dict(
        n_attr=count(bn.attractors()),
        n_fp=count(bn.fixedpoints()),
        n_attr_reachable=count(bn.attractors(reachable_from=cfgs[root])),
        n_fp_reachable=count(bn.fixedpoints(reachable_from=cfgs[root])),
    )
    data = pd.DataFrame(_data, index=[index])

    data["p_fp"] = data["n_fp"] / data["n_attr"]
    data["p_attr_reachable"] = data["n_attr_reachable"] / data["n_attr"]
    data["p_fp_reachable"] = data["n_fp_reachable"] / data["n_fp"]

    return data


def compute_attractor_statistics_ls(
    n_bn_cfg: _EnumeratedBonesisSolutions, root: str
) -> pd.DataFrame:
    """Apply `attractor_statistics()` to the list n_bn_cfg.
    :param `root` is the name of the initial configuration from which
                  the attractor reachability should be computed.
                  It will be passed to attractor_statistics()."""
    return pd.concat(
        (compute_attractor_statistics(bn, cfg, root, i) for i, bn, cfg in n_bn_cfg)
    )


## For eventual multiprocessing version:
# _diversity_args = [(bn, cfg, "init", idx) for idx, bn, cfg in diversity ]
# [type(_) for _ in _diversity_args[0]]
# attractor_statistics = pd.concat((attractor_statistics(*args) for args in _diversity_args))
## ^ this should be passed to multiprocessing.Pool.starmap()
