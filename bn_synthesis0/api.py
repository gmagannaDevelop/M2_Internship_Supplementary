"""
    Main Application Programming Interface
    for Synthetic Boolean Network Generation.
"""

from typing import Union
from pathlib import Path
import toml
import pandas as pd

_ENCODING: str = "utf-8"

# TODO: remove magic constants


def save_experiment(i_bn_cfg_ls, ig_kwargs, location: Path) -> bool:
    """
    Save the following:
        1.- A list of boolean networks and their respective
            configurations yielded by the solver.

        2.- The set of keyword arguments used to generate the influence
            graph from which the boolean networks were inferred.

    To the specified `location`.
    NOTE: the location MUST NOT exist, but its parent should.

    The `i_bn_cfg_ls` is expected to be formatted as follows:
    A list of tuples of the form:
        (integer, mpbn.MPBooleanNetwork, pandas.DataFrame)

    The integer just serves to uniquely identify each boolean network,
    which will be useful for eventually creating bundles.
    """
    location = location.resolve()
    location.mkdir()

    maxlength = lambda num_ls: len(str(max(num_ls, key=lambda _n: len(str(_n)))))
    _max_l = maxlength([i for i, bn, cfg in i_bn_cfg_ls])

    format_num = lambda num, digits: str(num).zfill(digits)

    with open(location / "ig_kwargs.toml", "w", encoding=_ENCODING) as _f_ig:
        toml.dump(ig_kwargs, _f_ig)

    for i, bn, cfg in i_bn_cfg_ls:
        _n = format_num(i, _max_l)
        with open(location / f"{_n}_bn.bnet", "w", encoding=_ENCODING) as _f_bn:
            _f_bn.write(bn.source())
        with open(
            location / f"{_n}_configurations.csv", "w", encoding=_ENCODING
        ) as _f_cfg:
            pd.DataFrame(cfg).to_csv(_f_cfg)
    # TODO: think about a better way of verifying that the dump has been successful.
    _all_saved = all(
        list(location.glob(_ext))
        for _ext in ["*.toml", "*_bn.bnet", "*_configurations.csv"]
    )
    return _all_saved


def load_experiment(location: Union[Path, str]):
    """Load the boolean networks and configurations,
    return a i_bn_cfg_ls, see `save_experiment()`"""
    location = Path(location) if isinstance(location, str) else location
    _bns = sorted(list(location.glob("*_bn.bnet")))
    _cfgs = sorted(list(location.glob("*_configurations.csv")))

    if len(_bns) != len(_cfgs):
        raise ValueError(
            "Cannot build a list with different number of "
            f"{len(_bns)} BNs and {len(_cfgs)} configs."
        )

    a = [(i, _bn, _cfg) in enumerate(zip(_bns, _cfgs))]

    return _bns, _cfgs
