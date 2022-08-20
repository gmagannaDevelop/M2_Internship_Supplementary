""" Generation of trajectories using bonesis"""

import bonesis
from bonesis0.asp_encoding import configurations_of_facts
import pandas as pd


class ConfigurationsView(bonesis.BonesisView):
    project = True
    show_templates = ["configuration"]

    def format_model(self, model):
        atoms = model.symbols(shown=True)
        return configurations_of_facts(atoms, keys="all")


# these are pylint configuration guards:
# pylint: disable=no-member, pointless-statement
def make_trajectory_v1(f, start, end, steps=None, **kwargs):
    """
    f: BooleanNetwork
    start, end: configurations
    steps: number of steps to reach end from start (default=distance between start and end)
    """
    if steps is None:
        steps = len([n for n in f if start.get(n) != end.get(n)])
    bo = bonesis.BoNesis(f)
    traj = [~bo.obs(start)] + [bo.cfg() for _ in range(1, steps)] + [~bo.obs(end)]
    for x, y in zip(traj[:-1], traj[1:]):
        x >= y

    bo.custom(
        """
hsize(0,0).
hsize(X,K) :- K = #count { N: mcfg(X,N,1),mcfg(X,N,-1) }; mcfg(X,_,_).
maxdst(M) :- M = #max { K: hsize(_,K) }.
#minimize { K@1: maxdst(K) }.
"""
    )
    clingo_opts = ["--opt-strategy=bb"] + kwargs.pop("clingo_opts", [])

    sol = next(
        iter(ConfigurationsView(bo, mode="optN", clingo_opts=clingo_opts, **kwargs))
    )

    cfg_order = [c.name for c in traj]
    col_order = list(sorted(f))
    return pd.DataFrame.from_dict(sol, orient="index")[col_order].loc[cfg_order]


# pylint: enable=no-member, pointless-statement


def labelled_trajectory(
    bn, cfgs, start: str, stop: str, _trans_label: str, steps=None, **kwargs
) -> pd.DataFrame:
    """
    Create a labeled trajectory using `bn_synthesis.trajectories.make_trajectory_v1`

    @param bn: A MostPermissive Boolean Network
    @param cfgs: A set of configurations, as a dict or pandas.DataFrame
    @param start: name of the initial configuration of the trajectory.
    @param stop: name of the last configuration of the trajectory.
    @param _trans_label: A label for intermediate configurations found by clingo.

    `steps` and `kwargs` are passed directly to `make_trajectory_v1`
    """
    cfgs = cfgs.to_dict() if isinstance(cfgs, pd.DataFrame) else cfgs
    if not isinstance(cfgs, dict):
        raise TypeError("`cfgs` should be a dictionnary or a pandas.DataFrame")

    traj_df = make_trajectory_v1(bn, cfgs[start], cfgs[stop], steps=steps, **kwargs)
    new_index = traj_df.index.map(
        lambda x: f"{_trans_label}_{x.replace('__cfg', '')}" if "__cfg" in x else x
    ).to_list()
    new_index[0] = start
    new_index[-1] = stop

    return traj_df.set_index([new_index])


def labelled_trajectory_from_reachability_constraints(
    bn, reach_contraints, candidate_cfgs
) -> pd.DataFrame:
    """Synthesise trajectories from reachability constraints"""
    trajectories = []
    for constraint in reach_contraints:
        # this parsing approach feels too hacky (and thus unpythonic), but it's the simplest
        # reachability constraints are of the form:
        # 'reach("Configuration(\'init\')", "Configuration(\'branch\')")'
        _pieces = str(constraint).split("'")
        start, stop = _pieces[1], _pieces[3]
        trajectories.append(
            labelled_trajectory(
                bn,
                candidate_cfgs,
                start=start,
                stop=stop,
                _trans_label=f"{start}_to_{stop}",
            ).drop_duplicates(keep="last")
        )

    return pd.concat(trajectories).drop_duplicates(keep="first")
