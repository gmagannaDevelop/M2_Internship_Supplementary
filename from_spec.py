#!/usr/bin/env python
# coding: utf-8
"""
Automated generation of Boolean Network ensembles
from anonymous dynamical constraints.
"""

import sys

import datetime as dt
from pathlib import Path
import toml

import random
import numpy as np
import pandas as pd
import seaborn as sns
import plotnine as p9

from sklearn.utils import Bunch

import bonesis

from bn_synthesis.api import save_experiment
from bn_synthesis.trajectories import labelled_trajectory_from_reachability_constraints

from bn_synthesis.inference import (
    specification_from_anonymous_paths,
    enforce_distance,
)
from bn_synthesis.statistics import (
    compute_multiple_cfg_metrics,
    compute_attractor_statistics_ls,
    metrics_summary_statistics,
)
from bn_synthesis.utils.formatting import rearrange_cols_first

if len(sys.argv) != 4:
    print("usage: " f"{__file__} spec.toml experiment_name rng_seed")
    sys.exit()

# locations
root = Path(".").resolve()
here = Path(".").resolve()
data_dir = root / "data"
# date
today = dt.datetime.today().strftime("%Y-%m-%d")

## EXPERIMENT IDENTIFIER
experiment_name = sys.argv[2]
seed = int(sys.argv[-1])
experiment_dir = data_dir / f"{today}_{experiment_name}_seed_{seed}"


_conf = Bunch(
    **{
        "root": root.as_posix(),
        "date": today,
        "workdir": here.relative_to(root).as_posix(),
        "data_dir": data_dir.relative_to(root).as_posix(),
        "experiment": experiment_dir.name,
    }
)

with open(sys.argv[1], "r", encoding="utf-8") as _spec:
    spec = toml.load(_spec, _dict=Bunch)
# seed = spec.seed
paths = [_path.split(">") for _path in spec.constraints.path_spec]
if spec.constraints.get("distances"):
    spec.constraints.distances = [
        tuple(dist.split("|")) for dist in spec.constraints.distances
    ]
    spec.constraints.distances = [
        (f, t, int(d)) for f, t, d in spec.constraints.distances
    ]
distances = spec.constraints.get("distances")
global_distance = spec.constraints.get("global_distance")


for k, v in spec.bonesis.items():
    bonesis.settings[k] = v


# pylint: disable=no-member, pointless-statement
# n_attempts_before_success = 0
n_abs = 0 # shorthand for n_attempts_before_success
new_diversity = []
rng = np.random.default_rng(seed)
while not new_diversity:
    print(f"Trying with seed = {seed}")
    # create ig
    dom1 = bonesis.InfluenceGraph.scale_free_(seed=seed, **spec.influence_graph)
    # create bonesis instance
    bo = bonesis.BoNesis(dom1)
    bo.settings["clingo_options"] = (f"--seed={seed}",)
    bo_specs = specification_from_anonymous_paths(bo, paths)
    configs = bo_specs.configurations
    # Universal constraints on fixed points
    for _fp_constraint in spec.constraints.fixed_points:
        _start, *_fps = _fp_constraint.split("|")
        configs[_start] >> "fixpoints" ^ {configs[_fp] for _fp in _fps}
    # Constraints on distances between configurations
    if distances or global_distance:
        enforce_distance(
            bo, bo_specs, distances=distances, global_distance=global_distance
        )
    random.seed(seed)
    for i, (bn, cfg) in enumerate(
        bo.diverse_boolean_networks(limit=150, extra="configurations")
    ):
        new_diversity.append((i, bn, cfg))
    if not new_diversity:
        seed = int(rng.integers(1, 20221231))
        n_abs += 1
    else:
        print(f"Found {len(new_diversity)} diverse boolean networks")

# pylint: enable=no-member, pointless-statement

dom_kwargs = spec.influence_graph
dom_kwargs.update({"seed": seed})
if not save_experiment(new_diversity, dom_kwargs, experiment_dir):
    raise IOError(
        f"Could not save experiment to {data_dir.relative_to(root).as_posix()}"
    )
print(f"Results saved to `{experiment_dir.relative_to(root).as_posix()}`")

with open(experiment_dir / "notes.txt", "a") as f:
    f.write(f"Found {len(new_diversity)} Boolean Networks after {n_abs} failed attempts\n")

diversity_metrics_df = compute_multiple_cfg_metrics(new_diversity)
diversity_metrics_df["phi_distance"] = -1.0 * diversity_metrics_df["phi_coeff"]
_resumed = metrics_summary_statistics(diversity_metrics_df)
_resumed["mean_total_distance"] = (
    _resumed["mean_jaccard_distance"] + _resumed["mean_phi_distance"]
)
_resumed = rearrange_cols_first(_resumed, ["mean_total_distance", "sum_of_deviations"])

diversity_plot = (
    p9.ggplot(
        _resumed,
        p9.aes(
            x="mean_jaccard_distance", y="mean_phi_distance", colour="sum_of_deviations"
        ),
    )
    + p9.geom_point()
)


_resumed_sorted = _resumed.sort_values(
    ["mean_total_distance", "sum_of_deviations"], ascending=[False, True]
)
_best_candidate = _resumed_sorted.index[0]

with open(experiment_dir / "notes.txt", "a") as f:
    f.write(f"Best candidate is bn #{_best_candidate}\n")


candidate_cfgs = new_diversity[_best_candidate][-1]
candidate_cfgs_df = pd.DataFrame(candidate_cfgs)
candidate_bn = new_diversity[_best_candidate][1]

# sns.displot(diversity_metrics_df, x="jaccard_distance", y="phi_distance", kind="kde")
# sns.displot(
#    _resumed_sorted, x="mean_jaccard_distance", y="mean_phi_distance", kind="kde"
# )

print("Computing attractor statistics...", end="\t", flush=True)
attractor_statistics_df = compute_attractor_statistics_ls(
    new_diversity, spec.constraints.initial_state
)
print("Done", flush=True)

print("Generating trajectories...", end="\t", flush=True)
full_traj_df = labelled_trajectory_from_reachability_constraints(
    candidate_bn, bo_specs.reachability_constraints, candidate_cfgs
)
print("Done")

full_traj_df.to_csv(experiment_dir / "full_trace.csv")
attractor_statistics_df.to_csv(experiment_dir / "attractor_statistics.csv")
diversity_metrics_df.to_csv(experiment_dir / "configuration_diversity_metrics.csv")
_resumed_sorted.to_csv(experiment_dir / "resumed_configuration_diversity_metrics.csv")
diversity_plot.save((experiment_dir / "diversity_plot.png").as_posix())

