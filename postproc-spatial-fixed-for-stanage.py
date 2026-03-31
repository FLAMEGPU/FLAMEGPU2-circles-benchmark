#! /usr/bin/env python3
import argparse
import math
import pathlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns

"""
Script for extracting data from multiple runs of the fixed density benchmark, only considering spatial3D and spatial3D RTC runs.

-i / --inputs takes a pair of values - a label and a directory, which should contain one directory per GPU model / node type, each with a fixed-density_perSimulationCSV.csv files inside

-o is a required output, and will contain the 3 generated CSVs and 1 plot (for now)

Example inputs would be: 

stanage-bench-data/
├── 2026-03-30-r550
│   ├── a100-sxm4
│   │   └── fixed-density_perSimulationCSV.csv
│   ├── h100-nvl
│   │   └── fixed-density_perSimulationCSV.csv
│   └── h100-pcie
│       └── fixed-density_perSimulationCSV.csv
└── 2026-03-31-r550
    ├── a100-sxm4
    │   └── fixed-density_perSimulationCSV.csv
    ├── h100-nvl
    │   └── fixed-density_perSimulationCSV.csv
    └── h100-pcie
        └── fixed-density_perSimulationCSV.csv

Produced outputs:

output_dir
└── combined
    ├── data.csv
    ├── data.png
    ├── relative.csv
    └── summary.csv

where relative.csv contains the relative performance for each gpu model compared to the first -i, 
"""


def cli():
    cli = argparse.ArgumentParser(
        description="Script for comparing spatial-only fixed-density benchamrks"
    )
    cli.add_argument(
        "-i",
        "--input",
        dest="inputs",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
        help="Pair of [LABEL] and [PATH] where PATH is a directory containing one or more child directories, each containing fixed-density_perSimulationCSV.CSV",
    )
    cli.add_argument(
        "-o",
        "--output",
        required=True,
        type=pathlib.Path,
        help="output path for aggregate CSV",
    )
    cli.add_argument("--show", action="store_true", help="Show the plots interactively")

    args = cli.parse_args()
    return args


def extract_data(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    drop_cols = ["repeat", "agent_density", "mean_message_count"]
    group_cols = [
        "GPU",
        "release_mode",
        "seatbelts_on",
        "model",
        "steps",
        "agent_count",
        "env_width",
        "comm_radius",
        "sort_period",
    ]
    time_cols = ["s_rtc", "s_simulation", "s_init", "s_exit", "s_step_mean"]

    models_to_drop = ["circles_bruteforce", "circles_bruteforce_rtc"]

    dfs = []

    # for each input csv:
    for label, dir_str in args.inputs:
        dir_path = pathlib.Path(dir_str)

        csv_files = list(dir_path.rglob("fixed-density_perSimulationCSV.csv"))

        for path in csv_files:
            # read fixed-density_perSimulationCSV.csv files into pandas dataframe
            df = pd.read_csv(path)

            # Drop some columns (repeat, agent_density, mean_message_count)
            df = df.drop(columns=drop_cols, errors="ignore")

            # Drop rows with models we don't care about (if re-using old data
            df = df[~df["model"].isin(models_to_drop)]

            # Compute the mean times (s_rtc,s_simulation,s_init,s_exit,s_step_mean) for repeats of the same simulation (same: GPU, release_mode,seatbelts_on,model,steps  agent_count,env_width,comm_radius,sort_period).
            df = df.groupby(group_cols)[time_cols].mean().reset_index()

            # Get the total time for each simulations (s_tot = s_rtc + s_simulation + s_init + s_exit + s_step_mean)
            df["s_total"] = (
                df["s_rtc"]
                + df["s_simulation"]
                + df["s_init"]
                + df["s_exit"]
                + df["s_step_mean"]
            )

            # Compute the number of agent updates per each simulation scale (agent_updates = steps * agent_count)
            df["agent_updates"] = df["steps"] * df["agent_count"]

            # Add the label to each row in the dataframe, so they can safely be combined
            df["label"] = label

            # store this df for later.
            dfs.append(df)

    # Combine the dataframes into a single dataframe
    df = pd.concat(dfs, ignore_index=True)

    # Extract summary information across the full sweep
    group_keys = ["label", "GPU", "model"]
    columns_to_sum = [
        "s_simulation",
        "s_init",
        "s_exit",
        "s_step_mean",
        "s_total",
        "agent_updates",
    ]
    summary_df = df.groupby(group_keys)[columns_to_sum].sum().reset_index()

    # Compute the agent_updates_per_s_total and agent_update_per_s_simulation for each (averaged) simulation
    df["agent_updates_per_s_total"] = df["agent_updates"] / df["s_total"]
    df["agent_updates_per_s_simulation"] = df["agent_updates"] / df["s_simulation"]

    # Compute the agent_updates_per_s_total and agent_update_per_s_simulation in the summary df
    summary_df["agent_updates_per_s_total"] = (
        summary_df["agent_updates"] / summary_df["s_total"]
    )
    summary_df["agent_updates_per_s_simulation"] = (
        summary_df["agent_updates"] / summary_df["s_simulation"]
    )

    # pivot so we can compute relative performance between label values for each gpu/model combination
    comparison_df = summary_df.pivot(
        index=["GPU", "model"], columns="label", values="agent_updates_per_s_simulation"
    )
    baseline = comparison_df.columns[0]
    relative_perf = comparison_df.divide(comparison_df[baseline], axis=0)
    relative_perf_df = relative_perf.reset_index()

    # Return the 3 dataframes
    return df, summary_df, relative_perf_df


def plot(df: pd.DataFrame, output_dir_path: pathlib.Path, show: bool) -> None:
    sns.set_style("darkgrid")
    palette = sns.color_palette("Dark2")

    g = sns.relplot(
        data=df,
        kind="line",
        x="agent_count",
        y="agent_updates_per_s_simulation",
        hue="label",
        style="GPU",
        col="model",
        col_wrap=2,
        markers=True,
        dashes=True,
        palette=palette,
        height=6,
        aspect=1.0,
        facet_kws={"sharey": True, "sharex": True},
    )

    global_y_max = df["agent_updates_per_s_simulation"].max()
    magnitude = 10 ** math.floor(math.log10(global_y_max))
    tick_step = magnitude / 2 if global_y_max / magnitude < 2 else magnitude
    rounded_max = math.ceil(global_y_max / tick_step) * tick_step
    for ax in g.axes.flat:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=rounded_max)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))

    output_file_path = output_dir_path / "data.png"
    g.savefig(output_file_path, dpi=100, bbox_inches="tight")
    if show:
        plt.show()


def main():
    # parse cli
    args = cli()

    # extract dat form inputs, into multiple dataframes
    df, summary_df, relative_perf_df = extract_data(args)

    # ensure the output directory exists
    output_dir_path = pathlib.Path(args.output)
    output_dir_path.mkdir(exist_ok=True, parents=True)

    # write out the CSVs and print to stdout
    df.to_csv(output_dir_path / "data.csv", index=False)
    print(df)
    summary_df.to_csv(output_dir_path / "summary.csv", index=False)
    print(summary_df)
    relative_perf_df.to_csv(output_dir_path / "relative.csv", index=False)
    print(relative_perf_df)

    # Plot performance per model per label per gpu
    plot(df, output_dir_path, args.show)


if __name__ == "__main__":
    main()
