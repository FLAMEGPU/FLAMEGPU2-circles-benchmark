#! /usr/bin/env python3

import os
import sys
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def cli():
    parser = argparse.ArgumentParser(description="Python script to generate figures from csv files")
    parser.add_argument(
        "-v", 
        "--verbose", 
        action="store_true", 
        help="increase verbosity of output"
    )
    parser.add_argument(
        "-f", 
        "--force", 
        action="store_true", 
        help="Force overwriting of files (surpress user confirmation)"
    )
    parser.add_argument(
        "-o", 
        "--output-dir", 
        type=str, 
        help="directory to output figures into."
    )
    parser.add_argument(
        "--dpi", 
        type=int, 
        help="DPI for output file",
        default=96
    )
    parser.add_argument(
        "-s",
        "--show", 
        action="store_true",
        help="Show the plot(s)"
    )
    parser.add_argument(
        "input_csv", 
        type=str, 
        help="csv containing one row per run"
        # @todo - add a default which works for this repo.
    )

    args = parser.parse_args()
    return args

def load_inputs(input_csv):
    # Read in the csv
    df = pd.read_csv(input_csv, sep=',', quotechar='"')
    # Strip any whitespace from column names
    df.columns = df.columns.str.strip()

    # @todo - validate that the expected columns are available.

    return df

def process_data(input_df):
    print(input_df.columns)

    # Columns to group data by - i.e. identify repetitions of a single run
    group_by_columns = ["GPU", "release_mode", "seatbelts", "model", "steps", "agentCount"]

    # aggregations to apply to each column.
    agg_dict = {
        'ms_rtc': ['mean'],
        'ms_simulation': ['mean'],
        'ms_init': ['mean'],
        'ms_exit': ['mean'],
        'ms_stepMean': ['mean'],
    }

    # New names for each aggregated column, by flattening the dict of lists.
    new_column_labels = [f"{op}_{col}" for col, ops in agg_dict.items() for op in ops]

    # Get the aggregated data
    grouped = input_df.groupby(by=group_by_columns).agg(agg_dict)
    # Apply the new column names
    grouped.columns = new_column_labels
    # Reset the index, 
    grouped = grouped.reset_index()

    return grouped

# @todo
def print_summary(processed_df, output_dir, force):

    if output_dir is not None:
        output_dir_check(output_dir)
        output_filename = "processed.csv"
        # Get the path for output 
        output_filepath = pathlib.Path(output_dir) / output_filename
        # If the file does not exist, or force is true write the output file, otherwise error.
        if not output_filepath.exists or force:
            try:
                print(f"writing csv to {output_filepath}")
                processed_df.to_csv(output_filepath, sep=",", header=True, index=False, quotechar='"', float_format='%.3f')
            except Exception as e:
                print(f"Error: could not write to {output_filepath}")
                return
        else:
            print(f"Error: {output_filepath} already exists. Specify a different `-o/--output-dir` or use `-f/--force`")
            return

def output_dir_check(output_dir):
    if output_dir is not None:
        p = pathlib.Path(output_dir)
        p.mkdir(exist_ok=True)

def plot(processed_df, output_dir, dpi, force, show):
    print(processed_df)
   
    # Use some seaborn deafault for fontsize etc.
    sns.set_context("talk", rc={"lines.linewidth": 2.5})  # notebook, talk, paper or poster

    # Add a background colour.
    sns.set_style("darkgrid")
    # sns.set_palette(sns.color_palette("Dark2"))

    # Have some ticks
    sns.set_style("ticks")


    fig, ax = plt.subplots(constrained_layout=True)

    # Set the size of the figure in inches
    fig.set_size_inches(11.7, 8.27)

    # Log scale on either axis
    # ax.set(xscale="log")
    # ax.set(yscale="log")

    xkey = "agentCount"
    xlabel = "Agent Count"

    ykey = "mean_ms_stepMean"
    ylabel = "Mean Iteration Runtime (ms)"

    huekey = "model"
    stylekey=huekey

    # For now, use seaborn?
    g = sns.lineplot(
        data=processed_df, 
        x=xkey, 
        y=ykey, 
        hue=huekey, 
        style=stylekey, 
        markers=True,
        ax=ax,
        # size=6,
        # legend_out=False
    )

    plt.ylim(0, None)
    plt.xlim(0, None)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.title(f"{ylabel} vs {xlabel}")

    # line_plot.ax.legend(loc=2)

    # g.set_ylabels("survival probability")
    # g.add_legend(bbox_to_anchor=(1.05, 0), loc=2, borderaxespad=0.)


    output_filename = f"{ykey}.png"
    # Make the output direcotry if needed.
    if output_dir is not None:
        output_dir_check(output_dir)

        # Get the path for output 
        output_filepath = pathlib.Path(output_dir) / output_filename
        # If the file does not exist, or force is true write the otuput file, otherwise error.
        if not output_filepath.exists or force:
            try:
                print(f"writing figure to {output_filepath}")
                fig.savefig(output_filepath, dpi=dpi)
            except Exception as e:
                print(f"Error: could not write to {output_filepath}")
                return
        else:
            print(f"Error: {output_filepath} already exists. Specify a different `-o/--output-dir` or use `-f/--force`")
            return

    # If not outputting, or if the show flag was set, show the plot.
    if show or output_dir is None:
        plt.show()

def main():
    # @todo - print some key info to stdout to complement the data? i.e. RTC time? This can just be fetched from the input csv.

    # Process the cli
    args = cli()
    # Load input files
    input_df = load_inputs(args.input_csv)
    # Construct the data to plot
    processed_df = process_data(input_df)

    # Print summary?
    print_summary(processed_df, args.output_dir, args.force)

    # Plot the data
    plot(processed_df, args.output_dir, args.dpi, args.force, args.show)



if __name__ == "__main__":
    args = cli()
    main()