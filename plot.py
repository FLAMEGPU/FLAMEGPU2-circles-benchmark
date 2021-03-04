#! /usr/bin/env python3

import os
import sys
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



""" def matt():
    initialPopSize = 0 
    finalPopSize = 0
    popSizeIncrement = 0

    initialNumSpecies = 0
    finalNumSpecies = 0
    numSpeciesIncrement = 0

    with open('params.csv') as f:
        csvreader = csv.reader(f)

        popRow = next(csvreader)
        popRow = list(map(int, popRow))
        initialPopSize = popRow[0]
        finalPopSize = popRow[1]
        popSizeIncrement = popRow[2]

        speciesRow = next(csvreader)
        speciesRow = list(map(int, speciesRow))
        initialNumSpecies = speciesRow[0]
        finalNumSpecies = speciesRow[1]
        numSpeciesIncrement = speciesRow[2]     

    with open('serial.csv') as serial:
        with open('concurrent.csv') as concurrent:
            fig, ax = plt.subplots()
            
            # Read in data
            serialRows = []
            concurrentRows = []
            
            csvreader = csv.reader(serial)
            for row in csvreader:
            row = list(map(float, row))
            serialRows.append(row)
            
            csvreader = csv.reader(concurrent)
            for row in csvreader:
            row = list(map(float, row))
            concurrentRows.append(row)
            
            # Plot serial results
            for row in serialRows:
            ax.plot(np.arange(initialNumSpecies, finalNumSpecies + 1, numSpeciesIncrement),row,"^--", linewidth=1)
            
            # Plot concurrent results
            for row in concurrentRows:
            ax.plot(np.arange(initialNumSpecies, finalNumSpecies + 1, numSpeciesIncrement),row,"^-", linewidth=1)
            
            # Display timing results  
            ax.grid(True)
            ax.set_title('Average step time against number of species')
            ax.set_ylabel('Average Step time (ms)')
            ax.set_xlabel('Number of species')
            ax.xaxis.set_ticks(np.arange(initialNumSpecies, finalNumSpecies + 1, numSpeciesIncrement))
            ax.legend(np.arange(initialPopSize, finalPopSize + 1, popSizeIncrement), title="Population Size")
            
            # Plot speedup
            fig2, ax2 = plt.subplots()
            for s, c in zip(serialRows, concurrentRows):
            r = []
            for i in range(len(s)):
                r.append(s[i] / c[i])
            ax2.plot(np.arange(initialNumSpecies, finalNumSpecies + 1, numSpeciesIncrement), r, "^-", linewidth=1)

            ax2.set_title('Speedup against number of species')
            ax2.set_ylabel('Speedup')
            ax2.set_xlabel('Number of species')
            ax2.xaxis.set_ticks(np.arange(initialNumSpecies, finalNumSpecies + 1, numSpeciesIncrement))
            ax2.legend(np.arange(initialPopSize, finalPopSize + 1, popSizeIncrement), title="Population Size")
            plt.show() """
    

def cli():
    parser = argparse.ArgumentParser(description="Python script to generate figures from csv files")
    parser.add_argument(
        "-v", 
        "--verbose", 
        action="store_true", 
        help="increase verbosity of output"
    )
    # parser.add_argument(
    #     "-f", 
    #     "--force", 
    #     action="store_true", 
    #     help="Force overwriting of files (surpress user confirmation)"
    # )
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
        "input_csv", 
        type=str, 
        help="csv containing one row per run"
        # @todo - add a default which works for this repo.
    )

    args = parser.parse_args()
    return args

def load_inputs(input_csv):
    # Read in the csv
    df = pd.read_csv(input_csv)
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

def plot(output_dir, dpi, processed_df):
    print("@todo - plot()")
    print(processed_df)


    # f, ax = plt.subplots(figsize=(7, 7))
    # ax.set(xscale="log", yscale="linear")
    # Some formatting
    sns.set_theme()
    sns.set_style("darkgrid")
    sns.set_palette(sns.color_palette("Dark2"))

    fig, ax = plt.subplots(constrained_layout=True)



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

    plt.show()


def main():
    # @todo - print some key info to stdout to complement the data? i.e. RTC time? This can just be fetched from the input csv.

    # Process the cli
    args = cli()
    # Load input files
    input_df = load_inputs(args.input_csv)
    # Construct the data to plot
    processed_df = process_data(input_df)
    # Plot the data
    plot(args.output_dir, args.dpi, processed_df)



if __name__ == "__main__":
    args = cli()
    main()