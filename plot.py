#! /usr/bin/env python3

import os
import sys
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from dataclasses import dataclass


# Maximum DPI
MAX_SANE_DPI = 1000
# Default DPI
DEFAULT_DPI = 96

LEGEND_BORDER_PAD = 0.5

# Size of figures in inches
FIGSIZE_INCHES = (16, 9)

# CSV files that should be preset in the input directory.
EXPECTED_CSV_FILES=[
    "fixed-density_perSimulationCSV.csv",
    "fixed-density_perStepPerSimulationCSV.csv",
    "variable-density_perSimulationCSV.csv",
    "variable-density_perStepPerSimulationCSV.csv"
]



# input cols for per step per sim
# GPU,release_mode,seatbelts_on,model,steps,agent_count,env_width,comm_radius,repeat,agent_density,step,ms_step

# Input cols for per sim.
# GPU,release_mode,seatbelts_on,model,steps,agent_count,env_width,comm_radius,repeat,agent_density,mean_message_count,ms_rtc,ms_simulation,ms_init,ms_exit,ms_step_mean

# input csv columns which identify a row as a duplicate of another repetition for aggregation, for per-step per-sim csvs
GROUP_BY_COLUMNS_PER_STEP_PER_SIM = ["GPU","release_mode","seatbelts_on","model","steps","agent_count","env_width","comm_radius","step"]

# input csv columns which identify a row as a duplicate of another repetition for aggregation, for per-sim csv
GROUP_BY_COLUMNS_PER_SIM = ["GPU","release_mode","seatbelts_on","model","steps","agent_count","env_width","comm_radius"]


# Aggregate operations to apply across grouped csv rows, for the per-step per-sim csvs
AGGREGATIONS_PER_STEP_PER_SIM = {
    'agent_density': ['mean'],
    'ms_step': ['mean'],
}

# Aggregate operations to apply across grouped csv rows, for the per-sim csvs
AGGREGATIONS_PER_SIM = {
    'agent_density': ['mean'],
    'mean_message_count': ['mean'],
    'ms_rtc': ['mean'],
    'ms_simulation': ['mean'],
    'ms_init': ['mean'],
    'ms_exit': ['mean'],
    'ms_step_mean': ['mean'],
}



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
        default=DEFAULT_DPI
    )
    parser.add_argument(
        "-s",
        "--show", 
        action="store_true",
        help="Show the plot(s)"
    )
    parser.add_argument(
        "input_dir", 
        type=str, 
        help="Input directory, containing the 4 expected input csv files",
        default="."
    )

    args = parser.parse_args()
    return args

def validate_args(args):
    valid = True

    # If output_dir is passed, create it, error if can't create it.
    if args.output_dir is not None:
        p = pathlib.Path(args.output_dir)
        try:
            p.mkdir(exist_ok=True)
        except Exception as e:
            print(f"Error: Could not create output directory {p}: {e}")
            valid = False

    # DPI must be positive, and add a max.
    if args.dpi is not None:
        if args.dpi < 1:
            print(f"Error: --dpi must be a positive value. {args.dpi}")
            valid = False
        if args.dpi > MAX_SANE_DPI:
            print(f"Error: --dpi should not be excessively large. {args.dpi} > {MAX_SANE_DPI}")
            valid = False

    # Ensure that the input direcotry exists, and that all required inputs are present.
    input_dir = pathlib.Path(args.input_dir) 
    if input_dir.is_dir():
        missing_files = []
        for csv_filename in EXPECTED_CSV_FILES:
            csv_path = input_dir / csv_filename
            if not csv_path.is_file():
                missing_files.append(csv_filename)
                valid = False
        if len(missing_files) > 0:
            print(f"Error: {input_dir} does not contain required files:")
            for missing_file in missing_files:
                print(f"  {missing_file}")
    else:
        print(f"Error: Invalid input_dir provided {args.input_dir}")
        valid = False

    return valid

def load_inputs(input_dir):
    dfs = {}
    input_dir = pathlib.Path(input_dir)

    for csv_name in EXPECTED_CSV_FILES:
        csv_path = input_dir / csv_name

        # Read in the csv
        df = pd.read_csv(csv_path, sep=',', quotechar='"')

        # Strip any whitespace from column names
        df.columns = df.columns.str.strip()

        # @todo - validate that the expected columns are available.

        dfs[csv_name] = df

    return dfs

def process_data(input_dataframes, verbose):
    output_dataframes = {}

    for csv_name, input_df in input_dataframes.items():
        if verbose:
            print(f"processing {csv_name}")

        if verbose:
            print(f"input columns:")
            for column in input_df.columns:
                print(f"  {column}")

        # If its a per-step file, use one set of operations, otherwise us a different set of operations.

        csv_is_per_step = "perStep" in csv_name

        # Columns to group data by - i.e. identify repetitions of a single run
        group_by_columns = GROUP_BY_COLUMNS_PER_STEP_PER_SIM if csv_is_per_step else GROUP_BY_COLUMNS_PER_SIM

        # fetch the appropriate list of aggregate operations to apply.
        aggregations = AGGREGATIONS_PER_STEP_PER_SIM if csv_is_per_step else AGGREGATIONS_PER_SIM

        # New names for each aggregated column, by flattening the dict of lists.
        new_column_labels = [f"{op}_{col}" for col, ops in aggregations.items() for op in ops]

        # Get the aggregated data
        grouped_df = input_df.groupby(by=group_by_columns).agg(aggregations)
        # Apply the new column names
        grouped_df.columns = new_column_labels
        # Reset the index, 
        grouped_df = grouped_df.reset_index()

        grouped_df["env_volume"] = grouped_df["env_width"] * grouped_df["env_width"] * grouped_df["env_width"]

        if verbose:
            print(f"output columns:")
            for column in grouped_df.columns:
                print(f"  {column}")

        # Store teh processed dataframe.
        output_dataframes[csv_name] = grouped_df

    return output_dataframes

def store_processed_data(input_dataframes, processed_dataframes, output_dir, force, verbose):
    success = True

    # If the output_dir is not none, save each processed csv to disk.
    if output_dir is not None:
        output_dir = pathlib.Path(output_dir)
        for csv_name, processed_df in processed_dataframes.items():
            output_csv_path = output_dir / f"processed_{csv_name}"
            if not output_csv_path.exists() or force:
                try:
                    if verbose:
                        print(f"Writing to {output_csv_path}")
                    processed_df.to_csv(output_csv_path, sep=",", header=True, index=False, quotechar='"', float_format='%.3f')
                except Exception as e:
                    print(f"Error: could not write to {output_csv_path} with exception {e}")
                    success = False
            else:
                print(f"Error: {output_csv_path} already exists. Use `-f/--force` to overwrite.")
                success = False

    # Print out some key values to stdout.

    # @todo - write to a summary txt file in the output dir, and / or stdout.
    for csv_name, input_df in input_dataframes.items():

        # Get the max rtc time from the input file, and also output the mean too for good measure.
        # @todo - might be better to have a threshold cutoff?
        if "ms_rtc" in input_df: 
            max_ms_rtc = input_df["ms_rtc"].max()
            mean_ms_rtc = input_df["ms_rtc"].mean()
            print(f"{csv_name}: max_ms_rtc {max_ms_rtc:.3f}, mean_ms_rtc {mean_ms_rtc:.3f}")

    return success

MANUAL_PRETTY_CSV_KEY_MAP = {
    "step": "Step",
    "GPU": "GPU",
    "release_mode": "Release Mode",
    "seatbelts_on": "Seatbelts On",
    "model": "Implementation",
    "steps": "Steps",
    "agent_count": "Agent Count",
    "env_width": "Environment Width",
    "comm_radius": "Communication Radius",
    "repeat": "Repeat",
    "agent_density": "Agent Density",
    "mean_message_count": "Average Message Count",
    "ms_rtc": "RTC Time (ms)",
    "ms_simulation": "Simulation Time (ms)",
    "ms_init": "Init Function Time (ms)",
    "ms_exit": "Exit Function Time (ms)",
    "ms_step_mean": "Average Step Time (ms)",
    "ms_step": "Step Time (ms)",
    "mean_ms_rtc": "Average RTC Time (ms)",
    "mean_ms_simulation": "Average Simulation Time (ms)",
    "mean_ms_init": "Average Init Function Time (ms)",
    "mean_ms_exit": "Average Exit Function Time (ms)",
    "mean_ms_step_mean": "Average Step Time (ms)",
    "mean_ms_step": "Average Step Time (ms)",
    "mean_agent_density": "Agent Density",
    "env_volume": "Environment Volume",
}

def pretty_csv_key(csv_key):
    if csv_key is None:
        return None
    pretty_key = csv_key.replace("_", " ")
    if csv_key in MANUAL_PRETTY_CSV_KEY_MAP:
        pretty_key = MANUAL_PRETTY_CSV_KEY_MAP[csv_key]
    return pretty_key

# Dataclass requires py 3.7, but saves a bunch of effort.
@dataclass 
class PlotOptions:
    """Class for options for a single plot"""
    xkey: str
    ykey: str
    huekey: str = None
    stylekey: str = None
    plot_type: str = "lineplot"
    filename: str = None
    logx: bool = False
    logy: bool = False
    minx: int = None
    maxx: int = None
    miny: int = None
    maxy: int = None
    # “auto”, “brief”, “full”, or False
    sns_legend: str = "auto"
    legend_outside: bool = True
    legend_y_offset: float = -0.00
    df_query: str = None
    sns_palette: str = "Dark2"
    sns_style: str = "darkgrid"
    # notebook, talk, paper or poster
    sns_context: str = "talk"


    def plot(self, df_in, output_prefix, output_dir, dpi, force, show, verbose):
        df = df_in

        # Set a filename if needed. 
        if self.filename is None:
            self.filename = f"{self.plot_type}--{self.xkey}--{self.ykey}--{self.huekey}--{self.stylekey}"

        # Use some seaborn deafault for fontsize etc.
        sns.set_context(self.sns_context, rc={"lines.linewidth": 2.5})  

        # Set the general style.
        sns.set_style(self.sns_style)

        # Filter the data using pandas queries if required.
        if self.df_query is not None and len(self.df_query):
            df = df.query(self.df_query)
            
        # If the df is empty, skip.
        if df.shape[0] == 0:
            print(f"Skipping plot {self.filename} - 0 rows of data")
            return False

        # Get the number of palette values required.
        huecount = len(df[self.huekey].unique()) if self.huekey is not None else 1 

        # Set palette.
        palette = sns.color_palette(self.sns_palette, huecount)
        sns.set_palette(palette)


        # create a matplotlib figure and axis, for a single plot.
        # Use constrained layout for better legend placement.
        fig, ax = plt.subplots(constrained_layout=True)

        # Set the size of the figure in inches
        fig.set_size_inches(FIGSIZE_INCHES[0], FIGSIZE_INCHES[1])


        # Generate labels / titles etc.
        xlabel = f"{pretty_csv_key(self.xkey)}"
        ylabel = f"{pretty_csv_key(self.ykey)}"
        huelabel = f"{pretty_csv_key(self.huekey)}"
        stylelabel = f"{pretty_csv_key(self.stylekey)}"
        hs_label = f"{huelabel} x {stylelabel}" if huelabel != stylelabel else f"{huelabel}"
        figtitle = f"{ylabel} vs {xlabel} (hs_label)"

        # @todo - validate keys.

        # Decide if using internal legend.
        external_legend = self.legend_outside

        g = None
        if self.plot_type == "lineplot":
            # plot the data @todo - lineplot vs scatter?
            g = sns.lineplot(
                data=df, 
                x=self.xkey, 
                y=self.ykey, 
                hue=self.huekey, 
                style=self.stylekey, 
                markers=True,
                ax=ax,
                # size=6,
                legend=self.sns_legend,
                palette=palette,
            )
        elif self.plot_type == "scatterplot":
            g = sns.scatterplot(
                data=df, 
                x=self.xkey, 
                y=self.ykey, 
                hue=self.huekey, 
                style=self.stylekey, 
                markers=True,
                ax=ax,
                # size=6,
                legend=self.sns_legend,
                palette=palette,
            )
        else:
            raise Exception(f"Bad plot_type {self.plot_type}")

        # Set a title
        # @disabled for now.
        # if len(figtitle):
        #     plt.title(figtitle)

        # adjust x axis if required.
        ax.set(xlabel=xlabel)
        if self.logx:
            ax.set(xscale="log")
        if self.minx is not None:
            ax.set_xlim(left=self.minx)
        if self.maxx is not None:
            ax.set_xlim(right=self.maxx)

        # adjust y axis if required.
        ax.set(ylabel=ylabel)
        if self.logy:
            ax.set(yscale="log")
        if self.miny is not None:
            ax.set_ylim(bottom=self.miny)
        if self.maxy is not None:
            ax.set_ylim(top=self.maxy)

        # Disable scientific notation on axes
        ax.ticklabel_format(useOffset=False, style='plain')

        # If there is reason to have a legend, do some extra processing.
        if ax.get_legend() is not None:
            legend = ax.get_legend()
            loc = None
            bbox_to_anchor = None
            if external_legend:
                # Set legend placement if not internal.
                loc = "upper left"
                # @todo - y offset should be LEGNED_BORDER_PAD trasnformed from font units to bbox.
                bbox_to_anchor = (1, 1 - self.legend_y_offset)

            # Get the handles and labels for the legend
            handles, labels = ax.get_legend_handles_labels()

            # Iterate the labels in the legend, looking for the huekey or stylekey as values
            # If either were found, replace with the pretty version
            found_count = 0
            for i, label in enumerate(labels):
                if label == self.huekey: 
                    labels[i] = huelabel
                    found_count += 1
                elif label == self.stylekey:
                    labels[i] = stylelabel
                    found_count += 1

            # If neither were found, set a legend title.
            if found_count == 0:
                # add an invisble patch with the appropriate label, like how seaborn does if multiple values are provided.
                handles.insert(0, mpatches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False, label=hs_label))
                labels.insert(0, hs_label)
                pass
            ax.legend(handles=handles, labels=labels, loc=loc, bbox_to_anchor=bbox_to_anchor, borderaxespad=LEGEND_BORDER_PAD)

        # if an output directory is provided, save the figure to disk. 
        if output_dir is not None:
            output_dir = pathlib.Path(output_dir)
            # Prefix the filename with the experiment prefix.
            output_filename = f"{output_prefix}--{self.filename}"

            # Get the path for output 
            output_filepath = output_dir / output_filename
            # If the file does not exist, or force is true write the otuput file, otherwise error.
            if not output_filepath.exists() or force:
                try:
                    if verbose:
                        print(f"writing figure to {output_filepath}")
                    fig.savefig(output_filepath, dpi=dpi, bbox_inches='tight')
                except Exception as e:
                    print(f"Error: could not write to {output_filepath}")
                    return False
            else:
                print(f"Error: {output_filepath} already exists. Specify a different `-o/--output-dir` or use `-f/--force`")
                return False

        # If not outputting, or if the show flag was set, show the plot.
        if show: # or output_dir is None:
            plt.show()

        return True

QUALITATIVE_PALETTE = "Dark2"
SEQUENTIAL_PALETTE = "viridis"

# Define the figures to generate for each input CSV.
PLOTS_PER_CSV={
    # No need for sequential colour pallete 
    "fixed-density_perSimulationCSV.csv": [
        PlotOptions(
            filename="volume--step-ms--model--all.png",
            plot_type="lineplot",
            xkey="env_volume",
            ykey="mean_ms_step_mean",
            huekey="model",
            stylekey="model",
            sns_palette=QUALITATIVE_PALETTE,
            minx=0,
            miny=0
        ),
        PlotOptions(
            filename="volume--step-ms--model--zoomed.png",
            plot_type="lineplot",
            xkey="env_volume",
            ykey="mean_ms_step_mean",
            huekey="model",
            stylekey="model",
            sns_palette=QUALITATIVE_PALETTE,
            # df_query="model == 'circles_spatial3D' or model == 'circles_spatial3D_rtc'",
            minx=0,
            miny=0,
            maxy=200,
        ),
        # PlotOptions(
        #     filename="volume--message-count--model--zoomed.png",
        #     plot_type="lineplot",
        #     xkey="env_volume",
        #     ykey="mean_mean_message_count",
        #     huekey="model",
        #     stylekey="model",
        #     sns_palette=QUALITATIVE_PALETTE,
        #     # df_query="model == 'circles_spatial3D' or model == 'circles_spatial3D_rtc'",
        #     minx=0,
        #     miny=0,
        #     maxy=1000,
        # )
    ],
    "fixed-density_perStepPerSimulationCSV.csv": [
        # PlotOptions(
        #     filename="scatterplot--step--mean_ms_step--volume--volume--spatial3D-rtc-only.png",
        #     plot_type="scatterplot",
        #     xkey="step",
        #     ykey="mean_ms_step",
        #     huekey="env_volume",
        #     stylekey="env_volume",
        #     df_query="model == 'circles_spatial3D_rtc'",
        #     sns_palette=SEQUENTIAL_PALETTE,
        # )
    ],
    # Worth using a sequential colour pallette here.
    "variable-density_perSimulationCSV.csv": [
        # PlotOptions(
        #     plot_type="lineplot",
        #     xkey="agent_count",
        #     ykey="mean_ms_step_mean",
        #     huekey="env_volume",
        #     stylekey="env_volume",
        #     df_query="model == 'circles_spatial3D_rtc'",
        #     sns_palette=SEQUENTIAL_PALETTE,
        # ),
        PlotOptions(
            filename="volume--step-ms--density--3drtc.png",
            plot_type="lineplot",
            xkey="env_volume",
            ykey="mean_ms_step_mean",
            huekey="mean_agent_density",
            stylekey="mean_agent_density",
            df_query="model == 'circles_spatial3D_rtc'",
            sns_palette=SEQUENTIAL_PALETTE,
        ),
        PlotOptions(
            filename="densit--step-ms--volume--3drtc.png",
            plot_type="lineplot",
            xkey="mean_agent_density",
            ykey="mean_ms_step_mean",
            huekey="env_volume",
            stylekey="env_volume",
            df_query="model == 'circles_spatial3D_rtc'",
            sns_palette=SEQUENTIAL_PALETTE,
        )
    ],
    "variable-density_perStepPerSimulationCSV.csv": [

    ]
}

def plot_figures(processed_dataframes, output_dir, dpi, force, show, verbose):

    # For each processed dataframe
    for csv_name, processed_df in processed_dataframes.items():
        csv_is_per_step = "perStep" in csv_name

        output_prefix = csv_name.split("_")[0]
        if csv_is_per_step:
            output_prefix = f"{output_prefix}_perStep"

        # Get the list of figures to generate, based on the type of csv / the csv name?
        if csv_name in PLOTS_PER_CSV:
            plots_to_generate = PLOTS_PER_CSV[csv_name]
            for plot_options in plots_to_generate:
                plotted = plot_options.plot(processed_df, output_prefix, output_dir, dpi, force, show, verbose)
    

def main():
    # @todo - print some key info to stdout to complement the data? i.e. RTC time? This can just be fetched from the input csv.

    # Process the cli
    args = cli()
    # Validate cli
    valid_args = validate_args(args)
    if not valid_args:
        return False

    # Load all input dataframes.
    input_dataframes = load_inputs(args.input_dir)

    # Process the dataframes. 
    processed_dataframes = process_data(input_dataframes, args.verbose)

    # Store the processed dataframes on disk if an output dir is provided and/or print some stuff to console.
    store_processed_data(input_dataframes, processed_dataframes, args.output_dir, args.force, args.verbose)

    # Plot the figures to disk, or interactively.
    plot_figures(processed_dataframes, args.output_dir, args.dpi, args.force, args.show, args.verbose)

# Run the main methood if this was not included as a module
if __name__ == "__main__":
    args = cli()
    main()