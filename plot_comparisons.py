#! /usr/bin/env python3

"""
Python script to produce plots comparing benchmark results from multiple runs of the same benchmark.
I.e. to allow comparison of hardware, software or compiler version etc.

Requires that `plot.py` be called on all relevant benchmarks first, to extract and aggregate data to avoid duplication.

Input experiment configuration via required yaml file, output location can be overridden by CLI.
"""

import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import yaml

SPEEDUP_COL_MODIFIER = "__speedup__"
MONOSPACE_FONT_FAMILY = "DejaVu Sans Mono"

class ComparisonPlotConfig():
    """ Class representing the configuration for this script, to be loaded from disk, but with default values where appropriate
    """
    def __init__(self, config_path, yaml_data, CLI_OUTPUT_DIR = None):
        """ Create an instance of the config object, either using the default values or loaded from disk if yaml_data is provided / when required.
        Implicitly validates?
        """
        # If the yaml contains the input directory map, use it and set teh parent dir as appropriate
        if yaml_data is not None and "LABELLED_INPUT_DIRS" in yaml_data and config_path is not None:
            self.input_path_base = config_path.resolve().parent
            self.LABELLED_INPUT_DIRS = {k:pathlib.Path(v) for k, v in yaml_data["LABELLED_INPUT_DIRS"].items()}
        else:
            self.input_path_base = pathlib.Path(__file__).resolve().parent
            self.LABELLED_INPUT_DIRS = {
                "V100 CUDA 11.0": pathlib.Path("./sample/figures/v100-515.65.01/2.0.0-rc-v100-11.0-beltsoff/"),
                "A100 CUDA 11.8": pathlib.Path("./sample/figures/a100-525.105.17/2.0.0-rc-a100-11.8-beltsoff/"),
                "H100 CUDA 11.8": pathlib.Path("./sample/figures/h100-525.105.17/2.0.0-rc-h100-11.8-beltsoff/"), 
            }
        self.INPUT_FILE_PER_DIR = yaml_data["INPUT_FILE_PER_DIR"] if "INPUT_FILE_PER_DIR" in yaml_data else "processed_fixed-density_perSimulationCSV.csv"
        # Set the output dir, by default to the input config file dir, otherwies use the value from file, or override from the cli.
        self.OUTPUT_DIR = self.input_path_base
        if "OUTPUT_DIR" in yaml_data:
            op = pathlib.Path(yaml_data["OUTPUT_DIR"])
            self.OUTPUT_DIR = op if op.is_absolute() else (self.input_path_base / op).resolve()
        # Override the output dir if one is passed in from the CLI, relative to the working directory.
        if CLI_OUTPUT_DIR is not None:
            self.OUTPUT_DIR = CLI_OUTPUT_DIR.resolve()
        self.OUTPUT_FILE_PREFIX = yaml_data["OUTPUT_FILE_PREFIX"] if "OUTPUT_FILE_PREFIX" in yaml_data else "plot"
        self.COMBINED_CSV_OUTPUT_FILENAME = yaml_data["COMBINED_CSV_OUTPUT_FILENAME"] if "COMBINED_CSV_OUTPUT_FILENAME" in yaml_data else "combined.csv"
        self.SIMULATORS = None
        if "SIMULATORS" in yaml_data:
            self.SIMULATORS = yaml_data["SIMULATORS"]
        else:
            self.SIMULATORS = {
                "Circles Brute Force": "circles_bruteforce",
                "Circles Spatial3D": "circles_spatial3D",
                "Circles Brute Force RTC": "circles_bruteforce_rtc",
                "Circles Spatial3D RTC": "circles_spatial3D_rtc",
            }
        self.TITLE_PREFIX = yaml_data["TITLE_PREFIX"] if "TITLE_PREFIX" in yaml_data else "FLAME GPU 2 Circles Benchmark: Fixed Density"
        self.X = yaml_data["X"] if "X" in yaml_data else "agent_count"
        self.XLABEL = yaml_data["XLABEL"] if "XLABEL" in yaml_data else "Population Size"
        self.Y = yaml_data["Y"] if "Y" in yaml_data else "mean_s_simulation"
        self.YLABEL = yaml_data["YLABEL"] if "YLABEL" in yaml_data else "Mean Simulation Time (s)"
        self.YLIM_BOTTOM = yaml_data["YLIM_BOTTOM"] if "YLIM_BOTTOM" in yaml_data else None
        self.HUE = yaml_data["HUE"] if "HUE" in yaml_data else "label"
        self.HUELABEL="GPU/CUDA"
        self.STYLE = yaml_data["STYLE"] if "STYLE" in yaml_data else "label"
        self.STYLELABEL="GPU/CUDA"
        self.FIGSIZE_INCHES = yaml_data["FIGSIZE_INCHES"] if "FIGSIZE_INCHES" in yaml_data else (16, 9)
        self.SNS_CONTEXT="talk"
        self.SNS_PALETTE="Dark2"
        self.SNS_STYLE="darkgrid"
        self.DPI = yaml_data["DPI"] if "DPI" in yaml_data else 96
        self.LEGEND_BORDER_PAD = yaml_data["LEGEND_BORDER_PAD"] if "LEGEND_BORDER_PAD" in yaml_data else 0.5
        self.EXTERNAL_LEGEND = yaml_data["EXTERNAL_LEGEND"] if "EXTERNAL_LEGEND" in yaml_data else False
        # Speedup related values
        self.SPEEDUP_OUTPUT_FILE_PREFIX = yaml_data["SPEEDUP_OUTPUT_FILE_PREFIX"] if "SPEEDUP_OUTPUT_FILE_PREFIX" in yaml_data else "plot-speedup-v100-fixed-density-max-pop"
        self.SPEEDUP_FILTER_COLUMN = yaml_data["SPEEDUP_FILTER_COLUMN"] if "SPEEDUP_FILTER_COLUMN" in yaml_data else "agent_count" 
        self.SPEEDUP_BASE_INPUT_LABEL = yaml_data["SPEEDUP_BASE_INPUT_LABEL"] if "SPEEDUP_BASE_INPUT_LABEL" in yaml_data else "V100 SXM2 CUDA 11.0"
        self.SPEEDUP_TITLE = yaml_data["SPEEDUP_TITLE"] if "SPEEDUP_TITLE" in yaml_data else "FLAME GPU 2 Circles Benchmark: Maximum Population Size Speedup vs V100 SXM2 CUDA 11.0"
        self.SPEEDUP_X = yaml_data["SPEEDUP_X"] if "SPEEDUP_X" in yaml_data else "model"
        self.SPEEDUP_XLABEL = yaml_data["SPEEDUP_XLABEL"] if "SPEEDUP_XLABEL" in yaml_data else "Model Implementation"
        self.SPEEDUP_Y = yaml_data["SPEEDUP_Y"] if "SPEEDUP_Y" in yaml_data else "mean_s_simulation"
        self.SPEEDUP_YLABEL = yaml_data["SPEEDUP_YLABEL"] if "SPEEDUP_YLABEL" in yaml_data else "Simulation Speedup"
        self.SPEEDUP_YLIM_TOP = yaml_data["SPEEDUP_YLIM_TOP"] if "SPEEDUP_YLIM_TOP" in yaml_data else None
        self.SPEEDUP_HUE = yaml_data["SPEEDUP_HUE"] if "SPEEDUP_HUE" in yaml_data else "label"
        self.SPEEDUP_HUE_LABEL = yaml_data["SPEEDUP_HUE_LABEL"] if "SPEEDUP_HUE_LABEL" in yaml_data else "GPU/CUDA"

    def to_yaml(self):
        """ Get a string containing the YAML representation of this object.
        """
        def yaml_custom_representer(dumper, data):
            return yaml.dumper.represent_dict(data.__dict__)
        yaml.add_representer(self, yaml_custom_representer)
        return yaml.dump(self, sort_keys=False)
    
    def get_abs_LABELLED_INPUT_DIRS(self):
        """ Get a map containing the absolute paths for the LABNELLED_INPUT_DIRS member variable
        """
        return {k:(self.input_path_base / v).resolve() if not v.is_absolute() else v for k, v in self.LABELLED_INPUT_DIRS.items()}
    
    def get_abs_OUTPUT_DIR(self):
        return (self.input_path_base / self.OUTPUT_DIR).resolve() if not self.OUTPUT_DIR.is_absolute() else self.OUTPUT_DIR

    def validate(self):
        """ Ensure that required member data is ok. This might not be perfect, as can't be fully sure until data is loaded.
        """
        # Error if no input directories
        if len(self.LABELLED_INPUT_DIRS) == 0:
            raise Exception("LABELLED_INPUT_DIRS must contain at least one element")
        # Error if no INPUT_FILE_PER_DIR is specified
        if len(self.INPUT_FILE_PER_DIR) == 0:
            raise Exception("INPUT_FILE_PER_DIR must be non-empty")
        for l, d in self.get_abs_LABELLED_INPUT_DIRS().items():
            if not d.is_dir():
                raise Exception(f"LABELLED_INPUT_DIRS[{l}] = {f} is not a valid directory")
            # Error if INPUT_FILE_PER_DIR is not specified. 
            f = d / self.INPUT_FILE_PER_DIR
            if not f.is_file():
                raise Exception(f"INPUT_FILE_PER_DIR for {l}: {f} does not exist.")
        # Simulators, X, Y, H, S must be non empty. Actual validation of the value can't be done until each file is loaded, and will raise an error later anyway.
        if len(self.SIMULATORS) == 0:
            raise Exception(f"config.SIMULATORS must be non-empty")
        if len(self.X) == 0:
            raise Exception(f"config.X must be non-empty")
        if len(self.Y) == 0:
            raise Exception(f"config.Y must be non-empty")
        if len(self.HUE) == 0:
            raise Exception(f"config.HUE must be non-empty")
        if len(self.STYLE) == 0:
            raise Exception(f"config.STYLE must be non-empty")
        # If no exceptions, validation passes
        return True

def main():
    # CLI parsing
    parser = argparse.ArgumentParser(description="Create plots comparing multiple runs of the same benchmark, configured via YAML file.")
    parser.add_argument("-c", "--config", required=True, type=pathlib.Path, help="Path to configuration YAML file containing configuration about which elements to plot")
    parser.add_argument("-o", "--output-dir", type=pathlib.Path, help="Path to output directory for plots")
    parser.add_argument("-f", "--force", action="store_true", help="Force output of files, overwriting existing files.")
    parser.add_argument("--no-plots", action="store_true", help="Prevent cration of plot files, i.e. only compute speedups.")
    args = parser.parse_args()

    # Validate output dir if passed via cli
    if args.output_dir is not None:
        if args.output_dir.is_file():
            raise Exception(f"-o, --output-dir {args.output_dir} is an existing file. Please specify an unused path or existing directory")

    # Load config from disk
    if not args.config.is_file():
        raise Exception(f"Input yaml file {args.config} is not a file")

    yaml_config = None
    with open(args.config, "r") as fp:
        yaml_config = yaml.safe_load(fp)
    if yaml_config is None:
        raise Exception(f"No data lodaded from YAML file {args.config}")

    config = ComparisonPlotConfig(args.config, yaml_config, args.output_dir)

    # Validate the yaml includes required elements, as best as we can at this stage.
    config.validate()

    # Aggregate and combine dataframes
    big_df, dataframes = prepare_data(config)

    # Print some data to stdout for tabular output
    tables(config, big_df, dataframes, args.force)

    # Speedup plots.
    if not args.no_plots:
        speedup_plots(config, big_df, args.force)

    # Non-speedup plots.
    if not args.no_plots:
        plots(config, big_df, args.force)

def prepare_data(config):
    """ Process input datafiles, computing new data as required
    """
    # Load raw data into many dataframes
    dataframes = {}
    for label, d in config.get_abs_LABELLED_INPUT_DIRS().items():
        filepath = d / config.INPUT_FILE_PER_DIR
        # Read in the csv
        df = pd.read_csv(filepath, sep=',', quotechar='"')
        # Strip any whitespace from column names
        df.columns = df.columns.str.strip()
        # Add the label for easier series plotting
        df["label"] = label
        # Store the dataframe for later
        dataframes[label] = df

    # Compute the relative speeudp to each dataframe, in each dataframe. Does not do it at the big df level because can't have non unique multi indexes.
    # Blindly trust the dataframes are the same shape and order. If not there are bigger problems anyway.
    for base_label, base_df in dataframes.items():
        if len(config.SPEEDUP_BASE_INPUT_LABEL) == 0 or base_label == config.SPEEDUP_BASE_INPUT_LABEL:
            new_column = f"{config.SPEEDUP_Y}{SPEEDUP_COL_MODIFIER}{base_label.replace(' ', '_')}"
            for inner_label, inner_df in dataframes.items():
                dataframes[inner_label][new_column] = base_df[config.SPEEDUP_Y] / inner_df[config.SPEEDUP_Y]

    # Concat all the dataframes into a big dataframe? (I could have done this in the first pass...)
    big_df = pd.concat(dataframes.values(), ignore_index=True)
    return big_df, dataframes

def tables(config, big_df, dataframes, force):
    """ Print / Save data for tabular output, to csv and stdout as appropraite"
    """
    # output the combined csv to disk, @todo parameterise and cli this?
    OUTPUT_CSV_FILENAME = f"{config.COMBINED_CSV_OUTPUT_FILENAME}"
    OUTPUT_CSV_FILEPATH = config.get_abs_OUTPUT_DIR() / OUTPUT_CSV_FILENAME
    big_df.to_csv(OUTPUT_CSV_FILEPATH, sep=",", header=True, index=False, quotechar='"', float_format='%.6f')

    # Print to stdout a markdown table for a specific query
    speedup_cols = [x for x in big_df.columns if x.startswith(f"{config.SPEEDUP_Y}{SPEEDUP_COL_MODIFIER}")]
    max_speedup_filter_col_value = big_df[config.SPEEDUP_FILTER_COLUMN].max()
    biggest_sim = big_df.query(f"{config.SPEEDUP_FILTER_COLUMN} == {max_speedup_filter_col_value}")[["label", "model", config.SPEEDUP_Y] + speedup_cols]

    # Pivot to get a table of per GPU/CUDA per model simulation time
    print(f"mean_s_simulation for {config.SPEEDUP_FILTER_COLUMN} == {max_speedup_filter_col_value}")
    pivot_simulation = biggest_sim.pivot(index="model", columns="label", values="mean_s_simulation")
    # reindex to get the deisred order of columns
    pivot_simulation = pivot_simulation.reindex(list(config.get_abs_LABELLED_INPUT_DIRS().keys()), axis=1)
    print(pivot_simulation.to_markdown(floatfmt=".3f"))
    print()

    # Pivot to get a table of per GPU/CUDA per model speedup against a base
    print(f"{speedup_cols[0]} for {config.SPEEDUP_FILTER_COLUMN} == {max_speedup_filter_col_value}")
    pivot_speedup = biggest_sim.pivot(index="model", columns="label", values=speedup_cols[0])
    # reindex to get the deisred order of columns
    pivot_speedup = pivot_speedup.reindex(list(config.get_abs_LABELLED_INPUT_DIRS().keys()), axis=1)
    print(pivot_speedup.to_markdown(floatfmt=".3f"))

def speedup_plots(config, big_df, force):
    """ Plot speedup data to disk
    """
    speedup_cols = [x for x in big_df.columns if x.startswith(f"{config.SPEEDUP_Y}{SPEEDUP_COL_MODIFIER}")][0:1]

    # Filter data to just be for the largest sim.
    max_speedup_filter_col_value = big_df[config.SPEEDUP_FILTER_COLUMN].max()
    df = big_df.query(f"{config.SPEEDUP_FILTER_COLUMN} == {max_speedup_filter_col_value}")

    # Do some plotting stuff.
    sns.set_context(config.SNS_CONTEXT, rc={"lines.linewidth": 2.5})  
    sns.set_style(config.SNS_STYLE)
    huecount = len(df[config.HUE].unique()) if config.HUE is not None else 1 
    palette = sns.color_palette(config.SNS_PALETTE, huecount)
    sns.set_palette(palette)

    fig, ax = plt.subplots(constrained_layout=True)
    fig.set_size_inches(config.FIGSIZE_INCHES[0], config.FIGSIZE_INCHES[1])

    for speedup_y_col in speedup_cols:
        speedup_base = speedup_y_col.replace(f"{config.SPEEDUP_Y}{SPEEDUP_COL_MODIFIER}", "")
        speedup_base_label = speedup_base.replace("_", " ")
        # Build the output path for this image. Error if it cannot be created.
        OUTPUT_FILENAME = f"{config.SPEEDUP_OUTPUT_FILE_PREFIX}-{speedup_base}.png"
        OUTPUT_FILEPATH = config.get_abs_OUTPUT_DIR() / OUTPUT_FILENAME
        if config.get_abs_OUTPUT_DIR().is_file():
            raise Exception(f"Unable to create output file {OUTPUT_FILEPATH}, directory {config.get_abs_OUTPUT_DIR()} is an existing file")
        if OUTPUT_FILEPATH.is_dir():
            raise Exception(f"Unable to create output file {OUTPUT_FILEPATH}, it is an existing directory")
        elif OUTPUT_FILEPATH.is_file() and not force:
            raise Exception(f"Unable to create output file {OUTPUT_FILEPATH}, it is an existing file. Please specify -f,--force to enable over-writing of existing output files")

        # Apply a query
        g = sns.barplot(
            data=df, 
            x=config.SPEEDUP_X,
            y=speedup_y_col, 
            hue=config.HUE, 
            ax=ax,
            palette=palette,
        )

        # Axis settings
        if config.SPEEDUP_XLABEL:
            ax.set(xlabel=config.SPEEDUP_XLABEL)
        if config.SPEEDUP_YLABEL:
            ax.set(ylabel=config.SPEEDUP_YLABEL)
        if config.SPEEDUP_YLIM_TOP is not None:
            ax.set_ylim(top=config.SPEEDUP_YLIM_TOP)
        # Compute the figure title
        plt.title(f"{config.SPEEDUP_TITLE} for {config.SPEEDUP_FILTER_COLUMN} == {max_speedup_filter_col_value}")
        # compute the legend label
        legend_title = f"{config.HUELABEL} x {config.STYLELABEL}" if config.HUELABEL != config.STYLELABEL else f"{config.HUELABEL}"
        # If using an external legend, do external placement. This is experimental.
        if config.EXTERNAL_LEGEND:
            # Set legend placement if not internal.
            loc = "upper left"
            # @todo - y offset should be LEGEND_BORDER_PAD transformed from font units to bbox.
            bbox_to_anchor = (1, 1 - 0.0)
            handles, labels = ax.get_legend_handles_labels()
            # add an invisble patch with the appropriate label, like how seaborn does if multiple values are provided.
            handles.insert(0, mpatches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False, label=hs_label))
            labels.insert(0, legend_title)
            legend = ax.legend(handles=handles, labels=labels, loc=loc, bbox_to_anchor=bbox_to_anchor, borderaxespad=config.LEGEND_BORDER_PAD)
            plt.setp(legend.texts, family=MONOSPACE_FONT_FAMILY)
        else:
            if ax.get_legend() is not None:
                legend = ax.get_legend()
                legend.set_title(legend_title)
                plt.setp(legend.texts, family=MONOSPACE_FONT_FAMILY)

        # Save teh figure to disk, creating the parent dir if needed.
        OUTPUT_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_FILEPATH, dpi=config.DPI, bbox_inches='tight')

def plots(config, big_df, force):
    """ Plot raw performance data to disk
    """
    for SIMULATOR_LABEL, SIMULATOR in config.SIMULATORS.items():
        # Build the output path for this image. Error if it cannot be created.
        OUTPUT_FILENAME = f"{config.OUTPUT_FILE_PREFIX}-{SIMULATOR}.png"
        OUTPUT_FILEPATH = config.get_abs_OUTPUT_DIR() / OUTPUT_FILENAME
        if config.get_abs_OUTPUT_DIR().is_file():
            raise Exception(f"Unable to create output file {OUTPUT_FILEPATH}, directory {config.get_abs_OUTPUT_DIR()} is an existing file")
        if OUTPUT_FILEPATH.is_dir():
            raise Exception(f"Unable to create output file {OUTPUT_FILEPATH}, it is an existing directory")
        elif OUTPUT_FILEPATH.is_file() and not force:
            raise Exception(f"Unable to create output file {OUTPUT_FILEPATH}, it is an existing file. Please specify -f,--force to enable over-writing of existing output files")

        # Apply a query
        query = f"model == '{SIMULATOR}'"
        df = big_df.query(query)

        # Do some plotting stuff.
        sns.set_context(config.SNS_CONTEXT, rc={"lines.linewidth": 2.5})  
        sns.set_style(config.SNS_STYLE)
        huecount = len(df[config.HUE].unique()) if config.HUE is not None else 1 
        palette = sns.color_palette(config.SNS_PALETTE, huecount)
        sns.set_palette(palette)

        fig, ax = plt.subplots(constrained_layout=True)
        fig.set_size_inches(config.FIGSIZE_INCHES[0], config.FIGSIZE_INCHES[1])
        # filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
        g = sns.lineplot(
            data=df, 
            x=config.X, 
            y=config.Y, 
            hue=config.HUE, 
            style=config.STYLE, 
            markers=True,
            dashes = True,
            ax=ax,
            # size=6,
            legend="full",
            palette=palette,
        )

        # Axis settings
        if config.XLABEL:
            ax.set(xlabel=config.XLABEL)
        if config.YLABEL:
            ax.set(ylabel=config.YLABEL)
        ax.ticklabel_format(useOffset=False, style='plain')
        if config.YLIM_BOTTOM is not None:
            ax.set_ylim(bottom=config.YLIM_BOTTOM)
        # Compute the figure title
        plt.title(f"{config.TITLE_PREFIX} {SIMULATOR_LABEL}")
        # compute the legend label
        legend_title = f"{config.HUELABEL} x {config.STYLELABEL}" if config.HUELABEL != config.STYLELABEL else f"{config.HUELABEL}"
        # If using an external legend, do external placement. This is experimental.
        if config.EXTERNAL_LEGEND:
            # Set legend placement if not internal.
            loc = "upper left"
            # @todo - y offset should be LEGEND_BORDER_PAD transformed from font units to bbox.
            bbox_to_anchor = (1, 1 - 0.0)
            handles, labels = ax.get_legend_handles_labels()
            # add an invisble patch with the appropriate label, like how seaborn does if multiple values are provided.
            handles.insert(0, mpatches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False, label=hs_label))
            labels.insert(0, legend_title)
            ax.legend(handles=handles, labels=labels, loc=loc, bbox_to_anchor=bbox_to_anchor, borderaxespad=config.LEGEND_BORDER_PAD)
            plt.setp(legend.texts, family=MONOSPACE_FONT_FAMILY)
        else:
            if ax.get_legend() is not None:
                legend = ax.get_legend()
                legend.set_title(legend_title)
                plt.setp(legend.texts, family=MONOSPACE_FONT_FAMILY)
        # Save teh figure to disk, creating the parent dir if needed.
        OUTPUT_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_FILEPATH, dpi=config.DPI, bbox_inches='tight')

if __name__ == "__main__":
    main()