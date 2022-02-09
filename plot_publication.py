#! /usr/bin/env python3
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from matplotlib import patches as mpatches
import argparse
import pathlib


# Default DPI
DEFAULT_DPI = 300

# Default directory for visualisation images
DEFAULT_INPUT_DIR= "." #"./sample/data/v100-470.82.01/alpha.2-v100-11.0-beltsoff"
DEFAULT_OUTPUT_DIR = "." #"./sample/figures/v100-470.82.01/alpha.2-v100-11.0-beltsoff"

# Drift csv filename from simulation output
VARIABLE_POP_CSV_FILENAME = "fixed-density_perSimulationCSV.csv"
VARIABLE_RADIUS_CSV_FILENAME = "comm-radius_perSimulationCSV.csv"
VARIABLE_DENSITY_CSV_FILENAME = "variable-density_perSimulationCSV.csv"
VARIABLE_SORT_PERIOD_CSV_FILENAME = "sort-period_perSimulationCSV.csv"

EXPECTED_INPUT_FILES = [VARIABLE_POP_CSV_FILENAME, VARIABLE_RADIUS_CSV_FILENAME, VARIABLE_SORT_PERIOD_CSV_FILENAME]

MODEL_NAME_MAP = {'circles_bruteforce': "Brute Force", 
                  'circles_spatial3D': "Spatial", 
                  'circles_bruteforce_rtc': "Brute Force (RTC)",
                  'circles_spatial3D_rtc': "Spatial (RTC)"}

def cli():
    parser = argparse.ArgumentParser(description="Python script to generate figure from csv files")

    parser.add_argument(
        '-o', 
        '--output-dir', 
        type=str, 
        help='directory to output figures into.',
        default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        '--dpi', 
        type=int, 
        help='DPI for output file',
        default=DEFAULT_DPI
    )

    parser.add_argument(
        '-i',
        '--input-dir', 
        type=str, 
        help='Input directory, containing the csv files',
        default=DEFAULT_INPUT_DIR
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

    # Ensure that the input directory exists, and that all required input is present.
    if args.input_dir is not None:
        input_dir = pathlib.Path(args.input_dir) 
        if input_dir.is_dir():
            missing_files = []
            for required_file in EXPECTED_INPUT_FILES:
                required_file_path = input_dir / required_file
                if not required_file_path.is_file():
                    missing_files.append(required_file_path)
                    valid = False
            if len(missing_files) > 0:
                print(f"Error: {input_dir} does not contain required files:")
                for missing_file in missing_files:
                    print(f"  {missing_file}")
        else:
            print(f"Error: Invalid input_dir provided {args.input_dir}")
            valid = False

    return valid


def main():

    # Validate cli
    args = cli()
    valid_args = validate_args(args)
    if not valid_args:
        return False
            
    # Set figure theme
    sns.set_theme(style='white')
    
    # setup sub plot using mosaic layout
    gs_kw = dict(width_ratios=[1, 1], height_ratios=[1, 1, 1])
    f, ax = plt.subplot_mosaic([['p1', 'p2'],
                                ['p3', 'p4'],
                                ['p5' , '.' ],
                                ],
                                  gridspec_kw=gs_kw, figsize=(7.5, 7.5),
                                  constrained_layout=True)
    input_dir = pathlib.Path(args.input_dir)
    
    # common palette
    custom_palette = {"circles_bruteforce": "g", "circles_bruteforce_rtc": "r", "circles_spatial3D": "b", "circles_spatial3D_rtc": "k"}

    
    # Load per simulation step data into data frame (strip any white space)
    pop_df = pd.read_csv(input_dir/VARIABLE_POP_CSV_FILENAME, sep=',', quotechar='"')
    pop_df.columns = pop_df.columns.str.strip()

    # Plot popsize brute force
    scale_df_bf = pop_df.query("model == 'circles_bruteforce_rtc' or model == 'circles_bruteforce'")
    plt_df_bf = sns.lineplot(x='agent_count', y='s_step_mean', hue='model', style='model', data=scale_df_bf, ax=ax['p1'], palette=custom_palette, ci="sd")
    # plt_df_bf.ticklabel_format(style='plain', axis='x') # no scientific notation
    plt_df_bf.set(xlabel='N', ylabel='Step time (s)')
    ax['p1'].set_title(label='A', loc='left', fontweight="bold")
    ax['p1'].legend().set_visible(False)
    
    # Plot spatial
    scale_df_spatial = pop_df.query("model == 'circles_spatial3D_rtc' or model == 'circles_spatial3D'")
    plt_df_s = sns.lineplot(x='agent_count', y='s_step_mean', hue='model', style='model', data=scale_df_spatial, ax=ax['p2'], palette=custom_palette, ci="sd")
    # plt_df_s.ticklabel_format(style='plain', axis='x') # no scientific notation
    plt_df_s.set(xlabel='N', ylabel='')
    ax['p2'].set_title(label='B', loc='left', fontweight="bold")
    ax['p2'].legend().set_visible(False)
    
    
    # Load per simulation step data into data frame (strip any white space)
    rad_df = pd.read_csv(input_dir/VARIABLE_RADIUS_CSV_FILENAME, sep=',', quotechar='"')
    rad_df.columns = rad_df.columns.str.strip()
    rad_df["comm_radius_as_percentage_env_width"] = rad_df["comm_radius"] / rad_df["env_width"] *100

    # Plot radius comparison
    rad_df = rad_df.query("model == 'circles_spatial3D' or model == 'circles_bruteforce'")
    plt_df_rad = sns.lineplot(x='comm_radius_as_percentage_env_width', y='s_step_mean', hue='model', data=rad_df, ax=ax['p3'], palette=custom_palette, ci="sd")
    plt_df_rad.ticklabel_format(style='plain', axis='x') # no scientific notation
    plt_df_rad.set(xlabel='r as % of W', ylabel='Step time (s)')
    ax['p3'].set_title(label='C', loc='left', fontweight="bold")
    ax['p3'].legend().set_visible(False)
    
    # plot data volume
    messages_df = rad_df.query("model == 'circles_spatial3D'")
    plt_df_rad = sns.lineplot(x='comm_radius_as_percentage_env_width', y='mean_message_count', hue='model', data=messages_df, ax=ax['p4'], palette=custom_palette, ci="sd")
    plt_df_rad.set(xlabel='r as % of W', ylabel='Messages / step')
    ax['p4'].set_title(label='D', loc='left', fontweight="bold")
    ax['p4'].legend().set_visible(False)
    
   
    # Load per simulation step data into data frame (strip any white space)
    sort_df = pd.read_csv(input_dir/VARIABLE_SORT_PERIOD_CSV_FILENAME, sep=',', quotechar='"')
    sort_df.columns = sort_df.columns.str.strip()

    # Plot sort period comparison
    plot_for_com_radius = 8
    sort_df = sort_df.query("comm_radius == " + str(plot_for_com_radius) + " and (model == 'circles_spatial3D' or model == 'circles_spatial3D_rtc')")
    plt_df_sort = sns.lineplot(x='sort_period', y='s_step_mean', hue='model', data=sort_df, ax=ax['p5'], palette=custom_palette, ci="sd")
    plt_df_sort.ticklabel_format(style='plain', axis='x') # no scientific notation
    plt_df_sort.set(xlabel='Sort period (steps)', ylabel='Step time (s)')
    ax['p5'].set_title(label='E', loc='left', fontweight="bold")
    ax['p5'].legend().set_visible(False)
    

    # Legend
    lines_labels = [ax.get_legend_handles_labels() for ax in f.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    labels = [MODEL_NAME_MAP[l] for l in labels] # Rename labels to provide readable legends
    unique = {k:v for k, v in zip(labels, lines)} 
    f.legend(unique.values(), unique.keys(), loc='lower right')

    
        
    # Save to image
    #f.tight_layout()
    output_dir = pathlib.Path(args.output_dir) 
    f.savefig(output_dir/"paper_figure.png", dpi=args.dpi) 
    f.savefig(output_dir/"paper_figure.pdf", format='pdf', dpi=args.dpi)
    
    #plt.show()


# Run the main method if this was not included as a module
if __name__ == "__main__":
    main()
