#!/usr/bin/env python3
############################################################################################################
### Author: Sebastian Stricker, Uni Heidelberg, sebastian.stricker@stud.uni-heidelberg.de
### Date: 08/2021
### License: Contact author
###
### Creates analysis data to assess algorithm accuracy, performance and outputs further measurements for refinement.
### 
############################################################################################################
# %% setup
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

import scipy.stats

import argparse
import logging
from pathlib import Path
import sys

# Imports from base dir of repository
parent_dir = Path(__file__).resolve().parents[1]
sys.path.append(parent_dir)

import src.setup as setup


LOGGER = logging.getLogger(__name__)

def draw_classification_rate(dataframe, axes):
    dataframe_ground_truth    = dataframe[dataframe['ground truth'] != 'NOT CLASSIFIED']
    dataframe_negatives     = dataframe[dataframe['ground truth'] == 'NA']

    # Not classified data
    nr_total = dataframe.shape[0]
    nr_no_ground_truth  = nr_total - (dataframe_ground_truth.shape[0])

    # Classified/not classified
    dataframe_classified = dataframe_ground_truth[dataframe_ground_truth['ground truth'] != 'NA']
    nr_couldnt_classify = dataframe_classified[dataframe_classified['Heartrate (BPM)'] == 'NA'].shape[0]
    nr_classified       = dataframe_classified.shape[0] - nr_couldnt_classify

    # False postive rate
    nr_false_positives  = dataframe_negatives[dataframe_negatives['Heartrate (BPM)'] != 'NA'].shape[0]
    nr_true_negatives   = dataframe_negatives.shape[0] - nr_false_positives
    
    # Draw Plot
    labels = 'No ground truth available', 'Classified', 'Couldn’t classify', 'False Positives', 'True negatives'
    sizes = [nr_no_ground_truth, nr_classified, nr_couldnt_classify, nr_false_positives, nr_true_negatives]
    colors = ['#000000', '#3973E6' , '#CF0000', '#FFDB70', '#E6E2E1']
    explode = (0.0, 0.0, 0.3, 0.0, 0.1)

    patches, _, pcts = axes.pie(sizes, explode=explode, colors=colors, startangle=90, autopct='%1.1f%%', pctdistance=1.1)

    axes.set_title('Classification Rate', fontsize=18)
    axes.legend(patches, labels, loc="best")
    axes.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    LOGGER.info("--------- Classification Rate ---------")
    LOGGER.info("Total               : " + str(round((nr_total/nr_total) * 100, 2)).rjust(6)             + "% - "+ str(nr_total))
    LOGGER.info("No ground truth data: " + str(round((nr_no_ground_truth/nr_total) * 100, 2)).rjust(6)   + "% - "+ str(nr_no_ground_truth))
    LOGGER.info("Classified          : " + str(round((nr_classified/nr_total) * 100, 2)).rjust(6)        + "% - "+ str(nr_classified))
    LOGGER.info("Not classified      : " + str(round((nr_couldnt_classify/nr_total) * 100, 2)).rjust(6)  + "% - "+ str(nr_couldnt_classify))
    LOGGER.info("False Positives     : " + str(round((nr_false_positives/nr_total) * 100, 2)).rjust(6)   + "% - "+ str(nr_false_positives))
    LOGGER.info("True Negatives      : " + str(round((nr_true_negatives/nr_total) * 100, 2)).rjust(6)    + "% - "+ str(nr_true_negatives))
    return axes

def draw_accuracy(dataframe, ax_line, ax_scatter):
    dataframe_ground_truth    = dataframe[dataframe['ground truth'] != 'NOT CLASSIFIED']
    dataframe_negatives     = dataframe[dataframe['ground truth'] == 'NA']

    dataframe_classified = dataframe_ground_truth[dataframe_ground_truth['ground truth'] != 'NA']
    dataframe_classified = dataframe_classified[dataframe_classified['Heartrate (BPM)'] != 'NA']
    
    # r2 value is not calculated if any nan is present. 
    # Origin unknown though, above commands should filter for those
    # Very weird...
    dataframe_classified = dataframe_classified.dropna(subset=['ground truth', 'Heartrate (BPM)'])

    # False & True positives
    nr_false_positives  = dataframe_negatives[dataframe_negatives['Heartrate (BPM)'] != 'NA'].shape[0]
    nr_true_negatives   = dataframe_negatives.shape[0] - nr_false_positives

    nr_total = dataframe_classified.shape[0] + nr_false_positives + nr_true_negatives

    # convert to numbers
    dataframe_classified['ground truth'] = pd.to_numeric(dataframe_classified['ground truth'] , downcast="float")
    dataframe_classified['Heartrate (BPM)'] = pd.to_numeric(dataframe_classified['Heartrate (BPM)'] , downcast="float")

    ####### LINE PLOT
    differences = (dataframe_classified['ground truth'] - dataframe_classified['Heartrate (BPM)']).abs()

    y = [nr_true_negatives + differences[differences < i].count() for i in range(1,101)]
    y = [round(i/nr_total, 3) for i in y]
    x = list(range(1, 101))

    ax_line.plot(x,y)

    ax_line.set(title="Accuracy", ylabel="Accuracy within error (%)", xlabel="Error (BPM)")
    ax_line.set(ylim=(0, 1), xlim=(0,100), yticks=[x/10.0 for x in range(0, 11)], xticks=list(range(0, 110, 10)))
    ax_line.grid()

    LOGGER.info("-------------- Accuracy ---------------")
    intervals = [2, 5, 10, 20, 30, 50, 75, 100]
    for x in intervals:
        LOGGER.info("Up to " + str(x).ljust(3-len(str(x))) + ": " + str(y[x-1]))

    LOGGER.info("")
    
    ###### SCATTER PLOT
    x = dataframe_classified['Heartrate (BPM)'].tolist()
    y = dataframe_classified['ground truth'].tolist()

    max_value = max(max(x), max(y))

    regression_result = scipy.stats.linregress(x, y)
    r2 = round(regression_result.rvalue**2, 2)
    line = [regression_result.intercept + regression_result.slope*x for x in x]
    
    ax_scatter.scatter(x,y, s=1)
    ax_scatter.plot(x, line, 'r')
    ax_scatter.set_title("Ground truth vs Algorithm. \n Linear regression. R2-value: " + str(r2))
    ax_scatter.set(xlabel="Algorithm (BPM)", ylabel="Ground truth (BPM)", xlim=(0, max_value), ylim=(0, max_value) )
    ax_scatter.set_aspect('equal')
    ax_scatter.grid()

    LOGGER.info("R2-Value: " + str(r2) + "\n")

    return ax_line, ax_scatter

def create_plots(dataframe, outdir, filename):
    # Make Main Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,5))

    # Pie Chart - Classification rate
    ax1 = draw_classification_rate(dataframe, ax1)

    # Line Plot - Accuracy within Error 
    ax2, ax3 = draw_accuracy(dataframe, ax2, ax3)

    plt.savefig(outdir / f"{filename}.svg")
    #plt.show()
    fig.clf()

def main(indir, outdir, path_ground_truths):
    LOGGER.info("######## Quality Control: Statistical Analysis ########")
    try:
        outdir = outdir / "statistics/"
        outdir.mkdir(parents=True, exist_ok=True)

        # Load in ground truth cvs into dataframe
        ground_truths = pd.read_csv(path_ground_truths, keep_default_na=False)

        # Load result.csv files into dataframe
        algorithm_results = []
        groups = set()

        subdirs = indir.glob('*/')

        for path in subdirs:
            if not path.is_dir():
                continue 
            
            results_files = list(path.glob('*.csv'))
            if not results_files:
                continue
            elif len(results_files) > 1:
                print("Error: More than one results file found")
                sys.exit()

            dataset_name = path.name
            
            # Cut away suffix
            idx = dataset_name.index('_medaka_bpm_out')
            dataset_name = dataset_name[:idx]

            LOGGER.info("Found results file for dataset " + dataset_name)

            # add DATASET column for merge later
            results = pd.read_csv(results_files[0], keep_default_na=False)
            results.insert(0,'DATASET','')
            results["DATASET"] = dataset_name

            algorithm_results.append(results)

            # Extract group names of valid datasets
            # Performance over datasets can vary greatly.
            # Analysing over different groups gives info on best and worst case performance.

            # DATASETGROUP1_13FPS_170814... ---> DATASETGROUP1
            # DATASETGROUP2_13FPS_170814... ---> DATASETGROUP2
            dataset_group = dataset_name.split('_')[0]
            groups.add(dataset_group)

        # list of dataframes to single unified dataframe
        algorithm_results = pd.concat(algorithm_results, axis=0, ignore_index=True, sort=False)

        # Merge the results
        output_df = pd.merge(algorithm_results, ground_truths, how="inner")

        # Output combined results
        output_df.to_csv(outdir / "merged.csv", index=False)

        # Make statistics over each group seperately
        LOGGER.info("")
        for group_name in groups:
            group_df = output_df[output_df['DATASET'].str.startswith(group_name + '_')]

            LOGGER.info(f"### {group_name} ###")
            group_df.to_csv(outdir / f"{group_name}.csv", index=False)
            create_plots(group_df, outdir, group_name)

        LOGGER.info("Done.")
    except Exception as e:
        LOGGER.exception("During creation of statistics on test set results")
    return

# For cluster mode, runable as toplevel script
if __name__ == '__main__':

    # Parse input arguments.
    parser = argparse.ArgumentParser(description='Combine ground truths with analysis output and calculate accuracy statistics')
    parser.add_argument('-o', '--outdir', action="store", dest='outdir', help='Where store the assessment report data',         default=False, required = True)
    parser.add_argument('-i', '--indir', action="store", dest='indir', help='Path to analysed folders',                         default=False, required = True)
    parser.add_argument('-g', '--ground_truth', action="store", dest='ground_truth_csv', help="Path to the ground truth csv",   default=False, required=True)

    args = parser.parse_args()
    args.indir = Path(args.indir)
    args.outdir = Path(args.outdir)
    args.ground_truth_csv = Path(args.ground_truth_csv)
    
    setup.config_logger(args.outdir, ("assessment" + ".log"))

    main(args.indir, args.outdir, args.ground_truth_csv)
