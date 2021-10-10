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
import glob2
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

import scipy.stats

import argparse
import logging
import os
import sys


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
    LOGGER.info("Total               : " + str(nr_total))
    LOGGER.info("No ground truth data: " + str(nr_no_ground_truth))
    LOGGER.info("Not classified      : " + str(nr_couldnt_classify))
    LOGGER.info("Classified          : " + str(nr_classified))
    LOGGER.info("False Positives     : " + str(nr_false_positives))
    LOGGER.info("True Negatives      : " + str(nr_true_negatives))
    return axes

def draw_accuracy(dataframe, ax_line, ax_scatter):
    dataframe_ground_truth    = dataframe[dataframe['ground truth'] != 'NOT CLASSIFIED']
    dataframe_negatives     = dataframe[dataframe['ground truth'] == 'NA']

    dataframe_classified = dataframe_ground_truth[dataframe_ground_truth['ground truth'] != 'NA']
    dataframe_classified = dataframe_classified[dataframe_classified['Heartrate (BPM)'] != 'NA']

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
    y = [round(i/nr_total, 2) for i in y]
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

    plt.savefig(os.path.join(outdir, filename + '.png'))
    #plt.show()
    fig.clf()

def main(indir, outdir, path_ground_truths):
    LOGGER.info("######## Quality Control: Statistical Analysis ########")
    try:
        outdir = os.path.join(outdir, "statistics/")
        os.makedirs(outdir, exist_ok=True)

        # Load in ground truth cvs into dataframe
        ground_truths = pd.read_csv(path_ground_truths, keep_default_na=False)

        # Load result.csv files into dataframe
        algorithm_results = []

        subdirs = {os.path.join(p, '') for p in glob2.glob(indir + '/*/')}

        for path in subdirs:
            if os.path.isdir(path):
                results_files = [f for f in glob2.glob(path + '/*.csv')]
                if not results_files:
                    continue
                elif len(results_files) > 1:
                    print("Error: More than one results file found")
                    sys.exit()

                dataset_name = os.path.basename(os.path.normpath(path))
                dataset_name = dataset_name[:-15]

                LOGGER.info("Found results file for dataset " + dataset_name)

                # add DATASET column for merge later
                results = pd.read_csv(results_files[0], keep_default_na=False)
                results.insert(0,'DATASET','')
                results["DATASET"] = dataset_name

                algorithm_results.append(results)

        algorithm_results = pd.concat(algorithm_results, axis=0, ignore_index=True)

        # Merge the results
        output_df = pd.merge(algorithm_results, ground_truths, how="left")

        # remove empty entries
        output_df = output_df[output_df['ground truth'] != '']    # removes empty cells, keeps 'NA' filled ones.
        csv = output_df.to_csv(os.path.join(outdir, "merged.csv"), index=False)

        #split 35C off, as unreliable for 13fps
        C21_28 = output_df[output_df['DATASET'].str.contains("21C|28C") ]
        C35 = output_df[output_df['DATASET'].str.contains("35C")]

        # create and store plots and csv files on disk
        LOGGER.info("")
        if not C21_28.empty:
            LOGGER.info("### 21C and 28C data ###")
            csv = C21_28.to_csv(os.path.join(outdir, "21c_28c.csv"), index=False)
            create_plots(C21_28, outdir, 'C21_28')

        if not C35.empty:
            LOGGER.info("####### 35C data #######")
            csv = C35.to_csv(os.path.join(outdir, "35c.csv"), index=False)
            create_plots(C35, outdir, 'C35')


        LOGGER.info("Done.")
    except Exception as e:
        LOGGER.exception("During creation of statistics on test set results")
    return

# For cluster mode, runable as toplevel script
if __name__ == '__main__':
    import setup
    # Parse input arguments.
    parser = argparse.ArgumentParser(description='Combine ground truths with analysis output and calculate accuracy statistics')
    parser.add_argument('-o', '--outdir', action="store", dest='outdir', help='Where store the assessment report data',         default=False, required = True)
    parser.add_argument('-i', '--indir', action="store", dest='indir', help='Path to analysed folders',                         default=False, required = True)
    parser.add_argument('-g', '--ground_truth', action="store", dest='ground_truth_csv', help="Path to the ground truth csv",   default=False, required=True)

    args = parser.parse_args()
    setup.config_logger(args.outdir, ("assessment" + ".log"))

    main(args.indir, args.outdir, args.ground_truth_csv)

# This is a workaround to ensure relative import works when using this file as both a  toplevel script and module.
else:
    from . import setup