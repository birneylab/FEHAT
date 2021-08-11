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
import scipy.stats

import argparse
import os
import sys
import time

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

    return ax_line, ax_scatter

def create_plots(dataframe, outdir, filename):
    # Make Main Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,5))

    # Pie Chart - Classification rate
    ax1 = draw_classification_rate(dataframe, ax1)

    # Line Plot - Accuracy within Error 
    ax2, ax3 = draw_accuracy(dataframe, ax2, ax3)

    plt.savefig(os.path.join(outdir, filename + '.png'))
    plt.show()
    fig.clf()

parser = argparse.ArgumentParser(description='Read in medaka heart video frames')
parser.add_argument('-o', '--outdir', action="store", dest='outdir', help='Where store the assessment report data',         default=False, required = True)
parser.add_argument('-i', '--indir', action="store", dest='indir', help='Path to analysed folders',                         default=False, required = True)
parser.add_argument('-g', '--ground_truth', action="store", dest='ground_truth_csv', help="Path to the ground truth csv",   default=False, required=True)

args = parser.parse_args()

# output directory
timestamp = time.strftime("%Y%m%d-%H%M%S")
os.makedirs(os.path.join(args.outdir, timestamp), exist_ok=True)

# Load in ground truth cvs into dataframe
ground_truths = pd.read_csv(args.ground_truth_csv, keep_default_na=False)

# Load result.csv files into dataframe
algorithm_results = []

subdirs = {os.path.join(p, '') for p in glob2.glob(args.indir + '/*/')}

for path in subdirs:
    if os.path.isdir(path):
        results_files = [f for f in glob2.glob(path + '/*.csv')]
        if not results_files:
            continue
        elif len(results_files) > 1:
            print("Error: More than one results file found")
            sys.exit()

        dataset_name = os.path.basename(os.path.normpath(path))
        datasetname = dataset_name[4:] # Remove "out_" prefix from folder names

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
csv = output_df.to_csv(os.path.join(args.outdir, "merged.csv"), index=False)

#split 35C off, as unreliable for 13fps
C21_28 = output_df[output_df['DATASET'].str.contains("21C|28C") ]
C35 = output_df[output_df['DATASET'].str.contains("35C")]
csv = C21_28.to_csv(os.path.join(args.outdir, "21c_28c.csv"), index=False)
csv = C35.to_csv(os.path.join(args.outdir, "35c.csv"), index=False)

# create and store plots on disk
create_plots(C21_28, args.outdir, 'C21_28')
create_plots(C35, args.outdir, 'C35')

# %%
