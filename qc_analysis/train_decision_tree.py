#!/usr/bin/env python
# coding: utf-8
############################################################################################################
# Authors:
#   Anirudh Bhashyam, Uni Heidelberg, anirudh.bhashyam@stud.uni-heidelberg.de   (Current Maintainer)
# Date: 04/2022
# License: GNU GENERAL PUBLIC LICENSE Version 3
###
# Algorithms for:
#   Decision tree evaluation by medaka_bpm.
###
############################################################################################################
import argparse
from pathlib import Path
import decision_tree.src.analysis as analyse

def process_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", help = "Results from medaka_bpm", type = str, required = True)
    parser.add_argument("-o", "--out_dir",  help = "Directory to write analysis results", type = str, required = True)

    args = parser.parse_args()
    args.input_file = Path(args.input_file)
    args.out_dir = Path(args.out_dir)
    return args

def main():
    args = process_args()
    raw_data = analyse.pd.read_csv(args.input_file)
    data, scale = analyse.process_data(raw_data, 20)
    classifier, classifier_results = analyse.decision_tree(data)
    limits = analyse.get_thresholds(raw_data, analyse.QC_FEATURES, classifier)
    analyse.write_results(raw_data, data, classifier, classifier_results, limits, args.out_dir)
    
if __name__ == "__main__":
    main()