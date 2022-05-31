#!/usr/bin/env python
# coding: utf-8
############################################################################################################
# Authors:
#   Anirudh Bhashyam, Uni Heidelberg, anirudh.bhashyam@stud.uni-heidelberg.de   (Current Maintainer)
# Date: 04/2022
# License: Contact authors
###
# Algorithms for:
#   comparing predicted heart regions and annotated heart regions.
###
############################################################################################################
import os
import sys
import argparse

import annotation.src.annotation_compare as annot

def process_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_frames", type = str, required = True, help = "Path to the input video.")
    parser.add_argument("-af", "--annotation_file", help = "Path to the annotation file", type = str, required = True)
    parser.add_argument("-o", "--out_dir",  help = "Directory to write analysis results", type = str, required = True)
    return parser.parse_args()

def main():
    args = process_args()
    frames = args.input_frames
    annotation_file = args.annotation_file
    out = args.out_dir
    roi_mask = annot.get_roi(frames_dir = frames, annotation_file = annotation_file, outdir = out)
    roi_positions = annot.extract_roi_positions(roi_mask)
    annotation = annot.AnnotationJSON(annotation_file).read()
    annot.write_results(annotation, roi_positions, roi_mask, annotation_file, out)
    
if __name__ == "__main__":
    main()

