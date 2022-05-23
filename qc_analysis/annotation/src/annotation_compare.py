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
import warnings
import glob2
import logging

from typing import List, Callable, Iterable
from abc import ABC, abstractmethod

import math

import matplotlib 
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import json
import xml.etree.ElementTree as et

import statistics
from statistics import mean

import numpy as np

sys.path.append(os.path.join(os.path.abspath("".join([__file__, 4 * "/.."])), "src"))

import io_operations
import segment_heart
import annotation_metrics

matplotlib.use('Agg')

PLOT_SAVE_DIR = os.path.relpath("figs")
DATA_SAVE_DIR = os.path.relpath("metrics")

# Parallelisation
# warnings.filterwarnings('ignore')
# warnings.filterwarnings("ignore", category = SyntaxWarning)

class AnnotationFormats:
    XML = "xml"
    JSON = "json"

class Annotation(ABC):
    @abstractmethod
    def read(self):
        pass
    
class AnnotationJSON(Annotation):
    def __init__(self, annotation_file: str):
        ext = os.path.basename(annotation_file).split(".")[-1]
        if ext != AnnotationFormats.JSON:
            raise ValueError("Annotation file must be in JSON format.")
        
        self.annotation_file = annotation_file
        
    def read(self) -> List[float]:
        # Parse json annotation data.
        with open(self.annotation_file) as f:
            data = json.load(f)
            
        self.annotations = [point for point in data["shapes"][0]["points"]]
        return self.annotations
    
class AnnotationXML(Annotation):
    def __init__(self, annotation_file: str):
        ext = os.path.basename(annotation_file).split(".")[-1]
        if ext != AnnotationFormats.XML:
            raise ValueError("Annotation file must be in XML format.")
        self.annotation_file = annotation_file
    
    def read(self) -> List[float]:
        # Parse xml annotation data.
        tree = et.parse(".".join([self.annotation_file, AnnotationFormats.XML]))
        root = tree.getroot()

        self.annotations = list()

        # Annotations are in bndbox element.
        for neighbour in root.iter("bndbox"):
            x_min = int(neighbour.find("xmin").text)
            y_min = int(neighbour.find("ymin").text)
            x_max = int(neighbour.find("xmax").text)
            y_max = int(neighbour.find("ymax").text)
            self.annotations.append([x_min, y_min, x_max, y_max])
            
        return self.annotations

        
class InvalidROI(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)
        
    def __str__(self):
        return f"{self.message}"
    
# Goes through all channels and loops and yields well data fields and paths to frames sorted by frame index.
def well_video_generator(indir, channels, loops):
    # -A001--PO01--LO001--CO1--SL001--PX32500--PW0070--IN0020--TM280--X014600--Y011401--Z214683--T0000000000--WE00001.tif

    # Grab all paths to all frames
    all_frames = glob2.glob(os.path.join(indir, "*.tif")) + glob2.glob(os.path.join(indir, "*.tiff"))

    # Channel
    for channel in channels:
        # LOGGER.info("### CHANNEL " + channel + " ###")
        channel_videos = [frame for frame in all_frames if channel in frame]
        # Loop
        for loop in loops:
            # LOGGER.info("### LOOP " + loop + " ###")
            loop_videos = [frame for frame in channel_videos if loop in frame]

            # Well
            for well_id in range(1, 97):
                well_id = ('WE00' + '{:03d}'.format(well_id))

                well_frames = [
                    frame for frame in loop_videos if well_id in frame]

                # Skipping costs (close to) no time and the logic is simpler than extracting the exact set of wells per loop/channel.
                # Could be solved cleaner though.
                if not well_frames:
                    continue

                # Sort frames in correct order
                frame_indices = [io_operations.frameIdx(path) for path in well_frames]
                _, well_frames_sorted = (list(t) for t in zip(
                    *sorted(zip(frame_indices, well_frames))))

                metadata = {'well_id': well_id, 'loop': loop, 'channel': channel}
                yield well_frames_sorted, metadata  
    
def get_roi(frames_dir: str, annotation_file: str, outdir: str):
    # LOGGER.info(f"""
    #                 Comparing video annotations to hroi -
    #                 Channel: {str(video_metadata["channel"])}
    #                 Loop: {str(video_metadata["loop"])}
    #                 Well: {str(video_metadata["well_id"])}
    #             """)
    annotation = AnnotationJSON(annotation_file).read()
    
    # All paths to the frames.
    all_frames = glob2.glob(os.path.join(frames_dir, "*.tif")) + glob2.glob(os.path.join(frames_dir, "*.tiff"))
    
    # Video data. 
    tiffs = glob2.glob(os.path.join(frames_dir, "*SL001" + "*.tif")) + glob2.glob(os.path.join(frames_dir, "*SL001" + "*.tiff"))
    nr_of_videos = len(tiffs)
    
    # Extract different channels
    # using a set, gives only unique values
    channels = {"CO" + tiff.split("-CO")[-1][0] for tiff in tiffs}
    channels = sorted(list(channels))

    # Extract different Loops
    # using a set, gives only unique values
    loops = {"LO" + tiff.split("-LO")[-1][0:3] for tiff in tiffs}
    loops = sorted(list(loops))
    
    for well_frame_paths, video_metadata in well_video_generator(frames_dir, channels, loops):
        bpm = None
        qc_attributes = {}

        ################################################################################ Setup
        # Add well position to output directory path
        out_dir = os.path.join(outdir,
                               "annotation_analysis", 
                               video_metadata["channel"],
                               "-".join([video_metadata["loop"], video_metadata["well_id"]]))

        os.makedirs(out_dir, exist_ok = True)
        
        # Extract the artificial timestamps for the video. 
        video_metadata['timestamps'] = io_operations.extract_timestamps(
        well_frame_paths)
        
        # Make the video.
        video = io_operations.load_video(well_frame_paths)

        # Ensures np array not lists.
        video = np.asarray(video)
        timestamps = np.asarray(video_metadata['timestamps'])

        video, timestamps = segment_heart.sort_frames(video, video_metadata['timestamps'])
        # fps = segment_heart.determine_fps(timestamps, args['fps'])
        fps = 24

        timestamps = segment_heart.equally_spaced_timestamps(len(timestamps), fps)

        ################################# Normalize Frames
        # LOGGER.info("Normalizing frames")
        #save_video(video, fps, out_dir, "before_norm.mp4")
        
        # Normalize frames
        normed_video = segment_heart.normVideo(video)
        del video

        # LOGGER.info("Writing video")
        segment_heart.save_video(normed_video, fps, out_dir, "embryo.mp4")

        ################################ Detect HROI and write into figure. 
        # LOGGER.info("Detecting HROI")
        # Runs the region detection in 8 bit (No effect if video loaded in 8bit anyway)
        video8  = segment_heart.assert_8bit(normed_video)
        
        frame2frame_changes = segment_heart.absdiff_between_frames(video8)
        frame2frame_changes_thresh= segment_heart.threshold_changes(frame2frame_changes)

        # Detect movement and stop analysis early
        stop_frame, max_change = segment_heart.detect_movement(frame2frame_changes_thresh)
        qc_attributes["Stop frame"] = str(stop_frame)
        qc_attributes["Movement detection max"] = max_change

        # Break condition
        if stop_frame < 3 * fps:
            # LOGGER.info("Movement before 3 seconds. Stopping analysis")
            return None, fps, qc_attributes

        # Shorten videos
        normed_video                = normed_video[:stop_frame]
        video8                      = video8[:stop_frame]
        frame2frame_changes         = frame2frame_changes[:stop_frame]
        frame2frame_changes_thresh  = frame2frame_changes_thresh[:stop_frame]

        try:
            #hroi_mask, all_roi, total_changes, top_changing_pixels = HROI2(frame2frame_changes_thresh)
            hroi_mask, _, _ = segment_heart.HROI3(normed_video, frame2frame_changes_thresh, timestamps, fps)
            return hroi_mask

        except InvalidROI as e: 
            raise e("No region of interest generated.")
        
        # Make a heatmap of all_roi.
        # sns.heatmap(all_roi, cmap = "YlGnBu", vmin = 0, vmax = 1)
        # plt.savefig(os.path.join(out_dir, "roi_heatmap.png")) 
        
        # # Make a heatmap of hroi_mask.
        # sns.heatmap(hroi_mask, cmap = "YlGnBu", vmin = 0, vmax = 1)
        # plt.savefig(os.path.join(out_dir, "hroi_heatmap.png"))

        # # Make a heatmap of hroi_pixels.
        # sns.heatmap(hroi_pixels, cmap = "YlGnBu", vmin = 0, vmax = 1)
        # plt.savefig(os.path.join(out_dir, "hroi_pixels_heatmap.png"))
        # print(hroi_pixels.shape)

def extract_roi_positions(hroi_mask: np.ndarray) -> list:
    """
    Extracts the positions of the roi in the video.
    """
    positions = np.where(hroi_mask == 1)
    return list(zip(positions[1], positions[0]))

def draw(polygon: np.ndarray, 
         hroi_mask: np.ndarray,
         out_dir: str, 
         save_name: str = "abc", 
         save_q: bool = True,) -> None:
    """
    Draws the annotation and roi regions.
    """
    plt.figure(figsize = (10, 10))
    
    # Draw the annotation polygon.
    for i in range(len(polygon) - 1):
       plt.plot([polygon[i, 0], polygon[i + 1, 0]], [polygon[i, 1], polygon[i + 1, 1]], c = "c")

    plt.plot([polygon[0, 0], polygon[-1, 0]], [polygon[0, 1], polygon[-1, 1]], c = "c")
    
    
    # Draw the roi.
    plt.imshow(hroi_mask, cmap = "YlGnBu", alpha = 0.7)

    
    plt.xlim((min(polygon[:, 0]) - 20, max(polygon[:, 0]) + 20))
    plt.ylim((min(polygon[:, 1]) - 20, max(polygon[:, 1]) + 20))
    
    plt.gca().invert_yaxis()
    
    if save_q:
        plt.savefig(os.path.join(out_dir, ".".join([save_name, "png"])), 
                    dpi = 180,
                    bbox_inches = "tight")
        
    plt.show()
     
def compare(annotation: List[List], roi_positions: List, metric: Callable):
    return metric(annotation, roi_positions)

    
def write_results(annotation: List[List], 
                  roi_positions: List[List],
                  roi_mask: np.ndarray,
                  annot_file: str,
                  out_dir: str) -> None:
    
    video_name = os.path.basename(annot_file).split(".")[0]
    results_dir = os.path.join(out_dir, "-".join(["annotation_comparison_results", video_name]))
    
    if os.path.exists(results_dir):
         print("Results directory already exists. Overwriting.")
    
    os.makedirs(results_dir, exist_ok = True)
    
    plots_dir = os.path.join(results_dir, PLOT_SAVE_DIR)
    data_dir = os.path.join(results_dir, DATA_SAVE_DIR)
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    # Plots.
    draw(np.array(annotation), roi_mask, save_name = "annotation_roi_plot", out_dir = plots_dir)
    
    # Metrics.
    perc_diff, count = compare(annotation, roi_positions, annotation_metrics.count_metric)
    com_difference = compare(np.array(annotation), np.array(roi_positions), annotation_metrics.com_metric)
    
    with open(os.path.join(data_dir, "metrics.txt"), "w") as f:
        f.write(f"% Area Difference: {perc_diff * 100:.4f}\n")
        f.write(f"% of ROI Inside the Annotation: {count * 100:.4f}\n")
        f.write(f"Centre of Mass Difference: {com_difference:.4f}")