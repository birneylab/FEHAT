```Still in development.```

# Decision Tree
Analyse the results of quality control parameters of the heart region detection from medaka. Used to train the decision tree.

## Usage
Training the decision tree saves the resulting model in 
`qc_analysis/data` and can be done as follows.

```
$ git clone --recursive https://github.com/birneylab/medaka_bpm
$ cd medaka_bpm/qc_analysis
$ python medaka_qc_analysis.py -i <input_results_file_csv> -o <output_results_directory>
```

The results file needs to have the following QC parameters as features:

	1. HROI Change Intensity
	2. Harmonic Intensity
	3. Heart size
	4. Movement detection max
	5. SNR
	6. Signal intensity
	7. Signal regional prominence
	8. Intensity/Harmonic Intensity (top 5 %)
	9. SNR Top 5%
	10. Signal Intensity Top 5%

The output results directory (specified) will contain training metrics and qc parameter plots. The **decision tree evaluation** happens **automatically** via `medaka_bpm.analyse` if the tree has been trained as above.

# Annotation
Compares annotated videos with the region detection from medaka.

## Usage

```
$ git clone --recursive https://github.com/birneylab/medaka_bpm
$ cd medaka_bpm/qc_analysis
$ python annotation_run.py -i <input_video (frames)> -af <annotation_file (JSON)> -o <output_results_directory>
```