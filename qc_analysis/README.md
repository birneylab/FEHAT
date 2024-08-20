Various functionalities for quality control. Experimental stage. 
At testing time, the decision tree was overfitting on training data and had poor performance when transferred to new unseen data.

# Decision Tree
Analyse the results of quality control parameters of the heart region detection from medaka. Used to train the decision tree.

## Usage
Training the decision tree can be done as follows:

```
$ python train_decision_tree.py -i <input_results_file_csv> -o <output_results_directory>
```

Example files can be found in ```decision_tree/data/```. The serialized pre-trained tree used ```train_set_F0.csv```.

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
Currently, it adds a flag in the results-file, specifying if the tree assumes the result to be an error (flag=1) or not an error (flag=0).