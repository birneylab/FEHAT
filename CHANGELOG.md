Date: May 14, 2021

In this version of the script, some improvements were made to the original one, developed by Jack in https://github.com/monahanj/medaka_embryo_heartRate. 

- The wrappers (sh files) create one subjob for each well and each loop in a plate simultaneously. This new method makes the results quicker.
After running the sh file, some helpful information is presented in the terminal regarding your analyses.
- Even the analyses for a specific well fails, a folder with the video of the particular embryo is created, making it easy to understand the problem.
- In the main folder, a child folder is created for each loop. Inside this loop child folder (e.g., LO001 folder), where every resulting well´s folder are (with results for the wells), a CSV report file is created with BPM results and error types, if it is the case, for each well. Inside this same folder, a JPEG file is made with the data distribution for that particular loop.
- A CSV file is also created in the main folder, merging all data from each loop in a single data frame. This way, it is easy to read the file in excel, for example. In addition, a graph file with data distribution (a general graph for every loop) is also created.

The changes below are regarding the heart detection accuracy:

- Added a grey mask before detecting the most changeable pixels, so the veins are not often chosen as heart region. The problem is that the veins present high variation and are an important confusion factor. As they are more clearer than the heart, we can mask them and avoid them being detected.
- Instead of choosing an ROI (region of interest) only if the region meets specific overlap requirements, failing analyses if the criteria are not met, the algorithm now always selects an area considering the most probable location, avoiding the analyses to fail.
- When the embryo moves around, the original script stopped working. In this new version, the script detects when the embryo moves and only delete those frames, keeping going with analyses. The remaining part of the video after this event has been shown enough to present accurate results.
- Now, there is an option to insert the expected average as an argument for the command line execution of the script. This argument is helpful because sometimes, the script detects more than one furrier peak, especially if the two chambers of the heart are segmented. One of those values is always more than double the correct peak (is always an outlier). It is tough to make the script recognize this without some values to compare. Then, you can do a previous screening of the data and insert the average as an argument; then, it will be easier for the algorithm to choose the correct peak.

Date: May 19, 2021

- Added option to run in a single server (not in a farm of servers)

Date: June 23, 2021

- General restructuring of the code into a more modular architecture. Removed unused legacy code.
- Integrated logging capeability and exception handling
- Cross platform support through python by removing dependencies with .sh files
- Providing the loops as arguments is no longer a necessity
- Noticable increase in analysis speed.
