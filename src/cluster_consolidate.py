#!/usr/bin/env python
############################################################################################################
# Authors: 
#   Marcio Ferreira,    EMBL-EBI,       marcio@ebi.ac.uk
#   Sebastian Stricker, Uni Heidelberg, sebastian.stricker@stud.uni-heidelberg.de
# Date: 08/2021
# License: Contact authors
###
# For cluster mode.
# Called as a dependend bsub job when all instances running cluster.py have finished.
# Gathers data from outdir/tmp and creates the final report.
###
############################################################################################################
import argparse
import pandas as pd
import os
# from functools import reduce
# import numpy as np
# import statistics
# import seaborn as sns
# from matplotlib import pyplot as plt

import glob2
from pathlib import Path
import logging

import io_operations
import setup
import shutil

LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Read in medaka heart video frames')
parser.add_argument('-o','--outdir', action="store", dest='outdir', help='Where to store the output report and the global log',    default=False, required = True)
parser.add_argument('-i','--indir', action="store", dest='indir', help='Path to temp folder with results',                      default=False, required = True)

args = parser.parse_args()

# Adds a trailing slash if it is missing.
args.indir  = os.path.join(args.indir, '')
args.outdir = os.path.join(args.outdir, '')

out_dir = args.outdir
indir = args.indir

# Number code for logfile and outfile respectively
experiment_name = os.path.basename(os.path.normpath(out_dir))
experiment_id = experiment_name.split('_')[0]

setup.config_logger(out_dir, ("logfile_" + experiment_id + ".log"))

LOGGER.info("Consolidating cluster results")

# path to central log file
logs_paths    = glob2.glob(indir + '*.log')
results_paths = glob2.glob(indir + '*.txt')

# log files and results files of one analysis should have same index in lists
logs_paths.sort()
results_paths.sort()

results = {'channel': [], 'loop': [], 'well': [], 'heartbeat': [], 'log': []}
for log, result in zip(logs_paths, results_paths):
    if (Path(log).stem.split('-') != Path(result).stem.split('-')):
        LOGGER.exception("Logfile and result file order not right")

    #/../CO6-LO001-WE00001.txt -> [CO6, LO001, WE00001]
    metadata = Path(log).stem.split('-')

    with open(log) as fp:
        log_text = fp.read()
        results['channel'].append(metadata[0])
        results['loop'].append(metadata[1])
        results['well'].append(metadata[2])
        results['log'].append(log_text)

    with open(result) as fp:
        bpm = fp.read()
        results['heartbeat'].append(bpm)

# Sort through pandas
results = pd.DataFrame.from_dict(results)
results.sort_values(by=['channel', 'loop', 'well'])

logs = results['log'].tolist()
LOGGER.info("Log reports from analyses: \n" + '\n'.join(logs))

results = results.to_dict(orient='list')

#  results: Dictionary {'channel': [], 'loop': [], 'well': [], 'heartbeat': []}
io_operations.write_to_spreadsheet(out_dir, results, experiment_id)

#TODO: Dangerous as it removes all conents of directory. Assert that in fact, only log and txt files are inside
#TODO: and that folder is named tmp/
# Clean up and remove tmp directory
#shutil.rmtree(indir)

# #######################################################################

# loops_folders = os.listdir(indir)

# total_nan = 0
# number_of_rows = 0

# if len(loops_folders) > 1:
#     #print('varios loops')
    
#     dataframes_list = []

#     try:
#         for f in loops_folders:

#             #The user could have inserted loops that does not exist, then just try to not raise an error
#             try: 
#                 #print(indir + "/" + f + '/' + 'general_report.csv')
#                 vars()[str(f)] = pd.read_csv(indir + "/" + f + '/general_report.csv')
                
#                 #summ the total nan to make a report
#                 total_nan = vars()[str(f)]['tbpm'].isna().sum() + total_nan

#                 #sum the total rows to make a report
#                 index = vars()[str(f)].index
#                 number_of_rows = number_of_rows + len(index)
            

#                 #create a real list, that is, loops that are really present            
#                 dataframes_list.append(f)
#             except:
#                 print("loop " + str(f) + " not found in loops list in consolidated.py")
#     except:
#         print("An error ocurred. Are you sure that the CSV files ended with \'[loopName]_data.csv\' are present in each loop folder?")



#     #merge all dataframes on the fly
#     outer_merged = reduce(lambda  left,right: pd.merge(left,right,on=['well'],
#                                                 how='outer'), [pd.read_csv(indir + "/" + f + '/general_report.csv') for f in dataframes_list])

   
#     blue_print_df = pd.DataFrame({
#     'id': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96'],
#     'well': ['WE00001', 'WE00002', 'WE00003', 'WE00004', 'WE00005', 'WE00006', 'WE00007', 'WE00008', 'WE00009', 'WE00010', 'WE00011', 'WE00012', 'WE00013', 'WE00014', 'WE00015', 'WE00016', 'WE00017', 'WE00018', 'WE00019', 'WE00020', 'WE00021', 'WE00022', 'WE00023', 'WE00024', 'WE00025', 'WE00026', 'WE00027', 'WE00028', 'WE00029', 'WE00030', 'WE00031', 'WE00032', 'WE00033', 'WE00034', 'WE00035', 'WE00036', 'WE00037', 'WE00038', 'WE00039', 'WE00040', 'WE00041', 'WE00042', 'WE00043', 'WE00044', 'WE00045', 'WE00046', 'WE00047', 'WE00048', 'WE00049', 'WE00050', 'WE00051', 'WE00052', 'WE00053', 'WE00054', 'WE00055', 'WE00056', 'WE00057', 'WE00058', 'WE00059', 'WE00060', 'WE00061', 'WE00062', 'WE00063', 'WE00064', 'WE00065', 'WE00066', 'WE00067', 'WE00068', 'WE00069', 'WE00070', 'WE00071', 'WE00072', 'WE00073', 'WE00074', 'WE00075', 'WE00076', 'WE00077', 'WE00078', 'WE00079', 'WE00080', 'WE00081', 'WE00082', 'WE00083', 'WE00084', 'WE00085', 'WE00086', 'WE00087', 'WE00088', 'WE00089', 'WE00090', 'WE00091', 'WE00092', 'WE00093', 'WE00094', 'WE00095', 'WE00096'],
#     'twell_prov_id': ['A001', 'A002', 'A003', 'A004', 'A005', 'A006', 'A007', 'A008', 'A009', 'A010', 'A0011', 'A012', 'B012', 'B011', 'B010', 'B009', 'B008', 'B007', 'B006', 'B005', 'B004', 'B003', 'B002', 'B001', 'C001', 'C002', 'C003', 'C004', 'C005', 'C006', 'C007', 'C008', 'C009', 'C010', 'C011', 'C012', 'D012', 'D011', 'D010', 'D009', 'D008', 'D007', 'D006', 'D005', 'D004', 'D003', 'D002', 'D001', 'E001', 'E002', 'E003', 'E004', 'E005', 'E006', 'E007', 'E008', 'E009', 'E010', 'E011', 'E012', 'F012', 'F011', 'F010', 'F009', 'F008', 'F007', 'F006', 'F005', 'F004', 'F003', 'F002', 'F001', 'G001', 'G002', 'G003', 'G004', 'G005', 'G006', 'G007', 'G008', 'G009', 'G010', 'G011', 'G012', 'H012', 'H011', 'H010', 'H009', 'H008', 'H007', 'H006', 'H005', 'H004', 'H003', 'H002', 'H001']})


#     outer_merged_final = pd.merge(outer_merged, blue_print_df, how="outer", on=["well"])
    
#     #outer_merged_final.columns = [outer_merged_final.columns[:-1], 'Test']

#     outer_merged_final = outer_merged_final.sort_values(by=['id'])


#     df_columns = outer_merged_final.columns.tolist()
#     delete_columns = [x for x in df_columns if 'twell_id' in x]

#     #delete_columns

#     outer_merged_final.drop(delete_columns, axis=1, inplace=True)






#     outer_merged_final = outer_merged_final.rename(columns={'twell_prov_id': 'twell_id'})

#     cols_names = outer_merged_final.columns.tolist()
    
    
    
    
#     index = 0
#     position_tbpm = 0
#     position_error = 0

#     for g in cols_names:    
#         if "tbpm" in g:
#             cols_names[index] = "tbpm " + dataframes_list[position_tbpm]
#             position_tbpm += 1
#         elif "message if any" in g:
#             cols_names[index] = "message if any " + dataframes_list[position_error]
#             position_error += 1        
#         index += 1


#     outer_merged_final.columns = cols_names

#     outer_merged_final

    
    
#     names = outer_merged_final.columns.tolist()


#     names.remove('id')
#     names.remove('twell_id')

#     names = ['id', 'twell_id'] + names

#     outer_merged_final = outer_merged_final[names]


#     outer_merged_final.to_csv(out_dir + '/consolidated_report.csv', index=False, header=True, na_rep='NaN')

    
    
# else:
#         print('only one loop')
        
#         outer_merged_final = pd.read_csv(indir + "/" + loops_folders[0] + '/general_report.csv')
        
#         #summ the total nan to make a report
#         total_nan = outer_merged_final['tbpm'].isna().sum() 

#         #sum the total rows to make a report
#         index = outer_merged_final.index
#         number_of_rows = len(index)
        
#         cols_names = outer_merged_final.columns.tolist()
        
#         #print(cols_names)
    
#         index = 0
#         position_tbpm = 0
#         position_error = 0

#         for g in cols_names:    
#             if "tbpm" in g:
#                 cols_names[index] = "tbpm " + loops_folders[0]
#                 position_tbpm += 1
#             elif "message if any" in g:
#                 cols_names[index] = "message if any " + loops_folders[0]
#                 position_error += 1        
#             index += 1

#         #print(cols_names)
        
        
#         outer_merged_final.columns = cols_names
        
#         #print(dataframe)
        
  
        

#         outer_merged_final.to_csv(out_dir + '/consolidated_report.csv', index=False, header=True, na_rep='NaN')

       
        
        
# tbpm_list = []
# cols_names = outer_merged_final.columns.tolist()

# for f in cols_names:
#     if f.startswith("tbpm"):
#         tbpm_list.append(f)


# tbpm_columns = outer_merged_final[tbpm_list]
  
# listTbpm = []
# for f in tbpm_list:    
#     listn = tbpm_columns[f].tolist()
#     listTbpm.append(listn)
    
# flattened  = [val for sublist in listTbpm for val in sublist]
  
        
# number_total_of_rows = len(flattened)



# listNoNan = [n for n in flattened if str(n) != 'nan']





# valid_number_of_rows = number_of_rows - total_nan



# #calculate error index
# error_index = total_nan/number_of_rows*100
# #print(data)

# #calculate Coeficient of variation

# cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100
# cv = round(cv(listNoNan),2)
# mean_data = round(statistics.mean(listNoNan),2)

# std_dev = round(np.std(listNoNan),2)


# final_text = ("total_rows: " + str(number_of_rows) + ";  " + "Error rows: " + str(total_nan) + ";  " + \
#                  "error index: " + str(round(error_index, 2)) + \
#                  " %;  " + "average: " + str(mean_data) + ";  std deviation: " + str(std_dev) + \
#                  ";  variation coeficient: " + str(cv))


# sns.set(font_scale = 1.2)
# fig, axs = plt.subplots(1,2, figsize=(15,10))
# double_plot = sns.histplot(data = listNoNan, bins = 20, ax=axs[0]) #.set_title("Frequency Histogram of tbpm data")
# ymin, ymax = double_plot.get_ylim()
# double_plot.set_title("Frequency Histogram of tbpm data")
# double_plot.set(ylim=(ymin, ymax+24))
# for p in double_plot.patches:
#     double_plot.annotate(format(p.get_height(), 'd'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
# double_plot = sns.boxplot(x=listNoNan, orient="h", color='salmon', ax=axs[1]).set_title("Boxplot of tbpm data")
# fig.suptitle(final_text, fontsize=12)
# fig.show()
# fig.savefig(out_dir + "/final_graph.jpg")
