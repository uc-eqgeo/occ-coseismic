from glob import glob
import pickle as pkl
import numpy as np
import pandas as pd

"""
This script was written to find the recurrance time of a single crustal fault in the CFM model (Wharekauhau),
as the initial outputs in the Wellington region show that there is no uplift associated with it, which is odd for a 
thrust fault. Net result was that the geodetic branches gave a recurrance that was too low (15,215 years), compared
to the geologic recurrance (2390 years). When all were used together, the recurrance was 4132 years, hence the seeming
lack of uplift related to Wharekauhau scenarios
"""

results_dir = '../results/crustal_Model_CFM_testing'


def make_branch_weight_dict(branch_weight_file_path, sheet_name):
    """
    This function reads in the excel file with the branch weights and returns a dictionary with the branch weights
    and other information (scaling values, solution file names, etc.).
    The dictionary keys are the unique ID strings based on the branch parameters

    :param branch_weight_file_path: string; path to the excel file with the branch weights
    :param sheet_name: string; name of the sheet in the excel file with the branch weights
    """

    # read in the Excel file with the branch weights and other metadata
    branch_weights = pd.read_excel(branch_weight_file_path, sheet_name=sheet_name, header=0)

    # make a dictionary with the branch weights and other metadata
    branch_weight_dict = {}
    for row in range(len(branch_weights)):

        N_val = branch_weights["N"][row]
        N_string = str(N_val).replace('.', '')
        b_val = branch_weights["b"][row]
        b_string = str(b_val).replace('.', '')
        C_val = branch_weights["C"][row]
        C_string = str(C_val).replace('.', '')
        S_val = branch_weights["S"][row]
        S_string = str(S_val).replace('.', '')
        def_model  = branch_weights["def_model"][row]
        time_dependence = branch_weights["time_dependence"][row]
        file_suffix = branch_weights["solution_file_suffix"][row]
        total_weight_RN = branch_weights["total_weight_RN"][row]

        # make a unique ID for each branch.
        # The NSHM solution files do not include the rate scaling factor (S) (i.e., they are all S=1)
        # These lines use the same solution file for 3 different S values
        unique_id = f"N{N_string}_b{b_string}_C{C_string}_S{S_string}_{time_dependence}_{def_model}{file_suffix}"

        branch_weight_dict[unique_id] = {"N": N_val, "b": b_val, "C": C_val, "S": S_val, "def_model": def_model,
                                    "time_dependence": time_dependence, "file_suffix": file_suffix, "total_weight_RN":
                                               total_weight_RN}

    return branch_weight_dict


branch_weight_dict = make_branch_weight_dict(branch_weight_file_path='../data/branch_weight_data.xlsx', sheet_name="crustal_weights_4_2")

gf_name = 'sites'

pkl_list = glob(results_dir + f'/{gf_name}*/branch_site_disp_dict*.pkl')

geodetic = []
branch_weight = []
sum_rate = []

site = '1772_5420'

for pkl_file in pkl_list:
    with open(pkl_file, 'rb') as f:
        disp_dict = pkl.load(f)
    
    branch_code = pkl_file.split('_')[-2]
    branch_id = glob(results_dir + f'/{gf_name}*/N*S10*{branch_code}_cumu_PPE.h5')[0].split('\\')[-1].replace('_cumu_PPE.h5', '')
    if 'geodetic' in branch_id:
        geodetic.append(True)
    else:
        geodetic.append(False)
    
    branch_weight.append(branch_weight_dict[branch_id]["total_weight_RN"])
    sum_rate.append(sum([rate for rate in disp_dict[site]['rates']]))


total_sum = np.average(sum_rate, weights=branch_weight)
geologic_sum = np.average([sum_rate[i] for i in range(len(sum_rate)) if not geodetic[i]], weights=[branch_weight[i] for i in range(len(branch_weight)) if not geodetic[i]])
geodetic_sum = np.average([sum_rate[i] for i in range(len(sum_rate)) if geodetic[i]], weights=[branch_weight[i] for i in range(len(branch_weight)) if geodetic[i]])
print('Total sum rate:', round(1/total_sum), 'years')
print('Geologic sum rate:', round(1/geologic_sum), 'years')
print('Geodetic sum rate:', round(1/geodetic_sum), 'years')