import pandas as pd
from probabalistic_displacement_scripts import plot_weighted_mean_haz_curves, \
    make_sz_crustal_paired_PPE_dict, make_fault_model_PPE_dict, get_weighted_mean_PPE_dict, \
    plot_weighted_mean_haz_curves_colorful
import pickle as pkl


#### USER INPUTS   #####
slip_taper = False                           # True or False, only matters if crustal. Defaults to False for sz.
fault_type = "sz"                       # "crustal or "sz"; only matters for single fault model
crustal_model_version = "_CFM"           # "_Model1", "_Model2", or "_CFM"
sz_model_version = "_deblob_steeperdip"                    # must match suffix in the subduction directory with gfs
outfile_extension = "deblob_steeperdip"               # Optional; something to tack on to the end so you don't overwrite files

probability_plot = True                # plots the probability of exceedance at the 0.2 m uplift and subsidence thresholds
displacement_chart = True                 # plots the displacement at the 10% and 2% probability of exceedance
# thresholds
make_hazcurves = False
make_colorful_hazcurves = False
#make_map = True


# Do you want to calculate the PPEs for a single fault model or a paired crustal/subduction model?
paired_crustal_sz = True                   # True or False

# Do you want to calculate PPEs for the fault model?
# This only has to be done once because it is saved a pickle file
# If False, it just makes figures and skips making the PPEs
calculate_fault_model_PPE = True            # True or False

figure_file_type_list = ["png", "pdf"]             # file types for figures

unique_id_keyphrase_list = ["N165", "N279"]         # sz
#unique_id_keyphrase_list = ["N27", "N46"]          # crustal
#unique_id_keyphrase_list = ["S066", "S141"]
#unique_id_keyphrase_list = ["S042", "S158"]

# set up file directories
crustal_directory = "crustal"
sz_directory = "subduction"
results_directory = "results"

#plot_order_temp = ["Porirua CBD north", "Porirua CBD south"]
######################################################

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
###############################

gf_name = "sites"
if fault_type == "crustal" and not paired_crustal_sz:
    fault_model_version = crustal_model_version
elif fault_type == "sz" and not paired_crustal_sz:
    fault_model_version = sz_model_version
    slip_taper = False

if slip_taper:
    taper_extension = "_tapered"
else:
    taper_extension = "_uniform"

# these directories should already be made from calculating displacements in a previous script
crustal_model_version_results_directory = f"{results_directory}/crustal{crustal_model_version}"
sz_model_version_results_directory = f"{results_directory}/sz{sz_model_version}"

# get branch weights from the saved Excel spreadsheet
branch_weight_file_path = f"../data/branch_weight_data.xlsx"
crustal_sheet_name = "crustal_weights_4_2"
sz_sheet_name = "sz_weights_4_0"
crustal_branch_weight_dict = make_branch_weight_dict(branch_weight_file_path=branch_weight_file_path,
                                                     sheet_name=crustal_sheet_name)
sz_branch_weight_dict = make_branch_weight_dict(branch_weight_file_path=branch_weight_file_path,
                                                sheet_name=sz_sheet_name)

# designate which branch weight dictionary to use based on the fault type
if not paired_crustal_sz and fault_type=="crustal":
    fault_model_branch_weight_dict = crustal_branch_weight_dict
if not paired_crustal_sz and fault_type=="sz":
    fault_model_branch_weight_dict = sz_branch_weight_dict

# extract the solution suffix based on the fault type and solution folder name
crustal_file_suffix_list = [crustal_branch_weight_dict[key]["file_suffix"] for key in crustal_branch_weight_dict.keys()]
sz_file_suffix_list = [sz_branch_weight_dict[key]["file_suffix"] for key in sz_branch_weight_dict.keys()]

# make list of file extensions with green's function type and solution suffix
crustal_extension1_list = [gf_name + suffix for suffix in crustal_file_suffix_list]
sz_extension1_list = [gf_name + suffix for suffix in sz_file_suffix_list]


### make a dictionary of all the branch probabilities, oranized by site within each branch
# option to skip this step if you've already run it once and saved to a pickle file
if not paired_crustal_sz:
    n_samples = 1000000
    model_version_results_directory = f"{results_directory}/{fault_type}{fault_model_version}"
    if calculate_fault_model_PPE:
        make_fault_model_PPE_dict(
            branch_weight_dict=fault_model_branch_weight_dict,
            model_version_results_directory=model_version_results_directory, n_samples=n_samples,
            slip_taper=slip_taper, outfile_extension=outfile_extension)

    fault_model_PPE_filepath = f"../{model_version_results_directory}/allbranch_PPE_dict_{outfile_extension}{taper_extension}.pkl"
    with open(fault_model_PPE_filepath, 'rb') as f:
        PPE_dict = pkl.load(f)

##### paired crustal and sz PPE
if paired_crustal_sz:
    n_samples = 100000
    model_version_results_directory = f"{results_directory}/paired_c{crustal_model_version}_sz{sz_model_version}"

    #### skip this part if you've already run it once and saved to a pickle file
    if calculate_fault_model_PPE:
        make_sz_crustal_paired_PPE_dict(
            crustal_branch_weight_dict=crustal_branch_weight_dict, sz_branch_weight_dict=sz_branch_weight_dict,
            crustal_model_version_results_directory=crustal_model_version_results_directory,
            sz_model_version_results_directory=sz_model_version_results_directory,
            slip_taper=slip_taper, n_samples=n_samples,
            out_directory=model_version_results_directory, outfile_extension=outfile_extension)

    paired_PPE_pickle_name = f"sz_crustal_paired_PPE_dict_{outfile_extension}{taper_extension}.pkl"
    paired_PPE_filepath = f"../{model_version_results_directory}/{paired_PPE_pickle_name}"
    with open(paired_PPE_filepath, 'rb') as f:
        PPE_dict = pkl.load(f)

# calculate weighted mean PPE for the branch or paired dataset
weighted_mean_PPE_dict = get_weighted_mean_PPE_dict(fault_model_PPE_dict=PPE_dict,
                                                    out_directory=model_version_results_directory,
                                                    outfile_extension=outfile_extension, slip_taper=slip_taper)

# open the saved weighted mean PPE dictionary
weighted_mean_PPE_filepath = f"../{model_version_results_directory}/weighted_mean_PPE_dict_{outfile_extension}" \
                             f"{taper_extension}.pkl"
with open(weighted_mean_PPE_filepath, 'rb') as f:
    weighted_mean_PPE_dict = pkl.load(f)


# plot hazard curves and save to file

if paired_crustal_sz:
    model_version_title = f"paired crustal{crustal_model_version} and sz{sz_model_version}"
else:
    model_version_title = f"{fault_type}{fault_model_version}"

if make_hazcurves:
    plot_weighted_mean_haz_curves(
        PPE_dictionary=PPE_dict, weighted_mean_PPE_dictionary=weighted_mean_PPE_dict,
        model_version_title=model_version_title, exceed_type_list=["up", "down", "total_abs"],
        out_directory=model_version_results_directory, file_type_list=figure_file_type_list, slip_taper=slip_taper)

if make_colorful_hazcurves:
    plot_weighted_mean_haz_curves_colorful(weighted_mean_PPE_dictionary=weighted_mean_PPE_dict, PPE_dictionary=PPE_dict,
                                           exceed_type_list=["down"],
                                           model_version_title=model_version_title,
                                           out_directory=model_version_results_directory,
                                           file_type_list=figure_file_type_list,
                                           slip_taper=slip_taper, file_name=f"colorful_lines{fault_model_version}",
                                           string_list=unique_id_keyphrase_list)



