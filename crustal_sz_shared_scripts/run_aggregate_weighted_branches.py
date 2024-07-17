import os
import pandas as pd
import numpy as np
from probabalistic_displacement_scripts import plot_weighted_mean_haz_curves, \
    make_sz_crustal_paired_PPE_dict, make_fault_model_PPE_dict, get_weighted_mean_PPE_dict, \
    plot_weighted_mean_haz_curves_colorful, save_disp_prob_tifs, save_disp_prob_xarrays
from helper_scripts import get_NSHM_directories, get_rupture_disp_dict
import pickle as pkl
from nesi_scripts import nesi_get_weighted_mean_PPE_dict


#### USER INPUTS   #####
slip_taper = False                           # True or False, only matters if crustal. Defaults to False for sz.
fault_type = "crustal"                       # "crustal", "sz" or "py"; only matters for single fault model + getting name of paired crustal subduction pickle files
crustal_model_version = "_Model_CFM_wellington_1km"           # "_Model1", "_Model2", or "_CFM"
sz_model_version = "_wellington_1km"                    # must match suffix in the subduction directory with gfs
outfile_extension = ""               # Optional; something to tack on to the end so you don't overwrite files
nesi = True    # Prepares code for NESI runs
testing = False   # Impacts number of samples runs, job time etc


# Processing Flags (True/False)
paired_crustal_sz = False       # Do you want to calculate the PPEs for a single fault model or a paired crustal/subduction model?
load_random = False             # Do you want to uses the same grid for scenarios for each site, or regenerate a new grid for each site?
calculate_fault_model_PPE = False   # Do you want to calculate PPEs for each branch?
remake_PPE = False              # Recalculate branch PPEs from scratch, rather than search for pre-existing files (useful if have to stop processing...)
calculate_weighted_mean_PPE = True   # Do you want to weighted mean calculate PPEs?
save_arrays = True             # Do you want to save the displacement and probability arrays?
default_plot_order = False       # Do you want to plot haz curves for all sites, or use your own selection of sites to plot? 
make_hazcurves = False       # Do you want to make hazard curves?
make_colorful_hazcurves = False # Do you want to make colorful hazard curves?
plot_order_csv = "../wellington_10km_grid_points.csv"  # csv file with the order you want the branches to be plotted in (must contain sites in order under column siteId). Does not need to contain all sites

# Processing Parameters
time_interval = 100     # Time span of hazard forecast (yrs)
sd = 0.4                # Standard deviation of the normal distribution to use for uncertainty in displacements

# Nesi Parameters
launch_sbatch = False   # Run sbatch command to launch nesi jobs 
nesi_step = 'prep'  # 'prep' or 'combine
n_array_tasks = 1000    # Number of array tasks
min_tasks_per_array = 100   # Minimum number of sites per array
min_branches_per_array = 1  # Minimum number of branches per array


# Parameters that shouldn't need to be changed
crustal_directory = "crustal"
sz_directory = "subduction"
results_directory = "results"
figure_file_type_list = ["png", "pdf"]             # file types for figures
unique_id_keyphrase_list = ["N165", "N279"]         # sz
#unique_id_keyphrase_list = ["N27", "N46"]          # crustal
#unique_id_keyphrase_list = ["S066", "S141"]
#unique_id_keyphrase_list = ["S042", "S158"]


if testing:
    n_samples = 1e4   # Number of scenarios to run
    job_time = 1    # Amount of time to allocate per site in the cumu_PPE task array
    mem = 5    # Memory allocation for cumu_PPE task array
else:
    n_samples = 1e5
    job_time = 3
    mem = 25

## Solving processing conflicts
if calculate_fault_model_PPE:
    calculate_weighted_mean_PPE = True  # If recalculating PPEs, you need to recalculate the weighted mean PPEs


if not default_plot_order and not os.path.exists(plot_order_csv):
    raise Exception("Manual plot order selected but no plot order csv found. Please create a csv file with the order you want the branches to be plotted in (must contain sites in order under column siteId)")

if paired_crustal_sz:
    if fault_type == 'all':
        print("Running combined crustal, hikurangi-kermadec, and puysegur models")
        fault_type = ['crustal', 'sz', 'py']
    elif not fault_type in ['sz', 'py']:
        raise Exception("Paired crustal and subduction model selected but fault type is not sz, py or all. Please select sz or py as fault type.")
        
    else:
        fault_type = ['crustal', fault_type]
else:
    if fault_type == 'all':
        raise Exception("Can't have fault type = 'all' and paired_crustal_sz = False")
    fault_type = [fault_type]

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
if not paired_crustal_sz:
    if fault_type[0] == "crustal":
        model_version_list = [crustal_model_version]
    else:
        model_version_list = [sz_model_version]
        slip_taper = False    
else:
    model_version_list = [crustal_model_version] + [sz_model_version] * len(fault_type[1:])

if slip_taper:
    taper_extension = "_tapered"
else:
    taper_extension = "_uniform"

# these directories should already be made from calculating displacements in a previous script
model_version_results_directory = []
for ix, model in enumerate(fault_type):
    model_version_results_directory.append(f"{results_directory}/{model}{model_version_list[ix]}")

# get branch weights from the saved Excel spreadsheet
branch_weight_file_path = f"../data/branch_weight_data.xlsx"
crustal_sheet_name = "crustal_weights_4_2"
sz_sheet_name = "sz_weights_4_0"
py_sheet_name = "py_weights_4_0"

sheet_list = []
if 'crustal' in fault_type:
    sheet_list.append(crustal_sheet_name)
if 'sz' in fault_type:
    sheet_list.append(sz_sheet_name)
if 'py' in fault_type:
    sheet_list.append(py_sheet_name)

branch_weight_dict_list = []
for sheet in sheet_list:
    branch_weight_dict_list.append(make_branch_weight_dict(branch_weight_file_path=branch_weight_file_path,
                                                            sheet_name=sheet))
#if paired_crustal_sz or fault_type=="crustal":
#    crustal_branch_weight_dict = make_branch_weight_dict(branch_weight_file_path=branch_weight_file_path,
#                                                        sheet_name=crustal_sheet_name)
#if fault_type in ['sz', 'py']:
#    sz_branch_weight_dict = make_branch_weight_dict(branch_weight_file_path=branch_weight_file_path,
#                                                    sheet_name=sz_sheet_name)

# designate which branch weight dictionary to use based on the fault type
fault_model_branch_weight_dict = {}
for ii in range(len(fault_type)):
    fault_model_branch_weight_dict = fault_model_branch_weight_dict | branch_weight_dict_list[ii]

#if not paired_crustal_sz and fault_type=="crustal":
#    fault_model_branch_weight_dict = crustal_branch_weight_dict
#if not paired_crustal_sz and any([fault_type=="sz", fault_type=="py"]):
#    fault_model_branch_weight_dict = sz_branch_weight_dict

# Is this section necessary?
## extract the solution suffix based on the fault type and solution folder name
#crustal_file_suffix_list = [crustal_branch_weight_dict[key]["file_suffix"] for key in crustal_branch_weight_dict.keys()]
#sz_file_suffix_list = [sz_branch_weight_dict[key]["file_suffix"] for key in sz_branch_weight_dict.keys()]
#
## make list of file extensions with green's function type and solution suffix
#crustal_extension1_list = [gf_name + suffix for suffix in crustal_file_suffix_list]
#sz_extension1_list = [gf_name + suffix for suffix in sz_file_suffix_list]

NSHM_directory_list, file_suffix_list, n_branches = get_NSHM_directories(fault_type, crustal_model_version, sz_model_version, deformation_model='geologic and geodetic', time_independent=True,
                         time_dependent=True, single_branch=False)

extension1_list = [gf_name + suffix for suffix in file_suffix_list]
get_rupture_dict = False

for ix, extension1 in enumerate(extension1_list):
    ftype = [(jj, ftype) for jj, ftype in enumerate(fault_type) if '_' + ftype.replace('rustal', '') + '_' in extension1][0]
    if not os.path.exists(f"../{model_version_results_directory[ftype[0]]}/{extension1}/all_rupture_disps_{extension1}{taper_extension}.pkl") or get_rupture_dict:
        print(f"\nbranch {ix + 1} of {len(extension1_list)}")
        get_rupture_disp_dict(NSHM_directory=NSHM_directory_list[ix], extension1=extension1_list[ix],
                                slip_taper=slip_taper, fault_type=ftype[1], gf_name=gf_name,
                                results_version_directory=model_version_results_directory[ftype[0]],
                                crustal_directory=crustal_directory, sz_directory=sz_directory,
                                model_version=model_version_list[ftype[0]], search_radius=9e5)

### make a dictionary of all the branch probabilities, oranized by site within each branch
# option to skip this step if you've already run it once and saved to a pickle file
if not paired_crustal_sz:
    fault_type = fault_type[0]
    out_version_results_directory = f"{results_directory}/{fault_type}{model_version_list[0]}"
    PPE_filepath = f"../{out_version_results_directory}/allbranch_PPE_dict{outfile_extension}{taper_extension}.pkl"
    if not os.path.exists(PPE_filepath):
        print('No fault model PPE pkl file found. Making a new one...')
        calculate_fault_model_PPE = True

    if calculate_fault_model_PPE:
        make_fault_model_PPE_dict(
            branch_weight_dict=fault_model_branch_weight_dict,
            model_version_results_directory=out_version_results_directory, n_samples=n_samples,
            slip_taper=slip_taper, outfile_extension=outfile_extension, nesi=nesi, nesi_step=nesi_step, sbatch=launch_sbatch, mem=mem,
            time_interval=time_interval, sd=sd, n_array_tasks=n_array_tasks, min_tasks_per_array=min_tasks_per_array, job_time=job_time,
            load_random=load_random, remake_PPE=remake_PPE)

    if not nesi and calculate_weighted_mean_PPE:
        print('Loading pre-prepared fault model PPE dictionary...')
        with open(PPE_filepath, 'rb') as f:
            PPE_dict = pkl.load(f)

##### paired crustal and sz PPE
if paired_crustal_sz:
    out_version_results_directory = f"{results_directory}/paired_c{crustal_model_version}"
    pickle_prefix = ''
    for sub in fault_type[1:]:
        out_version_results_directory += f"_{sub}{sz_model_version}"
        pickle_prefix += f"{sub}_"
    paired_PPE_pickle_name = f"{pickle_prefix}crustal_paired_PPE_dict_{outfile_extension}{taper_extension}.pkl"
    PPE_filepath = f"../{out_version_results_directory}/{paired_PPE_pickle_name}"

    if not os.path.exists(PPE_filepath):
        print(f"No crustal-{'-'.join(fault_type[1:])} paired PPE pkl file found. Making a new one...")
        calculate_fault_model_PPE = True

    #### skip this part if you've already run it once and saved to a pickle file
    if calculate_fault_model_PPE:
        make_sz_crustal_paired_PPE_dict(
            crustal_branch_weight_dict=branch_weight_dict_list[0], sz_branch_weight_dict_list=branch_weight_dict_list[1:],
            crustal_model_version_results_directory=model_version_results_directory[0],
            sz_model_version_results_directory_list=model_version_results_directory[1:],
            paired_PPE_pickle_name=paired_PPE_pickle_name, slip_taper=slip_taper, n_samples=int(n_samples),
            out_directory=out_version_results_directory, outfile_extension=outfile_extension, sz_type_list=fault_type[1:],
            nesi=nesi, nesi_step=nesi_step, n_array_tasks=n_array_tasks, min_tasks_per_array=min_tasks_per_array,
            mem=mem, time_interval=time_interval, sd=sd, job_time=job_time, remake_PPE=remake_PPE, load_random=load_random)

    if not nesi and calculate_weighted_mean_PPE:
        print('Loading fault model PPE dictionary...')
        with open(PPE_filepath, 'rb') as f:
            PPE_dict = pkl.load(f)

# calculate weighted mean PPE for the branch or paired dataset
weighted_mean_PPE_filepath = f"../{out_version_results_directory}/weighted_mean_PPE_dict_{outfile_extension}{taper_extension}.pkl"
if calculate_weighted_mean_PPE or not os.path.exists(weighted_mean_PPE_filepath):
    if nesi:
        print('Preparing NESI scripts for weighted mean PPE...')
        time_per_branch = 45 # Seconds
        n_branches = np.product(np.array(n_branches))
        total_time = n_branches * time_per_branch
        hours, rem = divmod(total_time, 3600)
        mins = np.ceil(rem / 60)
        nesi_get_weighted_mean_PPE_dict(out_directory=out_version_results_directory, ppe_name=os.path.basename(PPE_filepath),
                                        outfile_extension=outfile_extension, slip_taper=slip_taper, sbatch=launch_sbatch,
                                        hours=int(hours), mins=int(mins), mem=50, account='uc03610', cpus=1)
    else:
        print('Calculating weighted mean PPE...')
        weighted_mean_PPE_dict = get_weighted_mean_PPE_dict(fault_model_PPE_dict=PPE_dict,
                                                            out_directory=out_version_results_directory,
                                                            outfile_extension=outfile_extension, slip_taper=slip_taper)
else:
    # open the saved weighted mean PPE dictionary
    print('Loading pre-prepared weighted mean PPE dictionary...')
    with open(weighted_mean_PPE_filepath, 'rb') as f:
        weighted_mean_PPE_dict = pkl.load(f)

# plot hazard curves and save to file
if save_arrays:
    print('Saving data arrays...')
    ds = save_disp_prob_xarrays(outfile_extension, slip_taper=slip_taper, model_version_results_directory=out_version_results_directory,
                        thresh_lims=[0, 3], thresh_step=0.01, output_thresh=True, probs_lims = [0.01, 0.20], probs_step=0.01,
                        output_probs=True, grid=False, weighted=True)

if paired_crustal_sz:
    model_version_title = f"paired crustal{crustal_model_version} and "
    for ix, sub in enumerate(fault_type[1:]):
        model_version_title += f"{sub}{sz_model_version} and "
    model_version_title = model_version_title[:-5]
else:
    model_version_title = f"{fault_type[0]}{model_version_list[0]}"

if default_plot_order:
    plot_order = [key for key in weighted_mean_PPE_dict.keys() if key != 'branch_weights']
else:
    print('Using custom plot order from', plot_order_csv)
    plot_order = pd.read_csv(plot_order_csv)
    plot_order = list(plot_order['siteId'])

if make_hazcurves or make_colorful_hazcurves:
    print(f"\nOutput Directory: {out_version_results_directory}/weighted_mean_figures...")

#if make_geotiffs:
#    print(f"\nSaving hazard curve geotiffs...")
#    save_disp_prob_tifs(outfile_extension, slip_taper=slip_taper, 
#                            model_version_results_directory=model_version_results_directory,
#                            thresh_lims=[0, 3], thresh_step=0.25, output_thresh=True,
#                            probs_lims = [0.02, 0.5], probs_step=0.02, output_probs=True,
#                            weighted=True)

if make_hazcurves:
    print(f"\nMaking hazard curves...")
    plot_weighted_mean_haz_curves(
        PPE_dictionary=PPE_dict, weighted_mean_PPE_dictionary=weighted_mean_PPE_dict,
        model_version_title=model_version_title, exceed_type_list=["up", "down", "total_abs"],
        out_directory=out_version_results_directory, file_type_list=figure_file_type_list, slip_taper=slip_taper, plot_order=plot_order)

if make_colorful_hazcurves:
    print(f"\nMaking colourful hazard curves...")
    plot_weighted_mean_haz_curves_colorful(weighted_mean_PPE_dictionary=weighted_mean_PPE_dict, PPE_dictionary=PPE_dict,
                                           exceed_type_list=["down"],
                                           model_version_title=model_version_title,
                                           out_directory=out_version_results_directory,
                                           file_type_list=figure_file_type_list,
                                           slip_taper=slip_taper, file_name=f"colorful_lines_{''.join(model_version_list)}",
                                           string_list=unique_id_keyphrase_list, plot_order=plot_order)