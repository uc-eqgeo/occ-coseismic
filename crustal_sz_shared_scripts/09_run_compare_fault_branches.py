
import os
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib
from single_branch_plotting import map_and_plot_probabilities
from compare_fault_model import compare_faultmodel_prob_plot, compare_disps_chart, compare_mean_hazcurves, \
    compare_disps_with_net

os.chdir(os.path.dirname(__file__))
########## USER INPUTS #######################
plot_order_name = "from_csv"                 # "JDE sites", "from_csv", or "default"
results_directory = "results"
exceed_type = "down"                     # "down", "up", or "total_abs"
slip_taper = False
transect = False  # Whether to assume that all points are in order along a transect, and that a distance should be calculated rather than site names labeled
sigma_lims = "2sig"   # "minmax", "2sig" or "1sig" for the error bars on the displacement chart
file_id = 'numba_empty'

# Choose what models to compare. These names should be in the results folder already.
model_subdirectory_dict = {#"fq_hikkerk" : ["sz_fq_3nub110", "sz_fq_pnub110", "sz_fq_3nhb110", "sz_fq_pnhb110", "sz_fq_3lhb110", "sz_fq_plhb110"],
#                                           "sz_fq_3lhb110C1", "sz_fq_3lhb110C100", "sz_fq_3lhb110C1000", "sz_fq_3nhb110C1", "sz_fq_3nhb110C100"],
                           "hikkerk" : ["sz_NzEx"]}
#model_subdirectory_names = ["crustal_CFM","crustal_Model1", "crustal_Model2"]

# used for plot labels/titles. must be in same order as model_subdirectory_names
pretty_names = ["Crustal", "NZNSHM SZ"]
pretty_names = []
for key in model_subdirectory_dict.keys():
    pretty_names += model_subdirectory_dict[key]

file_type_list = ["png"]     # generally png and/or pdf
probability_plot = False            # plots the probability of exceedance at the 0.2 m uplift and subsidence thresholds
displacement_chart = False          # plots the displacement at the 10% and 2% probability of exceedance thresholds
compare_hazcurves = True        # plots the different hazard curves on the same plot
disps_net = False
make_map = False
labels_on = False                # displacement number labels for bar charts and probability plots

plot_order_csv = "../sites/paper_sites_points.csv"  # csv file with site order

#### script ###################
crustal_sheet_name = "crustal_weights_4_2"
sz_sheet_name = "sz_weights_4_0"
py_sheet_name = "py_weights_4_0"

sheet_list = []
for key in model_subdirectory_dict.keys():
    if 'fq_hikkerk' in key:
        sheet_list.append(sz_sheet_name + "_fq")
    elif 'hikkerk' in key:
        sheet_list.append(sz_sheet_name)
    elif 'puysegur' in key:
        sheet_list.append(py_sheet_name)
    else:
        sheet_list.append(crustal_sheet_name)

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

        N_val = branch_weights["N"][row].astype(float)
        N_string = str(N_val).replace('.', '')
        b_val = branch_weights["b"][row].astype(float)
        b_string = str(b_val).replace('.', '')
        C_val = branch_weights["C"][row]
        C_string = str(C_val).replace('.', '')
        S_val = branch_weights["S"][row].astype(float)
        S_string = str(S_val).replace('.', '')
        def_model  = branch_weights["def_model"][row]
        time_dependence = branch_weights["time_dependence"][row]
        file_suffix = branch_weights["PCDHM_file_suffix"][row]
        total_weight_RN = branch_weights["total_weight_RN"][row]

        # make a unique ID for each branch.
        # The NSHM solution files do not include the rate scaling factor (S) (i.e., they are all S=1)
        # These lines use the same solution file for 3 different S values
        unique_id = f"N{N_string}_b{b_string}_C{C_string}_S{S_string}_{time_dependence}_{def_model}{file_suffix}"

        branch_weight_dict[unique_id] = {"N": N_val, "b": b_val, "C": C_val, "S": S_val, "def_model": def_model,
                                    "time_dependence": time_dependence, "file_suffix": file_suffix, "total_weight_RN":
                                               total_weight_RN}

    return branch_weight_dict

branch_weight_dict = {}
branch_weight_file_path = os.path.relpath(os.path.join(os.path.dirname(__file__), f"../data/branch_weight_data.xlsx"))
for sheet in sheet_list:
    branch_weight_dict = branch_weight_dict | make_branch_weight_dict(branch_weight_file_path=branch_weight_file_path,
                                                                      sheet_name=sheet)

file_suffix_list = []
model_dict_list = []
for key in model_subdirectory_dict.keys():
    file_suffix_list += model_subdirectory_dict[key]
    model_dict_list += [key] * len(model_subdirectory_dict[key])
branch_key = [key for key in branch_weight_dict.keys() if any(["_S10_" in key, "_S1_" in key]) and any([suffix in key for suffix in file_suffix_list])]
branch_label_dict = {}
for key in branch_key:
    branch_label_dict[branch_weight_dict[key]["file_suffix"].strip('_')] = key

PPE_path_list = []
for ix, suffix in enumerate(file_suffix_list):
    PPE_path_list.append(f"../{results_directory}/{model_dict_list[ix]}/sites_{suffix}/{branch_label_dict[suffix]}_cumu_PPE.h5")

# makes the text editable upon export to a pdf
matplotlib.rcParams['pdf.fonttype'] = 42

displacement_threshold_list = [0.2]

title = " vs ".join(pretty_names)
dir_name = "_".join(pretty_names)
dir_name = dir_name.replace(" ", "_")

if slip_taper: slip_taper_extension = "_tapered"
else: slip_taper_extension = "_uniform"

mean_PPE_path_list = []
for name in model_subdirectory_dict.keys():
    for branch in model_subdirectory_dict[name]:
        mean_PPE_path_i = f"../{results_directory}/{name}/sites_{branch}/branch_site_disp_dict_sites_{branch}_S10.h5"
        mean_PPE_path_list.append(mean_PPE_path_i)

outfile_directory = f"{results_directory}/compare_fault_models/{dir_name}"
if not os.path.exists(f"../{outfile_directory}"):
        os.makedirs(f"../{outfile_directory}", exist_ok=True)
file_name = file_id + '_' + sigma_lims

pretty_site_names = []
if plot_order_name == "from_csv":
    print('Using custom plot order from', plot_order_csv)
    plot_order = pd.read_csv(plot_order_csv)
    if 'Pretty Name' in plot_order.columns:
        pretty_site_names = list(plot_order['Pretty Name'])
    plot_order = list(plot_order['siteId'])

elif plot_order_name == "JDE sites":
    plot_order = ["Paraparaumu", "Porirua CBD north", "South Coast", "Wellington Airport", "Wellington CBD", "Petone",
                   "Seaview", "Eastbourne", "Turakirae Head", "Lake Ferry", "Cape Palliser", "Flat Point"]


# Calculate distances along the transect.
# Assumptions:
# 1. The first site is the starting point of the transect.
# 2. The transect is a straight line.
# 3. SiteIds are coordinates in UTM, rounded to km
if transect:
    lonlat = np.array([[float(lon), float(lat)] for lon, lat in (siteId.split('_') for siteId in plot_order)])
    lonlat -= lonlat[0, :]
    distance = np.round(np.sqrt(np.sum(lonlat**2, axis=1))) * 1e-3
    plot_ix = np.argsort(distance)
    plot_order = np.vstack(([plot_order[ix] for ix in plot_ix], distance[plot_ix])).T
    labels_on = False
else:
    plot_order = np.vstack((plot_order, np.arange(len(plot_order)))).T

if probability_plot:
    compare_faultmodel_prob_plot(PPE_paths=PPE_path_list, plot_name=file_name,
                                 outfile_directory=outfile_directory, title=title, pretty_names=pretty_names,
                                 plot_order=plot_order,
                                 labels_on=labels_on,
                                 file_type_list=file_type_list,
                                 threshold=0.2,
                                 transect=transect, site_names=pretty_site_names, sigma=sigma_lims)

if displacement_chart:
    compare_disps_chart(PPE_paths=PPE_path_list, plot_name=file_name, outfile_directory=outfile_directory,
                        title=title, pretty_names=pretty_names,
                        plot_order=plot_order,
                        labels_on=labels_on, file_type_list=file_type_list,
                        transect=transect, site_names=pretty_site_names, plot_bars=False, sigma=sigma_lims)

if compare_hazcurves:
    compare_mean_hazcurves(PPE_paths=PPE_path_list, plot_name=file_name, outfile_directory=outfile_directory,
                           title=title, pretty_names=pretty_names, plot_order=plot_order,
                           file_type_list=file_type_list, site_names=pretty_site_names, sigma=sigma_lims)

if disps_net:
    site_dists = plot_order[:, 1].astype(float) if transect else []
    compare_disps_with_net(PPE_paths=PPE_path_list, plot_name=file_name, outfile_directory=outfile_directory,
                           title=title, pretty_names=pretty_names, site_dists = site_dists,
                           file_type_list=file_type_list, sites=plot_order, transect=transect, site_names=pretty_site_names)
if make_map:
    PPE_dicts = []
    for PPE_path in PPE_path_list:
        for disp in displacement_threshold_list:
            map_and_plot_probabilities(PPE_path=PPE_path,
                                       plot_name=PPE_path.split("/")[-2],
                                       exceed_type=exceed_type,
                                       title=title,
                                       outfile_directory=outfile_directory,
                                       plot_order=plot_order,
                                       labels_on=labels_on,
                                       file_type_list=file_type_list,
                                       threshold=disp,
                                       colorbar_max=0.3,
                                       transect=transect)
