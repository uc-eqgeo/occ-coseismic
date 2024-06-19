import pickle as pkl
import pandas as pd
import random
import os
import matplotlib
from shapely.geometry import Point
from helper_scripts import get_rupture_disp_dict, save_target_rates
from rupture_scenario_plotting_scripts import vertical_disp_figure
from probabalistic_displacement_scripts import get_site_disp_dict, get_cumu_PPE, plot_branch_hazard_curve, \
    make_10_2_disp_plot, make_branch_prob_plot, save_10_2_disp , \
    make_branch_prob_grid
    # plot_cumu_disp_hazard_map

##### USER INPUTS   #####
# must run crustal and subduction lists/loops separately
results_directory = "results"

slip_taper = False                    # True or False, only matter if crustal otherwise it defaults to false later.
fault_type = "py"                  # "crustal or "sz" or "py"

# How many branches do you want to run?
# True or False; this just picks the most central branch (geologic, time independent, mid b and N) for crustal
single_branch = False

# True: Skip making a random sample of rupture IDs and just use the ones you know we want to look at
# False: Make a random sample of rupture IDs
specific_rupture_ids = True

#can only run one type of GF and fault geometry at a time
gf_name = "grid"                       # "sites" or "grid" or "coastal"

crustal_model_extension = "_Model_CFM_50km"         # "_Model1", "_Model2", or "_CFM"
sz_model_version = "_southland_10km"                # must match suffix in the subduction directory with gfs

default_plot_order = True
plot_order_csv = "../national_10km_grid_points_trim.csv"  # csv file with the order you want the branches to be plotted in (must contain sites in order under column siteId). Does not need to contain all sites

if default_plot_order:
    plot_order = None
elif not default_plot_order and not os.path.exists(plot_order_csv):
    print("Manual plot order selected but no plot order csv found. Please create a csv file with the order you want the branches to be plotted in (must contain sites in order under column siteId)")
    exit()
else:
    print('Using custom plot order from', plot_order_csv)
    plot_order = pd.read_csv(plot_order_csv)
    plot_order = list(plot_order['siteId'])

# Can run more than one type of deformation model at a time (only matters for crustal)
deformation_model = "geologic and geodetic"          # "geologic" or "geodetic" or "geologic and geodetic"

# can process both time dependent and independent (only matters for crustal)
time_dependent = True       # True or False
time_independent = True     # True or False

# Just want to make some figures?
# False: calculate displacements and probabilities and saves them as dictionaries (and continues with figures)
# True: uses saved displacement and probability dictionaries to make probability and displacement figures
only_make_figures = False
file_type_list=["png", "pdf"]

# Skip the displacements and jump to probabilities
# True: this skips calculating displacements and making displacement figures (assumes you've already done it)
# False: this calculates displacements (and makes disp figures) and probabilities
skip_displacements = False
calculate_cumu_PPE = False

testing = True

# this makes so when you export fonts as pdfs, they are editable in Illustrator
matplotlib.rcParams['pdf.fonttype'] = 42

# Set up which branches you want to calculate displacements and probabilities for
file_suffix_list = []
NSHM_directory_list = []
if fault_type == "crustal":
    model_version = crustal_model_extension
    if time_independent and not single_branch:
        if "geologic" in deformation_model:
            file_suffix_list_i = ["_c_MDA2", "_c_MDEz", "_c_MDE1", "_c_MDA3", "_c_MDA4", "_c_MDA5", "_c_MDEw",
                                  "_c_MDEx", "_c_MDEy"]
            NSHM_directory_list_i = ["crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDA2",
                                   "crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDEz",
                                   "crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDE1",
                                   "crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDA3",
                                   "crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDA4",
                                   "crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDA5",
                                   "crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDEw",
                                   "crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDEx",
                                   "crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDEy",
                                    ]
            file_suffix_list.extend(file_suffix_list_i)
            NSHM_directory_list.extend(NSHM_directory_list_i)
        if "geodetic" in deformation_model:
            file_suffix_list_i = ["_c_MDE2", "_c_MDE3", "_c_MDE4", "_c_MDE5", "_c_MDIw", "_c_MDIx", "_c_MDIz",
                                  "_c_MDIy", "_c_MDI0"]
            NSHM_directory_list_i = ["crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDE2",
                                     "crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDE3",
                                     "crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDE4",
                                     "crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDE5",
                                     "crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDIw",
                                     "crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDIx",
                                     "crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDIz",
                                     "crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDIy",
                                     "crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDI0"]
            file_suffix_list.extend(file_suffix_list_i)
            NSHM_directory_list.extend(NSHM_directory_list_i)

    if time_dependent and not single_branch:
        if "geologic" in deformation_model:
            file_suffix_list_i = ["_c_NjE5", "_c_MjIw", "_c_MjIx", "_c_NjIy", "_c_NjIz", "_c_NjI0", "_c_NjI1",
                                  "_c_NjI2", "_c_NjI3"]
            NSHM_directory_list_i = ["crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjE5",
                                     "crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjIw",
                                     "crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjIx",
                                     "crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjIy",
                                     "crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjIz",
                                     "crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjI0",
                                     "crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjI1",
                                     "crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjI2",
                                     "crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjI3"]
            file_suffix_list.extend(file_suffix_list_i)
            NSHM_directory_list.extend(NSHM_directory_list_i)
        if "geodetic" in deformation_model:
            file_suffix_list_i = ["_c_NjI5", "_c_NjMw", "_c_NjMx", "_c_NjMy", "_c_NjMz", "_c_NjM0", "_c_NjM1",
                                  "_c_NjM2", "_c_NjM3"]
            NSHM_directory_list_i = ["crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjI5",
                                     "crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjMw",
                                     "crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjMx",
                                     "crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjMy",
                                     "crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjMz",
                                     "crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjM0",
                                     "crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjM1",
                                     "crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjM2",
                                     "crustal_solutions/NZSHM22_TimeDependentInversionSolution-QXV0b21hdGlvblRhc2s6MTExNjM3"]
            file_suffix_list.extend(file_suffix_list_i)
            NSHM_directory_list.extend(NSHM_directory_list_i)

    if single_branch:
        file_suffix_list_i = ["_c_MDEz"]
        NSHM_directory_list_i = ["crustal_solutions/NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTA3MDEz"]
        file_suffix_list.extend(file_suffix_list_i)
        NSHM_directory_list.extend(NSHM_directory_list_i)

elif fault_type == "sz":
    model_version = sz_model_version
    slip_taper = False
    if not single_branch:
        file_suffix_list_i = ["_sz_NJk2", "_sz_NTE2",  "_sz_NzE0"]
        NSHM_directory_list_i = ["sz_solutions/NZSHM22_ScaledInversionSolution-QXV0b21hdGlvblRhc2s6MTA3Njk2",
                                "sz_solutions/NZSHM22_ScaledInversionSolution-QXV0b21hdGlvblRhc2s6MTA3NzEx",
                                "sz_solutions/NZSHM22_ScaledInversionSolution-QXV0b21hdGlvblRhc2s6MTA3NzE0"]

        file_suffix_list.extend(file_suffix_list_i)
        NSHM_directory_list.extend(NSHM_directory_list_i)
    if single_branch:
        file_suffix_list_i = ["_sz_NzE0"]  # "_sz_NzE0" Highest weighted Branch
        NSHM_directory_list_i = ["sz_solutions/NZSHM22_ScaledInversionSolution-QXV0b21hdGlvblRhc2s6MTA3NzE0"]
        file_suffix_list.extend(file_suffix_list_i)
        NSHM_directory_list.extend(NSHM_directory_list_i)
elif fault_type == "py":
    model_version = sz_model_version
    slip_taper = False
    if not single_branch:
        file_suffix_list_i = ["_py_M5NQ"]
        NSHM_directory_list_i = ["sz_solutions/NZSHM22_ScaledInversionSolution-QXV0b21hdGlvblRhc2s6MTMyNzM5NQ=="]
        file_suffix_list.extend(file_suffix_list_i)
        NSHM_directory_list.extend(NSHM_directory_list_i)
    if single_branch:
        file_suffix_list_i = ["_py_M5NQ"]
        NSHM_directory_list_i = ["sz_solutions/NZSHM22_ScaledInversionSolution-QXV0b21hdGlvblRhc2s6MTMyNzM5NQ=="]
        file_suffix_list.extend(file_suffix_list_i)
        NSHM_directory_list.extend(NSHM_directory_list_i)

if len(file_suffix_list) != len(NSHM_directory_list):
    raise ValueError("Number of file suffixes and NSHM directories must be equal")

extension1_list = [gf_name + suffix for suffix in file_suffix_list]
crustal_directory = "crustal"
sz_directory = "subduction"
model_version_results_directory = f"{results_directory}/{fault_type}{model_version}"
if gf_name == "grid":
    grid = True
else:
    grid = False

if not os.path.exists(f"../{results_directory}"):
    os.mkdir(f"../{results_directory}")
if not os.path.exists(f"../{model_version_results_directory}"):
    os.mkdir(f"../{model_version_results_directory}")

if only_make_figures is False and skip_displacements is False:
    Wellington = Point(1749150, 5428092)  # Wellington coordinates in NZTM
    # Calculate displacements and make displacement dictionary once per branch. Save to pickle file in branch directory.
    for i in range(len(extension1_list)):
        print (f"\nbranch {i} of {len(extension1_list)}")
        get_rupture_disp_dict(NSHM_directory=NSHM_directory_list[i], extension1=extension1_list[i],
                              slip_taper=slip_taper, fault_type=fault_type, gf_name=gf_name,
                              results_version_directory=model_version_results_directory,
                              crustal_directory=crustal_directory, sz_directory=sz_directory,
                              model_version=model_version, location=Wellington, search_radius=2.5e6)

### make vertical displacement figures (random sample of ~10 ruptures per branch)
print(f"\nOut Directory: {model_version_results_directory}")
if skip_displacements is False:
    if slip_taper:
        taper_extension = "_tapered"
    else:
        taper_extension = "_uniform"
    for i in range(len(extension1_list)):
        with open(f"../{model_version_results_directory}/{extension1_list[i]}/all_rupture_disps_{extension1_list[i]}"
                  f"{taper_extension}.pkl",
                   "rb") as fid:
            all_ruptures_disp_dict = pkl.load(fid)
        rupture_id = list(all_ruptures_disp_dict.keys())

        if specific_rupture_ids:
            target_rupture_ids = []
        else:
            if fault_type == "crustal": num_ruptures = 5
            if fault_type == "sz" or fault_type == "py": num_ruptures = 10
            target_rupture_ids = random.sample(rupture_id, num_ruptures)

        # tack on a few rupture scenarios that we want to see results from. Only include if they are in the rupture list
        if fault_type == "sz":
            if 948 in rupture_id: target_rupture_ids.append(948)
        elif fault_type == 'py':
            if 247 in rupture_id: target_rupture_ids.append(247)
        elif fault_type == "crustal":
            ids = [20890, 96084, 97010, 166970, 305270, 368024, 375389, 401491]
            for id in ids:
                if id in rupture_id: target_rupture_ids.append(id)
        if len(target_rupture_ids) == 0:
            print(f"No target rupture IDs found for {extension1_list[i]}")
            quit()

        # make figures and save rates
        print(f"\n*~ Making displacement figures for {extension1_list[i]} ~*")
        vertical_disp_figure(NSHM_directory=NSHM_directory_list[i], all_ruptures_disp_dict=all_ruptures_disp_dict,
                             target_rupture_ids=target_rupture_ids, extension1=extension1_list[i],
                             extent="same", slip_taper=slip_taper, grid=grid, fault_type=fault_type,
                             results_version_directory=model_version_results_directory, crustal_directory=crustal_directory,
                             sz_directory=sz_directory, model_version=model_version, file_type_list=file_type_list, save_arrays=False)

        # save_target_rates(NSHM_directory_list[i], target_rupture_ids=target_rupture_ids, extension1=extension1_list[i],
        #                   results_version_directory=model_version_results_directory)

if testing:
    n_samples = 1e4
else:
    n_samples = 1e6

if gf_name == "sites":
    ## calculate rupture branch probabilities and make plots
    for i in range(len(extension1_list)):

        print(f"*~ Processing site information for {extension1_list[i]} ~*")

        taper_extension = "_tapered" if slip_taper else "_uniform"
        pkl_file = f"../{model_version_results_directory}/{extension1_list[i]}/cumu_exceed_prob_{extension1_list[i]}{taper_extension}.pkl"

        if not os.path.exists(pkl_file) or calculate_cumu_PPE:
            print('\tMaking exceedence probability dictionary for each site...')
            ## step 1: get site displacement dictionary
            branch_site_disp_dict = get_site_disp_dict(extension1_list[i], slip_taper=slip_taper,
                                model_version_results_directory=model_version_results_directory)

            ### step 2: get exceedance probability dictionary
            get_cumu_PPE(extension1=extension1_list[i], branch_site_disp_dict=branch_site_disp_dict,
                        model_version_results_directory=model_version_results_directory, slip_taper=slip_taper,
                        time_interval=100, n_samples=n_samples)  # n_samples reduced from 1e6 for testing speed

        ## step 3 (optional): plot hazard curves
        print(f"*~ Making probability figures~*")
        print(f"\tBranch Hazard Curves....")
        plot_branch_hazard_curve(extension1=extension1_list[i],
                            model_version_results_directory=model_version_results_directory,
                            slip_taper=slip_taper, file_type_list=file_type_list, plot_order=plot_order)

        # step 4 (optional): plot hazard maps (Needs to be imported from subduction/sz_probability_plotting_scripts.py)
        #plot_cumu_disp_hazard_map(extension1=extension1_list[i], slip_taper=slip_taper, grid=grid, fault_type=fault_type,
        #                          model_version_results_directory=model_version_results_directory,
        #                          crustal_directory=crustal_directory,
        #                          sz_directory=sz_directory, model_version=model_version)

        ## step 5: plot bar charts
        max_sites = 12  # Max number of sites to show on one bar chart [default 12]
        print(f"\tBranch Probability Plots....")
        make_branch_prob_plot(extension1_list[i], slip_taper=slip_taper, threshold=0.2,
                            model_version_results_directory=model_version_results_directory,
                            model_version=model_version, plot_order=plot_order, max_sites=max_sites)

        print(f"\t10/2 Displacement Plots....")
        make_10_2_disp_plot(extension1=extension1_list[i], slip_taper=slip_taper,
                                 model_version_results_directory=model_version_results_directory,
                                 file_type_list=["png", "pdf"], probability_list=[0.1, 0.02],
                                 plot_order=plot_order, max_sites=max_sites)

        print('\tWriting 10/2 Displacement to geoJSON...\n')
        save_10_2_disp(extension1=extension1_list[i], slip_taper=slip_taper,
                                    model_version_results_directory=model_version_results_directory)

if gf_name == "grid":
    ## calculate rupture branch probabilities and make plots
    for i in range(len(extension1_list)):

        print(f"*~ Processing site information for {extension1_list[i]} ~*")

        taper_extension = "_tapered" if slip_taper else "_uniform"
        pkl_file = f"../{model_version_results_directory}/{extension1_list[i]}/cumu_exceed_prob_{extension1_list[i]}{taper_extension}.pkl"

        if not os.path.exists(pkl_file) or calculate_cumu_PPE:
            print('\tMaking exceedence probability dictionary for each site...')
            ## step 1: get site displacement dictionary (where each site is a grid pixel)
            branch_site_disp_dict = get_site_disp_dict(extension1_list[i], slip_taper=slip_taper,
                                model_version_results_directory=model_version_results_directory)

            ### step 2: get exceedance probability dictionary
            get_cumu_PPE(extension1=extension1_list[i], branch_site_disp_dict=branch_site_disp_dict,
                        model_version_results_directory=model_version_results_directory, slip_taper=slip_taper,
                        time_interval=100, n_samples=n_samples)  # n_samples reduced from 1e6 for testing speed

        ## step 3 (optional): plot hazard curves
        print(f"*~ Making probability figures~*")
        print(f"\tBranch Hazard Curves....")
        #plot_branch_hazard_curve(extension1=extension1_list[i],
        #                    model_version_results_directory=model_version_results_directory,
        #                    slip_taper=slip_taper, file_type_list=file_type_list, plot_order=plot_order)

        # step 4 (optional): plot hazard maps (Needs to be imported from subduction/sz_probability_plotting_scripts.py)
        #plot_cumu_disp_hazard_map(extension1=extension1_list[i], slip_taper=slip_taper, grid=grid, fault_type=fault_type,
        #                          model_version_results_directory=model_version_results_directory,
        #                          crustal_directory=crustal_directory,
        #                          sz_directory=sz_directory, model_version=model_version)

        ## step 5: plot bar charts
        max_sites = 12  # Max number of sites to show on one bar chart [default 12]
        print(f"\tBranch Probability Plots....")
        make_branch_prob_grid(extension1_list[i], slip_taper=slip_taper, threshold=0.2,
                            model_version_results_directory=model_version_results_directory,
                            model_version=model_version, plot_order=plot_order, max_sites=max_sites)

        print(f"\t10/2 Displacement Plots....")
        make_10_2_disp_plot(extension1=extension1_list[i], slip_taper=slip_taper,
                                 model_version_results_directory=model_version_results_directory,
                                 file_type_list=["png", "pdf"], probability_list=[0.1, 0.02],
                                 plot_order=plot_order, max_sites=max_sites)

        print('\tWriting 10/2 Displacement to geoJSON...\n')
        save_10_2_disp(extension1=extension1_list[i], slip_taper=slip_taper,
                                    model_version_results_directory=model_version_results_directory)