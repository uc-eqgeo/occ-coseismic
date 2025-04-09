import os
import pandas as pd
from probabalistic_displacement_scripts import plot_weighted_mean_haz_curves, plot_single_branch_haz_curves, \
    make_sz_crustal_paired_PPE_dict, make_fault_model_PPE_dict, get_weighted_mean_PPE_dict, \
    save_disp_prob_xarrays
from helper_scripts import get_NSHM_directories, get_rupture_disp_dict
import pickle as pkl
try:
    import geopandas as gpd
except ImportError:
    print("Running on NESI. Site geojsons won't be output....")

os.chdir(os.path.dirname(__file__))
#### USER INPUTS #####
slip_taper = False                           # True or False, only matters if crustal. Defaults to False for sz.
fault_type = "sz"                       # "crustal", "sz" or "py"; only matters for single fault model + getting name of paired crustal subduction pickle files
crustal_mesh_version = "_CFM"           # Name of the crustal mesh model version (e.g. "_CFM", "_CFM_steeperdip", "_CFM_gentlerdip")
crustal_site_names = "_EastCoastNI_10km"   # Name of the sites geojson
sz_site_names = ["_EastCoastNI_5km", "_SouthIsland_5km"]       # Name of the sites geojson
sz_list_order = ["sz", "py"]         # Order of the subduction zones
sz_names = ["hikkerk", "puysegur"]   # Name of the subduction zone
outfile_extension = ""               # Optional; something to tack on to the end so you don't overwrite files
nesi = False   # Prepares code for NESI runs
testing = False   # Impacts number of samples runs, job time etc
fakequakes = True  # Use fakequakes for the subduction zone (applied only to hikkerk)

# Processing Flags (True/False)
single_branch = ["_sz_fq_3lhb110max9", "_sz_fq_plhb110max9"] # Allows you to specifically select which branches to calculate PPEs for. If None, all branches will be calculated
rate_scaling = False           # Do you want to calculate PPEs for a single branch with different rate scalings?
paired_crustal_sz = False      # Do you want to calculate the PPEs for a single fault model or a paired crustal/subduction model?
load_random = True             # Do you want to uses the same grid for scenarios for each site, or regenerate a new grid for each site?
calculate_fault_model_PPE = True   # Do you want to calculate PPEs for each branch?
remake_PPE = False            # Recalculate branch PPEs from scratch, rather than search for pre-existing files (useful if have to stop processing...)
calculate_weighted_mean_PPE = False   # Do you want to weighted mean calculate PPEs?
remake_weighted_PPE = False    # Recalculate weighted branch PPEs from scratch, rather than search for pre-existing files (useful if have to stop processing...)
save_arrays = True         # Do you want to save the displacement and probability arrays?
default_plot_order = True       # Do you want to plot haz curves for all sites, or use your own selection of sites to plot? 
make_hazcurves = False     # Do you want to make hazard curves?
plot_order_csv = "../sites/EastCoastNI_5km_transect_points.csv"  # csv file with the order you want the branches to be plotted in (must contain sites in order under column siteId). Does not need to contain all sites
use_saved_dictionary = True   # Use a saved dictionary if it exists

# Processing Parameters
time_interval = [100]     # Time span of hazard forecast (yrs)
sd = 0.4                # Standard deviation of the normal distribution to use for uncertainty in displacements
n_cpus = 10
thresh_lims = [0, 30]
thresh_step = 0.01

# Nesi Parameters
prep_sbatch = True   # Prep jobs for sbatch
nesi_step = 'combine'  # 'prep' or 'combine'
n_array_tasks = 250    # Number of array tasks
min_tasks_per_array = 250   # Minimum number of sites per array
min_branches_per_array = 1  # Minimum number of branches per array
account = 'uc03610' # NESI account to use

# Parameters that shouldn't need to be changed
crustal_directory = "crustal"
sz_directory = "subduction"
results_directory = "results"
figure_file_type_list = ["png", "pdf"]             # file types for figures
figure_file_type_list = ["png"]

## Set parameters based on user inputs
if testing:
    n_samples = 1e5   # Number of scenarios to run
    job_time = 1    # Amount of time to allocate per site in the cumu_PPE task array
    mem = 1    # Memory allocation for cumu_PPE task array
else:
    n_samples = 1e6   # Number of scenarios to run
    job_time = 3    # Amount of time to allocate per site in the cumu_PPE task array
    mem = 3    # Memory allocation for cumu_PPE task array

if paired_crustal_sz and nesi_step == 'prep':
    if n_array_tasks < 500:
        n_array_tasks = 500
    if job_time < 5:
        job_time = 5

if fault_type == 'crustal' and n_array_tasks < 500:
    n_array_tasks = 500

if fault_type == 'all':
    job_time = 120
    mem = 3
    min_tasks_per_array = 5

if isinstance(sz_site_names, str):
    sz_site_names = [sz_site_names]

n_samples, job_time, mem, n_array_tasks, min_tasks_per_array = int(n_samples), int(job_time), int(mem), int(n_array_tasks), int(min_tasks_per_array)
## Solving processing conflicts
# if calculate_fault_model_PPE and not nesi:
#     calculate_weighted_mean_PPE = True  # If recalculating PPEs, you need to recalculate the weighted mean PPEs

if paired_crustal_sz and calculate_weighted_mean_PPE:
    calculate_fault_model_PPE = True

if nesi and calculate_weighted_mean_PPE and paired_crustal_sz:
    mem = 5

if fakequakes and fault_type == 'crustal' and not paired_crustal_sz:
    fakequakes = False

if fakequakes and all([fault_type != 'all', fault_type != 'sz', ]):
    raise Exception('Fakequakes selected but fault type must be all or sz')

if not nesi and fakequakes:
    load_random = True

if not default_plot_order and not os.path.exists(plot_order_csv) and make_hazcurves:
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

if single_branch is not None:
    if isinstance(single_branch, str):
        single_branch = [single_branch]
    if paired_crustal_sz:
        raise Exception("Can't have single branch and paired crustal sz")
    if calculate_weighted_mean_PPE:
        print("Can't have single branch and calculate weighted mean PPE. Stopping after fault model PPE calculation.")
        calculate_weighted_mean_PPE = False
        save_arrays = False

time_interval = [str(int(interval)) for interval in time_interval]
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
###############################

gf_name = "sites"

if not paired_crustal_sz:
    if fault_type[0] == "crustal":
        site_names_list = [crustal_site_names]
        disc_version_list = [crustal_mesh_version.strip('_')]
    else:
        if len(sz_site_names) > 1:
            sz_ix = sz_list_order.index(fault_type[0])
            sz_site_names = [sz_site_names[sz_ix]]
            sz_disc_version = [f"{sz_names[sz_list_order.index(fault_type[0])]}"]
        else:
            sz_disc_version = sz_names
        if fakequakes and sz_site_names[0][:3] != "_fq":
            sz_disc_version = ["fq_" + sz_disc_version[0]]
        site_names_list = [sz_site_names[0]]
        disc_version_list = [sz_disc_version[0]]
        slip_taper = False        
else:
    if len(sz_site_names) == 1:
        site_names_list = [crustal_site_names] + [sz_site_names] * len(fault_type[1:])
        disc_version_list = [crustal_mesh_version.strip('_')] + sz_names[:1] * len(fault_type[1:])
    elif len(sz_site_names) == len(sz_list_order):
        site_names_list = [crustal_site_names]
        disc_version_list = [crustal_mesh_version.strip('_')]
        for ftype in fault_type[1:]:
            site_names_list += [sz_site_names[sz_list_order.index(ftype)]]
            disc_version_list += [sz_names[sz_list_order.index(ftype)]]
    else:
        raise Exception("Length of sz_site_names must be 1 or equal to the number of subduction fault types")
    if fakequakes:
        sz_ix = 1 + sz_list_order.index('sz')
        site_names_list[sz_ix] = "fq_" + site_names_list[sz_ix]
        disc_version_list[sz_ix] = "fq_" + disc_version_list[sz_ix]

if slip_taper:
    taper_extension = "_tapered"
else:
    taper_extension = "_uniform"

# these directories should already be made from calculating displacements in a previous script
version_discretise_directory = []
for ix, model in enumerate(fault_type):
    version_discretise_directory.append(f"{results_directory}/{disc_version_list[ix]}")

# get branch weights from the saved Excel spreadsheet
branch_weight_file_path = os.path.relpath(os.path.join(os.path.dirname(__file__), f"../data/branch_weight_data.xlsx"))
crustal_sheet_name = "crustal_weights_4_2"
sz_sheet_name = "sz_weights_4_0"
py_sheet_name = "py_weights_4_0"
if fakequakes:
    sz_sheet_name += "_fq"

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

# designate which branch weight dictionary to use based on the fault type
if not paired_crustal_sz:
    print('Checking rupture_disp_dicts...')
    fault_model_branch_weight_dict = {}
    for ii in range(len(fault_type)):
        fault_model_branch_weight_dict = fault_model_branch_weight_dict | branch_weight_dict_list[ii]
    
    NSHM_directory_list, file_suffix_list, n_branches = get_NSHM_directories(fault_type, deformation_model='geologic and geodetic', time_independent=True,
                            time_dependent=True, single_branch=single_branch, fakequakes=fakequakes)
    
    if single_branch:
        branch_keys = list(fault_model_branch_weight_dict.keys())
        if rate_scaling:
            branch_key = [key for key in branch_keys if any([suffix in key[-len(suffix):] for suffix in file_suffix_list])]
        else:
            branch_key = [key for key in branch_keys if any(["_S10_" in key, "_S1_" in key]) and any([suffix in key[-len(suffix):] for suffix in file_suffix_list])]
        fault_model_single_branch_weight_dict = {}
        for key in branch_key:
            fault_model_single_branch_weight_dict = fault_model_single_branch_weight_dict | {key: fault_model_branch_weight_dict[key]}
        fault_model_branch_weight_dict = fault_model_single_branch_weight_dict

    extension1_list = [gf_name + suffix for suffix in file_suffix_list]

    if "crustal" in fault_type:
        site_geojson = f"../crustal/discretised_{version_discretise_directory[0].split('/')[-1]}/{fault_type[0]}_site_locations{crustal_site_names}.geojson"
    else:
        site_geojson = f"../subduction/discretised_{version_discretise_directory[0].split('/')[-1]}/{fault_type[0]}_site_locations{['_fq' if fakequakes else ''][0]}{sz_site_names[0]}.geojson"
    
    if nesi:
        with open(site_geojson, 'r') as f:
            sites = f.readlines()
            inv_sites = [site.split('"')[9] for site in sites if 'siteId' in site]
    else:
        site_gdf = gpd.read_file(site_geojson)
        inv_sites = site_gdf['siteId'].values.tolist()

    for ix, extension1 in enumerate(extension1_list):
        get_rupture_dict = False
        ftype = [(jj, ftype) for jj, ftype in enumerate(fault_type) if '_' + ftype.replace('rustal', '') + '_' in extension1][0]
        all_rupture_disp_file = f"../{version_discretise_directory[ftype[0]]}/{extension1}/all_rupture_disps_{extension1}{taper_extension}_sites.pkl"

        print(f"\n\tbranch {ix + 1} of {len(extension1_list)}: {extension1}")
        if os.path.exists(all_rupture_disp_file):
            with open(all_rupture_disp_file, 'rb') as fid:
                rupt = pkl.load(fid)
            sites = rupt['site_name_list']
            if len(set(inv_sites).intersection(sites)) < len(inv_sites):
                get_rupture_dict = True
        else:
            get_rupture_dict = True

        if get_rupture_dict:
            calculate_fault_model_PPE = True
            get_rupture_disp_dict(NSHM_directory=NSHM_directory_list[ix], extension1=extension1_list[ix],
                                    slip_taper=slip_taper, fault_type=ftype[1], gf_name=gf_name,
                                    results_version_directory=site_names_list[ftype[0]],
                                    disc_version_directory=version_discretise_directory[ftype[0]],
                                    crustal_directory=crustal_directory, sz_directory=sz_directory,
                                    search_radius=9e5, fakequakes=fakequakes)

### make a dictionary of all the branch probabilities, oranized by site within each branch
# option to skip this step if you've already run it once and saved to a pickle file
if not paired_crustal_sz:
    fault_type = fault_type[0]
    out_version_results_directory = version_discretise_directory[ftype[0]]
    PPE_filepath = f"../{out_version_results_directory}/all_branch_PPE_dict{outfile_extension}{taper_extension}.pkl"
    if not os.path.exists(PPE_filepath):
        print('No fault model PPE pkl file found. Making a new one...')
        calculate_fault_model_PPE = True
        use_saved_dictionary = False

    if calculate_fault_model_PPE or not use_saved_dictionary:
        print('\nCreating fault model PPE dictionaries...')
        PPE_dict = make_fault_model_PPE_dict(
                    branch_weight_dict=fault_model_branch_weight_dict,
                    model_version_results_directory=out_version_results_directory, n_samples=n_samples,
                    slip_taper=slip_taper, outfile_extension=outfile_extension, nesi=nesi, nesi_step=nesi_step, sbatch=prep_sbatch, mem=mem,
                    time_interval=time_interval, sd=sd, n_array_tasks=n_array_tasks, min_tasks_per_array=min_tasks_per_array, job_time=job_time,
                    load_random=load_random, remake_PPE=remake_PPE, account=account, thresh_lims=thresh_lims, thresh_step=thresh_step, inv_sites=inv_sites)
    else:
        print('Loading pre-prepared fault model PPE dictionary...')
        with open(PPE_filepath, 'rb') as f:
            PPE_dict = pkl.load(f)

##### paired crustal and sz PPE
if paired_crustal_sz:
    out_version_results_directory = f"{results_directory}/paired_c{crustal_site_names}"
    pickle_prefix = ''
    for sub in fault_type[1:]:
        out_version_results_directory += f"_{sub}{sz_site_names[sz_list_order.index(sub)]}"
        pickle_prefix += f"{sub}_"
    if not os.path.exists(f"../{out_version_results_directory}"):
        os.mkdir(f"../{out_version_results_directory}")
    for ix, branch_weight_dict in enumerate(branch_weight_dict_list):
        with open(f"../{out_version_results_directory}/branch_weight_dict_{fault_type[ix]}.pkl", 'wb') as f:
            pkl.dump(branch_weight_dict, f)

    paired_PPE_pickle_name = f"weighted_mean_PPE_dict{outfile_extension}{taper_extension}.h5"
    PPE_filepath = f"../{out_version_results_directory}/{paired_PPE_pickle_name}"

    if not os.path.exists(PPE_filepath):
        print(f"No crustal-{'-'.join(fault_type[1:])} paired PPE .h5 file found. Making a new one...")
        calculate_fault_model_PPE = True
        use_saved_dictionary = False

    # Get sites that we want to calculate PPE for
    site_geojson = f"../crustal/discretised_{version_discretise_directory[0].split('/')[-1]}/{fault_type[0]}_site_locations{crustal_site_names}.geojson"
    if nesi:
        with open(site_geojson, 'r') as f:
            sites = f.readlines()
            site_gdf = [site.split('"')[9] for site in sites if 'siteId' in site]
    else:
        site_gdf = gpd.read_file(site_geojson)   
    
    for ix, ftype in enumerate(fault_type[1:]):
        fq = '' if not all([fakequakes, ftype == 'sz']) else '_fq'
        site_geojson = f"../subduction/discretised_{version_discretise_directory[ix + 1].split('/')[-1]}/{ftype}_site_locations{fq}{sz_site_names[sz_list_order.index(ftype)]}.geojson"
        if nesi:
            with open(site_geojson, 'r') as f:
                sites = f.readlines()
                site_gdf = list(set(site_gdf + [site.split('"')[9] for site in sites if 'siteId' in site]))
                inv_sites = site_gdf
                
        else:
            sz_gdf = gpd.read_file(site_geojson)
            site_gdf = pd.concat([site_gdf, sz_gdf]).drop_duplicates().reset_index(drop=True)
            inv_sites = site_gdf['siteId'].values.tolist()

    #### skip this part if you've already run it once and saved to a pickle file
    if calculate_fault_model_PPE or not use_saved_dictionary:

        make_sz_crustal_paired_PPE_dict(
            crustal_branch_weight_dict=branch_weight_dict_list[0], sz_branch_weight_dict_list=branch_weight_dict_list[1:],
            crustal_model_version_results_directory=version_discretise_directory[0],
            sz_model_version_results_directory_list=version_discretise_directory[1:],
            slip_taper=slip_taper, n_samples=int(n_samples),
            out_directory=out_version_results_directory, outfile_extension=outfile_extension,
            nesi=nesi, nesi_step=nesi_step, n_array_tasks=n_array_tasks, min_tasks_per_array=min_tasks_per_array,
            mem=mem, time_interval=time_interval, job_time=job_time, remake_PPE=remake_PPE, account=account,
            thresh_lims=thresh_lims, thresh_step=thresh_step, site_gdf=site_gdf)

# calculate weighted mean PPE for the branch or paired dataset
weighted_mean_PPE_filepath = f"../{out_version_results_directory}/weighted_mean_PPE_dict{outfile_extension}{taper_extension}.h5"
if not paired_crustal_sz and calculate_weighted_mean_PPE or not os.path.exists(weighted_mean_PPE_filepath):
    print('Calculating weighted mean PPE...')
    get_weighted_mean_PPE_dict(fault_model_PPE_dict=PPE_dict, out_directory=out_version_results_directory, outfile_extension=outfile_extension,
                               slip_taper=slip_taper, nesi=nesi, nesi_step=nesi_step, account=account, n_samples=n_samples, min_tasks_per_array=10,
                               n_array_tasks=n_array_tasks, mem=mem, cpus=n_cpus, job_time=job_time, thresh_lims=thresh_lims, thresh_step=thresh_step,
                               site_list=inv_sites, remake_PPE=remake_weighted_PPE, time_interval=time_interval)

# plot hazard curves and save to file
if save_arrays:
    if single_branch:
        weighted = False
    else:
        weighted = True
        branch_key = ['']
    for key in branch_key:
        ds = save_disp_prob_xarrays(outfile_extension, slip_taper=slip_taper, model_version_results_directory=out_version_results_directory,
                            thresh_lims=[0, 3], thresh_step=0.05, output_thresh=True, probs_lims = [0.00, 0.20], probs_step=0.01,
                            output_probs=True, weighted=weighted, sites=inv_sites, out_tag=site_names_list[0], single_branch=key,
                            time_intervals=time_interval)

if paired_crustal_sz:
    site_names_title = f"paired crustal{crustal_site_names} and "
    for ix, sub in enumerate(fault_type[1:]):
        site_names_title += f"{sub}{sz_site_names[ix]} and "
    site_names_title = site_names_title[:-5]
else:
    if not isinstance(fault_type, list): 
        fault_type = [fault_type]
    site_names_title = f"{fault_type[0]}{site_names_list[0]}"

if make_hazcurves:
    if default_plot_order:
        site_gdf = gpd.read_file(site_geojson)
        plot_order = site_gdf['siteId'].values.tolist()

    else:
        print('Using custom plot order from', plot_order_csv)
        plot_order = pd.read_csv(plot_order_csv)
        plot_order = list(plot_order['siteId'])

    print(f"\nMaking hazard curves...")
    if single_branch:
        for branch, key in zip(single_branch, branch_key):
            out_dir = f"{out_version_results_directory}/sites{branch}/hazard_curves{outfile_extension}"
            PPE_filepath = f"../{out_version_results_directory}/sites{branch}/{key}_cumu_PPE.h5"
            print(f"Output Directory: {out_dir}")
            for interval in time_interval:
                print('Plotting hazard curves for', branch, 'at', interval, 'years')
                plot_single_branch_haz_curves(
                    PPE_dictionary=PPE_filepath, model_version_title=site_names_title,
                    exceed_type_list=["up", "down", "total_abs"], out_directory=out_dir,
                    file_type_list=figure_file_type_list, slip_taper=slip_taper, plot_order=plot_order, interval=interval)   
    else:
        out_dir =  f"{out_version_results_directory}/weighted_mean_figures"
        print(f"Output Directory: {out_dir}")
        for interval in time_interval:
            plot_weighted_mean_haz_curves(
                weighted_mean_PPE_dictionary=weighted_mean_PPE_filepath,
                model_version_title=site_names_title, exceed_type_list=["up", "down", "total_abs"],
                out_directory=out_version_results_directory, file_type_list=figure_file_type_list, slip_taper=slip_taper, plot_order=plot_order,
                sigma=2, intervals=[interval])
    