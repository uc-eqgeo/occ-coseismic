
import os
import h5py as h5
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib
from pcdhm.weighted_mean_plotting import map_and_plot_probabilities
from pcdhm.compare_fault_model import compare_faultmodel_prob_plot, compare_disps_chart, compare_mean_hazcurves, \
    compare_disps_with_net

########## USER INPUTS #######################
plot_order_name = "from_csv"                 # "JDE sites", "from_csv", or "default"
results_directory = "results"
exceed_type = "down"                     # "down", "up", or "total_abs"
slip_taper = False

# Choose what models to compare. These names should be in the results folder already.
model_subdirectory_names = ["CFM", "hikkerk"]
#model_subdirectory_names = ["crustal_CFM","crustal_Model1", "crustal_Model2"]

# used for plot labels/titles. must be in same order as model_subdirectory_names
pretty_names = ["Crustal", "NZNSHM SZ"]
#pretty_names = ["crustal_CFM", "crustal_Model1", "crustal_Model2"]

file_type_list = ["png"]     # generally png and/or pdf
probability_plot = True             # plots the probability of exceedance at the 0.2 m uplift and subsidence thresholds
displacement_chart = True           # plots the displacement at the 10% and 2% probability of exceedance thresholds
compare_hazcurves = True        # plots the different hazard curves on the same plot
make_map = True
disps_net = True
labels_on = False                # displacement number labels for bar charts and probability plots

plot_order_csv = "../sites/JDE_sites_12.csv"  # csv file with site order

#### script ###################
# makes the text editable upon export to a pdf
matplotlib.rcParams['pdf.fonttype'] = 42

displacement_threshold_list = [0.2]

title = " vs ".join(pretty_names)
file_name = "_".join(pretty_names)
file_name = file_name.replace(" ", "_")

if slip_taper: slip_taper_extension = "_tapered"
else: slip_taper_extension = "_uniform"

mean_PPE_path_list = []
for name in model_subdirectory_names:
    mean_PPE_path_i = f"../{results_directory}/{name}/weighted_mean_PPE_dict{slip_taper_extension}.h5"
    mean_PPE_path_list.append(mean_PPE_path_i)

outfile_directory = f"{results_directory}/compare_fault_models/{file_name}"
if not os.path.exists(f"../{outfile_directory}"):
        os.makedirs(f"../{outfile_directory}", exist_ok=True)


if plot_order_name == "from_csv":
    print('Using custom plot order from', plot_order_csv)
    plot_order = pd.read_csv(plot_order_csv)
    plot_order = list(plot_order['siteId'])
elif plot_order_name == "JDE sites":
    plot_order = ["Paraparaumu", "Porirua CBD north", "South Coast", "Wellington Airport", "Wellington CBD", "Petone",
                   "Seaview", "Eastbourne", "Turakirae Head", "Lake Ferry", "Cape Palliser", "Flat Point"]
else:
    weighted_mean_PPE_dict = h5.File(mean_PPE_path_list[0], 'r')
    plot_order = [key for key in weighted_mean_PPE_dict.keys() if key not in ["branch_weights", "branch_ids", "thresholds", "sigma_lims", "threshold_vals"]]
    weighted_mean_PPE_dict.close()

if probability_plot:
    compare_faultmodel_prob_plot(PPE_paths=mean_PPE_path_list, plot_name=file_name,
                                 outfile_directory=outfile_directory, title=title, pretty_names=pretty_names,
                                 plot_order=plot_order,
                                 labels_on=labels_on,
                                 file_type_list=file_type_list,
                                 threshold=0.2)

if displacement_chart:
    compare_disps_chart(PPE_paths=mean_PPE_path_list, plot_name=file_name, outfile_directory=outfile_directory,
                        title=title, pretty_names=pretty_names,
                        plot_order=plot_order,
                        labels_on=labels_on, file_type_list=file_type_list)

if compare_hazcurves:
    compare_mean_hazcurves(PPE_paths=mean_PPE_path_list, plot_name=file_name, outfile_directory=outfile_directory,
                           title=title, pretty_names=pretty_names, exceed_type=exceed_type,
                           plot_order=plot_order,
                           file_type_list=file_type_list)

if disps_net:
    compare_disps_with_net(PPE_paths=mean_PPE_path_list, plot_name=file_name, outfile_directory=outfile_directory,
                           title=title, pretty_names=pretty_names,
                           file_type_list=file_type_list, sites=plot_order)
if make_map:
    PPE_dicts = []
    for PPE_path in mean_PPE_path_list:
        for disp in displacement_threshold_list:
            map_and_plot_probabilities(PPE_path=PPE_path,
                                       plot_name=file_name,
                                       exceed_type=exceed_type,
                                       title=title,
                                       outfile_directory=outfile_directory,
                                       plot_order=plot_order,
                                       labels_on=True,
                                       file_type_list=file_type_list,
                                       threshold=disp,
                                       colorbar_max=0.3)
