import os
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pandas as pd
from array_operations import read_gmt_grid, read_tiff
import matplotlib.pyplot as plt
from helper_scripts import printProgressBar, get_probability_bar_chart_data, get_exceedance_bar_chart_data
from time import time
import h5py as h5
import xarray as xr


def profile_windows(n_sites, thresholds, n_windows, chunksize, cumu_disp, sigma_lims):
    n_exceedance_up = np.zeros((n_sites, len(thresholds)))
    n_exceedance_down = np.zeros((n_sites, len(thresholds)))
    n_exceedance_abs = np.zeros((n_sites, len(thresholds)))

    for ix, threshold in enumerate(thresholds):
        n_exceedance_up[:, ix] = np.sum(cumu_disp > threshold, axis=1)
        n_exceedance_down[:, ix] = np.sum(cumu_disp < -threshold, axis=1)
        n_exceedance_abs[:, ix] = np.sum(np.abs(cumu_disp) > threshold, axis=1)

    exceedance_probs_up = n_exceedance_up / n_windows
    exceedance_probs_down = n_exceedance_down / n_windows
    exceedance_probs_abs = n_exceedance_abs / n_windows

    # %
    n_chunks = int(n_windows / chunksize)

    chunk_disp = cumu_disp[:, :n_chunks * chunksize].reshape((n_sites, n_chunks, chunksize))
    # %
    err_exceedance_up = np.zeros((n_sites, len(thresholds), len(sigma_lims)))
    err_exceedance_down = np.zeros((n_sites, len(thresholds), len(sigma_lims)))
    err_exceedance_abs = np.zeros((n_sites, len(thresholds), len(sigma_lims)))

    for ix, threshold in enumerate(thresholds):
        err_exceedance_up[:, ix, :] = np.percentile(np.sum(chunk_disp > threshold, axis=2) / chunksize, sigma_lims, axis=1).T
        err_exceedance_down[:, ix, :] = np.percentile(np.sum(chunk_disp < -threshold, axis=2) / chunksize, sigma_lims, axis=1).T
        err_exceedance_abs[:, ix, :] = np.percentile(np.sum(np.abs(chunk_disp) > threshold, axis=2) / chunksize, sigma_lims, axis=1).T

    return exceedance_probs_up, exceedance_probs_down, exceedance_probs_abs, err_exceedance_up, err_exceedance_down, err_exceedance_abs

procDir = 'C:/Users/jmc753/Work/RSQSim/Aotearoa/whole_nz_rsqsim'
siteCSV = '../sites/national_5km_grid_points.csv'
grdDir = 'grds_5km'
catalogue = 'whole_nz_Mw6_5-10_0.csv'
geoTiff = True
timeSpan = 100.
t0 = 1e4
tlen = 1e5
thresholds = '0/10/0.01'   # Dislacement thresholds to use for exceedance probability calculations
chunksize = 100 # Number of windows to include in each subset for error calculation (pseudo-bootstrapping)

recalculate_displacements = False
reload_displacements = False

events_h5file = os.path.join(procDir, os.path.basename(siteCSV).replace('.csv', f'_events_{tlen:.0e}.h5'.replace('+','')))
probs_h5_file = os.path.join(procDir, os.path.basename(siteCSV).replace('.csv', f'_PPE_{tlen:.0e}.h5'.replace('+','')))

if not os.path.exists(events_h5file):
    recalculate_displacements = True

sites_df = pd.read_csv(siteCSV)

if recalculate_displacements:
    # Set paths
    procDir = os.path.abspath(procDir)
    grdDir = os.path.abspath(os.path.join(procDir, grdDir))
    catalogue_csv = os.path.join(procDir, catalogue)

    assert os.path.exists(catalogue_csv), f"Catalogue file {catalogue_csv} not found"


    # Load in grid locations from CSV file
    sites_lon = sites_df.Lon.values
    sites_lat = sites_df.Lat.values

    n_sites = sites_df.shape[0]

    # Load catalogue, trim to contain only the required time window
    catalogue = pd.read_csv(catalogue_csv)
    event_list = catalogue.event_id.values

    time_window = 365.25 * 24 * 3600 * timeSpan
    min_time = t0 * 365.25 * 24 * 3600
    max_time = min_time + tlen * 365.25 * 24 * 3600

    bool_array = np.vstack([np.array(catalogue.t0 >= min_time), np.array(catalogue.t0 <= max_time)])
    catalogue = catalogue[bool_array.all(axis=0)]

    # Assign each event to a time window
    t0 = catalogue['t0'].min()
    n_windows = np.ceil((catalogue['t0'].max() - t0) / time_window).astype(int)
    catalogue['window'] = np.floor((catalogue['t0'] - t0) / time_window).astype(int)

    n_events = len(catalogue.event_id.to_list())
    cumu_disp = np.zeros((n_sites, n_windows))

    not_found = 0
    if not reload_displacements or not os.path.exists(events_h5file):
        print('Regenerating Event Data...')
        printProgressBar(0, n_events, prefix=f"0/{n_events}", suffix='', decimals=1, length=100, fill='█', printEnd="\r")
        disp_array = np.zeros((n_sites, n_events))
        for ix, event in enumerate(catalogue.event_id):
            if geoTiff and os.path.exists(os.path.join(grdDir, f"ev{event}.tif")):
                inlon, inlat, data = read_tiff(os.path.join(grdDir, f"ev{event}.tif"))
            elif os.path.exists(os.path.join(grdDir, f"ev{event}.grd")):
                inlon, inlat, data = read_gmt_grid(os.path.join(grdDir, f"ev{event}.grd"))
            else:
                not_found += 1
                # Many missing events will be because they are in the Kermadecs, and therefore displacement grids were not generated for them
                printProgressBar(0, n_events, prefix=f"0/{n_events}",
                                 suffix=f'\tEvent {event} not found in grid directory ({not_found} (missing [{(not_found / ix) * 100:.02f}%])', decimals=1, length=100, fill='█', printEnd="\r")
                continue
            interp = RegularGridInterpolator((inlat, inlon), data, bounds_error=False, fill_value=None)
            disp_array[:, ix] = interp((sites_lat, sites_lon))
            printProgressBar(ix + 1, n_events, prefix=f"{ix + 1}/{n_events}", suffix='', decimals=1, length=100, fill='█', printEnd="\r")
        with h5.File(events_h5file, 'w') as h5file:
            h5file.create_dataset('disp_array', data=disp_array, compression='gzip')
        del disp_array
    else:
        print('Reloading Event Data...')

    disp_h5 = h5.File(events_h5file, 'r')
        
    windows = catalogue.window.values
    event_number = catalogue.index.values

    print('\nCalculating cumulative window displacements....')
    start = time()
    printProgressBar(0, n_windows, prefix=f'0/{n_windows}', suffix='{0.00 seconds}', decimals=1, length=100, fill='█', printEnd="\r")
    for window in range(n_windows):
        displacements = disp_h5['disp_array'][:, np.isin(catalogue.window.values, window)]
        cumu_disp[:, window] = np.nansum(displacements, axis=1)
        printProgressBar(window + 1, n_windows, prefix=f'{window + 1}/{n_windows}', suffix=f'{time()-start:.2f} seconds', decimals=1, length=100, fill='█', printEnd="\r")
    disp_h5.close()

    cumu_disp[np.isnan(cumu_disp)] = 0
    thesh_min, thresh_max, thresh_step = [np.float64(val) for val in thresholds.split('/')]
    thresholds = np.arange(thesh_min, thresh_max + thresh_step, thresh_step)
    sigma_lims = [0, 2.275, 15.865, 84.135, 97.725, 100]
    # %%
    start = time()
    exceedance_probs_up, exceedance_probs_down, exceedance_probs_abs, err_exceedance_up, err_exceedance_down, err_exceedance_abs = profile_windows(n_sites, thresholds, n_windows, chunksize, cumu_disp, sigma_lims)
    print(f"\nComplete in {time()-start:.5f} seconds")

    # %% Write outputs to hdf5 file

    if os.path.exists(probs_h5_file):
        os.remove(probs_h5_file)

    printProgressBar(0, n_sites, prefix=f"Writing 0/{n_sites} sites to h5file", suffix='', decimals=1, length=100, fill='█', printEnd="\r")
    with h5.File(probs_h5_file, 'w') as h5file:
        h5file.create_dataset('thresholds', data=thresholds)
        h5file.create_dataset('sigma_lims', data=sigma_lims)
        for site_ix in range(n_sites):
            site = f'{round(sites_lon[site_ix] / 1000)}_{round(sites_lat[site_ix] / 1000)}'
            site_group = h5file.create_group(site)
            site_group.create_dataset('site_coords', data=[sites_lon[site_ix], sites_lat[site_ix]])
            site_group.create_dataset('cumu_disp', data=cumu_disp[site_ix, :])
            site_group.create_dataset('exceedance_probs_up', data=exceedance_probs_up[site_ix, :])
            site_group.create_dataset('exceedance_probs_down', data=exceedance_probs_down[site_ix, :])
            site_group.create_dataset('exceedance_probs_abs', data=exceedance_probs_abs[site_ix, :])
            site_group.create_dataset('error_exceedance_up', data=err_exceedance_up[site_ix, :, :])
            site_group.create_dataset('error_exceedance_down', data=err_exceedance_down[site_ix, :, :])
            site_group.create_dataset('error_exceedance_abs', data=err_exceedance_abs[site_ix, :, :])
            printProgressBar(site_ix + 1, n_sites, prefix=f"Writing {site_ix + 1}/{n_sites} sites to h5file", suffix='', decimals=1, length=100, fill='█', printEnd="\r")


print('Add results to x_array datasets, and save as netcdf files')

# Define File Paths
exceed_type_list = ["abs", "up", "down"]

PPEh5 = h5.File(probs_h5_file, 'r')

sites = [*PPEh5.keys()]
metadata_keys = ['thresholds', 'sigma_lims']
for meta in metadata_keys:
    if meta in sites:
        sites.remove(meta)

thresholds = PPEh5["thresholds"]
thresholds = [round(val, 4) for val in thresholds]   # Rounding to try and deal with the floating point errors

site_x = [pixel['site_coords'][:][0] for key, pixel in PPEh5.items() if key not in metadata_keys]
site_y = [pixel['site_coords'][:][1] for key, pixel in PPEh5.items() if key not in metadata_keys]

x_data = np.unique(site_x)
y_data = np.unique(site_y)

xmin, xmax = min(x_data), max(x_data)
ymin, ymax = min(y_data), max(y_data)
x_res, y_res = min(np.diff(x_data)), min(np.diff(y_data))

x_data = np.arange(xmin, xmax + x_res, x_res)
y_data = np.arange(ymin, ymax + y_res, y_res)

site_x = (np.array(site_x) - x_data[0]) / x_res
site_y = (np.array(site_y) - y_data[0]) / y_res

# Create Datasets
da = {}
ds = xr.Dataset()

sigmas = [2.275, 97.725]
sigmas.sort()

out_name = probs_h5_file.replace('.h5', '')

do_disps = False
if do_disps:
    print(f"\tAdding Displacement Probability DataArrays....")

    if not all(np.isin(thresholds, np.round(PPEh5["thresholds"][:], 4))):
        dropped_thresholds = thresholds[np.isin(thresholds, PPEh5["thresholds"][:], invert=True)]
        thresholds = thresholds[np.isin(thresholds, PPEh5["thresholds"][:])]
        if len(thresholds) == 0:
            print('No requested thresholds were in the PPE dictionary. Change requested thresholds')
            pass
        else:
            print('Not all requested thresholds were in PPE dictionary.\nMissing thresholds:\n', dropped_thresholds)
            print('Running available thresholds:\n', thresholds)

    for exceed_type in exceed_type_list:
        thresh_grd = np.zeros([len(thresholds), len(y_data), len(x_data)]) * np.nan
        probs = np.zeros([len(sites), len(thresholds)])
        printProgressBar(0, len(thresholds), prefix=f'\tProcessing 0.00 m', suffix=f'{exceed_type}', length=50)
        for ii, threshold in enumerate(thresholds):
            probs[:, ii] = get_probability_bar_chart_data(site_PPE_dictionary=PPEh5, exceed_type=exceed_type,
                                                        threshold=threshold, site_list=sites, sigmas=sigmas)
            printProgressBar(ii + 1, len(thresholds), prefix=f'\tProcessing {threshold:.2f} m', suffix=f'{exceed_type}', length=50)
        for jj in range(len(sites)):
            thresh_grd[:, int(site_y[jj]), int(site_x[jj])] = probs[jj, :]

        da[exceed_type] = xr.DataArray(thresh_grd, dims=['threshold', 'lat', 'lon'], coords={'threshold': thresholds, 'lat': y_data, 'lon': x_data})
        da[exceed_type].attrs['exceed_type'] = exceed_type
        da[exceed_type].attrs['threshold'] = 'Displacement (m)'
        da[exceed_type].attrs['crs'] = 'EPSG:2193'

        ds['disp_' + exceed_type] = da[exceed_type]
    out_name += '_disp'

do_probs = True
if do_probs:
    print(f"\tAdding Probability Exceedence DataArrays....")
    probs_lims = [0.02, 0.1]
    probs_step = 0.02
    probabilities = np.round(np.arange(probs_lims[0], probs_lims[1] + probs_step, probs_step), 4)
    probabilities = np.array([0.02, 0.1])

    for exceed_type in exceed_type_list:
        thresh_grd = np.zeros([len(probabilities), len(y_data), len(x_data)]) * np.nan
        err_low_grd = np.zeros([len(probabilities), len(y_data), len(x_data)]) * np.nan
        err_high_grd = np.zeros([len(probabilities), len(y_data), len(x_data)]) * np.nan
        disps = np.zeros([len(sites), len(probabilities)])
        disps_err_low = np.zeros([len(sites), len(probabilities)])
        disps_err_high = np.zeros([len(sites), len(probabilities)])
        printProgressBar(0, len(probabilities), prefix=f'\tProcessing 00 %', suffix=f'{exceed_type}', length=50)
        for ii, probability in enumerate(probabilities):
            disps[:, ii], disps_err_low[:, ii], disps_err_high[:, ii] = get_exceedance_bar_chart_data(site_PPE_dictionary=PPEh5, exceed_type=exceed_type,
                                                                                                    site_list=sites, probability=probability, sigmas=sigmas)
            printProgressBar(ii + 1, len(probabilities), prefix=f'\tProcessing {int(100 * probability):0>2} %', suffix=f'{exceed_type}', length=50)
            if exceed_type == 'down':
                disps[:, ii] = -1 * disps[:, ii]
        for jj in range(len(sites)):
            thresh_grd[:, int(site_y[jj]), int(site_x[jj])] = disps[jj, :]
            err_low_grd[:, int(site_y[jj]), int(site_x[jj])] = disps_err_low[jj, :]
            err_high_grd[:, int(site_y[jj]), int(site_x[jj])] = disps_err_high[jj, :]

        da[exceed_type] = xr.DataArray(thresh_grd, dims=['probability', 'lat', 'lon'], coords={'probability': (probabilities * 100).astype(int), 'lat': y_data, 'lon': x_data})
        da[exceed_type].attrs['exceed_type'] = exceed_type
        da[exceed_type].attrs['threshold'] = 'Exceedance Probability (%)'
        da[exceed_type].attrs['crs'] = 'EPSG:2193'
        ds['prob_' + exceed_type] = da[exceed_type]

        da[exceed_type + '_err_high'] = xr.DataArray(err_high_grd, dims=['probability', 'lat', 'lon'], coords={'probability': (probabilities * 100).astype(int), 'lat': y_data, 'lon': x_data})
        da[exceed_type + '_err_high'].attrs['exceed_type'] = exceed_type + '_'+ str(sigmas[1])
        da[exceed_type + '_err_high'].attrs['threshold'] = 'Exceedance Probability (%)'
        da[exceed_type + '_err_high'].attrs['crs'] = 'EPSG:2193'
        ds['prob_' + exceed_type + '_err_high'] = da[exceed_type + '_err_high']

        da[exceed_type + '_err_low'] = xr.DataArray(err_low_grd, dims=['probability', 'lat', 'lon'], coords={'probability': (probabilities * 100).astype(int), 'lat': y_data, 'lon': x_data})
        da[exceed_type + '_err_low'].attrs['exceed_type'] = exceed_type + '_'+ str(sigmas[0])
        da[exceed_type + '_err_low'].attrs['threshold'] = 'Exceedance Probability (%)'
        da[exceed_type + '_err_low'].attrs['crs'] = 'EPSG:2193'
        ds['prob_' + exceed_type + '_err_low'] = da[exceed_type + '_err_low']

    out_name += '_prob'

ds.to_netcdf(out_name + '.nc')

for ii in sites_df.index[sites_df['siteId'] == '1749_5427'].to_list():
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(thresholds, err_exceedance_up[ii, :, 0], color='blue')
    ax.plot(thresholds, err_exceedance_up[ii, :, 1], color='red')
    ax.plot(thresholds, exceedance_probs_up[ii, :], color='black')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim([0.00001, 2])
    ax.set_yticks([0.00001,0.0001, 0.001, 0.01, 0.1, 1])
    ax.set_yticklabels([0.001, 0.01, 0.1,1,10,100])
    ax.set_xlim([0.001, 3])
    ax.set_xticks([0.01,0.1, 1.])
    ax.set_xticklabels([0.01, 0.1, 1])
    plt.show()

print('Complete')