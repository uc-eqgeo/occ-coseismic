import h5py as h5
import os
from glob import glob
import geopandas as gpd

"""
Remove sites from the cumulative PPE h5 files that are not in a given geojson file.
Useful if you are trying to reduce the number of pairs when running paired crustal-subduction
"""

results_dir = 'fq_hikkerm'
fault_type = 'subduction'
keep_geojson = 'sz_site_locations_fq_version_0-1N.geojson'


keep_geojson = os.path.join('.', fault_type, f'discretised_{results_dir}', keep_geojson)
results_dir = os.path.join('.', 'results', results_dir)
cumu_h5_list = glob(os.path.join(results_dir, 'sites*', '*_cumu_PPE.h5'))


sites = gpd.read_file(keep_geojson)
sites = set(sites['siteId'].values)

for cumu_h5 in cumu_h5_list:
    print(f'Processing {cumu_h5}')
    removed_sites = 0
    kept_sites = 0
    with h5.File(cumu_h5, 'a') as f:
        h5_sites = list(f.keys())
        for meta in ['branch_weight', 'thresholds']:
            if meta in h5_sites:
                h5_sites.remove(meta)
        for site in h5_sites:
            if site not in sites:
                removed_sites += 1
                del f[site]
            else:
                kept_sites += 1
    print(f'Removed {removed_sites} sites, kept {kept_sites} sites')
