import h5py as h5
import os
from glob import glob
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import subprocess

"""
Reduce the size of the final cumu PPE h5 files.
This can include removing sites from the cumulative PPE h5 files, but h5 files will require repacking to actually reduce in size on disk.
Sites to be kept are listed in the keep geojson file, which should have a 'siteId' property that matches the site names in the h5 files.
Useful if you are trying to reduce the number of pairs when running paired crustal-subduction
"""

results_dir = 'CFM'
fault_type = 'crustal'
keep_geojson = 'v0-0-1_geoval'
h5_search_term = f'*_cumu_PPE.h5'
max_workers = 12   # tune to your I/O bandwidth
force_repack = False

keep_geojson = os.path.join('.', fault_type, f'discretised_{results_dir}', f'{fault_type}_site_locations_{keep_geojson}.geojson')
results_dir = os.path.join('.', 'results', results_dir)

sites = set(gpd.read_file(keep_geojson)['siteId'])
print(f"{len(sites)} to keep in {keep_geojson}...\n")

h5_search_files = os.path.join(results_dir, 'sites*', h5_search_term)
cumu_h5_list = glob(h5_search_files)

print(f"{len(cumu_h5_list)} cumulative PPE files to process from {h5_search_files}...\n")

META_KEYS = {'branch_weight', 'thresholds'}


# ── Per-file worker (runs in a subprocess) ────────────────────────────────────
def process_file(cumu_h5: str, sites: set, force_repack: bool) -> str:
    """Delete unwanted sites and repack a single h5 file. Returns a status string."""
    with h5.File(cumu_h5, 'a') as f:
        h5_sites  = set(f.keys()) - META_KEYS
        to_remove = h5_sites - sites
 
        if not to_remove:
            if not force_repack:
                return f"[SKIP]  {cumu_h5} — nothing to remove"
        else:
            for site in to_remove:
                del f[site]
 
    # Repack atomically: write to a temp file beside the original, then replace.
    # Using a sibling temp file keeps the rename on the same filesystem (instant).
    dir_, base = os.path.split(cumu_h5)
    size_before = os.path.getsize(cumu_h5) / 1024**3
 
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix='.repack_tmp.h5')
    os.close(fd)
    try:
        subprocess.run(
            ["h5repack", cumu_h5, tmp_path],
            check=True,
            capture_output=True,
        )
        size_after = os.path.getsize(tmp_path) / 1024**3
        os.replace(tmp_path, cumu_h5)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
 
    removed = len(to_remove) if to_remove else 0
    return (
        f"[DONE]  {base} — removed {removed} sites, "
        f"{size_before:.2f} GB → {size_after:.2f} GB "
        f"(saved {size_before - size_after:.2f} GB)"
    )
 
 
# ── Parallel dispatch ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    max_workers   = min(max_workers, os.cpu_count(), len(cumu_h5_list)) 
    if max_workers > 1:
        print(f"Processing with {max_workers} parallel workers...\n")
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(process_file, path, sites, force_repack): path
                for path in cumu_h5_list
            }
            for future in as_completed(futures):
                path = futures[future]
                try:
                    print(future.result())
                except Exception as exc:
                    print(f"[ERROR] {os.path.basename(path)}: {exc}")
    else:
        for path in cumu_h5_list:
            try:
                print(process_file(path, sites, force_repack))
            except Exception as exc:
                print(f"[ERROR] {os.path.basename(path)}: {exc}")
    
    print("\nAll done.")