import os

directory = 'sz_wellington_1km'
directory = 'crustal_Model_CFM_wellington_1km'

site_file = 'site_name_list.txt'


site_dir = f'../results/{directory}'

with open(f'{site_dir}/{site_file}', "r") as f:
    all_sites = f.read().splitlines()

print(f'Checking {len(all_sites)} jobs in {directory} for missing .h5 files')

missing_h5 = []
n_missing = 0
print('Missing 0/0....', end='\r')
for ix, site_details in enumerate(all_sites):
    site, branch_dir, scaling = site_details.split()
    if not os.path.exists(f'../{branch_dir}/site_cumu_exceed{scaling}/{site}.h5'):
        missing_h5.append([ix, site, branch_dir, scaling])
        n_missing += 1
    
    print(f'Missing {n_missing}/{ix + 1}....', end='\r')


with open(f'{site_dir}/missing_h5.txt', 'w') as f:
    for ix, site, branch_dir, scaling in missing_h5:
        f.write(f'{ix} ../{branch_dir}/site_cumu_exceed{scaling}/{site}.h5\n')

with open(f'{site_dir}/missing_{site_file}', 'w') as f:
    for ix, site, branch_dir, scaling in missing_h5:
        f.write(f'{site} {branch_dir} {scaling}\n')

print(f'Missing {len(missing_h5)}/{len(all_sites)} .h5 files: {site_dir}/missing_h5.txt')
