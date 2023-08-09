# this file is meant to be run on the server, not locally
# this is why the directories are different

import os
from scipy.io import loadmat, savemat
import numpy as np

# set run id number here
i = 0

# extract windows automatically
csi_dict = loadmat(f"./data/demo_data/csi{i}.mat")

lr_timings = list(range(15,301,15))
pp_timings = list(range(20,306,15))

print("Extracting Windows")
os.makedirs(f"./data/demo_data/extracted_windows_{i}")

count_lr = 0
count_pp = 0
count_ng = 0

for temp in range(54, 1261):
    t = temp/4 # need to increment by 0.25

    status = 0
    for lr in lr_timings:
        if t >= (lr-1.5) and t <= lr:
            status = 1
    if status == 0:
        for pp in pp_timings:
            if t >= (pp-1.5) and t <= pp:
                status = 2
    
    start_idx = np.argmin(np.abs(csi_dict["time"].squeeze() - 5 - t))
    end_idx = start_idx + 400
    window = csi_dict["csi_amp"][start_idx:end_idx,:]
    if status == 1:
        savemat(f"./data/demo_data/extracted_windows_{i}/lr_run{i}_rep{count_lr}.mat", {"window": window})
        count_lr = count_lr + 1
    elif status == 2:
        savemat(f"./data/demo_data/extracted_windows_{i}/pp_run{i}_rep{count_pp}.mat", {"window": window})
        count_pp = count_pp + 1
    else:
        savemat(f"./data/demo_data/extracted_windows_{i}/ng_run{i}_rep{count_ng}.mat", {"window": window})
        count_ng = count_ng + 1
    
print(count_lr, count_pp, count_ng)
print("FINISHED")