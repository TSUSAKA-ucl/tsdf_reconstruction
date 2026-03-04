#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# filename = "tsdf_snapshot.h5"   # ←変更してください
filename = sys.argv[1] if len(sys.argv) > 1 else "tsdf_snapshot.h5"

start_total = time.time()

# --- HDF5読み込み ---
start = time.time()
with h5py.File(filename, "r") as f:
    tsdf = f["tsdf"][:]      # float32
    weight = f["weight"][:]  # float32
print("Load time:", time.time() - start, "sec")

# --- flatten（高速） ---
tsdf_flat = tsdf.ravel()
weight_flat = weight.ravel()

# --- weight>0 mask ---
mask = weight_flat > 0

# --- 統計 ---
total_voxels = tsdf_flat.size
observed_voxels = np.count_nonzero(mask)

tsdf_obs = tsdf_flat[mask]

pos_count = np.count_nonzero(tsdf_obs > 0)
neg_count = np.count_nonzero(tsdf_obs < 0)
zero_count = np.count_nonzero(tsdf_obs == 0)

print("----- Statistics -----")
print("Total voxels:", total_voxels)
print("Weight>0 voxels:", observed_voxels)
print("TSDF positive:", pos_count)
print("TSDF negative:", neg_count)
print("TSDF zero:", zero_count)

# ===============================
# Histogram 1: Weight
# ===============================
plt.figure()
plt.hist(weight_flat[mask], bins=100)
plt.title("Weight Histogram (weight>0)")
plt.xlabel("weight")
plt.ylabel("count")
plt.show()

# ===============================
# Histogram 2: TSDF (observed only)
# ===============================
plt.figure()
plt.hist(tsdf_obs, bins=100)
plt.title("TSDF Histogram (weight>0)")
plt.xlabel("tsdf")
plt.ylabel("count")
plt.show()

print("Total elapsed:", time.time() - start_total, "sec")
