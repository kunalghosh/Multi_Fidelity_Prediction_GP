import sys
import numpy as np

def get_directories(arg_list):
    print(len(sys.argv))
    if len(sys.argv) == 1:
        """ Need atleast one """
        print("Need atleast one other directory to compare against")
        sys.exit(1)
    elif len(sys.argv) == 2:
        """ Set the provided directory as the other. Use the current directory as reference."""
        reference = "."
        other_dir = sys.argv[1]
    elif len(sys.argv) == 3:
        """ First one as refence and the second one as the other."""
        reference = sys.argv[1]
        other_dir = sys.argv[2]
    print(f"Reference dir = {reference}, Other directory = {other_dir}")
    return reference, other_dir

reference, other_dir = get_directories(sys.argv)

data = np.load(f"{reference}/test-A-1k_1_full_idxs.npz")['prediction_idxs']
data2 = np.load(f"{reference}/test-A-1k_2_full_idxs.npz")['prediction_idxs']
print(f"len set diff {len(set(data2).difference(data))} If the previous set has elements then two iterations of the new code produce different indices (this is desirable.)")

old_data = np.load(f"{other_dir}/test-A-1k_1_full_idxs.npz")['prediction_idxs']
print(f"len set diff {len(set(data).difference(old_data))} If the previous set is empty then index set 1 is same between old and new")

old_data2 = np.load(f"{other_dir}/test-A-1k_2_full_idxs.npz")['prediction_idxs']
print(f"len set diff {len(set(data2).difference(old_data2))} If the previous set is empty then index set 2 is same between old and new")

old_data3 = np.load(f"{other_dir}/test-A-1k_3_full_idxs.npz")['prediction_idxs']
data3 = np.load(f"{reference}/test-A-1k_3_full_idxs.npz")['prediction_idxs']
print(f"len set diff {len(set(data3).difference(old_data3))} If the previous set is empty then index set 3 is same between old and new")

data4 = np.load(f"{reference}/test-A-1k_4_full_idxs.npz")['prediction_idxs']
old_data4 = np.load(f"{other_dir}/test-A-1k_4_full_idxs.npz")['prediction_idxs']
print(f"len set diff {len(set(data4).difference(old_data4))} If the previous set is empty then index set 4 is same between old and new")
# %history -g -f test_data_old_new.py
