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


def get_prediction_idxs(file_path):
    """
    Gets path to the file and prints exception if there was a problem.
    Doesn't terminate execution.
    """
    data = None

    try:
        data = np.load(file_path)['prediction_idxs']
    except (Exception, OSError) as e:
        print(f"{e}. Couldn't load file {file_path}")

    return data

def compare_two_data_lists(data1, data2):
    """
    Gets two lists and returns set difference of the two lists.
    But if one of them is None (file loading error) then the return value is None
    """
    set_difference = None
    if data1 is None or data2 is None:
        set_difference = None
    else:
        set_difference = len(set(data1).difference(data2))
    return set_difference


data1 = get_prediction_idxs(f"{reference}/test-D-1k_1_full_idxs.npz")
data2 = get_prediction_idxs(f"{reference}/test-D-1k_2_full_idxs.npz")
set_difference = compare_two_data_lists(data1, data2)
print(f"len set diff {set_difference} If the previous set has elements then two iterations of the new code produce different indices (this is desirable.). None means there was an error.")

for i in range(1,16):
    data = get_prediction_idxs(f"{reference}/test-D-1k_{i}_full_idxs.npz")
    old_data = get_prediction_idxs(f"{other_dir}/test-D-1k_{i}_full_idxs.npz")
    set_difference = compare_two_data_lists(data, old_data)
    print(f"len set diff {set_difference} If the previous set is empty then index set {i} is same between old and new. None means there was an error.")

# # %history -g -f test_data_old_new.py
