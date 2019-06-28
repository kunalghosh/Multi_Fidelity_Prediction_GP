from scipy.sparse import lil_matrix


def create_mbtr(mbtr_desc, n_features, i_samples):
    """This is the function that is called by each process but with different
    parts of the data.
    """
    n_i_samples = len(i_samples)
    i_res = lil_matrix((n_i_samples, n_features))
    for i, i_sample in enumerate(i_samples):
        feat = mbtr_desc.create(i_sample)
        i_res[i, :] = feat
        print("{} %".format((i + 1) / n_i_samples * 100))
    return i_res
