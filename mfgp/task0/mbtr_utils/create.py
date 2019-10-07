from scipy.sparse import lil_matrix


def create_mbtr(mbtr_desc, n_features, i_samples):
    """This is the function that is called by each process but with different
    parts of the data.
    """
    n_i_samples = len(i_samples)
    feat = mbtr_desc.create(i_samples[0])
    i_resk1 = lil_matrix((n_i_samples, len(feat['k1'].flatten())))
    i_resk2 = lil_matrix((n_i_samples, len(feat['k2'].flatten())))
    i_resk3 = lil_matrix((n_i_samples, len(feat['k3'].flatten())))
    for i, i_sample in enumerate(i_samples):
        feat = mbtr_desc.create(i_sample)
        # print(feat)
        # print(type(feat))
        # print(len(feat['k1'].flatten()))
        # print(len(feat['k2'].flatten()))
        # print(len(feat['k3'].flatten()))
        i_resk1[i, :] = feat['k1'].flatten()
        i_resk2[i, :] = feat['k2'].flatten()
        i_resk3[i, :] = feat['k3'].flatten()
        print("{} %".format((i + 1) / n_i_samples * 100))
    return (i_resk1, i_resk2, i_resk3)
