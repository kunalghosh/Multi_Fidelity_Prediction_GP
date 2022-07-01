import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from mfgp_refactor.utils import desc_pp, desc_pp_notest, pre_rem_split
from mfgp_refactor.io_utils import (
    append_write,
    out_time,
    fig_MDS_scatter_std,
    fig_MDS_scatter_label,
    log_timing,
)


def save_data(conf, text, data, iter):
    """
    Save some debug and temporary data to disk
    """
    name = conf.out_name
    if text is not None:
        name = name + f"_{text}"
    np.save(name + f"_{iter}_idxs.npz", data)


def acq_fn(
    conf,
    fn_name,
    i,
    prediction_idxs,
    remaining_idxs,
    prediction_set_size,
    rnd_size,
    mbtr_data,
    homo_lowfid,
    K_high,
    gpr,
    preprocess,
    out_name,
    random_seed,
):
    if fn_name == "mean_pred_with_uncertainty":
        """
        I.
        We take GP predictions and (Predictions + std) which lie in the range we are interested in.
        And randomly pick from there. In practice, we do predictions + std > range_low
        """
        assert (
            conf.range_low is not None
            or conf.range_high is not None  # atleast one of them has to be not None
        ), "conf.range_low and conf.range_high are both None, acquisition strategy cannot work"

        print(
            f"prediction_set_size={prediction_set_size}, rnd_size={rnd_size}, K_high={K_high}"
        )
        # prediction_idxs_bef = prediction_idxs
        # prediction_idxs = remaining_idxs
        X_train = mbtr_data[remaining_idxs, :]
        y_train = homo_lowfid[remaining_idxs]

        # -- Preprocessing
        X_train_pp = desc_pp_notest(preprocess, X_train)

        # -- check mean and std in next dataset
        with log_timing(conf, "\nGPR Prediction"):
            mu_s, std_s = gpr.predict(X_train_pp, return_std=True)

        save_data(conf, "debug_mean_pred", data=mu_s, iter=i)
        save_data(conf, "debug_std_pred", data=std_s, iter=i)
        # -- unsorted top K idxs
        K = prediction_set_size
        idxs_above_lowlimit = np.where(mu_s + std_s > conf.range_low)[0] # NOTE : This is the only difference from H:
        save_data(conf, "debug_idxs_above_lowlimit", data=idxs_above_lowlimit, iter=i)

        # randomly pick number we need
        K_idxs_within_limit = np.random.choice(
            idxs_above_lowlimit, size=int(K), replace=False
        )
        save_data(conf, "debug_K_idxs_within_limit", data=K_idxs_within_limit, iter=i)

        # TODO : How many picked were actually in the range (we have the true labels)
        # The better the model gets the fewer false positives we have

        heldout_idxs_add_to_train = np.array(remaining_idxs)[K_idxs_within_limit]
        updated_prediction_idxs = np.r_[prediction_idxs, heldout_idxs_add_to_train]
        updated_remaining_idxs = np.setdiff1d(remaining_idxs, heldout_idxs_add_to_train)

        np.savez(
            out_name + "_" + str(i + 1) + "_idxs.npz",
            remaining_idxs=updated_remaining_idxs,
            prediction_idxs=updated_prediction_idxs,
            pick_idxs=K_idxs_within_limit,
        )

        # re-writing these variables as these are the ones that are returned by this function
        prediction_idxs = updated_prediction_idxs
        remaining_idxs = updated_remaining_idxs
    else if fn_name == "mean_pred":
        """
        H.
        We take GP predictions and only pick the ones in the range [conf.range_low, conf.range_high] we are interested in.
        also assert conf.range_low < conf.range_high (in the Input)
        if conf.range_low is not None:
            take predictions > conf.range_low
        if conf.range_high is not None:
            take predictions < conf.range_high
        """
        assert (
            conf.range_low is not None
            or conf.range_high is not None  # atleast one of them has to be not None
        ), "conf.range_low and conf.range_high are both None, acquisition strategy cannot work"

        print(
            f"prediction_set_size={prediction_set_size}, rnd_size={rnd_size}, K_high={K_high}"
        )
        # prediction_idxs_bef = prediction_idxs
        # prediction_idxs = remaining_idxs
        X_train = mbtr_data[remaining_idxs, :]
        y_train = homo_lowfid[remaining_idxs]

        # -- Preprocessing
        X_train_pp = desc_pp_notest(preprocess, X_train)

        # -- check mean and std in next dataset
        with log_timing(conf, "\nGPR Prediction"):
            mu_s = gpr.predict(X_train_pp, return_std=False)

        save_data(conf, "debug_mean_pred", data=mu_s, iter=i)
        # -- unsorted top K idxs
        K = prediction_set_size
        idxs_above_lowlimit = np.where(mu_s > conf.range_low)[0]
        save_data(conf, "debug_idxs_above_lowlimit", data=idxs_above_lowlimit, iter=i)

        # randomly pick number we need
        K_idxs_within_limit = np.random.choice(
            idxs_above_lowlimit, size=int(K), replace=False
        )
        save_data(conf, "debug_K_idxs_within_limit", data=K_idxs_within_limit, iter=i)

        # TODO : How many picked were actually in the range (we have the true labels)
        # The better the model gets the fewer false positives we have

        heldout_idxs_add_to_train = np.array(remaining_idxs)[K_idxs_within_limit]
        updated_prediction_idxs = np.r_[prediction_idxs, heldout_idxs_add_to_train]
        updated_remaining_idxs = np.setdiff1d(remaining_idxs, heldout_idxs_add_to_train)

        np.savez(
            out_name + "_" + str(i + 1) + "_idxs.npz",
            remaining_idxs=updated_remaining_idxs,
            prediction_idxs=updated_prediction_idxs,
            pick_idxs=K_idxs_within_limit,
        )

        # re-writing these variables as these are the ones that are returned by this function
        prediction_idxs = updated_prediction_idxs
        remaining_idxs = updated_remaining_idxs

    elif fn_name == "none":
        """
        A. random sampling
        """
        prediction_idxs_bef = prediction_idxs

        prediction_idxs, remaining_idxs = pre_rem_split(
            prediction_set_size, remaining_idxs, random_seed
        )

        K_idxs_within_limit = prediction_idxs

        prediction_idxs = np.r_[prediction_idxs_bef, K_idxs_within_limit]

        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        # -- Preprocessing
        X_train_pp = desc_pp_notest(preprocess, X_train)

        # -- save the values
        np.savez(
            out_name + "_" + str(i + 1) + "_idxs.npz",
            remaining_idxs=remaining_idxs,
            prediction_idxs=prediction_idxs,
            pick_idxs=K_idxs_within_limit,
        )

    elif fn_name == "rnd":
        """
        random sampling with chunk
        """
        prediction_idxs_bef = prediction_idxs

        prediction_idxs, remaining_idxs = pre_rem_split(
            prediction_set_size, remaining_idxs, random_seed
        )

        K_idxs_within_limit, _ = train_test_split(
            prediction_idxs, train_size=rnd_size, random_state=random_seed
        )

        prediction_idxs = np.r_[prediction_idxs_bef, K_idxs_within_limit]

        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        # -- Preprocessing
        start = time.time()
        X_train_pp = desc_pp_notest(preprocess, X_train)
        process_time = time.time() - start

        # -- save the values
        np.savez(
            out_name + "_" + str(i + 1) + "_idxs.npz",
            remaining_idxs=remaining_idxs,
            prediction_idxs=prediction_idxs,
            pick_idxs=K_idxs_within_limit,
        )

    elif fn_name == "high":
        """
        high std with chunk
        """
        prediction_idxs_bef = prediction_idxs

        prediction_idxs, remaining_idxs = pre_rem_split(
            prediction_set_size, remaining_idxs, random_seed
        )

        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        # -- Preprocessing
        start = time.time()
        X_train_pp = desc_pp_notest(preprocess, X_train)
        process_time = time.time() - start

        # check mean and std in next dataset
        mu_s, std_s = gpr.predict(X_train_pp, return_std=True)  # mu->mean? yes
        K = K_high

        K_idxs_within_limit = np.argpartition(-std_s, K)[:K]
        K_idxs_within_limit = np.array(prediction_idxs)[K_idxs_within_limit]

        prediction_idxs = np.r_[prediction_idxs_bef, K_idxs_within_limit]

        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        # -- Preprocessing
        start = time.time()
        X_train_pp = desc_pp_notest(preprocess, X_train)
        process_time = time.time() - start

        # -- save the values
        np.savez(
            out_name + "_" + str(i + 1) + "_idxs.npz",
            remaining_idxs=remaining_idxs,
            prediction_idxs=prediction_idxs,
            pick_idxs=K_idxs_within_limit,
        )

    elif fn_name == "cluster":
        from sklearn import cluster
        from sklearn.metrics import pairwise_distances_argmin_min

        """
        C. Clustering without chunk
        """
        K_high = prediction_set_size
        prediction_idxs_bef = prediction_idxs

        # -- without chunk
        prediction_idxs = remaining_idxs

        X_train = mbtr_data[prediction_idxs, :]

        # -- Preprocessing
        X_train_pp = desc_pp_notest(preprocess, X_train)

        num_clusters = K_high

        # -- clustering
        start = time.time()
        append_write(out_name, "starting clustering \n")
        km = cluster.KMeans(n_clusters=num_clusters, n_jobs=24, random_state=random_seed)
        z_km = km.fit(X_train_pp)
        process_time = time.time() - start
        out_time(out_name, process_time)

        labels = np.array(z_km.labels_)
        centers = np.array(z_km.cluster_centers_)
        append_write(out_name, "length of centers " + str(len(centers)) + "\n")

        start = time.time()
        append_write(out_name, "starting calculat nearest points of centers \n")
        closest, _ = pairwise_distances_argmin_min(centers, X_train_pp)
        append_write(out_name, "number of closest points " + str(len(closest)) + "\n")
        process_time = time.time() - start
        out_time(out_name, process_time)

        K_idxs_within_limit = np.array(prediction_idxs)[closest]
        append_write(
            out_name, "length of pick idxs " + str(len(K_idxs_within_limit)) + "\n"
        )

        # -- length of cluster
        #        cluster_idxs = np.empty(num_clusters)
        #        cluster_len = np.empty(num_clusters)

        #        for j in range(num_clusters):
        #            cluster_idxs[j] = np.array(np.where(labels == j)).flatten()
        #            cluster_len[j] = len(cluster_idxs[j])

        #        np.save(out_name + "_" + str(i+1) + "_cluster_len", cluster_len )

        prediction_idxs = np.r_[prediction_idxs_bef, K_idxs_within_limit]
        remaining_idxs = np.setdiff1d(remaining_idxs, K_idxs_within_limit)

        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]
        # -- Preprocessing
        X_train_pp = desc_pp_notest(preprocess, X_train)

        # -- save the values
        np.savez(
            out_name + "_" + str(i + 1) + "_idxs.npz",
            remaining_idxs=remaining_idxs,
            prediction_idxs=prediction_idxs,
            pick_idxs=K_idxs_within_limit,
        )

    elif fn_name == "cluster_highest":
        from sklearn import cluster
        from sklearn.metrics import pairwise_distances_argmin_min

        """
        G. Clustering and choose highest std mols without chunk
        """

        num_clusters = prediction_set_size
        prediction_idxs_bef = prediction_idxs

        # -- without chunk
        prediction_idxs = remaining_idxs

        X_train = mbtr_data[prediction_idxs, :]

        # -- Preprocessing
        X_train_pp = desc_pp_notest(preprocess, X_train)

        # -- check mean and std in next dataset
        append_write(out_name, "starting prediction \n")
        start = time.time()
        mu_s, std_s = gpr.predict(X_train_pp, return_std=True)  # mu->mean? yes
        process_time = time.time() - start
        out_time(out_name, process_time)

        # -- Histgram of high std
        plt.figure()
        plt.title("", fontsize=20)
        plt.xlabel("std.", fontsize=16)
        plt.ylabel("Number of molecules", fontsize=16)
        plt.tick_params(labelsize=14)
        (a_hist2, a_bins2, _) = plt.hist(std_s[:], bins=170)
        plt.savefig(out_name + "_" + str(i + 1) + "_std.eps")

        # -- check
        #        append_write(out_name,"Max value of std within pick_idxs " + str(np.max(std_s[pick_idxs])) + "\n" )
        #        append_write(out_name,"Min value of std within pick_idxs " + str(np.min(std_s[pick_idxs])) + "\n" )
        #        append_write(out_name,"Max value of std within all remaining_idxs " + str(np.max(std_s[np.setdiff1d(range(len(prediction_idxs)), pick_idxs)])) + "\n" )
        #        append_write(out_name,"Min value of std within all remaining_idxs " + str(np.min(std_s[:])) + "\n" )
        #        append_write(out_name, "The # of zero values " + str(np.count_nonzero(std_s < 1e-10)) + "\n" )
        #        append_write(out_name, "The # of not zero values " + str(np.count_nonzero(std_s > 1e-10)) + "\n" )

        # -- clustering
        start = time.time()
        append_write(out_name, "starting clustering \n")
        km = cluster.KMeans(n_clusters=num_clusters, n_jobs=24, random_state=random_seed)
        z_km = km.fit(X_train_pp)
        process_time = time.time() - start
        out_time(out_name, process_time)

        # -- label and center
        labels = np.array(z_km.labels_)
        centers = np.array(z_km.cluster_centers_)
        append_write(out_name, "length of centers " + str(len(centers)) + "\n")

        # -- Choose mols in each cluster
        start = time.time()
        append_write(
            out_name,
            "starting calculate find the mol which have highest std in each cluster \n",
        )
        cluster_idxs = {}
        K_idxs_within_limit = np.empty(num_clusters, dtype=int)

        for j in range(num_clusters):
            cluster_idxs = np.array(np.where(labels == j)).flatten()
            prediction_clu_idxs = np.array(prediction_idxs)[cluster_idxs]
            std_s_clu = std_s[cluster_idxs]
            if len(std_s_clu) == 1:
                K_idxs_within_limit[j] = prediction_clu_idxs
            else:
                pick_idxs_temp = np.argpartition(-std_s_clu, 1)[:1]
                K_idxs_within_limit[j] = prediction_clu_idxs[pick_idxs_temp]

        process_time = time.time() - start
        out_time(out_name, process_time)
        append_write(
            out_name, "length of pick idxs " + str(len(K_idxs_within_limit)) + "\n"
        )

        prediction_idxs = np.r_[prediction_idxs_bef, K_idxs_within_limit]
        remaining_idxs = np.setdiff1d(remaining_idxs, K_idxs_within_limit)

        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        # -- Preprocessing
        X_train_pp = desc_pp_notest(preprocess, X_train)

        # -- save the values
        np.savez(
            out_name + "_" + str(i + 1) + "_idxs.npz",
            remaining_idxs=remaining_idxs,
            prediction_idxs=prediction_idxs,
            pick_idxs=K_idxs_within_limit,
        )

    elif fn_name == "rnd2":
        """
        A .random sampling without chunk (same as none)
        """
        prediction_idxs_bef = prediction_idxs

        prediction_idxs, remaining_idxs = pre_rem_split(
            prediction_set_size, remaining_idxs, random_seed
        )

        K_idxs_within_limit = prediction_idxs

        prediction_idxs = np.r_[prediction_idxs_bef, K_idxs_within_limit]

        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        # -- Preprocessing
        start = time.time()
        X_train_pp = desc_pp_notest(preprocess, X_train)
        process_time = time.time() - start

        # -- save the values
        np.savez(
            out_name + "_" + str(i + 1) + "_idxs.npz",
            remaining_idxs=remaining_idxs,
            prediction_idxs=prediction_idxs,
            pick_idxs=K_idxs_within_limit,
        )

    elif fn_name == "cluster2":
        from sklearn import cluster
        from sklearn.metrics import pairwise_distances_argmin_min

        """
        clustering with checking each cluster (Too heavy calculation)
        """

        prediction_idxs_bef = prediction_idxs

        # -- without chunk
        prediction_idxs = remaining_idxs

        X_train = mbtr_data[prediction_idxs, :]

        # -- Preprocessing
        X_train_pp = desc_pp_notest(preprocess, X_train)

        num_clusters = K_high

        # -- clustering
        start = time.time()
        append_write(out_name, "starting clustering" + "\n")
        km = cluster.KMeans(n_clusters=num_clusters, n_jobs=1, random_state=random_seed)
        z_km = km.fit(X_train_pp)
        process_time = time.time() - start
        out_time(out_name, process_time)

        labels = np.array(z_km.labels_)
        centers = np.array(z_km.cluster_centers_)
        append_write(out_name, "length of centers " + str(len(centers)) + "\n")

        start = time.time()
        append_write(out_name, "starting calculat nearest points of centers \n")
        closest, _ = pairwise_distances_argmin_min(centers, X_train_pp)
        append_write(out_name, "number of closest points " + str(len(closest)) + "\n")
        process_time = time.time() - start
        out_time(out_name, process_time)

        K_idxs_within_limit = np.array(prediction_idxs)[closest]
        append_write(
            out_name, "length of pick idxs " + str(len(K_idxs_within_limit)) + "\n"
        )

        # -- cluster check
        cluster_idxs = {}
        cluster_std = {}
        cluster_avg = np.empty(num_clusters)
        cluster_len = np.empty(num_clusters)
        cluster_max = np.empty(num_clusters)
        #            cluster_pick_idxs = {}
        #            cluster_pick_idxs_t = {}
        #            cluster_pick_idxs_all = np.array([], dtype=int)
        #            tol_cluster = {}

        for j in range(num_clusters):
            print(j)
            cluster_idxs[j] = np.array(np.where(labels == j)).flatten()
            prediction_clu_idxs = np.array(prediction_idxs)[cluster_idxs[j]]

            X_train = mbtr_data[prediction_clu_idxs, :]

            # -- Preprocessing
            X_train_pp = desc_pp_notest(preprocess, X_train)

            mu_s, std_s = gpr.predict(X_train_pp, return_std=True)  # mu->mean? yes

            cluster_len[j] = len(cluster_idxs[j])
            cluster_std[j] = std_s
            cluster_avg[j] = np.average(cluster_std[j])
            cluster_max[j] = np.max(cluster_std[j])

        print(cluster_len)
        print(cluster_avg)

        np.save(out_name + "_" + str(i + 1) + "_cluster_std", cluster_std)
        np.save(out_name + "_" + str(i + 1) + "_cluster_len", cluster_len)
        np.save(out_name + "_" + str(i + 1) + "_cluster_avg", cluster_avg)
        np.save(out_name + "_" + str(i + 1) + "_cluster_max", cluster_max)
        np.save(out_name + "_" + str(i + 1) + "_cluster_idxs", K_idxs_within_limit)

        prediction_idxs = np.r_[prediction_idxs_bef, K_idxs_within_limit]
        remaining_idxs = np.setdiff1d(remaining_idxs, K_idxs_within_limit)

        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        # -- Preprocessing
        X_train_pp = desc_pp_notest(preprocess, X_train)

        # -- save the values
        np.savez(
            out_name + "_" + str(i + 1) + "_idxs.npz",
            remaining_idxs=remaining_idxs,
            prediction_idxs=prediction_idxs,
            pick_idxs=K_idxs_within_limit,
        )

    #            tol_cluster[j] = np.average(cluster_std[j])
    #            cluster_pick_idxs_t[j] = np.array(np.where(cluster_std[j] > tol_cluster[j])).flatten()

    #            cluster_pick_idxs[j] = cluster_idxs[j][cluster_pick_idxs_t[j]]
    #            prediction_clu_pick_idxs = np.array(prediction_idxs)[cluster_pick_idxs[j]]
    #            X_train = mbtr_data[prediction_clu_pick_idxs, :]
    #            mu_s, std_s = gpr.predict(X_train.toarray(), return_std=True) #mu->mean? yes

    #            cluster_pick_idxs_all = np.concatenate([cluster_pick_idxs_all,prediction_clu_pick_idxs]

    elif fn_name == "high2":
        """
        B. High std without chunk
        """
        K_high = prediction_set_size
        prediction_idxs_bef = prediction_idxs

        if len(remaining_idxs) == K_high:
            K_idxs_within_limit = remaining_idxs
            remaining_idxs = np.array([])
            prediction_idxs = np.r_[prediction_idxs_bef, K_idxs_within_limit]

        elif len(remaining_idxs) != K_high:

            prediction_idxs = remaining_idxs

            X_train = mbtr_data[prediction_idxs, :]

            # -- Preprocessing
            X_train_pp = desc_pp_notest(preprocess, X_train)

            # -- check mean and std in next dataset
            append_write(out_name, "starting prediction \n")
            start = time.time()
            mu_s, std_s = gpr.predict(X_train_pp, return_std=True)  # mu->mean? yes
            process_time = time.time() - start
            out_time(out_name, process_time)

            # -- Histgram of high std
            plt.figure()
            plt.title("", fontsize=20)
            plt.xlabel("std.", fontsize=16)
            plt.ylabel("Number of molecules", fontsize=16)
            plt.tick_params(labelsize=14)
            (a_hist2, a_bins2, _) = plt.hist(std_s[:], bins=170)
            plt.savefig(out_name + "_" + str(i + 1) + "_std.eps")

            # -- unsorted top K idxs
            K = K_high
            K_idxs_within_limit = np.argpartition(-std_s, K)[:K]

            # -- check
            append_write(
                out_name,
                "Max value of std within pick_idxs "
                + str(np.max(std_s[K_idxs_within_limit]))
                + "\n",
            )
            append_write(
                out_name,
                "Min value of std within pick_idxs "
                + str(np.min(std_s[K_idxs_within_limit]))
                + "\n",
            )
            append_write(
                out_name,
                "Max value of std within all remaining_idxs "
                + str(
                    np.max(
                        std_s[
                            np.setdiff1d(range(len(prediction_idxs)), K_idxs_within_limit)
                        ]
                    )
                )
                + "\n",
            )
            append_write(
                out_name,
                "Min value of std within all remaining_idxs "
                + str(np.min(std_s[:]))
                + "\n",
            )

            append_write(
                out_name,
                "The # of zero values " + str(np.count_nonzero(std_s < 1e-10)) + "\n",
            )
            append_write(
                out_name,
                "The # of not zero values " + str(np.count_nonzero(std_s > 1e-10)) + "\n",
            )

            K_idxs_within_limit = np.array(prediction_idxs)[K_idxs_within_limit]
            prediction_idxs = np.r_[prediction_idxs_bef, K_idxs_within_limit]
            remaining_idxs = np.setdiff1d(remaining_idxs, K_idxs_within_limit)

            np.save(out_name + "_high_std_" + str(i + 1), std_s)

        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        # -- Preprocessing
        X_train_pp = desc_pp_notest(preprocess, X_train)

        np.savez(
            out_name + "_" + str(i + 1) + "_idxs.npz",
            remaining_idxs=remaining_idxs,
            prediction_idxs=prediction_idxs,
            pick_idxs=K_idxs_within_limit,
        )

    elif fn_name == "high_and_cluster":
        from sklearn import cluster
        from sklearn.metrics import pairwise_distances_argmin_min

        """
        D. Combination of B and C
        without chunk,
        1. Choose mols with high std 
        2. Make cluster
        3. Choose mols which is near the center of clusters
        """
        K_pre = rnd_size  # highest std
        K_high = prediction_set_size  # num_cluster
        prediction_idxs_bef = prediction_idxs

        if len(remaining_idxs) == K_high:
            K_idxs_within_limit = remaining_idxs
            remaining_idxs = np.array([])
            prediction_idxs = np.r_[prediction_idxs_bef, K_idxs_within_limit]

        elif len(remaining_idxs) != K_high:

            prediction_idxs = remaining_idxs

            X_train = mbtr_data[prediction_idxs, :]

            # -- Preprocessing
            X_train_pp = desc_pp_notest(preprocess, X_train)

            # -- check mean and std in next dataset
            append_write(out_name, "starting prediction \n")
            start = time.time()
            mu_s, std_s = gpr.predict(X_train_pp, return_std=True)  # mu->mean? yes
            process_time = time.time() - start
            out_time(out_name, process_time)

            # -- Histgram of high std
            plt.figure()
            plt.title("", fontsize=20)
            plt.xlabel("std.", fontsize=16)
            plt.ylabel("Number of molecules", fontsize=16)
            plt.tick_params(labelsize=14)
            (a_hist2, a_bins2, _) = plt.hist(std_s[:], bins=170)
            plt.savefig(out_name + "_" + str(i + 1) + "_std.eps")

            # -- unsorted top K idxs
            #            K = int(len(remaining_idxs)/2.0)
            K = int(len(remaining_idxs) / K_pre)
            pick_idxs_tmp = np.argpartition(-std_s, K)[:K]

            # -- check
            append_write(
                out_name,
                "Max value of std within pick_idxs "
                + str(np.max(std_s[pick_idxs_tmp]))
                + "\n",
            )
            append_write(
                out_name,
                "Min value of std within pick_idxs "
                + str(np.min(std_s[pick_idxs_tmp]))
                + "\n",
            )
            append_write(
                out_name,
                "Max value of std within all remaining_idxs "
                + str(
                    np.max(
                        std_s[np.setdiff1d(range(len(prediction_idxs)), pick_idxs_tmp)]
                    )
                )
                + "\n",
            )
            append_write(
                out_name,
                "Min value of std within all remaining_idxs "
                + str(np.min(std_s[:]))
                + "\n",
            )
            append_write(
                out_name,
                "The # of zero values " + str(np.count_nonzero(std_s < 1e-10)) + "\n",
            )
            append_write(
                out_name,
                "The # of not zero values " + str(np.count_nonzero(std_s > 1e-10)) + "\n",
            )
            K_idxs_within_limit = np.array(prediction_idxs)[pick_idxs_tmp]

            # --
            X_train = mbtr_data[K_idxs_within_limit, :]
            y_train = homo_lowfid[K_idxs_within_limit]

            # -- Preprocessing
            X_train_pp = desc_pp_notest(preprocess, X_train)

            # --
            num_clusters = K_high

            # -- Clustering
            start = time.time()
            append_write(out_name, "starting clustering \n")
            try:
                km = cluster.KMeans(
                    n_clusters=num_clusters, n_jobs=24, random_state=random_seed
                )
                z_km = km.fit(X_train_pp)
            except ValueError as e:
                print(f"Couldn't cluster the datapoints. {e}")
                print(f"Continuing without clustering..")
                append_write(out_name, "Continuing without clustering...\n")
                append_write(out_name, f"Pick idxs length = {len(K_idxs_within_limit)}")
                np.savez(
                    out_name + "_" + str(i + 1) + "pickidxs_valuerror.npz",
                    pick_idxs=K_idxs_within_limit,
                )
            else:
                # No exception raised
                process_time = time.time() - start
                out_time(out_name, process_time)

                labels = np.array(z_km.labels_)
                centers = np.array(z_km.cluster_centers_)
                append_write(out_name, "length of centers " + str(len(centers)) + "\n")

                # -- Figure
                #                fig_MDS_scatter_std(mbtr_data[pick_idxs,:].toarray(), std_s[pick_idxs_tmp], out_name + "_" + str(i+1) + "_MDS_std.eps")
                #                fig_MDS_scatter_label(mbtr_data[pick_idxs,:].toarray(), labels, out_name + "_" + str(i+1) + "_MDS_label.eps")

                start = time.time()
                append_write(out_name, "starting calculate nearest points of centers \n")
                closest, _ = pairwise_distances_argmin_min(centers, X_train_pp)
                append_write(
                    out_name, "number of closest points " + str(len(closest)) + "\n"
                )
                process_time = time.time() - start
                out_time(out_name, process_time)

                # -- Calculate centers
                K_idxs_within_limit = np.array(K_idxs_within_limit)[closest]

            append_write(
                out_name, "length of pick idxs " + str(len(K_idxs_within_limit)) + "\n"
            )

            prediction_idxs = np.r_[prediction_idxs_bef, K_idxs_within_limit]
            remaining_idxs = np.setdiff1d(remaining_idxs, K_idxs_within_limit)

            # -- Histgram of high std
            plt.figure()
            plt.title("", fontsize=20)
            plt.xlabel("std.", fontsize=16)
            plt.ylabel("Number of molecules", fontsize=16)
            plt.tick_params(labelsize=14)
            (a_hist2, a_bins2, _) = plt.hist(std_s[pick_idxs_tmp[closest]], bins=170)
            plt.savefig(out_name + "_" + str(i + 1) + "_picked_std.eps")

        # --
        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        # -- Preprocessing
        X_train_pp = desc_pp_notest(preprocess, X_train)

        # -- save the values
        np.savez(
            out_name + "_" + str(i + 1) + "_idxs.npz",
            remaining_idxs=remaining_idxs,
            prediction_idxs=prediction_idxs,
            pick_idxs=K_idxs_within_limit,
        )

    elif fn_name == "high_and_cluster_dis":
        from sklearn import cluster
        from sklearn.metrics import pairwise_distances_argmin_min

        """
        uncompleted
        """
        K_pre = rnd_size  # highest std
        K_high = prediction_set_size  # num_cluster
        prediction_idxs_bef = prediction_idxs

        if len(remaining_idxs) == K_high:
            K_idxs_within_limit = remaining_idxs
            remaining_idxs = np.array([])
            prediction_idxs = np.r_[prediction_idxs_bef, K_idxs_within_limit]

        elif len(remaining_idxs) != K_high:

            prediction_idxs = remaining_idxs

            X_train = mbtr_data[prediction_idxs, :]

            # -- Preprocessing
            X_train_pp = desc_pp_notest(preprocess, X_train)

            # -- check mean and std in next dataset
            append_write(out_name, "starting prediction \n")
            start = time.time()
            mu_s, std_s = gpr.predict(X_train_pp, return_std=True)  # mu->mean? yes
            process_time = time.time() - start
            out_time(out_name, process_time)

            # -- Histgram of high std
            plt.figure()
            plt.title("", fontsize=20)
            plt.xlabel("std.", fontsize=16)
            plt.ylabel("Number of molecules", fontsize=16)
            plt.tick_params(labelsize=14)
            (a_hist2, a_bins2, _) = plt.hist(std_s[:], bins=170)
            plt.savefig(out_name + "_" + str(i + 1) + "_std.eps")

            # -- unsorted top K idxs
            #            K = int(len(remaining_idxs)/2.0)
            K = int(len(remaining_idxs) / K_pre)
            K_idxs_within_limit = np.argpartition(-std_s, K)[:K]

            # -- check
            append_write(
                out_name,
                "Max value of std within pick_idxs "
                + str(np.max(std_s[K_idxs_within_limit]))
                + "\n",
            )
            append_write(
                out_name,
                "Min value of std within pick_idxs "
                + str(np.min(std_s[K_idxs_within_limit]))
                + "\n",
            )
            append_write(
                out_name,
                "Max value of std within all remaining_idxs "
                + str(
                    np.max(
                        std_s[
                            np.setdiff1d(range(len(prediction_idxs)), K_idxs_within_limit)
                        ]
                    )
                )
                + "\n",
            )
            append_write(
                out_name,
                "Min value of std within all remaining_idxs "
                + str(np.min(std_s[:]))
                + "\n",
            )
            append_write(
                out_name,
                "The # of zero values " + str(np.count_nonzero(std_s < 1e-10)) + "\n",
            )
            append_write(
                out_name,
                "The # of not zero values " + str(np.count_nonzero(std_s > 1e-10)) + "\n",
            )
            K_idxs_within_limit = np.array(prediction_idxs)[K_idxs_within_limit]

            # --
            X_train = mbtr_data[K_idxs_within_limit, :]
            y_train = homo_lowfid[K_idxs_within_limit]

            # -- Preprocessing
            X_train_pp = desc_pp_notest(preprocess, X_train)

            # --
            num_clusters = K_high

            # -- Clustering
            start = time.time()
            append_write(out_name, "starting clustering \n")
            km = cluster.KMeans(
                n_clusters=num_clusters, n_jobs=24, random_state=random_seed
            )
            z_km = km.fit(X_train_pp)
            process_time = time.time() - start
            out_time(out_name, process_time)

            labels = np.array(z_km.labels_)
            centers = np.array(z_km.cluster_centers_)
            append_write(out_name, "length of centers " + str(len(centers)) + "\n")

            # -- Figure
            fig_MDS_scatter_std(
                mbtr_data[K_idxs_within_limit, :].toarray(),
                std[K_idxs_within_limit],
                out_name + "_" + str(i + 1) + "_MDS_std.eps",
            )
            fig_MDS_scatter_label(
                mbtr_data[K_idxs_within_limit, :].toarray(),
                labels,
                out_name + "_" + str(i + 1) + "_MDS_label.eps",
            )

            start = time.time()
            append_write(out_name, "starting calculat nearest points of centers \n")
            closest, _ = pairwise_distances_argmin_min(centers, X_train_pp)
            append_write(out_name, "number of closest points " + str(len(closest)) + "\n")
            process_time = time.time() - start
            out_time(out_name, process_time)

            # -- Calculate centers
            K_idxs_within_limit = np.array(K_idxs_within_limit)[closest]
            append_write(
                out_name, "length of pick idxs " + str(len(K_idxs_within_limit)) + "\n"
            )

            prediction_idxs = np.r_[prediction_idxs_bef, K_idxs_within_limit]
            remaining_idxs = np.setdiff1d(remaining_idxs, K_idxs_within_limit)

        # --
        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        # -- Preprocessing
        X_train_pp = desc_pp_notest(preprocess, X_train)

        # -- save the values
        np.savez(
            out_name + "_" + str(i + 1) + "_idxs.npz",
            remaining_idxs=remaining_idxs,
            prediction_idxs=prediction_idxs,
            pick_idxs=K_idxs_within_limit,
        )

    elif fn_name == "high_and_cluster_highest":
        from sklearn import cluster
        from sklearn.metrics import pairwise_distances_argmin_min

        """
        E. Combination of B and C
        without chunk,
        1. Choose mols with high std 
        2. Make cluster
        3. Choose mols which have highest std.
        """
        K_pre = rnd_size  # highest std
        K_high = prediction_set_size  # num_cluster
        prediction_idxs_bef = prediction_idxs
        # --
        num_clusters = K_high

        if len(remaining_idxs) == K_high:
            K_idxs_within_limit = remaining_idxs
            remaining_idxs = np.array([])
            prediction_idxs = np.r_[prediction_idxs_bef, K_idxs_within_limit]

        elif len(remaining_idxs) != K_high:

            prediction_idxs = remaining_idxs

            X_train = mbtr_data[prediction_idxs, :]

            # -- Preprocessing
            X_train_pp = desc_pp_notest(preprocess, X_train)

            # -- check mean and std in next dataset
            append_write(out_name, "starting prediction \n")
            start = time.time()
            mu_s, std_s = gpr.predict(X_train_pp, return_std=True)  # mu->mean? yes
            process_time = time.time() - start
            out_time(out_name, process_time)

            # -- Histgram of high std
            plt.figure()
            plt.title("", fontsize=20)
            plt.xlabel("std.", fontsize=16)
            plt.ylabel("Number of molecules", fontsize=16)
            plt.tick_params(labelsize=14)
            (a_hist2, a_bins2, _) = plt.hist(std_s[:], bins=170)
            plt.savefig(out_name + "_" + str(i + 1) + "_std.eps")

            # -- unsorted top K idxs
            K = int(len(remaining_idxs) / K_pre)
            pick_idxs_tmp = np.argpartition(-std_s, K)[:K]

            # -- check
            append_write(
                out_name,
                "Max value of std within pick_idxs "
                + str(np.max(std_s[pick_idxs_tmp]))
                + "\n",
            )
            append_write(
                out_name,
                "Min value of std within pick_idxs "
                + str(np.min(std_s[pick_idxs_tmp]))
                + "\n",
            )
            append_write(
                out_name,
                "Max value of std within all remaining_idxs "
                + str(
                    np.max(
                        std_s[np.setdiff1d(range(len(prediction_idxs)), pick_idxs_tmp)]
                    )
                )
                + "\n",
            )
            append_write(
                out_name,
                "Min value of std within all remaining_idxs "
                + str(np.min(std_s[:]))
                + "\n",
            )
            append_write(
                out_name,
                "The # of zero values " + str(np.count_nonzero(std_s < 1e-10)) + "\n",
            )
            append_write(
                out_name,
                "The # of not zero values " + str(np.count_nonzero(std_s > 1e-10)) + "\n",
            )
            pick_idxs2 = np.array(prediction_idxs)[pick_idxs_tmp]

            # --
            X_train = mbtr_data[pick_idxs2, :]
            y_train = homo_lowfid[pick_idxs2]

            # -- Preprocessing
            X_train_pp = desc_pp_notest(preprocess, X_train)

            # -- Clustering
            start = time.time()
            append_write(out_name, "starting clustering \n")
            km = cluster.KMeans(
                n_clusters=num_clusters, n_jobs=24, random_state=random_seed
            )
            z_km = km.fit(X_train_pp)
            process_time = time.time() - start
            out_time(out_name, process_time)

            labels = np.array(z_km.labels_)
            centers = np.array(z_km.cluster_centers_)
            append_write(out_name, "length of centers " + str(len(centers)) + "\n")

            # -- Figure
            #            fig_MDS_scatter_std(mbtr_data[pick_idxs,:].toarray(), std[pick_idxs], out_name + "_" + str(i+1) + "_MDS_std.eps")
            #            fig_MDS_scatter_label(mbtr_data[pick_idxs,:].toarray(), labels, out_name + "_" + str(i+1) + "_MDS_label.eps")

            # -- Choose mols in each cluster
            start = time.time()
            append_write(
                out_name,
                "starting calculate find the mol which have highest std in each cluster \n",
            )
            cluster_idxs = {}
            K_idxs_within_limit = np.empty(num_clusters, dtype=int)

            for j in range(num_clusters):
                cluster_idxs = np.array(np.where(labels == j)).flatten()
                prediction_clu_idxs = np.array(pick_idxs2)[cluster_idxs]
                std_s_clu = std_s[pick_idxs_tmp[cluster_idxs]]
                if len(std_s_clu) == 1:
                    K_idxs_within_limit[j] = prediction_clu_idxs
                else:
                    pick_idxs_temp = np.argpartition(-std_s_clu, 1)[:1]
                    K_idxs_within_limit[j] = prediction_clu_idxs[pick_idxs_temp]

            prediction_idxs = np.r_[prediction_idxs_bef, K_idxs_within_limit]
            remaining_idxs = np.setdiff1d(remaining_idxs, K_idxs_within_limit)

        # --
        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        # -- Preprocessing
        X_train_pp = desc_pp_notest(preprocess, X_train)

        # -- save the values
        np.savez(
            out_name + "_" + str(i + 1) + "_idxs.npz",
            remaining_idxs=remaining_idxs,
            prediction_idxs=prediction_idxs,
            pick_idxs=K_idxs_within_limit,
        )

    elif fn_name == "cluster_and_high":
        from sklearn import cluster
        from sklearn.metrics import pairwise_distances_argmin_min

        """
        uncompleted
        """

        prediction_idxs_bef = prediction_idxs
        num_clusters = int(K_high / 2.0)

        if len(remaining_idxs) == prediction_set_size:
            K_idxs_within_limit = remaining_idxs
            remaining_idxs = np.array([])
            prediction_idxs = np.r_[prediction_idxs_bef, K_idxs_within_limit]

        elif len(remaining_idxs) != prediction_set_size:

            prediction_idxs = remaining_idxs

            X_train = mbtr_data[prediction_idxs, :]

            # -- Preprocessing
            X_train_pp = desc_pp_notest(preprocess, X_train)

            # -- check mean and std in next dataset
            append_write(out_name, "starting prediction \n")
            start = time.time()
            mu_s, std_s = gpr.predict(X_train_pp, return_std=True)  # mu->mean? yes
            process_time = time.time() - start
            out_time(out_name, process_time)

            # -- Histgram of high std
            plt.figure()
            plt.title("", fontsize=20)
            plt.xlabel("std.", fontsize=16)
            plt.ylabel("Number of molecules", fontsize=16)
            plt.tick_params(labelsize=14)
            (a_hist2, a_bins2, _) = plt.hist(std_s[:], bins=170)
            plt.savefig(out_name + +"_" + str(i + 1) + "_std.eps")

            # -- unsorted top K idxs
            #            K = int(len(remaining_idxs)/2.0)
            #            pick_idxs = np.argpartition(-std_s, K)[:K]

            # -- check
            #            append_write(out_name,"Max value of std within pick_idxs " + str(np.max(std_s[pick_idxs])) + "\n" )
            #            append_write(out_name,"Min value of std within pick_idxs " + str(np.min(std_s[pick_idxs])) + "\n" )
            #            append_write(out_name,"Max value of std within all remaining_idxs " + str(np.max(std_s[np.setdiff1d(range(len(prediction_idxs)), pick_idxs)])) + "\n" )
            #            append_write(out_name,"Min value of std within all remaining_idxs " + str(np.min(std_s[:])) + "\n" )

            #            pick_idxs = np.array(prediction_idxs)[pick_idxs]

            # --
            #            X_train = mbtr_data[pick_idxs, :]
            #            y_train = homo_lowfid[pick_idxs]

            # -- Preprocessing
            X_train_pp = desc_pp_notest(preprocess, X_train)

            # -- clustering
            start = time.time()
            append_write(out_name, "starting clustering \n")
            km = cluster.KMeans(
                n_clusters=num_clusters, n_jobs=24, random_state=random_seed
            )
            z_km = km.fit(X_train_pp)
            process_time = time.time() - start
            out_time(out_name, process_time)

            labels = np.array(z_km.labels_)
            centers = np.array(z_km.cluster_centers_)
            append_write(out_name, "length of centers " + str(len(centers)) + "\n")

            start = time.time()
            append_write(out_name, "starting calculat nearest points of centers \n")
            closest, _ = pairwise_distances_argmin_min(centers, X_train_pp)
            append_write(out_name, "number of closest points " + str(len(closest)) + "\n")
            process_time = time.time() - start
            out_time(out_name, process_time)

            K_idxs_within_limit = np.array(K_idxs_within_limit)[closest]
            append_write(
                out_name, "length of pick idxs " + str(len(K_idxs_within_limit)) + "\n"
            )

            # -- cluster check
            cluster_idxs = {}
            cluster_len = np.empty(num_clusters)
            #            cluster_pick_idxs = {}
            #            cluster_pick_idxs_t = {}
            #            cluster_pick_idxs_all = np.array([], dtype=int)

            #            int(float(prediction_set_size)/float(num_clusters))

            while len(prediction_clu_idxs) == prediction_set_size:
                for j in range(num_clusters):
                    cluster_idxs[j] = np.array(np.where(labels == j)).flatten()
                    cluster_len[j] = len(cluster_idxs[j])  # length
                    prediction_clu_idxs = np.array(prediction_idxs)[cluster_idxs[j]]
                    std_s_clu = std_s[prediction_clu_idxs]

                    K = float(cluster_len[j]) / float(2)
                    K_int = K.is_integer()

                    if K_int:
                        pick_idxs_temp = np.argpartition(-std_s_clu, K)[:K]
                    else:
                        K = float(cluster_len[j] + 1) / float(2)
                        pick_idxs_temp = np.argpartition(-std_s_clu, K)[:K]

            np.save(out_name + "_" + str(i + 1) + "_cluster_len", cluster_len)

            prediction_idxs = np.r_[prediction_idxs_bef, K_idxs_within_limit]
            remaining_idxs = np.setdiff1d(remaining_idxs, K_idxs_within_limit)

        # --
        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        # -- Preprocessing
        X_train_pp = desc_pp_notest(preprocess, X_train)

        # -- save the values
        np.savez(
            out_name + "_" + str(i + 1) + "_idxs.npz",
            remaining_idxs=remaining_idxs,
            prediction_idxs=prediction_idxs,
            pick_idxs=K_idxs_within_limit,
        )

    elif fn_name == "high_and_random":
        from sklearn import cluster
        from sklearn.metrics import pairwise_distances_argmin_min

        """
        F. Combination of A and B
        without chunk,
        1. Choose mols with high std 
        2. random sampling
        """
        K_pre = rnd_size  # highest std
        K_high = prediction_set_size  # num_cluster
        prediction_idxs_bef = prediction_idxs

        if len(remaining_idxs) == K_high:
            K_idxs_within_limit = remaining_idxs
            remaining_idxs = np.array([])
            prediction_idxs = np.r_[prediction_idxs_bef, K_idxs_within_limit]

        elif len(remaining_idxs) != K_high:

            prediction_idxs = remaining_idxs

            X_train = mbtr_data[prediction_idxs, :]

            # -- Preprocessing
            X_train_pp = desc_pp_notest(preprocess, X_train)

            # -- check mean and std in next dataset
            append_write(out_name, "starting prediction \n")
            start = time.time()
            mu_s, std_s = gpr.predict(X_train_pp, return_std=True)  # mu->mean? yes
            process_time = time.time() - start
            out_time(out_name, process_time)

            # -- Histgram of high std
            plt.figure()
            plt.title("", fontsize=20)
            plt.xlabel("std.", fontsize=16)
            plt.ylabel("Number of molecules", fontsize=16)
            plt.tick_params(labelsize=14)
            (a_hist2, a_bins2, _) = plt.hist(std_s[:], bins=170)
            plt.savefig(out_name + "_" + str(i + 1) + "_std.eps")

            # -- unsorted top K idxs
            #            K = int(len(remaining_idxs)/2.0)
            K = int(len(remaining_idxs) / K_pre)
            pick_idxs_tmp = np.argpartition(-std_s, K)[:K]

            # -- check
            append_write(
                out_name,
                "Max value of std within pick_idxs "
                + str(np.max(std_s[pick_idxs_tmp]))
                + "\n",
            )
            append_write(
                out_name,
                "Min value of std within pick_idxs "
                + str(np.min(std_s[pick_idxs_tmp]))
                + "\n",
            )
            append_write(
                out_name,
                "Max value of std within all remaining_idxs "
                + str(
                    np.max(
                        std_s[np.setdiff1d(range(len(prediction_idxs)), pick_idxs_tmp)]
                    )
                )
                + "\n",
            )
            append_write(
                out_name,
                "Min value of std within all remaining_idxs "
                + str(np.min(std_s[:]))
                + "\n",
            )
            append_write(
                out_name,
                "The # of zero values " + str(np.count_nonzero(std_s < 1e-10)) + "\n",
            )
            append_write(
                out_name,
                "The # of not zero values " + str(np.count_nonzero(std_s > 1e-10)) + "\n",
            )
            K_idxs_within_limit = np.array(prediction_idxs)[pick_idxs_tmp]

            # --
            X_train = mbtr_data[K_idxs_within_limit, :]
            y_train = homo_lowfid[K_idxs_within_limit]

            # -- Preprocessing
            X_train_pp = desc_pp_notest(preprocess, X_train)

            # -- Random
            K_idxs_within_limit, _ = pre_rem_split(
                prediction_set_size, K_idxs_within_limit, random_seed
            )

            prediction_idxs = np.r_[prediction_idxs_bef, K_idxs_within_limit]
            remaining_idxs = np.setdiff1d(remaining_idxs, K_idxs_within_limit)

            # #-- Histgram of high std
            # plt.figure()
            # plt.title('', fontsize = 20)
            # plt.xlabel('std.', fontsize = 16)
            # plt.ylabel('Number of molecules', fontsize = 16)
            # plt.tick_params(labelsize = 14)
            # (a_hist2, a_bins2, _) = plt.hist(std_s[pick_idxs_tmp[closest]], bins=170)
            # plt.savefig(out_name + "_" + str(i+1) + "_picked_std.eps")

        # --
        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        # -- Preprocessing
        X_train_pp = desc_pp_notest(preprocess, X_train)

        # -- save the values
        np.savez(
            out_name + "_" + str(i + 1) + "_idxs.npz",
            remaining_idxs=remaining_idxs,
            prediction_idxs=prediction_idxs,
            pick_idxs=K_idxs_within_limit,
        )

    else:
        append_write(out_name, fn_name + "\n")
        append_write(out_name, "You should use defined acquisition function ! \n")
        append_write(out_name, "program stopped ! \n")
        sys.exit()

    return prediction_idxs, remaining_idxs, X_train_pp, y_train
