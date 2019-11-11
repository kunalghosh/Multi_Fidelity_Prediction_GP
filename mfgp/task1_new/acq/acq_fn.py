import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from utils import desc_pp, desc_pp_notest,pre_rem_split

def acq_fn(fn_name, i, prediction_idxs, remaining_idxs, prediction_set_size, rnd_size, mbtr_data,homo_lowfid, K_high , gpr, preprocess, out_name):
    if  fn_name == "none":
        """
        none
        """
        prediction_idxs_bef = prediction_idxs

        prediction_idxs, remaining_idxs = pre_rem_split(prediction_set_size, remaining_idxs)

        pick_idxs = prediction_idxs
        
        prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]

        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        #-- normalization
        start = time.time()
        X_train_pp = desc_pp_notest(preprocess, X_train)   
        process_time = time.time() - start

        #-- save the values 
        np.savez(out_name + "_" + str(i+1) + "_idxs.npz", remaining_idxs=remaining_idxs, prediction_idxs = prediction_idxs, pick_idxs = pick_idxs)

    if  fn_name == "rnd":
        """
        random sampling with chunk
        """
        prediction_idxs_bef = prediction_idxs

        prediction_idxs, remaining_idxs = pre_rem_split(prediction_set_size, remaining_idxs)

        pick_idxs,a = train_test_split(prediction_idxs, train_size = rnd_size)#, random_state=0)
        
        prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]

        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        #-- normalization
        start = time.time()
        X_train_pp = desc_pp_notest(preprocess, X_train)   
        process_time = time.time() - start

        #-- save the values 
        np.savez(out_name + "_" + str(i+1) + "_idxs.npz", remaining_idxs=remaining_idxs, prediction_idxs=prediction_idxs, pick_idxs = pick_idxs)

    elif fn_name == "high":
        """
        high std with chunk
        """
        prediction_idxs_bef = prediction_idxs
        
        prediction_idxs, remaining_idxs = pre_rem_split(prediction_set_size, remaining_idxs)

        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        #-- normalization
        start = time.time()
        X_train_pp = desc_pp_notest(preprocess, X_train)               
        process_time = time.time() - start
                                                                
        #check mean and std in next dataset
        mu_s, std_s = gpr.predict(X_train_pp, return_std=True) #mu->mean? yes
        K = K_high

        pick_idxs = np.argpartition(-std_s, K)[:K]
        pick_idxs = np.array(prediction_idxs)[pick_idxs]

        prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]
        
        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        #-- normalization
        start = time.time()
        X_train_pp = desc_pp_notest(preprocess, X_train)   
        process_time = time.time() - start

        #-- save the values 
        np.savez(out_name + "_" + str(i+1) + "_idxs.npz", remaining_idxs=remaining_idxs, prediction_idxs=prediction_idxs, pick_idxs = pick_idxs)

    elif fn_name == "cluster":
        from sklearn import cluster
        from sklearn.metrics import pairwise_distances_argmin_min
        """
        clustering
        """
        f2 = open(out_name,"a")
        f2.write("aaaaaa" + "\n"),f2.flush()
        prediction_idxs_bef = prediction_idxs

        prediction_idxs = remaining_idxs

        X_train = mbtr_data[prediction_idxs, :]
        
        #-- normalization
        start = time.time()
        X_train_pp = desc_pp_notest(preprocess, X_train)            
        process_time = time.time() - start

        num_clusters = K_high

        #-- clustering
        start = time.time()
        f2.write("starting clustering" + "\n"),f2.flush()
        km = cluster.KMeans(n_clusters = num_clusters, n_jobs = 24)
        z_km = km.fit(X_train_pp)
        process_time = time.time() - start
        f2.write("time "),f2.write(str(process_time) + "[s]" + "\n"),f2.flush()
        
        labels = np.array(z_km.labels_)
        centers = np.array(z_km.cluster_centers_)
        f2.write("length of centers "),f2.write(str(len(centers))),f2.flush()
        
        start = time.time()
        f2.write("starting calculat nearest points of centers" + "\n"),f2.flush()
        closest, _ = pairwise_distances_argmin_min(centers, X_train_pp)
        f2.write("number of closest points " + str(len(closest))),f2.flush()
        process_time = time.time() - start
        f2.write("time "),f2.write(str(process_time) + "[s]" + "\n"),f2.flush()
        
        pick_idxs = np.array(prediction_idxs)[closest]
        f2.write("length of pick idxs "),f2.write(str(len(pick_idxs)) + "\n"),f2.flush()
        
        prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]
        
        remaining_idxs = np.setdiff1d(remaining_idxs, pick_idxs)

        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        #-- normalization
        start = time.time()
        X_train_pp = desc_pp_notest(preprocess, X_train)
        process_time = time.time() - start

        #-- save the values 
        np.savez(out_name + "_" + str(i+1) + "_idxs.npz", remaining_idxs=remaining_idxs, prediction_idxs=prediction_idxs, pick_idxs = pick_idxs)        

        f2.close()        
    elif  fn_name == "rnd2":
        """
        random sampling without chunk
        """
        prediction_idxs_bef = prediction_idxs

        prediction_idxs, remaining_idxs = pre_rem_split(prediction_set_size, remaining_idxs)        
            
        pick_idxs = prediction_idxs

        prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]
        
        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        #-- normalization
        start = time.time()
        X_train_pp = desc_pp_notest(preprocess, X_train)
        process_time = time.time() - start

        #-- save the values 
        np.savez(out_name + "_" + str(i+1) + "_idxs.npz", remaining_idxs=remaining_idxs, prediction_idxs=prediction_idxs, pick_idxs = pick_idxs)

    elif fn_name == "high2":
        """
        high std without chunk
        """
        prediction_idxs_bef = prediction_idxs

        if len(remaining_idxs) == K_high :
            pick_idxs = remaining_idxs
#            remaining_idxs = {}
            remaining_idxs = np.array([])
            prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]

        elif len(remaining_idxs) != K_high :
        
            prediction_idxs = remaining_idxs

            X_train = mbtr_data[prediction_idxs, :]

            #-- normalization
            start = time.time()
            X_train_pp = desc_pp_notest(preprocess, X_train)               
            process_time = time.time() - start
            
            #-- check mean and std in next dataset
            mu_s, std_s = gpr.predict(X_train_pp, return_std=True) #mu->mean? yes
            K = K_high

            pick_idxs = np.argpartition(-std_s, K)[:K]
            pick_idxs = np.array(prediction_idxs)[pick_idxs]

            prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]

            remaining_idxs = np.setdiff1d(remaining_idxs, pick_idxs)
    
        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        #-- normalization
        start = time.time()
        X_train_pp = desc_pp_notest(preprocess, X_train)   
        process_time = time.time() - start
            
        #-- save the values 
        np.savez(out_name + "_" + str(i+1) + "_idxs.npz", remaining_idxs=remaining_idxs, prediction_idxs=prediction_idxs, pick_idxs = pick_idxs)

    return prediction_idxs, remaining_idxs, X_train_pp, y_train
                                    
