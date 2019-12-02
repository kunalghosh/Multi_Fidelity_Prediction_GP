import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from utils import desc_pp, desc_pp_notest,pre_rem_split
from io_utils import append_write, out_time

def acq_fn(fn_name, i, prediction_idxs, remaining_idxs, prediction_set_size, rnd_size, mbtr_data,homo_lowfid, K_high , gpr, preprocess, out_name):
    if  fn_name == "none":
        """
        A. random sampling 
        """
        prediction_idxs_bef = prediction_idxs

        prediction_idxs, remaining_idxs = pre_rem_split(prediction_set_size, remaining_idxs)

        pick_idxs = prediction_idxs
        
        prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]

        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        #-- Preprocessing
        X_train_pp = desc_pp_notest(preprocess, X_train)   

        #-- save the values 
        np.savez(out_name + "_" + str(i+1) + "_idxs.npz", remaining_idxs=remaining_idxs, prediction_idxs = prediction_idxs, pick_idxs = pick_idxs)

    elif  fn_name == "rnd":
        """
        random sampling with chunk
        """
        prediction_idxs_bef = prediction_idxs

        prediction_idxs, remaining_idxs = pre_rem_split(prediction_set_size, remaining_idxs)
       
        pick_idxs,_ = train_test_split(prediction_idxs, train_size = rnd_size)
        
        prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]
       
        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]
       
        #-- Preprocessing
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

        #-- Preprocessing
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


        #-- Preprocessing
        start = time.time()
        X_train_pp = desc_pp_notest(preprocess, X_train)   
        process_time = time.time() - start

        #-- save the values 
        np.savez(out_name + "_" + str(i+1) + "_idxs.npz", remaining_idxs=remaining_idxs, prediction_idxs=prediction_idxs, pick_idxs = pick_idxs)

    elif fn_name == "cluster":
        from sklearn import cluster
        from sklearn.metrics import pairwise_distances_argmin_min

        """
        C. Clustering without chunk
        """

        prediction_idxs_bef = prediction_idxs

        #-- without chunk
        prediction_idxs = remaining_idxs

        X_train = mbtr_data[prediction_idxs, :]

        #-- Preprocessing                                
        X_train_pp = desc_pp_notest(preprocess, X_train)            

        num_clusters = K_high

        #-- clustering
        start = time.time()
        append_write(out_name,"starting clustering \n")
        km = cluster.KMeans(n_clusters = num_clusters, n_jobs = 24)
        z_km = km.fit(X_train_pp)
        process_time = time.time() - start
        out_time(out_name, process_time)
        
        labels = np.array(z_km.labels_)
        centers = np.array(z_km.cluster_centers_)
        append_write(out_name,"length of centers " + str(len(centers)) + "\n")
        
        start = time.time()
        append_write(out_name,"starting calculat nearest points of centers \n")
        closest, _ = pairwise_distances_argmin_min(centers, X_train_pp)
        append_write(out_name,"number of closest points " + str(len(closest)) + "\n")
        process_time = time.time() - start
        out_time(out_name, process_time)
        
        pick_idxs = np.array(prediction_idxs)[closest]
        append_write(out_name,"length of pick idxs " + str(len(pick_idxs)) + "\n")

        #-- length of cluster
#        cluster_idxs = np.empty(num_clusters)
#        cluster_len = np.empty(num_clusters)

#        for j in range(num_clusters):
#            cluster_idxs[j] = np.array(np.where(labels == j)).flatten()
#            cluster_len[j] = len(cluster_idxs[j])
            
#        np.save(out_name + "_" + str(i+1) + "_cluster_len", cluster_len )
        
        prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]
        remaining_idxs = np.setdiff1d(remaining_idxs, pick_idxs)

        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]
        #-- Preprocessing                                                        
        X_train_pp = desc_pp_notest(preprocess, X_train)

        #-- save the values 
        np.savez(out_name + "_" + str(i+1) + "_idxs.npz", remaining_idxs=remaining_idxs, prediction_idxs=prediction_idxs, pick_idxs = pick_idxs)        

    elif  fn_name == "rnd2":
        """
        A .random sampling without chunk (same as none)
        """
        prediction_idxs_bef = prediction_idxs

        prediction_idxs, remaining_idxs = pre_rem_split(prediction_set_size, remaining_idxs)        
            
        pick_idxs = prediction_idxs

        prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]
        
        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        #-- Preprocessing                                                                                                                                                   
        start = time.time()
        X_train_pp = desc_pp_notest(preprocess, X_train)
        process_time = time.time() - start

        #-- save the values 
        np.savez(out_name + "_" + str(i+1) + "_idxs.npz", remaining_idxs=remaining_idxs, prediction_idxs=prediction_idxs, pick_idxs = pick_idxs)


    elif fn_name == "cluster2":
        from sklearn import cluster
        from sklearn.metrics import pairwise_distances_argmin_min
        """
        clustering with checking each cluster (Too heavy calculation)
        """

        prediction_idxs_bef = prediction_idxs

        #-- without chunk
        prediction_idxs = remaining_idxs

        X_train = mbtr_data[prediction_idxs, :]

        #-- Preprocessing                                                                                                                                
        X_train_pp = desc_pp_notest(preprocess, X_train)            

        num_clusters = K_high

        #-- clustering
        start = time.time()
        append_write(out_name,"starting clustering" + "\n")
        km = cluster.KMeans(n_clusters = num_clusters, n_jobs = 1)
        z_km = km.fit(X_train_pp)
        process_time = time.time() - start
        out_time(out_name, process_time)
        
        labels = np.array(z_km.labels_)
        centers = np.array(z_km.cluster_centers_)
        append_write(out_name,"length of centers " + str(len(centers)) + "\n")
        
        start = time.time()
        append_write(out_name,"starting calculat nearest points of centers \n")
        closest, _ = pairwise_distances_argmin_min(centers, X_train_pp)
        append_write(out_name,"number of closest points " + str(len(closest)) + "\n")
        process_time = time.time() - start
        out_time(out_name, process_time)
        
        pick_idxs = np.array(prediction_idxs)[closest]
        append_write(out_name,"length of pick idxs " + str(len(pick_idxs)) + "\n")

        #-- cluster check
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
            
            #-- Preprocessing                                                               
            X_train_pp = desc_pp_notest(preprocess, X_train)
            
            mu_s, std_s = gpr.predict(X_train_pp, return_std=True) #mu->mean? yes

            cluster_len[j] = len(cluster_idxs[j])
            cluster_std[j] = std_s
            cluster_avg[j] = np.average(cluster_std[j]) 
            cluster_max[j] = np.max(cluster_std[j]) 
            
        print(cluster_len)
        print(cluster_avg)

        np.save(out_name + "_" + str(i+1) + "_cluster_std", cluster_std )
        np.save(out_name + "_" + str(i+1) + "_cluster_len", cluster_len )
        np.save(out_name + "_" + str(i+1) + "_cluster_avg", cluster_avg )
        np.save(out_name + "_" + str(i+1) + "_cluster_max", cluster_max )        
        np.save(out_name + "_" + str(i+1) + "_cluster_idxs", pick_idxs )        
        
        prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]
        remaining_idxs = np.setdiff1d(remaining_idxs, pick_idxs)

        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        #-- Preprocessing                                                                                                                                                 
        X_train_pp = desc_pp_notest(preprocess, X_train)

        #-- save the values 
        np.savez(out_name + "_" + str(i+1) + "_idxs.npz", remaining_idxs=remaining_idxs, prediction_idxs=prediction_idxs, pick_idxs = pick_idxs)  


            
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
        prediction_idxs_bef = prediction_idxs

        if len(remaining_idxs) == K_high :
            pick_idxs = remaining_idxs
            remaining_idxs = np.array([])
            prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]
            
        elif len(remaining_idxs) != K_high :
        
            prediction_idxs = remaining_idxs

            X_train = mbtr_data[prediction_idxs, :]

            #-- Preprocessing                                                                                                                                                    
            start = time.time()
            X_train_pp = desc_pp_notest(preprocess, X_train)               
            process_time = time.time() - start
            
            #-- check mean and std in next dataset
            append_write(out_name,"starting prediction \n")
            start = time.time()
            mu_s, std_s = gpr.predict(X_train_pp, return_std=True) #mu->mean? yes
            process_time = time.time() - start
            out_time(out_name, process_time)
            
            #-- Histgram of high std
            plt.figure()
            plt.title('', fontsize = 20)
            plt.xlabel('std.', fontsize = 16)
            plt.ylabel('Number of molecules', fontsize = 16)
            plt.tick_params(labelsize = 14)
            #    plt.xlim(0, 90)
            #    plt.ylim(0, 2500)
            (a_hist2, a_bins2, _) = plt.hist(std_s[:], bins=170)
            plt.savefig(out_name + "_" + str(i+1) + "_std.eps")
            
            #-- unsorted top K idxs  
            K = K_high
            pick_idxs = np.argpartition(-std_s, K)[:K]

            #-- check 
            append_write(out_name, "Max value of std within pick_idxs " + str(np.max(std_s[pick_idxs])) + "\n" )
            append_write(out_name, "Min value of std within pick_idxs " + str(np.min(std_s[pick_idxs])) + "\n" )
            append_write(out_name, "Max value of std within all remaining_idxs " + str(np.max(std_s[np.setdiff1d(range(len(prediction_idxs)), pick_idxs)])) + "\n" )
            append_write(out_name, "Min value of std within all remaining_idxs " + str(np.min(std_s[:])) + "\n" )

            append_write(out_name, "The # of zero values " + str(np.count_nonzero(std_s < 1e-10)) + "\n" )
            append_write(out_name, "The # of not zero values " + str(np.count_nonzero(std_s > 1e-10)) + "\n" )

            
            pick_idxs = np.array(prediction_idxs)[pick_idxs]

            prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]

            remaining_idxs = np.setdiff1d(remaining_idxs, pick_idxs)
    
            np.save(out_name + "_high_std_" + str(i+1), std_s)

        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        #-- Preprocessing                                                                                                                                                  
        start = time.time()
        X_train_pp = desc_pp_notest(preprocess, X_train)   
        process_time = time.time() - start
            


        np.savez(out_name + "_" + str(i+1) + "_idxs.npz", remaining_idxs=remaining_idxs, prediction_idxs=prediction_idxs, pick_idxs = pick_idxs)

    elif fn_name == "high_and_cluster":
        """
        D. Combination of B and C
        without chunk,
        1. Choose mols with high std 
        2. Make cluster
        3. Choose mols which is near the center of clusters
        """
        prediction_idxs_bef = prediction_idxs

        if len(remaining_idxs) == K_high :
            pick_idxs = remaining_idxs
            remaining_idxs = np.array([])
            prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]

        elif len(remaining_idxs) != K_high :
        
            prediction_idxs = remaining_idxs

            X_train = mbtr_data[prediction_idxs, :]

            #-- Preprocessing                                                                                                                                                    
            start = time.time()
            X_train_pp = desc_pp_notest(preprocess, X_train)               
            process_time = time.time() - start
            
            #-- check mean and std in next dataset
            append_write(out_name,"starting prediction \n")
            start = time.time()
            mu_s, std_s = gpr.predict(X_train_pp, return_std=True) #mu->mean? yes
            process_time = time.time() - start
            out_time(out_name, process_time)

            #-- Histgram of high std
            plt.figure()
            plt.title('', fontsize = 20)
            plt.xlabel('std.', fontsize = 16)
            plt.ylabel('Number of molecules', fontsize = 16)
            plt.tick_params(labelsize = 14)
            #    plt.xlim(0, 90)
            #    plt.ylim(0, 2500)
            (a_hist2, a_bins2, _) = plt.hist(std_s[:], bins=170)
            plt.savefig(out_name +  + "_" + str(i+1) + "_std.eps")
            
            #-- unsorted top K idxs  
            K = int(len(remaining_idxs)/2.0)
            pick_idxs = np.argpartition(-std_s, K)[:K]

            #-- check 
            append_write(out_name,"Max value of std within pick_idxs " + str(np.max(std_s[pick_idxs])) + "\n" )
            append_write(out_name,"Min value of std within pick_idxs " + str(np.min(std_s[pick_idxs])) + "\n" )
            append_write(out_name,"Max value of std within all remaining_idxs " + str(np.max(std_s[np.setdiff1d(range(len(prediction_idxs)), pick_idxs)])) + "\n" )
            append_write(out_name,"Min value of std within all remaining_idxs " + str(np.min(std_s[:])) + "\n" )

            pick_idxs = np.array(prediction_idxs)[pick_idxs]
            
#            prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]
#            remaining_idxs = np.setdiff1d(remaining_idxs, pick_idxs)


        #-------
        X_train = mbtr_data[pick_idxs, :]
        y_train = homo_lowfid[pick_idxs]

        #--
        num_clusters = K_high

        #-- clustering
        start = time.time()
        append_write(out_name,"starting clustering \n")
        km = cluster.KMeans(n_clusters = num_clusters, n_jobs = 24)
        z_km = km.fit(X_train_pp)
        process_time = time.time() - start
        out_time(out_name, process_time)
        
        labels = np.array(z_km.labels_)
        centers = np.array(z_km.cluster_centers_)
        append_write(out_name,"length of centers " + str(len(centers)) + "\n")
        
        start = time.time()
        append_write(out_name,"starting calculat nearest points of centers \n")
        closest, _ = pairwise_distances_argmin_min(centers, X_train_pp)
        append_write(out_name,"number of closest points " + str(len(closest)) + "\n")
        process_time = time.time() - start
        out_time(out_name, process_time)
        
#        pick_idxs = np.array(prediction_idxs)[closest]
        pick_idxs = np.array(pick_idxs)[closest]
        append_write(out_name,"length of pick idxs " + str(len(pick_idxs)) + "\n")
        
        prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]
        remaining_idxs = np.setdiff1d(remaining_idxs, pick_idxs)

        #-------
        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        #-- Preprocessing                                                                                                                        
        start = time.time()
        X_train_pp = desc_pp_notest(preprocess, X_train)   
        process_time = time.time() - start
            
        #-- save the values 
        np.savez(out_name + "_" + str(i+1) + "_idxs.npz", remaining_idxs=remaining_idxs, prediction_idxs=prediction_idxs, pick_idxs = pick_idxs)






        

    elif fn_name == "cluster_and_high":
        """
        D. Combination of B and C
        aaaaa
        aaaaa
        without chunk,
        1. Choose mols with high std 
        2. Make cluster
        3. Choose mols which is near the center of clusters
        """
        prediction_idxs_bef = prediction_idxs

        if len(remaining_idxs) == K_high :
            pick_idxs = remaining_idxs
            remaining_idxs = np.array([])
            prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]

        elif len(remaining_idxs) != K_high :
        
            prediction_idxs = remaining_idxs

            X_train = mbtr_data[prediction_idxs, :]

            #-- Preprocessing                                                                                                                                                    
            start = time.time()
            X_train_pp = desc_pp_notest(preprocess, X_train)               
            process_time = time.time() - start
            
            #-- check mean and std in next dataset
            append_write(out_name,"starting prediction \n")
            start = time.time()
            mu_s, std_s = gpr.predict(X_train_pp, return_std=True) #mu->mean? yes
            process_time = time.time() - start
            out_time(out_name, process_time)

            #-- Histgram of high std
            plt.figure()
            plt.title('', fontsize = 20)
            plt.xlabel('std.', fontsize = 16)
            plt.ylabel('Number of molecules', fontsize = 16)
            plt.tick_params(labelsize = 14)
            #    plt.xlim(0, 90)
            #    plt.ylim(0, 2500)
            (a_hist2, a_bins2, _) = plt.hist(std_s[:], bins=170)
            plt.savefig(out_name +  + "_" + str(i+1) + "_std.eps")
            
            #-- unsorted top K idxs  
            K = int(len(remaining_idxs)/2.0)
            pick_idxs = np.argpartition(-std_s, K)[:K]

            #-- check 
            append_write(out_name,"Max value of std within pick_idxs " + str(np.max(std_s[pick_idxs])) + "\n" )
            append_write(out_name,"Min value of std within pick_idxs " + str(np.min(std_s[pick_idxs])) + "\n" )
            append_write(out_name,"Max value of std within all remaining_idxs " + str(np.max(std_s[np.setdiff1d(range(len(prediction_idxs)), pick_idxs)])) + "\n" )
            append_write(out_name,"Min value of std within all remaining_idxs " + str(np.min(std_s[:])) + "\n" )

            pick_idxs = np.array(prediction_idxs)[pick_idxs]
            
#            prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]
#            remaining_idxs = np.setdiff1d(remaining_idxs, pick_idxs)


        #-------
        X_train = mbtr_data[pick_idxs, :]
        y_train = homo_lowfid[pick_idxs]

        #-- Preprocessing                                                                                                                                                    
        start = time.time()
        X_train_pp = desc_pp_notest(preprocess, X_train)               
        process_time = time.time() - start
        
        #--
        num_clusters = K_high

        #-- clustering
        start = time.time()
        append_write(out_name,"starting clustering \n")
        km = cluster.KMeans(n_clusters = num_clusters, n_jobs = 24)
        z_km = km.fit(X_train_pp)
        process_time = time.time() - start
        out_time(out_name, process_time)
        
        labels = np.array(z_km.labels_)
        centers = np.array(z_km.cluster_centers_)
        append_write(out_name,"length of centers " + str(len(centers)) + "\n")
        
        start = time.time()
        append_write(out_name,"starting calculat nearest points of centers \n")
        closest, _ = pairwise_distances_argmin_min(centers, X_train_pp)
        append_write(out_name,"number of closest points " + str(len(closest)) + "\n")
        process_time = time.time() - start
        out_time(out_name, process_time)
        
#        pick_idxs = np.array(prediction_idxs)[closest]
        pick_idxs = np.array(pick_idxs)[closest]
        append_write(out_name,"length of pick idxs " + str(len(pick_idxs)) + "\n")
        
        prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]
        remaining_idxs = np.setdiff1d(remaining_idxs, pick_idxs)

        #-------
        X_train = mbtr_data[prediction_idxs, :]
        y_train = homo_lowfid[prediction_idxs]

        #-- Preprocessing                                                                                                                        
        start = time.time()
        X_train_pp = desc_pp_notest(preprocess, X_train)   
        process_time = time.time() - start
            
        #-- save the values 
        np.savez(out_name + "_" + str(i+1) + "_idxs.npz", remaining_idxs=remaining_idxs, prediction_idxs=prediction_idxs, pick_idxs = pick_idxs)

    else:
        append_write(out_name,fn_name + "\n")    
        append_write(out_name,"You should use defined acquisition function ! \n")    
        append_write(out_name,"program stopped ! \n")    
        sys.exit()        
    return prediction_idxs, remaining_idxs, X_train_pp, y_train
                                    
