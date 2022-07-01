import matplotlib.pyplot as plt
from sklearn import manifold

def fig_atom(df_62k, idxs, fig_name):
    plt.figure()
    num_atoms = df_62k["number_of_atoms"].values
    plt.title('', fontsize = 20)
    plt.xlabel('Number of atoms', fontsize = 16)
    plt.ylabel('Number of molecules', fontsize = 16)
    plt.tick_params(labelsize=14)
#    plt.xlim(0, 90)
#    plt.ylim(0, 2500)
    (a_hist2, a_bins2, _) = plt.hist(num_atoms[idxs], bins=170)
    plt.savefig(fig_name)
    
def fig_HOMO(homo_lowfid, idxs, fig_name):
    plt.figure()
    plt.title('Histogram of HOMO energy', fontsize = 20)
    plt.xlabel('Energy', fontsize = 16)
    plt.ylabel('', fontsize = 16)
    plt.tick_params(labelsize=14)
#    plt.xlim(-10, 0)
#    plt.ylim(0, 6000)
    (a_hist2, a_bins2, _) = plt.hist(homo_lowfid[idxs], bins=70)
    plt.savefig(fig_name) 

def fig_high_std(std_s, fig_name):
    #-- Histgram of high std
    plt.figure()
    plt.title('', fontsize = 20)
    plt.xlabel('std.', fontsize = 16)
    plt.ylabel('Number of molecules', fontsize = 16)
    plt.tick_params(labelsize = 14)
    #    plt.xlim(0, 90)
    #    plt.ylim(0, 2500)
    (a_hist2, a_bins2, _) = plt.hist(std_s[:], bins=170)
    plt.savefig(fig_name)

def fig_scatter_r2(y_test, mu_s, fig_name):

    plt.figure()
    plt.title('', fontsize = 20)
    plt.xlabel('Reference HOMO [eV]', fontsize = 16)
    plt.ylabel('Predicted HOMO [eV]', fontsize = 16)
    plt.tick_params(labelsize = 14)
    plt.scatter(y_test, mu_s)
    plt.savefig(fig_name)
    
def fig_MDS_scatter_std(x, std, fig_name):
    plt.figure()
    mds = manifold.MDS(n_components=2, dissimilarity="euclidean", random_state=6)
    pos = mds.fit_transform(x)
    plt.scatter(pos[:, 0], pos[:, 1], marker = 'o', c = std)    
    plt.show()

def fig_MDS_scatter_label(x, label, fig_name):
    plt.figure()
    mds = manifold.MDS(n_components=2, dissimilarity="euclidean", random_state=6)
    pos = mds.fit_transform(x)
    plt.scatter(pos[:, 0], pos[:, 1], marker = 'o', c = label)
    plt.show()
