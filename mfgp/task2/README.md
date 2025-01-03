In this task we do active learning by growing the training set of GP iteratively, 
by picking the test points with the highest predictive variance.

1. Run `init_train_idxs.py <int: dataset size> <int: initial training set size>` : Creates a `train_idxs.npz` file with the initial
set of training indices. e.g `python init_train_idxs.py 64000 1000` 

2. Run `train_gp.py train_idxs.npz` : Trains an exact GP on the entire list of indices in `train_idxs.npz`. If there are multiple rows
it is flattened and the training is done on the entire dataset. This also generates a new file `predictive_means_and_vars_iter{i}.npz`
which contains the predictive means and variances of the test set in the i^{th} iteration. This also generates a histogram of 
predictive variances and absolute errors. `predictive_variances_hist_{i}.png` and `absolute_errors_hist_{i}.png`.

3. Run `grow_training_data.py predictive_means_and_vars_iter{i}.npz` appends another row in the `train_idxs.npz` by picking the top 100
molecules (inidices) with highest predictive variance.

Steps 2 and 3 are repeated until the highest predictive variance in `predictive_variances_hist_{i}.png` drops below a pre-determined
(given by domain experts) threshold.

4. test shingo teranishi