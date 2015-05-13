%% Load MNIST data
addpath('../HMC/mnist');
X_tr = loadMNISTImages('train-images-idx3-ubyte')';
X_te = loadMNISTImages('t10k-images-idx3-ubyte')';
y_tr = loadMNISTLabels('train-labels-idx1-ubyte');
y_te = loadMNISTLabels('t10k-labels-idx1-ubyte');

%% Set label names
tr_l_label = sprintf('tr_l');
tr_r_label = sprintf('tr_r');
te_l_label = sprintf('te_l');
te_r_label = sprintf('te_r');

%% Crop left
X_tr = reshape(X_tr, [60000, 28, 28]);
X_tr_l = X_tr(:, :, 1:14);
X_tr_r = X_tr(:, :, 15:end);

%% Crop right
X_te = reshape(X_te, [10000, 28, 28]);
X_te_l = X_te(:, :, 1:14);
X_te_r = X_te(:, :, 15:end);

%% Change to row-major
X_tr_l = permute(X_tr_l, [1, 3, 2]);
X_tr_r = permute(X_tr_r, [1, 3, 2]);
X_te_l = permute(X_te_l, [1, 3, 2]);
X_te_r = permute(X_te_r, [1, 3, 2]);

%% export as leveldb
export_leveldb(reshape(X_tr_l, [60000, 28*14]),y_tr,28,14,1,tr_l_label);
export_leveldb(reshape(X_tr_r, [60000, 28*14]),y_tr,28,14,1,tr_r_label);
export_leveldb(reshape(X_te_l, [10000, 28*14]),y_te,28,14,1,te_l_label);
export_leveldb(reshape(X_te_r, [10000, 28*14]),y_te,28,14,1,te_r_label);