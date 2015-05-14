# train and test for left half of mnist dataset
caffe test -model=model/lenet_train_test_joint2.prototxt -weights=model/lenet_joint_iter_10000.caffemodel -iterations=100