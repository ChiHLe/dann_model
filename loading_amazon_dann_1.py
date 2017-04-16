from sklearn.datasets import load_svmlight_files
from sklearn import svm
import numpy as np
def load_amazon(source_name, target_name, data_folder=None, verbose=False):
    """
    Load the amazon sentiment datasets from svmlight format files
    inputs:
        source_name : name of the source dataset
        target_name : name of the target dataset
        data_folder : path to the folder containing the files
    outputs:
        xs : training source data matrix
        ys : training source label vector
        xt : training target data matrix
        yt : training target label vector
        xtest : testing target data matrix
        ytest : testing target label vector
    """

    if data_folder is None:
        data_folder = 'data/'

    source_file = data_folder + source_name + '_train.svmlight'
    source_test_file = data_folder + source_name + '_test.svmlight'
    test_file = data_folder + target_name + '_test.svmlight'
    target_file = data_folder + target_name +'_train.svmlight'
    if verbose:
        print('source file:', source_file)
        print('source test file:', source_test_file)
        print('target test file:  ', test_file)
        print('target file', target_file)

    xs, ys, xt, yt, xtest, ytest, x_target, y_target = load_svmlight_files([source_file, source_test_file, test_file, target_file])

    # Convert sparse matrices to numpy 2D array
    xs, xt, xtest, x_target = (np.array(X.todense()) for X in (xs, xt, xtest, x_target))

    # Convert {-1,1} labels to {0,1} labels
    ys, yt, ytest = (np.array((y + 1) / 2, dtype=int) for y in (ys, yt, ytest))
    d_train = xs[:1000,:]
    domain_train = np.zeros((1000,5001))
    domain_train[:,:-1] = d_train

    d_target = x_target[:1000,:]
    domain_target = np.ones((1000,5001))
    domain_target[:,:-1] = d_target
    
    
   
    domain_data = np.concatenate((domain_train, domain_target))
    
    domain_data = np.random.permutation(domain_data)
   # print(domain_data)
    
    x_domain = np.asarray(domain_data[:, :-1])
    y_domain = np.asarray(domain_data[:,-1:])
   # print(x_domain)
   # print(y_domain)
    return xs, ys, xt, yt, xtest, ytest, x_domain, y_domain
