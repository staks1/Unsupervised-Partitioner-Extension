import torch
from data_loader import MyDataset
from loss_fn import MyLoss

import h5py
from modelforest import ModelForest

import utils
import prepare

from torch.nn.functional import normalize

import numpy as np
from sklearn.decomposition import PCA

import os.path
import os
from pipeline_pq_function import quantizeQueryPoints

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:700"

def read_paths():
    with open('paths.txt', 'r') as file:
        lines = file.readlines()

    paths = {}
    for line in lines:

        if line[0] == '#' or '=' not in line:
            continue
        line_l = line.split('=')
        paths[line_l[0].strip()] = line_l[1].strip()
    if 'path_to_mnist' not in paths or 'path_to_sift' not in paths or 'path_to_knn_matrix' not in paths or 'path_to_models' not in paths or 'path_to_tensors' not in paths:
        raise Exception('Paths must have path_to_mnist, path_to_sift, path_to_knn_matrix, path_to_tensors, and path_to_models')
    return paths




def run(n_bins, epochs, param_lr, n_hidden_params=128, num_levels=0, tree_branching=1, model_type='neural', custom_dataset=None, do_training=False, data='mnist', prepare_knn=True, num_models=2, batch_size=1024, k_train=10, k_inference=10, n_bins_to_search=1, continue_train=False, eta_value=7, pca_comp=0, distance_metric='euclidean',pipeline_pvq=False,model_comb=None):
    
    ######## model combination only #####
    mods = []
    with_comb = False
    #model_type = opt.model_type                                                                                                                                                                                                                               
    #Check if model combination or single model is selected for the whole ensemble_forest                                                                                                                                                                      
    if(model_comb==[]):
        print('n OK training and prediction on the forest will be done using one model only ')
        # probably function should take care of things here                                                                                                                                                                                                    
    else :
        # get unique models                                                                                                                                                                                                                                    
        for mod in model_comb:
            mods.append(mod)
        mods = list(set(mods))
        if(len(mods)==1):
            raise ValueError('You selected combination of models but only picked one model')
        with_comb = True
        #just for checking                                                                                                                                                                                                                                     
        print(mods)
        print('n OK training and prediction on the forest will be done using combination of selected models ')
        
        ##### for model combination only #####
        model_type=None    
        ######################################

    
    levels = num_levels # no of levels of models
    branching = tree_branching # no of children per model each level

    bin_count = n_bins_to_search

    if levels == 0:
        branching = n_bins # build only a single model, not tree


    print("RUNNING WITH: epochs="+str(epochs) +
          "; lr="+str(param_lr)+"; n_bins="+str(n_bins)+"; levels="+str(levels) + "; branching="+str(branching) +"; model type="+str(model_type))

    print(" -- READING PATHS -- ")

    paths = read_paths()


    primary_device = utils.primary_device
    secondary_device = utils.secondary_device


    # get training dataset
    if data == 'sift':
        dataset = MyDataset(paths['path_to_sift'])
    elif data == 'mnist':
      dataset = MyDataset(paths['path_to_mnist'])
    pass

    # if data != 'custom':
    #     print(dataset.get_file().keys())



    if custom_dataset is not None:
        X = custom_dataset
    else:
        X = dataset.get_file()['train']

    n_data = X.shape[0]

    ###########################################################
    # Check if PCA dimensionality reduction should be applied
    if pca_comp is not None and pca_comp > 0:

        # Convert pca_comp to integer if it is greater than 1
        # Assuming that when pca_comp > 1, it represents the number of components
        if pca_comp > 1:
            num_components = int(pca_comp)
        else:
            num_components = pca_comp  # when 0 < pca_comp <= 1, it represents variance

        # Apply PCA transformation
        # If num_components < 1, it indicates the ratio of variance to be retained
        if num_components < 1:
            pca = PCA(n_components=num_components, svd_solver='full')
            X_transformed = pca.fit_transform(X)

            explained_variance = pca.explained_variance_ratio_
            sum_explained_variance = sum(explained_variance)
            num_components_kept = X_transformed.shape[1]

            print(f'\nYou have kept variance >= {num_components}, '
                  f'Explained variance for each component: {explained_variance}\n')
            print(f'You have kept {num_components_kept} out of {X.shape[1]} features.\n')

        # If num_components >= 1, it indicates the number of components to be retained
        else:
            pca = PCA(n_components=num_components)
            X_transformed = pca.fit_transform(X)

            explained_variance = pca.explained_variance_ratio_
            sum_explained_variance = sum(explained_variance)

            print(f'\nYou have kept {num_components} out of {X.shape[1]} features, '
                  f'Explained variance for each component: {explained_variance}\n')
            print(f'Sum of explained variances: {sum_explained_variance}\n')
    
    
    ###### add pq ##################
    if(pipeline_pvq):
        X_train_pq  = dataset.get_file()['train']
        X_test_pq = dataset.get_file()['test']
        Y_test_pq = dataset.get_file()['neighbors']

    ###########################################################


    # prepare knn ground truth
    class options(object):
        if data == 'sift':
            normalize_data=False
            sift=True
        elif data == 'mnist':
            normalize_data=False
            sift=False
            pass
        pass

    pass


    ###################### If the KNN file does not exist --> we find the Knn and create it ##############
    if prepare_knn:

        print('preparing knn with k = ', k_train)

        Y = prepare.dist_rank(torch.tensor(X, dtype=float), k_train, opt=options, data=data, distance_metric = distance_metric)


        f = h5py.File(paths['path_to_knn_matrix'] +  '/' + data + '-' + str(k_train) + '-nn.hdf5', 'w')

        arr = f.create_dataset('train', data=Y)
        f.close()
    else:
    ###################################### the knn file exists so we just load it ###########################
        if data == 'sift':
             train_dataset = MyDataset(paths['path_to_knn_matrix'] +  '/sift-' + str(k_train) + '-nn.hdf5')
        else:
            train_dataset = MyDataset(paths['path_to_knn_matrix'] + '/fashion_mnist_' + str(k_train) + '_nn.hdf5')
        pass

        Y = train_dataset.get_file()['train']
    pass


    if data != 'custom':
        test_dataset = dataset.get_file()['test']





    print("Preparing dataset tensor")

    if custom_dataset is None:
        if(os.path.isfile(paths['path_to_models'] + '/' + data +  '-X.pt')):
            X = torch.load(paths['path_to_models'] + '/' + data +  '-X.pt')
        else:
            X = torch.tensor(X, dtype=torch.double, device=primary_device)
            torch.save(X, paths['path_to_models'] + '/' + data + '-X.pt')

    else:

        if(os.path.isfile(paths['path_to_models'] + '/' + data +  '-X.pt') and not do_training):
            X = torch.load(paths['path_to_models'] + '/' + data +  '-X.pt')
        else:
            X = torch.tensor(X, dtype=torch.double, device=primary_device)
            torch.save(X, paths['path_to_models'] + '/' + data + '-X.pt')



    pass

    print("prepping k-NN Matrix tensor")
    if(os.path.isfile(paths['path_to_models'] + '/' + data +  '-Y.pt') and not prepare_knn):
        Y = torch.load(paths['path_to_models'] + '/' + data +  '-Y.pt')
    else:
        Y = torch.tensor(Y, dtype=torch.double , device=primary_device)
        torch.save(Y, paths['path_to_models'] + '/' + data + '-Y.pt')

    pass



    # save tensor to file
    if data != 'custom':
        X_test = torch.tensor(test_dataset, device=primary_device)
        Y_test = torch.tensor(dataset.get_file()['neighbors'], device=primary_device)
    else:
        X_test = X
        Y_test = Y

    # build model

    n_bins = n_bins
    input_dim = X.shape[1]



    # BUILD MODEL FOREST
    ############## create the model forest here ########################
    model_forest = ModelForest(with_comb,mods,num_models, input_dim, branching, levels, n_bins, n_hidden_params, model_type=model_type)



    print('BUILDING TREE')
    if do_training:
        if continue_train:
            model_forest.build_forest(load_from_file=True, data=data, models_path=paths['path_to_models'])
        else:
            model_forest.build_forest(load_from_file=False, data=data, models_path=paths['path_to_models'])
    else:
        model_forest.build_forest(load_from_file=True, data=data, models_path=paths['path_to_models'])




    ## criterion
    ## crit is the Loss function !!!
    crit = MyLoss(eta=eta_value)



    # start training!
    losses = []



    n = X.shape[0]  # no of points


    single_model = model_forest.trees[0].root.model

    print('no of parameters in one model: {}'.format(sum(p.numel() for p in single_model.parameters())))
    if do_training:

        print('training forest')
        ############################ train the forest ##################################################
        model_forest.train_forest(X, Y, crit, epochs, batch_size, param_lr, data, paths['path_to_models'])


    model_forest.eval()
    with torch.no_grad():



        if custom_dataset is not None:
            X_test = X
            Y_test = Y
        # print("calculating accuracy: Y_test: {}, X_test: {}, X: {}".format(Y_test.shape, X_test.shape, X.shape))

        train_bins = None

        del X
        torch.cuda.empty_cache()


        print(' --- FINDING TEST ACCURACY --- \n')
        ############ get TEST ACCCURACY ####################
        final_candidates,bins,plot_x,plot_y = utils.get_test_accuracy(model_forest, knn=Y_test, X_test=X_test, k=k_inference, batch_size=batch_size, bin_count_param=bin_count, models_path=paths['path_to_models'],pq=pipeline_pvq)
        ### for pq only ###
        if(pipeline_pvq):
                # we wil create function here to do the product vector quantization                                                                                                                                                                            
                # quantizeQueryPoints(final_candidates,X_test_pq,Y_test_pq,X_train_pq)                                                                                                                                                                         
                # we will add option to search for query point 0 or query point 1 or...                                                                                                                                                                        
                # so this is finegrained for each query point                                                                                                                                                                                                  
                top_k = k_inference
                quantizeQueryPoints(final_candidates,X_test_pq,Y_test_pq,X_train_pq,top_k)



    return (losses, bins, train_bins, plot_x, plot_y)

if __name__ == "__main__":

    opt = utils.parse_args()

    # run(n_bins=16, epochs=100, param_lr=1e-3, n_hidden_params=128, model_type='neural', do_training=True, data='mnist', prepare_knn=True, num_models=2, batch_size=3000, k_train=10, k_inference=10, n_bins_to_search=2)

    run(n_bins=opt.n_bins, epochs=opt.n_epochs, param_lr=opt.lr, n_hidden_params=opt.n_hidden, model_type=opt.model_type, do_training=True, data=opt.dataset_name, prepare_knn=not opt.load_knn, num_models=opt.n_trees, batch_size=opt.batch_size, k_train=opt.k_train, k_inference=opt.k_test, n_bins_to_search=opt.n_bins_to_search, continue_train=opt.continue_train, eta_value=opt.eta_value, pca_comp=opt.pca_comp, distance_metric=opt.distance_metric,pipeline_pvq=opt.pipeline_pvq,model_comb=opt.model_combination)
