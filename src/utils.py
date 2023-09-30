import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import argparse


### DEFAULT PARAMETERS ###

N_BINS = 2 #16
N_HIDDEN = 128
N_EPOCHS = 80
LR = 1e-3
K = 10
DATASET_NAME = 'sift'
BATCH_SIZE = 128 #256 #2048
N_BINS_TO_SEARCH = 2
N_TREES = 2
N_LEVELS = 0 #0
TREE_BRANCHING = 1
MODEL_TYPE = 'neural'
DISTANCE_METRIC = 'euclidean'
ETA_VALUE = 7


#lets run with cpu
cpu = torch.device('cpu')
cuda = torch.device('cuda')
if torch.cuda.is_available():
    #run with gpu first
    primary_device = cuda
    secondary_device = cpu
else:
    primary_device = cpu
    secondary_device = cpu



### sos arg parser ###
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_bins', default=N_BINS, type=int, help='number of bins' )
    parser.add_argument('--k_train', default=K, type=int, help='number of neighbors during training')
    parser.add_argument('--k_test', default=K, type=int, help='number of neighbors to construct knn graph')
    parser.add_argument('--dataset_name', default=DATASET_NAME, type=str, help='Specify dataset name, can be one of "sift", "mnist"')
    parser.add_argument('--n_hidden', default=N_HIDDEN, type=int, help='hidden dimension')
    parser.add_argument('--n_epochs', default=N_EPOCHS, type=int, help='number of epochs for training')
    parser.add_argument('--lr', default=LR, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='batch size')
    parser.add_argument('--n_bins_to_search', default=N_BINS_TO_SEARCH, type=int, help='number of bins to use')
    parser.add_argument('--n_trees', default=N_TREES, type=int, help='number of trees')
    parser.add_argument('--n_levels', default=N_LEVELS, type=int, help='number of levels in tree')
    parser.add_argument('--tree_branching', default=TREE_BRANCHING, type=int, help='number of children per node in tree')
    parser.add_argument('--model_type', default=MODEL_TYPE, type=str, help='Type of model to use')
    parser.add_argument('--eta_value', default=ETA_VALUE, type=int, help='Balance parameter for the loss function, takes only integer values')
    parser.add_argument('--pca_comp',default=0,type=float,help='number components to keep in PCA')
    parser.add_argument('--distance_metric', default=DISTANCE_METRIC, type=str, help='Specify distance metric, can be one of "euclidean", "mahalanobis"')
    # modify following 2 lines to run for python 3.8
    #parser.add_argument('--load_knn', action=argparse.BooleanOptionalAction, help='Load existing k-NN matrix from file')
    #parser.add_argument('--continue_train', action=argparse.BooleanOptionalAction, help='Load existing models from file')
    parser.add_argument('--load_knn', action='store_true', help='Load existing k-NN matrix from file')
    parser.add_argument('--continue_train', action='store_true', help='Load existing models from file')
    parser.add_argument('-pl','--pipeline_pvq',action='store_true',help='Use the Ensemble Model first in a 2 step pipeline.The results will be fed into Product Vector Quantization Sim Search')
    parser.add_argument('-mc','--model_combination', nargs='+', default=[], help='Select if you want a pseudorandom combination of models in the forest instead of classic ensembling.The models must exist of course')

    #parse args
    opt = parser.parse_args()
    if opt.dataset_name not in ['sift','mnist']:
        raise ValueError('dataset_name must be one of "sift", "mnist"')

    if (opt.model_type not in ['neural', 'linear','cnn'] and opt.model_combination == None):
        raise ValueError('model_type must be one of "neural", "linear","cnn" or combination or them')
    
    
    ####### for model combination only ######                                                                                                                                                                                                                         
    if opt.model_combination != None:
        mods = []
        for mod in opt.model_combination:
            mods.append(mod)
            if mod not in ['neural', 'linear','cnn']:
                raise ValueError('model_type must be one of "neural", "linear","cnn" or combination or them')
            #print(mod)        
    #######################################                                                                                                                                                                                                                                

    return opt




def get_test_accuracy(model_forest, knn, X_test, k, batch_size=1024, bin_count_param=1, models_path=None,pq=False):



    if models_path == None:
        print('no file directory for models found')
        return
    # assert isinstance(root_model, Model)




    print('-----DOING MODEL INFERENCE ------- ')


    # do the nearest neighbors search here
    query_bins, scores, dataset_bins = model_forest.infer(X_test, batch_size, bin_count_param, models_path)

    print('----- MODEL INFERENCE DONE ------- ')




    n_q = query_bins.shape[1] # no of points in test set only

    all_points_bins = []

    #model ensembling accuracies
    ensemble_accuracies = [] # array of accuracies
    ensemble_cand_set_sizes = []

    single_model = model_forest.trees[0].root.model

    print('no of parameters in one model: {}'.format(sum(p.numel() for p in single_model.parameters())))

    n_trees = model_forest.n_trees

    del model_forest

    torch.cuda.empty_cache()


    print('----- CALCULATING K-NN RECALL FOR EACH POINT ------- ')


    #set up total query time for all points
    # maybe try starting time inside the num models as well (and keep timing)

    for num_models in range(n_trees):
    # for num_models in range(n_trees- 1, n_trees): # TAKING ALL TREES AT ONCE

        accuracies = []

        X = []

        # should we time the total time for all queries ???
        for bin_count in range(1, bin_count_param + 1, 1):



            num_knns = torch.randn(n_q, 1)
            candidate_set_sizes = torch.randn(n_q, 1)


            print("%d models, %d bins "%(num_models + 1, bin_count))
            print()

            # Save total running time for test
            running_time = []

            # loop through all query points
            for point in range(n_q):
                c2_time = time.time()

                #print('\rpoint ' + str(point) + ' / ' + str(n_q), end='')
                max_i = -1


                max_i = torch.argmax(scores[:(num_models+1)], 0)[point].flatten()

                # bins for all query points
                assigned_bins = query_bins[max_i, point, :].flatten()

                all_points_bins.append(assigned_bins[0].item())

                #get candidate set_points
                candidate_set_points = sum(dataset_bins[max_i].flatten() == b for b in assigned_bins[:bin_count]).nonzero(as_tuple=False).flatten()




                c3_time = time.time()
                t2_time = c3_time - c2_time
                running_time.append(t2_time)

                # FIND CANDIDATE SET OF QUERY POINT END
                candidate_set_size = candidate_set_points.shape[0]
                #actual neighbors
                knn_points = knn[point][:k] # choose first k points for testing
                knn_points_size = knn_points.shape[0]


                # find size of overlap between knn_points and bin_points
                knn_and_bin_points = torch.cat((candidate_set_points.cuda(), knn_points.cuda()))

                uniques = torch.unique(knn_and_bin_points)

                uniques_size = uniques.shape[0]

                overlap = candidate_set_size + knn_points_size - uniques_size

                num_knns[point] = overlap

                candidate_set_sizes[point] = candidate_set_size



            pass



            accuracy = num_knns / k
            print()

            accuracy = torch.mean(accuracy)

            print('mean accuracy ', accuracy)
            candidate_set_size = torch.mean(candidate_set_sizes)
            print("mean candidate set size", candidate_set_size)
            print()

            # Computing average running time
            average_query_time = sum(running_time) / n_q

            # Computing standard deviation of running times
            std_query_time = np.std(running_time)
            del running_time

            # Printing in milliseconds
            print('average query time: %.2f, standard deviation: %.2f' % (average_query_time * 1000, std_query_time * 1000))

            accuracies.append(accuracy.item()) # for each bin_count

            X.append(candidate_set_size.item())
        pass
        ensemble_accuracies.append(accuracies)
        ensemble_cand_set_sizes.append(X)


    
    
    final_candidates_dict = None
    
    #########################  only for pq  ###########################
    if(pq==True):
            final_candidates = []
            #final_candidates = torch.empty(n_q,dataset_bins.shape[1])                                                                                                                                                                                                 
            for point in range(n_q):
                # find most confident model for each point                                                                                                                                                                                                             
                max_i = torch.argmax(scores[:(bin_count_param)], 0)[point].flatten()
                assigned_bins = query_bins[max_i, point, :].flatten()
                candidate_set_points = sum(dataset_bins[max_i].flatten() == b for b in assigned_bins[:bin_count_param]).nonzero(as_tuple=False).flatten()
                final_candidates.append(( point,candidate_set_points.tolist()) )
                #final_candidates[point,:] = candidate_set_points.flatten()                                                                                                                                                                                            
        
            # create dictionary of final candidates for each query point #                                                                                                                                                                                             
            final_candidates_dict = dict(final_candidates)
            print(final_candidates_dict[0])

            
    if bin_count_param > 0:

        print("first bin accuracy")
        print(accuracies[0])

        print("candidate_set_size of first bin on average")
        print(X[0])
        
        plt.figure()
        for m, acc in enumerate(ensemble_accuracies):
            plt.plot(ensemble_cand_set_sizes[m], acc, marker="x", label="no of models: " + str(m+1))

        plt.legend()
        plt.title("Average k-NN Recall vs Candidate Set Size")
        # Create 'outputs' directory if not exists (done elsewhere in the code, so not needed here)
        # if not os.path.exists('outputs'):
        #   os.makedirs('outputs')

        #plt.savefig('/kaggle/working/knn_recall_vs_cand_set_size.png')
        plt.savefig('outputs/knn_recall_vs_cand_set_size.png')
       # plt.show()

    return final_candidates_dict,all_points_bins, ensemble_cand_set_sizes, ensemble_accuracies
