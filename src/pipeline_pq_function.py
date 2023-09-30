
import numpy as np
from product_quantizer import VectorQuantizer
import time

def quantizeQueryPoints(candidates,X_test,Y_test,X_train,top_k):
    # get the X_test samples in each bin
    # dims depends on dataset (784 mnist/128 sift)
    dims=X_test.shape[1]

    # this dictionary will keep for all query points it's candidate size
    len_dict = dict ( [ (x,len(candidates[x]) )  for x in candidates.keys() ] )

    # find max len
    max_len = 0
    for x in candidates.values():
        if(len(x) > max_len):
            max_len = len(x)


    # test sizes match
    assert len(candidates)==X_test.shape[0] ,"The candidates and y_test sizes should be the same"

    # tensor of shape (test_samples, max_bins, dims)
    # initialize with negatives
    # candidate size + 1 to insert thr query point on top
    X = -1 * np.ones( (X_test.shape[0] , max_len + 1, dims  ) )

    # we are going to add each query point on top (element 0)
    # in order to get their neighbors then just by simsearching(:1,top_k)

    # just for testing purposes
    # print the original X_test query points
    #print('-----\n------')
    #print(X_test[0])
    #print(X_test[1])
    #print('-----\n------')
    # construct the tensor
    for x in candidates.keys():
        # add x test query point on top
        X[x,0,:] = X_test[x]

        for i in range(1,len_dict[x] ):
            X[x,i,:]= X_train[candidates[x][i]]

    # now we can feed each query point to the quantizer
    # we will test only with first query point
    # so we get the first 2d matrix from the X tensor
    # also we slice and get only the correct vectors
    # not the '-1' paddings
    #-----------------------------------------------#


    # we will do quantization for each query point and search it's top k neighbors inside it's candidate set
    # this array will keep the indices of the returned top_k neighbors for each query point
    all_query_neighbors = np.zeros( (X_test.shape[0],top_k),dtype=int)

    # for all query points
    # just use range(0,0) to probe only the first query point for example
    # doing it for all query points is very slow so here i apply pq for only the first  query point
    #for qi in range(X_test.shape[0]):
    for qi in range(0,1):
        # SELECT QUERY AND IT'S RETURNED CANDIDATE SET FROM ENSEMBLING
        # ALSO SET SHAPE,DIMS FOR PQ
        X_query = X[qi,:len_dict[qi],:]
        n_data = X_query.shape[0]
        d = X_query.shape[1]


        print('\nTHE QUERY SAMPLES HAVE SHAPE : {X_query.shape} \n')
        #---------------------------------------------#
        print('\n########################################################################################\n')
        print(f'##------------STARTING QUANTIZATION----------------------------------------------------##\n')
        print('##########################################################################################\n')


        start_time = time.time()

        # using quantizer on mnist
        index = VectorQuantizer(X_query,d,8,512)
        # creat the centroids
        index.trainCentroids(X_query)
        # encode the dataset of vectors
        index.insertVectors(X_query)
        # searh for vectors neighbors
        # top 10 can change neighbors of sample 0
        # should customize to find neighbors of given query point (not only the first/0)
        # top_k + 1 ,since it returns itself be default so we drop itself
        # we can also find more general approximate neighbors from the sample by just using simSearch(:N,..)

        _,indices = index.SimSearch(X_query[:1],top_k+1)
        # WE SHOULD MAP BACK TO THE ORIGINAL X_TRAIN DATASET #
        # candidates of '0' because we want the neighbors of query point 0
        # this should change depending on query point we are probing
        # value - 1 since the first point in the array is itself so the order starts from 1
        indices = indices.flatten()[1:]
        indices_map = [candidates[qi][value-1] for value in indices]

        # store this query point's neighbors that pq found
        all_query_neighbors[qi,:]=indices_map

        print('\n########################################################################################\n')
        print(f'##------------RETURNING INDICES--------------------------------------------------------##\n')
        print('##########################################################################################\n')
        print('\n',indices_map)
        #print('\n',len(indices_map))

    # print total time for all queries  to apply pq and return the neighbors
    print( f'\nTIME TAKEN  : {time.time()-start_time}\n' )

    print('\n###################################################################################################\n')
    print(f'##------------ ALL QUERIES TOP {top_k} INDICES----------------------------------------------------##\n')
    print('#####################################################################################################\n')
    print(all_query_neighbors)
