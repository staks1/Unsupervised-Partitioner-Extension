
from random import randint
import numpy as np
import sklearn
from sklearn.metrics import euclidean_distances
from sklearn.cluster import KMeans

# x=given vector
# give d=dimensions,
# give m=number of subvectors
# ds = dimensionality of each subvector
# u = set of subvectors
# c_num = centroid total possible number
# c_psub = total centroids per subvector (subspace) , customizable = c_num/m
# codebooks = the reproduction values to reconstruct whole vector

# we should find the max value of X_train vectors to
# define the max value for each centroid in the subspaces

class VectorQuantizer:
    # initialize the subvectors ,codebooks
    def __init__(self,X,d,m,c_num):
        assert d % m == 0 , "ERROR,subvector number should be divisor of dimensionality d"
        # self.d=d
        #self.bits = 2**nbits
        self.is_trained = False
        self.ds=int(d/m)
        self.m=m
        print(f'Dimensions of subvectors : {self.ds}\n')
        print(f'Number of subvectors : {self.m}\n')

        #print(self.m)
        # define the subvectors
        #self.u=np.zeros( (X.shape[0],X.shape[1],self.m,self.ds))
        # visualization
        self.u=[]
        '''
        for i in range(X.shape[0]):
            u1=[]
            for ind in range(0,d,self.ds):
                u1.append(list( X[i,ind:ind + self.ds]))
            self.u.append(u1)
        self.u = np.array(self.u)
        print(self.u)
        print(self.u.shape)
        '''

        # probably add ceil/floor to int
        assert c_num % m == 0 , "ERROR,total centroid number should divide the number of subvectors"
        self.c_num = c_num
        self.c_psub=int(self.c_num / m)
        print(f'Centroids per subvector : {self.c_psub}')
        # centroid estimators for each subvector
        # should be trainable
        self.estimators = [KMeans(n_clusters = self.c_psub) for _ in range(m) ]

        print(self.estimators)



    def trainCentroids(self,X):
        if self.is_trained :
            raise ValueError ("Centroids are already trained .")
        print("\n---------TRAINING CENTROIDS.Please wait ...\n")
        for i in range(self.m):
            estimator = self.estimators[i]
            # apply for all samples in the dataset
            # pick only the current subvector ans fit estimator
            X_i = X[:,i*self.ds :(i+1)*self.ds]
            estimator.fit(X_i)
            print( f'\nDimension has : {estimator.cluster_centers_}\n')
        #print(self.estimators.cluster_centers_)
        self.is_trained==True
        print("\n---------CENTROIDS trained ------------\n")




    # do the assignment of Codes
    def assignCode(self,X):
        # initialize the codebook
        codes = np.empty ((len(X),self.m))
        # training and assignment should move to other function #
        for j in range(self.m):
            # assign code to centroid for each subvector
            estimator = self.estimators[j]
            X_j = X[:,j*self.ds :(j+1)*self.ds]
            codes[:,j] = estimator.predict(X_j)
        return codes

    # encode and store the embeddings
    def insertVectors(self,X):
        # encode so we can use their encoded version
        self.codebook = self.assignCode(X)
        # should be intrger ()
        self.codebook=self.codebook.astype(int)
        print(f'\nSetting Codebooks :  \n {self.codebook} \n')
        print(f'\nCodebooks Shapes:  \n {self.codebook.shape} \n')

    # calculate the distances from all embeddings
    def computeDistance(self,X):
        # how many queries we are going to feed ??
        n_queries = len(X)
        n_codes = len(self.codebook)
        # initialize distance table
        # should define k !!
        distance_table = np.empty( (n_queries,self.m,self.c_psub) )

        # for each subvector
        for i in range(self.m):
            X_i = X[:,i*self.ds : (i+1) * self.ds]
            # get centroids values from trained centroids
            centers = self.estimators[i].cluster_centers_
            # could also add other distances
            distance_table[:,i,:]=euclidean_distances(X_i,centers,squared=True)

        distances = np.zeros((n_queries,n_codes))


        # update distances -- > total distance should be the sum of the distances of all subvectors centroids
        for j in range (self.m):
            distances += distance_table[:,i,self.codebook[:,i] ]

        return distances

    # iterate over all embeddings in dataset and find the query's neighbors
    # those samples with minimum distance (euclidean)
    def SimSearch(self,X,top_k):
        n_queries=len(X)
        # distance of our queries from all embeddings
        distances_total = self.computeDistance(X)
        # find indices of subvectors of topk minimum distances
        indices = np.argsort(distances_total,axis=1)[:,:top_k]

        distances = np.empty ((n_queries,top_k),dtype=np.float32)
        for i in range(n_queries):
            distances[i] = distances_total[i][indices[i]]
        return distances , indices
