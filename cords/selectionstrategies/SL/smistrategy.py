import math
from operator import gt
import re
import numpy as np
import torch
import time
import random
from scipy.sparse import csr_matrix
from torch.utils.data.sampler import SubsetRandomSampler
from .dataselectionstrategy import DataSelectionStrategy
from torch.utils.data import Subset, DataLoader
import submodlib
from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool
from itertools import starmap
import tqdm
#from p_tqdm import p_umap
# from cuml.cluster import KMeans
# from submodlib import FacilityLocationMutualInformationFunction, FacilityLocationVariantMutualInformationFunction
from sklearn.metrics.pairwise import cosine_similarity

def query_generator(train_rep, query_rep, private_rep, partition_indices, partition_budget, smi_func_type, metric, is_sparse):
    """
    Generate queries for imap_unordered
    """
    for partition in partition_indices:
        yield (train_rep[partition], query_rep, private_rep, smi_func_type, metric, partition, partition_budget, is_sparse)

def partition_subset_star(args):
        return partition_subset_selection(*args)
    
def partition_subset_selection(partition_train_rep, partition_query_rep, partition_private_rep, smi_func_type, metric, partition_indices, partition_budget, is_sparse):
    kernel_time = time.time()
    
    if smi_func_type in ['fl', 'gc', 'logdet', 'fl1mi', 'logdetmi', 'flcg', 'logdetcg', 'gccg']:
        if is_sparse:
            data_sijs=cosine_similarity(partition_train_rep)
        else:
            data_sijs = submodlib.helper.create_kernel(X=partition_train_rep,
                                                        metric=metric, 
                                                        method='sklearn')

    
    if smi_func_type in ['fl1mi', 'fl2mi', 'logdetmi', 'gcmi']:
        query_sijs = submodlib.helper.create_kernel(X=partition_query_rep, X_rep=partition_train_rep, 
                                                metric=metric, 
                                                method='sklearn')
    
    if smi_func_type in ['flcg', 'logdetcg', 'gccg']:
        
        private_sijs = submodlib.helper.create_kernel(X=partition_private_rep, X_rep=partition_train_rep, 
                                                metric=metric, 
                                                method='sklearn')
                                                #method='np_numba')
        if smi_func_type in ['logdetcg']:
            private_private_sijs = submodlib.helper.create_kernel(X=partition_private_rep,
                                                metric=metric, 
                                                method='sklearn')
                                                #method='np_numba')

    if smi_func_type in ['logdetmi']:
        query_query_sijs = submodlib.helper.create_kernel(X=partition_query_rep,
                                                        metric=metric,
                                                        method='sklearn')
                                                        #method='np_numba')
    
        
    #self.logger.info("Kernel Computation Time: {}".format(time.time()-kernel_time))
    
    greedy_selection_start_time = time.time()
    if smi_func_type == 'fl1mi':
        obj = submodlib.FacilityLocationMutualInformationFunction(n=partition_train_rep.shape[0],
                                                    num_queries=partition_query_rep.shape[0],
                                                    data_sijs=data_sijs,
                                                    query_sijs=query_sijs,
                                                    magnificationEta=1)               
    
    if smi_func_type == 'fl2mi':
        obj = submodlib.FacilityLocationVariantMutualInformationFunction(n=partition_train_rep.shape[0],
                                                    num_queries=partition_query_rep.shape[0],
                                                    query_sijs=query_sijs,
                                                    queryDiversityEta=1)

    if smi_func_type == 'logdetmi':
        obj = submodlib.LogDeterminantMutualInformationFunction(n=partition_train_rep.shape[0],
                                                            num_queries=partition_query_rep.shape[0],
                                                            data_sijs=data_sijs,
                                                            lambdaVal=1,
                                                            query_sijs=query_sijs,
                                                            query_query_sijs=query_query_sijs,
                                                            magnificationEta=1
                                                            )

    if smi_func_type == 'gcmi':
        obj = submodlib.GraphCutMutualInformationFunction(n=partition_train_rep.shape[0],
                                                            num_queries=partition_query_rep.shape[0],
                                                            query_sijs=query_sijs)

    if smi_func_type == 'fl':
        obj = submodlib.FacilityLocationFunction(n = partition_train_rep.shape[0],
                                                separate_rep=False,
                                                mode = 'dense',
                                                sijs = data_sijs)

    if smi_func_type == 'logdet':
        obj = submodlib.LogDeterminantFunction(n = partition_train_rep.shape[0],
                                                mode = 'dense',
                                                lambdaVal = 1,
                                                sijs = data_sijs)
    
    if smi_func_type == 'gc':
        obj = submodlib.GraphCutFunction(n = partition_train_rep.shape[0],
                                        mode = 'dense',
                                        lambdaVal = 1,
                                        separate_rep=False,
                                        ggsijs = data_sijs)
    
    if smi_func_type == 'flcg':
        obj = submodlib.FacilityLocationConditionalGainFunction(n=partition_train_rep.shape[0], 
                                                                num_privates=partition_private_rep.shape[0],
                                                                data_sijs=data_sijs,
                                                                private_sijs=private_sijs)
                                            

    if smi_func_type == 'logdetcg':
        obj = submodlib.LogDeterminantConditionalGainFunction(n=partition_train_rep.shape[0], 
                                                                num_privates=partition_private_rep.shape[0],
                                                                lambdaVal=1,
                                                                data_sijs=data_sijs,
                                                                private_sijs=private_sijs,
                                                                private_private_sijs=private_private_sijs)

    if smi_func_type == 'gccg':
        obj = submodlib.GraphCutConditionalGainFunction(n=partition_train_rep.shape[0], 
                                                        num_privates=partition_private_rep.shape[0],
                                                        lambdaVal=1,
                                                        data_sijs=data_sijs,
                                                        private_sijs=private_sijs)


    greedyList = obj.maximize(budget=partition_budget, optimizer='LazierThanLazyGreedy', stopIfZeroGain=False,
                            stopIfNegativeGain=False, verbose=False)
    
    #self.logger.info('Submodular function optimization complete. Greedy Selection time is %f' % (time.time() - greedy_selection_start_time))
    del partition_train_rep
    del obj
    if smi_func_type in ['fl1mi', 'fl2mi', 'logdetmi', 'gcmi']:
        del query_sijs
    if smi_func_type in ['fl', 'gc', 'logdet', 'fl1mi', 'logdetmi', 'flcg', 'logdetcg', 'gccg']:
        del data_sijs
    if smi_func_type in ['logdetmi']:
        del query_query_sijs
    if smi_func_type in ['logdetcg']:
        del private_private_sijs
    #Converting selected indices to global indices
    return([partition_indices[x[0]] for x in greedyList])  

class SMIStrategy():
    def __init__(self, logger, smi_func_type, 
                 num_partitions=20, partition_strategy='random',
                 optimizer='LazyGreedy', similarity_criterion='feature', 
                 metric='cosine', eta=1, stopIfZeroGain=False, 
                 stopIfNegativeGain=False, verbose=False, lambdaVal=1, is_sparse=False):
        """
        Constructor method
        """
        # super().__init__(train_representations, query_representations, original_indices, smi_func_type, logger)
        self.train_rep = None
        self.query_rep = None
        self.private_rep = None
        self.indices = None
        self.logger = logger
        self.optimizer = optimizer
        self.smi_func_type = smi_func_type
        self.num_partitions = num_partitions
        self.partition_strategy = partition_strategy
        self.metric = metric
        self.eta = eta
        self.stopIfZeroGain = stopIfZeroGain
        self.stopIfNegativeGain = stopIfNegativeGain
        self.verbose = verbose
        self.lambdaVal = lambdaVal
        self.similarity_criterion = similarity_criterion
        self.is_sparse=is_sparse
    
    # def update_representations(self, train_representations, query_representations, indices):
    #     self.train_rep = train_representations
    #     self.query_rep = query_representations
    #     self.indices = indices
    #     assert len(self.indices) == self.train_rep.shape[0], "Indices and representations must have same length"

    def random_partition(self, num_partitions, indices):
        """
        Randomly partition the data into num_partitions
        Parameters
        ----------
        num_partitions : int
            Number of partitions
        indices : list
            List of indices to partition
        Returns
        -------
        partition_indices : list
            List of lists of indices
        """
        partition_indices = []
        partition_size = int(math.ceil(len(indices)/num_partitions))
        random_indices = list(range(len(indices)))
        random.shuffle(random_indices)
        for i in range(num_partitions):
            partition_indices.append(random_indices[i*partition_size:(i+1)*partition_size])
        return partition_indices

    def kmeans(self, num_partitions, indices, partition_budget_split):
        # partition_indices=[[] for i in range(num_partitions)]
        # kmeans=KMeans(n_clusters=num_partitions)
        # kmeans.fit(self.train_rep)
        # for i, lab in enumerate(kmeans.labels_):
        #     partition_indices[lab].append(indices[i])
        # for l in partition_indices:
        #     assert len(l)>=partition_budget_split, "Budget must be less than effective ground set size"
        # return partition_indices
        partition_indices = []
        partition_size = int(math.ceil(len(indices)/num_partitions))
        random_indices = list(range(len(indices)))
        random.shuffle(random_indices)
        for i in range(num_partitions):
            partition_indices.append(random_indices[i*partition_size:(i+1)*partition_size])
        return partition_indices


    def select(self, budget, indices, representations, 
                query_representations=None, 
                private_representations=None,
                parallel_processes=50):
        """

        Parameters
        ----------
        budget :
        model_params :

        Returns
        -------

        """
        #self.train_rep = representations
        #self.query_rep = query_representations
        #self.private_rep = private_representations
        partition_budget_split = math.ceil(budget/self.num_partitions)
        smi_start_time = time.time()
        
        #Return Partitions of the data for subset selection
        if self.partition_strategy == 'random':
            partition_indices = self.random_partition(self.num_partitions, indices) 
        elif self.partition_strategy == 'kmeans':
            partition_indices = self.kmeans(self.num_partitions, indices, partition_budget_split)
        else:
            partition_indices = [list(range(len(indices)))]
        
        if self.partition_strategy not in ['random', 'kmeans']:
            assert self.num_partitions == 1, "Partition strategy {} not implemented for {} partitions".format(self.partition_strategy, self.num_partitions)

        if self.smi_func_type in ['flcg', 'logdetcg', 'gccg']:
            assert private_representations is not None, "CG functions requires private set"

        if self.smi_func_type in ['fl1mi', 'fl2mi', 'logdetmi', 'gcmi']:
            assert query_representations is not None, "SMI functions requires query set"

        greedyIdxs = []
        
        #Parallel computation of subsets
        #queries = [(x, partition_budget_split) for x in partition_indices]
        with Pool(parallel_processes) as pool:
            #greedyIdxs = pool.starmap(self.partition_subset_selection, queries)
            greedyIdxs_list = list(tqdm.tqdm(pool.imap_unordered(partition_subset_star, 
                                                query_generator(representations, query_representations, private_representations, partition_indices, partition_budget_split, self.smi_func_type, self.metric, self.is_sparse)), total=len(partition_indices)))
        #greedyIdxs_list = p_umap(partition_subset_star, queries)
        greedyIdxs = []
        for idxs in greedyIdxs_list:
            greedyIdxs.extend(idxs)

        originalIdxs = [indices[x] for x in greedyIdxs]
        assert len(set(originalIdxs)) == (partition_budget_split * self.num_partitions), "Selected subset must be equal to the budget"
        smi_end_time = time.time()
        self.logger.info("SMI algorithm Subset Selection time is: %.4f", smi_end_time - smi_start_time)
        return originalIdxs