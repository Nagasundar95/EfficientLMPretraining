import time
from .helper import create_sparse_kernel_faiss_innerproduct
import sys
from pysubmodlib import FacilityLocationFunction
import numpy as np
from scipy import sparse

class KnnSubmodStrategy():
    def __init__(self, logger, smi_func_type, 
                optimizer="LazyGreedy", similarity_criterion='feature',
                metric="cosine", eta=1, stopIfZeroGain=False,
                stopIfNegativeGain=False, verbose=False, lambdaVal=1):
        self.logger=logger
        self.optimizer=optimizer
        self.smi_func_type=smi_func_type
        self.metric=metric
        self.eta=eta
        self.stopIfZeroGain=stopIfZeroGain
        self.stopIfNegativeGain=stopIfNegativeGain
        self.verbose=verbose
        self.lambdaVal=lambdaVal
        self.similarity_criterion=similarity_criterion
        self.data_sijs=None
    
    def compute_sparse_kernel(self, representations,
                index_key, ngpu=-1, tempmem=-1, altadd=False,
                use_float16=True, use_precomputed_tables=True,
                replicas=1, max_add=-1, add_batch_size=1048576,
                query_batch_size=1024, nprobe=128, nnn=10):
        if self.smi_func_type not in ["fl", "gc", "logdet"]:
            assert False, f"{self.smi_func_type} not yet supported by this script"
        kernel_time=time.time()
        self.data_sijs=create_sparse_kernel_faiss_innerproduct(
            X=representations, index_key=index_key, logger=self.logger, ngpu=ngpu, 
            tempmem=tempmem, altadd=altadd, 
            use_float16=use_float16, use_precomputed_tables=use_precomputed_tables, 
            replicas=replicas, max_add=max_add, add_batch_size=add_batch_size, 
            query_batch_size=query_batch_size, nprobe=nprobe, nnn=nnn,
        )
        # D=np.load(open("/home/sumbhati/ingenious/D.npy", "rb"))
        # I=np.load(open("/home/sumbhati/ingenious/I.npy", "rb"))
        # nb=D.shape[0]
        # data=np.reshape(D, (-1,))
        # row_ind=np.repeat(np.arange(nb), nnn)
        # col_ind=np.reshape(I, (-1,))
        # self.data_sijs = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(nb,nb))
        self.logger.info(f"Time taken for sparse kernel construction: {kernel_time-time.time()}")

    def select(self, budget, nnn=10):
        self.logger.info("started select")
        n=self.data_sijs.shape[0]
        greedy_selection_start_time=time.time()
        if self.smi_func_type=="fl":
            obj=FacilityLocationFunction(n=n,
                mode="sparse",
                sijs=self.data_sijs,
                num_neighbors=nnn)

        if self.smi_func_type=="gc" or self.smi_func_type=="logdet":
            raise Exception(f"{self.smi_func_type} not yet supported by the script")

        self.logger.info("Beginning the greedy algorithm")
        greedyList=obj.maximize(budget=budget, optimizer=self.optimizer, verbose=False)
        self.logger.info(f"Greedy Selection Time: {time.time()-greedy_selection_start_time}")
        return [p[0] for p in greedyList]