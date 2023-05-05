import numpy as np
import faiss
import time
import re
import sys
from scipy import sparse
from tqdm.auto import tqdm
from numba import jit

def create_sparse_kernel_faiss_innerproduct(
        X, index_key, logger, ngpu=-1, 
        tempmem=-1, altadd=False, 
        use_float16=True, use_precomputed_tables=True, 
        replicas=1, max_add=-1, add_batch_size=32768, 
        query_batch_size=16384, nprobe=128, nnn=10
    ):
    
    if nnn>np.shape(X)[0]:
        raise Exception("ERROR: num of neighbors can't be more than no of datapoints")
    if nnn==-1:
        nnn=np.shape(X)[0] #default is total no of datapoints
    # Parse index_key
    # The index_key is a valid factory key that would work, but we decompose the training to do it faster
    pat = re.compile('(OPQ[0-9]+(_[0-9]+)?,|PCAR[0-9]+,)?' +
                    '(IVF[0-9]+),' +
                    '(PQ[0-9]+|Flat)')
    matchobject=pat.match(index_key)
    assert matchobject, "could not parse "+index_key
    mog=matchobject.groups()
    preproc_str=mog[0]
    ivf_str=mog[2]
    pqflat_str=mog[3]
    ncent=int(ivf_str[3:])

    class IdentPreproc:
        """a pre-processor is either a faiss.VectorTransform or an IndentPreproc"""
        def __init__(self, d):
            self.d_in=self.d_out=d
        
        def apply_py(self, x):
            return x

    # Wake up GPUs
    logger.info(f"preparing resources for {ngpu} GPUs")
    gpu_resources=[]
    for i in range(ngpu):
        res=faiss.StandardGpuResources()
        if tempmem>=0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    def make_vres_vdev(i0=0, i1=-1):
        " return vectors of device ids and resources useful for gpu_multiple"
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        if i1 == -1:
            i1 = ngpu
        for i in range(i0, i1):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        return vres, vdev

    # get preprocessor
    if preproc_str:
        logger.info(f"train preproc {preproc_str}")
        d = X.shape[1]
        t0 = time.time()
        if preproc_str.startswith('OPQ'):
            fi = preproc_str[3:-1].split('_')
            m = int(fi[0])
            dout = int(fi[1]) if len(fi) == 2 else d
            preproc = faiss.OPQMatrix(d, m, dout)
        elif preproc_str.startswith('PCAR'):
            dout = int(preproc_str[4:-1])
            preproc = faiss.PCAMatrix(d, dout, 0, True)
        else:
            assert False
        preproc.train(np.ascontiguousarray(X.astype("float32")))
        logger.info("preproc train done in %.3f s" % (time.time() - t0))
    else:
        d=X.shape[1]
        preproc=IdentPreproc(d)

    # coarse_quantizer=prepare_coarse_quantizer(preproc)
    nt=max(1000000, 256*ncent)
    logger.info("train coarse quantizer...")
    t0=time.time()
    # centroids=train_coarse_quantizer(X[:nt], ncent, preproc)
    d=preproc.d_out
    clus=faiss.Clustering(d, ncent)
    clus.spherical=True
    clus.verbose=True
    clus.max_points_per_centroid=10000000
    logger.info(f"apply preproc on shape  {X[:nt].shape}, k=, {ncent}")
    t0=time.time()
    x=preproc.apply_py(np.ascontiguousarray(X[:nt]))
    logger.info("preproc %.3f s output shape %s"%(time.time()-t0, x.shape))
    vres, vdev = make_vres_vdev()
    index = faiss.index_cpu_to_gpu_multiple(
        vres, vdev, faiss.IndexFlatIP(d))
    clus.train(x, index)
    centroids=faiss.vector_float_to_array(clus.centroids)
    del x
    centroids=centroids.reshape(ncent, d)
    logger.info("Coarse train time: %.3f s"%(time.time()-t0))
    coarse_quantizer=faiss.IndexFlatIP(preproc.d_out)
    coarse_quantizer.add(centroids)
    d=preproc.d_out
    if pqflat_str=="Flat":
        logger.info("making an IVFFlat Index")
        indexall=faiss.IndexIVFFlat(coarse_quantizer, d, ncent, faiss.METRIC_INNER_PRODUCT)
    else:
        m=int(pqflat_str[2:])
        assert m<56 or use_float16, f"PQ{m} will work only with -float16"
        logger.info(f"making an IVFPQ index, m={m}")
        indexall=faiss.IndexIVFPQ(coarse_quantizer, d, ncent, m, 8)
    coarse_quantizer.this.disown()
    indexall.own_fields=True
    # finish the training on CPU
    t0=time.time()
    logger.info("Training vector codes")
    indexall.train(preproc.apply_py(np.ascontiguousarray(X.astype("float32"))))
    logger.info("done %.3f s"%(time.time()-t0))
    # Prepare the index
    if not altadd:
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = use_float16
        co.useFloat16CoarseQuantizer = False
        co.usePrecomputed = use_precomputed_tables
        co.indicesOptions = faiss.INDICES_CPU
        co.verbose = True
        co.reserveVecs = max_add if max_add > 0 else X.shape[0]
        co.shard = True
        assert co.shard_type in (0, 1, 2)
        vres, vdev = make_vres_vdev()
        gpu_index = faiss.index_cpu_to_gpu_multiple(
            vres, vdev, indexall, co)
        logger.info("add...")
        t0 = time.time()
        nb=X.shape[0]
        block_ranges=[(i0, min(nb, i0+add_batch_size)) for i0 in range(0, nb, add_batch_size)]
        for i01 in block_ranges:
            i0, i1=i01
            xs=preproc.apply_py(np.ascontiguousarray(X[i0:i1].astype("float32")))
            gpu_index.add_with_ids(xs, np.arange(i0, i1))
            if max_add>0 and gpu_index.ntotal>max_add:
                logger.info("Flush indexes to CPU")
                for i in range(ngpu):
                    index_src_gpu=faiss.downcast_index(gpu_index.at(i))
                    index_src=faiss.index_gpu_to_cpu(index_src_gpu)
                    logger.info(f"index {i} size {index_src.ntotal}")
                    index_src.copy_subset_to(indexall, 0, 0, nb)
                    index_src_gpu.reset()
                    index_src_gpu.reserveMemory(max_add)
                gpu_index.syncWithSubIndexes()
            logger.info('\r%d/%d (%.3f s)  ' % (
                i0, nb, time.time() - t0))
            sys.stdout.flush()
        logger.info("Add time: %.3f s"%(time.time()-t0))
        # logger.info("Aggregate indexes to CPU")
        # t0=time.time()
        # if hasattr(gpu_index, "at"):
        #     # it is a sharded index
        #     for i in range(ngpu):
        #         index_src=faiss.index_gpu_to_cpu(gpu_index.at(i))
        #         logger.info(f"index {i} size {index_src.ntotal}")
        #         index_src.copy_subset_to(indexall, 0, 0, nb)
        # else:
        #     # simple index
        #     index_src=faiss.index_gpu_to_cpu(gpu_index)
        #     index_src.copy_subset_to(indexall, 0, 0, nb)
        # logger.info("done in %.3f s"%(time.time()-t0))
        if max_add>0:
            # it does not contain all the vectors
            gpu_index=None
    else:
        # set up a 3-stage pipeline that does:
        # - stage 1: load + preproc
        # - stage 2: assign on GPU
        # - stage 3: add to index
        vres, vdev = make_vres_vdev()
        coarse_quantizer_gpu = faiss.index_cpu_to_gpu_multiple(
            vres, vdev, indexall.quantizer)
        nb=X.shape[0]
        block_ranges=[(i0, min(nb, i0+add_batch_size)) for i0 in range(0, nb, add_batch_size)]
        logger.info("add...")
        t0 = time.time()
        for i01 in block_ranges:
            i0, i1=i01
            xs=preproc.apply_py(np.ascontiguousarray(X[i0:i1].astype("float32")))
            _, assign=coarse_quantizer_gpu.search(xs, 1)
            if indexall.__class__==faiss.IndexIVFPQ:
                indexall.add_core_o(i1-i0, faiss.swig_ptr(xs), None, None, faiss.swig_ptr(assign))
            elif indexall.__class__==faiss.IndexIVFFlat:
                indexall.add_core(i1-i0, faiss.swig_ptr(xs), None, faiss.swig_ptr(assign))
            else:
                assert False
            logger.info('\r%d/%d (%.3f s)  ' % (
                i0, nb, time.time() - t0))
            sys.stdout.flush()
        logger.info("Add time: %.3f s"%(time.time()-t0))
        gpu_index=None

    
    co=faiss.GpuMultipleClonerOptions()
    co.useFloat16=use_float16
    co.useFloat16CoarseQuantizer=False
    co.usePrecomputed=use_precomputed_tables
    co.indicesOptions=0
    co.verbose=True
    co.shard=True # The replicas will be made "manually"
    t0=time.time()
    logger.info(f"CPU index contains {indexall.ntotal} vectors, move to GPU")
    if replicas==1:
        if not gpu_index:
            logger.info("copying loaded index to GPUs")
            vres, vdev = make_vres_vdev()
            index = faiss.index_cpu_to_gpu_multiple(
                vres, vdev, indexall, co)
        else:
            index = gpu_index
    else:
        del gpu_index # We override the GPU index
        logger.info(f"Copy CPU index to {replicas} sharded GPU indexes")
        index=faiss.IndexReplicas()
        for i in range(replicas):
            gpu0 = ngpu * i / replicas
            gpu1 = ngpu * (i + 1) / replicas
            vres, vdev = make_vres_vdev(gpu0, gpu1)
            logger.info(f"dispatch to GPUs {gpu0}:{gpu1}")
            index1 = faiss.index_cpu_to_gpu_multiple(
                vres, vdev, indexall, co)
            index1.this.disown()
            index.addIndex(index1)
            index.own_fields = True
    del indexall
    logger.info("move to GPU done in %.3f s"%(time.time()-t0))
    
    ps=faiss.GpuParameterSpace()
    ps.initialize(index)
    # index=indexall
    logger.info("search...")
    nq=X.shape[0]
    # index.nprobe=nprobe
    ps.set_index_parameter(index, 'nprobe', nprobe)
    t0=time.time()
    if query_batch_size==0:
        D, I=index.search(preproc.apply_py(np.ascontiguousarray(X.astype("float32"))), nnn)
    else:
        I=np.empty((nq, nnn), dtype="int32")
        D=np.empty((nq, nnn), dtype="float32")
        block_ranges=[(i0, min(nb, i0+query_batch_size)) for i0 in range(0, nb, query_batch_size)]
        pbar=tqdm(range(len(block_ranges)))
        for i01 in block_ranges:
            i0, i1=i01
            xs=preproc.apply_py(np.ascontiguousarray(X[i0:i1].astype("float32")))
            Di, Ii=index.search(xs, nnn)
            I[i0:i1]=Ii
            D[i0:i1]=Di
            pbar.update(1)
    logger.info("search completed in %.3f s"%(time.time()-t0))
    data=np.reshape(D, (-1,))
    row_ind=np.repeat(np.arange(nb), nnn)
    col_ind=np.reshape(I, (-1,))
    np.save(open("data.npy", "wb"), data)
    np.save(open("row_ind.npy", "wb"), row_ind)
    np.save(open("col_ind.npy", "wb"), col_ind)
    sparse_csr = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(nb,nb))
    return 
    
@jit(nopython=True, parallel=True)
def get_rbf_kernel(dist_mat, kw=0.1):
	sim = np.exp(-dist_mat/(kw*dist_mat.mean()))
	return sim