import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import gc
import os
from io import StringIO


from .create_ggm import create_ggm
from .create_ggm import FDRResults
from .create_ggm_multi import create_ggm_multi

def save_ggm(self, file_path):
    """
    save create_ggm object to HDF5 file, 
    including metadata, expression matrix, partial correlation results, and subsequent generated data (such as FDR, modules, etc.).
    
    Parameters:
        file_path: str, the path to save the HDF5 file.
    """
    # delete the h5 file if it already exists
    if os.path.exists(file_path):
        os.remove(file_path)
    with h5py.File(file_path, 'w') as f:
        # Save metadata
        meta_grp = f.create_group("metadata")
        # Save method if it is a create_ggm_multi object
        if hasattr(self, 'method'):
            meta_grp.attrs['method'] = self.method
        meta_grp.attrs['round_num'] = self.round_num
        meta_grp.attrs['selected_num'] = self.selected_num
        meta_grp.attrs['target_sampling_time'] = self.target_sampling_time
        meta_grp.attrs['gene_num'] = self.gene_num
        meta_grp.attrs['project_name'] = self.project_name
        meta_grp.attrs['cut_off_pcor'] = self.cut_off_pcor
        meta_grp.attrs['cut_off_coex_cell'] = self.cut_off_coex_cell
        meta_grp.attrs['seed_used'] = self.seed_used
        meta_grp.attrs['run_mode'] = self.run_mode
        meta_grp.attrs['double_precision'] = self.double_precision
        meta_grp.attrs['use_chunking'] = self.use_chunking
        meta_grp.attrs['chunk_size'] = self.chunk_size
        meta_grp.attrs['stop_threshold'] = self.stop_threshold
        meta_grp.attrs['FDR_control'] = self.FDR_control
        meta_grp.attrs['FDR_threshold'] = self.FDR_threshold
        meta_grp.attrs['auto_adjust'] = self.auto_adjust
        meta_grp.attrs['auto_find_modules'] = self.auto_find_modules
        meta_grp.create_dataset("gene_name", data=np.array(self.gene_name, dtype="S"))
        meta_grp.create_dataset("sample_name", data=np.array(self.sample_name, dtype="S"))
        meta_grp.attrs['samples_num'] = self.samples_num

        # Save expression matrix as CSR format
        matrix_grp = f.create_group("matrix")
        matrix_grp.create_dataset("data", data=self.matrix.data)
        matrix_grp.create_dataset("indices", data=self.matrix.indices)
        matrix_grp.create_dataset("indptr", data=self.matrix.indptr)
        matrix_grp.attrs["shape"] = self.matrix.shape

        # Save partial correlation results
        results_grp = f.create_group("results")
        results_grp.create_dataset("coexpressed_cell_num", data=self.coexpressed_cell_num)
        results_grp.create_dataset("pcor_all", data=self.pcor_all)
        results_grp.create_dataset("pcor_sampling_num", data=self.pcor_sampling_num)
        results_grp.create_dataset("rho_all", data=self.rho_all)
        if self.SigEdges is not None:
            csv_str = self.SigEdges.to_csv(index=False)
            dt = h5py.string_dtype(encoding='utf-8')
            results_grp.create_dataset("SigEdges_csv", data=np.array([csv_str], dtype=dt))


        # save FDR results(if exists)
        if self.fdr is not None:
            fdr_grp = f.create_group("FDR")
            fdr_grp.attrs["permutation_fraction"] = self.fdr.permutation_fraction
            fdr_grp.attrs["FDR_threshold"] = self.fdr.FDR_threshold
            fdr_matrix_grp = fdr_grp.create_group("permutation_matrix")
            fdr_matrix_grp.create_dataset("data", data=self.fdr.permutation_matrix.data)
            fdr_matrix_grp.create_dataset("indices", data=self.fdr.permutation_matrix.indices)
            fdr_matrix_grp.create_dataset("indptr", data=self.fdr.permutation_matrix.indptr)
            fdr_matrix_grp.attrs["shape"] = self.fdr.permutation_matrix.shape
            fdr_grp.create_dataset("permutation_gene_name", data=np.array(self.fdr.permutation_gene_name, dtype="S"))
            fdr_grp.create_dataset("permutation_coexpressed_cell_num", data=self.fdr.permutation_coexpressed_cell_num)
            fdr_grp.create_dataset("permutation_pcor_all", data=self.fdr.permutation_pcor_all)
            fdr_grp.create_dataset("permutation_pcor_sampling_num", data=self.fdr.permutation_pcor_sampling_num)
            fdr_grp.create_dataset("permutation_rho_all", data=self.fdr.permutation_rho_all)
            if self.fdr.permutation_SigEdges is not None:
                csv_str_fdr = self.fdr.permutation_SigEdges.to_csv(index=False)
                dt = h5py.string_dtype(encoding='utf-8')
                fdr_grp.create_dataset("permutation_SigEdges_csv", data=np.array([csv_str_fdr], dtype=dt))
            if hasattr(self.fdr, "summary") and self.fdr.summary is not None:
                csv_str_summary = self.fdr.summary.to_csv(index=False)
                dt = h5py.string_dtype(encoding='utf-8')
                fdr_grp.create_dataset("fdr_summary_csv", data=np.array([csv_str_summary], dtype=dt))


        # Save modules, modules_summary, go_enrichment, mp_enrichment (if exists)
        if hasattr(self, "modules") and self.modules is not None:
            csv_str_mod = self.modules.to_csv(index=False)
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset("modules_csv", data=np.array([csv_str_mod], dtype=dt))
        if hasattr(self, "modules_summary") and self.modules_summary is not None:
            csv_str_mod_sum = self.modules_summary.to_csv(index=False)
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset("modules_summary_csv", data=np.array([csv_str_mod_sum], dtype=dt))
        if hasattr(self, "go_enrichment") and self.go_enrichment is not None:
            csv_str_go = self.go_enrichment.to_csv(index=False)
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset("go_enrichment_csv", data=np.array([csv_str_go], dtype=dt))
        if hasattr(self, "mp_enrichment") and self.mp_enrichment is not None:
            csv_str_mp = self.mp_enrichment.to_csv(index=False)
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset("mp_enrichment_csv", data=np.array([csv_str_mp], dtype=dt))
    
        gc.collect()


def load_ggm(file_path):
    """
    Load create_ggm object from HDF5 file.

    Parameters:
        file_path: str, the path to the HDF5 file.
    return:
        create_ggm object    
    """
    with h5py.File(file_path, 'r') as f:
        # Load metadata
        meta_grp = f['metadata']
        # Check if it is a create_ggm_multi object
        if "method" in meta_grp.attrs:
            method = meta_grp.attrs['method']
            cls_to_use = create_ggm_multi
        else:
            cls_to_use = create_ggm
        round_num = meta_grp.attrs['round_num'].item()        
        selected_num = meta_grp.attrs['selected_num'].item()
        target_sampling_time = meta_grp.attrs['target_sampling_time'].item()
        gene_num = meta_grp.attrs['gene_num'].item()
        project_name = meta_grp.attrs['project_name']
        cut_off_pcor = meta_grp.attrs['cut_off_pcor'].item()
        cut_off_coex_cell = meta_grp.attrs['cut_off_coex_cell'].item()
        seed_used = meta_grp.attrs['seed_used'].item()
        run_mode = meta_grp.attrs['run_mode'].item()
        double_precision = bool(meta_grp.attrs['double_precision'])
        use_chunking = bool(meta_grp.attrs['use_chunking'])
        chunk_size = meta_grp.attrs['chunk_size'].item()
        stop_threshold = meta_grp.attrs['stop_threshold'].item()
        FDR_control = bool(meta_grp.attrs['FDR_control'])
        FDR_threshold = meta_grp.attrs['FDR_threshold'].item()
        auto_adjust = bool(meta_grp.attrs['auto_adjust'])
        auto_find_modules = bool(meta_grp.attrs['auto_find_modules']) 
        gene_name = np.array(meta_grp['gene_name'], dtype='U')
        sample_name = np.array(meta_grp['sample_name'], dtype='U')
        samples_num = meta_grp.attrs['samples_num'].item()

        # Load expression matrix
        matrix_grp = f['matrix']
        data = matrix_grp['data'][:]
        indices = matrix_grp['indices'][:]
        indptr = matrix_grp['indptr'][:]
        shape = matrix_grp.attrs['shape']
        matrix = sp.csr_matrix((data, indices, indptr), shape=shape)

        # Load partial correlation results
        results_grp = f['results']
        coexp = results_grp['coexpressed_cell_num'][:]
        pcor_all = results_grp['pcor_all'][:]
        pcor_sampling_num = results_grp['pcor_sampling_num'][:]
        rho_all = results_grp['rho_all'][:]
        if "SigEdges_csv" in results_grp:
            csv_bytes = results_grp["SigEdges_csv"][()][0]
            csv_str = csv_bytes.decode('utf-8') if isinstance(csv_bytes, bytes) else csv_bytes
            SigEdges = pd.read_csv(StringIO(csv_str))
        else:
            SigEdges = None

        # Load FDR results (if exists)
        if "FDR" in f:
            fdr_grp = f["FDR"]
            fdr = FDRResults()
            fdr.permutation_fraction = fdr_grp.attrs["permutation_fraction"].item()
            fdr.FDR_threshold = fdr_grp.attrs["FDR_threshold"].item()
            fdr_matrix_grp = fdr_grp["permutation_matrix"]
            fdr_data = fdr_matrix_grp["data"][:]
            fdr_indices = fdr_matrix_grp["indices"][:]
            fdr_indptr = fdr_matrix_grp["indptr"][:]
            fdr_shape = fdr_matrix_grp.attrs["shape"]
            fdr.permutation_matrix = sp.csr_matrix((fdr_data, fdr_indices, fdr_indptr), shape=fdr_shape)
            fdr.permutation_gene_name = np.array(fdr_grp["permutation_gene_name"], dtype="U")
            perm_coexp = fdr_grp["permutation_coexpressed_cell_num"][:]
            perm_pcor_all = fdr_grp["permutation_pcor_all"][:]
            perm_pcor_sampling_num = fdr_grp["permutation_pcor_sampling_num"][:]
            perm_rho_all = fdr_grp["permutation_rho_all"][:]
            if perm_coexp.shape != (gene_num, gene_num):
                raise ValueError("FDR: permutation_coexpressed_cell_num shape mismatch")
            if perm_pcor_all.shape != (gene_num, gene_num):
                raise ValueError("FDR: permutation_pcor_all shape mismatch")
            if perm_pcor_sampling_num.shape != (gene_num, gene_num):
                raise ValueError("FDR: permutation_pcor_sampling_num shape mismatch")
            if perm_rho_all.shape != (gene_num, gene_num):
                raise ValueError("FDR: permutation_rho_all shape mismatch")
            fdr.permutation_coexpressed_cell_num = perm_coexp
            fdr.permutation_pcor_all = perm_pcor_all
            fdr.permutation_pcor_sampling_num = perm_pcor_sampling_num
            fdr.permutation_rho_all = perm_rho_all
            if "permutation_SigEdges_csv" in fdr_grp:
                csv_bytes_fdr = fdr_grp["permutation_SigEdges_csv"][()][0]
                csv_str_fdr = csv_bytes_fdr.decode('utf-8') if isinstance(csv_bytes_fdr, bytes) else csv_bytes_fdr
                fdr.permutation_SigEdges = pd.read_csv(StringIO(csv_str_fdr))
            else:
                fdr.permutation_SigEdges = None
            if "fdr_summary_csv" in fdr_grp:
                csv_bytes_summary = fdr_grp["fdr_summary_csv"][()][0]
                csv_str_summary = csv_bytes_summary.decode('utf-8') if isinstance(csv_bytes_summary, bytes) else csv_bytes_summary
                fdr.summary = pd.read_csv(StringIO(csv_str_summary))
            else:
                fdr.summary = None
        else:
            fdr = None

        # Load modules, modules_summary, go_enrichment, mp_enrichment (if exists)
        if "modules_csv" in f:
            csv_bytes_mod = f["modules_csv"][()][0]
            csv_str_mod = csv_bytes_mod.decode('utf-8') if isinstance(csv_bytes_mod, bytes) else csv_bytes_mod
            modules = pd.read_csv(StringIO(csv_str_mod))
        else:
            modules = None
        if "modules_summary_csv" in f:
            csv_bytes_mod_sum = f["modules_summary_csv"][()][0]
            csv_str_mod_sum = csv_bytes_mod_sum.decode('utf-8') if isinstance(csv_bytes_mod_sum, bytes) else csv_bytes_mod_sum
            modules_summary = pd.read_csv(StringIO(csv_str_mod_sum))
        else:
            modules_summary = None
        if "go_enrichment_csv" in f:
            csv_bytes_go = f["go_enrichment_csv"][()][0]
            csv_str_go = csv_bytes_go.decode('utf-8') if isinstance(csv_bytes_go, bytes) else csv_bytes_go
            go_enrichment = pd.read_csv(StringIO(csv_str_go))
        else:
            go_enrichment = None
        if "mp_enrichment_csv" in f:
            csv_bytes_mp = f["mp_enrichment_csv"][()][0]
            csv_str_mp = csv_bytes_mp.decode('utf-8') if isinstance(csv_bytes_mp, bytes) else csv_bytes_mp
            mp_enrichment = pd.read_csv(StringIO(csv_str_mp))
        else:
            mp_enrichment = None

        gc.collect()

    # Create an empty object using __new__ method not calling __init__
    obj = cls_to_use.__new__(cls_to_use)
    # Initialize the object with the loaded data
    if cls_to_use is create_ggm_multi:
        obj.method = method
    obj.matrix = matrix
    obj.round_num = round_num
    obj.selected_num = selected_num
    obj.target_sampling_time = target_sampling_time
    obj.gene_num = gene_num
    obj.gene_name = gene_name
    obj.samples_num = samples_num
    obj.sample_name = sample_name
    obj.project_name = project_name
    obj.cut_off_pcor = cut_off_pcor
    obj.cut_off_coex_cell = cut_off_coex_cell
    obj.seed_used = seed_used
    obj.run_mode = run_mode
    obj.double_precision = double_precision
    obj.use_chunking = use_chunking
    obj.chunk_size = chunk_size
    obj.stop_threshold = stop_threshold
    obj.FDR_control = FDR_control
    obj.FDR_threshold = FDR_threshold
    obj.auto_adjust = auto_adjust
    obj.auto_find_modules = auto_find_modules
    obj.coexpressed_cell_num = coexp
    obj.pcor_all = pcor_all
    obj.pcor_sampling_num = pcor_sampling_num
    obj.rho_all = rho_all
    obj.SigEdges = SigEdges
    obj.fdr = fdr
    obj.modules = modules
    obj.modules_summary = modules_summary
    obj.go_enrichment = go_enrichment
    obj.mp_enrichment = mp_enrichment

    gc.collect()
    return obj
