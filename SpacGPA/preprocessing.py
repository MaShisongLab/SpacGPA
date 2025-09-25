
import numpy as np
import scipy.sparse as sp
import gc

# detect_duplicated_genes
def detect_duplicated_genes(adata, tol=1e-8, remove=False):
    """
    Detect duplicate genes (linearly dependent columns) from an AnnData object by comparing normalized nonzero patterns.
    
    Parameters:
        adata (AnnData): an AnnData object with expression data in .X and gene names in .var_names.
        tol (float): tolerance for numerical comparison (default 1e-8).
        remove (bool): whether to remove duplicate genes (default False).

    Returns:
        AnnData: a new AnnData object with duplicate genes removed.
    """
    if adata.X is None or adata.var_names is None:
        raise ValueError("AnnData object must have X and var_names.")
    if not np.issubdtype(adata.X.dtype, np.number):
        raise ValueError("Expression data must be numeric.")
    
    x_matrix = adata.X
    if isinstance(x_matrix, np.matrix):
        x_csc = sp.csc_matrix(np.asarray(x_matrix))
    elif isinstance(x_matrix, np.ndarray):
        x_csc = sp.csc_matrix(x_matrix)
    elif sp.issparse(x_matrix):
        x_csc = x_matrix.tocsc()
    else:
        x_type = type(x_matrix)
        raise ValueError(f"This type of matrix |{x_type}| is not supported for run create_ggm.\nPlease convert to numpy.matrix, numpy.ndarray or scipy.sparse.csr_matrix.")
    
    del x_matrix
    gc.collect()

    var_names = adata.var_names

    seen = {}
    keep_cols = []
    
    for j in range(x_csc.shape[1]):
        col_start = x_csc.indptr[j]
        col_end = x_csc.indptr[j+1]
        row_idx = x_csc.indices[col_start:col_end]
        col_data = x_csc.data[col_start:col_end]
        
        if len(col_data) == 0:
            key = "zero"
        else:
            factor = col_data[0]
            if np.abs(factor) < tol:
                norm_data = col_data
            else:
                norm_data = col_data / factor
            norm_data = np.round(norm_data / tol) * tol
            key = (tuple(row_idx), tuple(norm_data))
        
        if key not in seen:
            seen[key] = j
            keep_cols.append(j)
        else:
            print(f"Gene {var_names[j]} is duplicated with Gene {var_names[seen[key]]}")
    
    if remove:
        adata_new = adata[:, keep_cols].copy()
        return adata_new
    else:
        return adata

#  
def detect_zero_in_csr(adata, remove=False):
    """
    Detect unexpected zero in a CSR matrix of an AnnData object.
    where unexpected zero is defined as zero values in .X.data but have indices in .X.indices.
    
    Parameters:
        adata (AnnData): an AnnData object with expression data in .X, which must be in CSR format.
        remove (bool): whether to remove unexpected zero  (default False).
    
    Returns:
        AnnData: a new AnnData object with zero values removed.
    """
    if adata.X is None:
        raise ValueError("AnnData object must have X.")
    if not sp.issparse(adata.X):
        print("This Function is only for adata.X in CSR format.")
        return adata
    else:
        if np.any(adata.X.data == 0):
            print("Zero values detected in adata.X.")
            if remove:
                adata.X.eliminate_zeros()
                print("Zero values removed.")
        else:
            print("No zero values detected in adata.X.")
    return adata
