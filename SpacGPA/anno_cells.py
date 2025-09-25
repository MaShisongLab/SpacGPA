
import pandas as pd
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import leidenalg
import sys
import random
from anndata import AnnData
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.stats import skew, rankdata
from igraph import Graph
from sklearn.neighbors import NearestNeighbors, KernelDensity
from matplotlib import colors as mcolors
from scanpy.plotting.palettes import default_20, vega_10, vega_20


#################### support functions ####################
# construct_spatial_weights 
def construct_spatial_weights(
        coords, 
        k_neighbors=6) -> sp.csr_matrix:
    """
    Construct a spatial weights matrix using kNN and 1/d as weights.
    The resulting W is NOT row-normalized.
    Diagonal entries are set to 0.
    
    Parameters:
        coords (np.array): Spatial coordinates of cells, shape (N, d).
        k_neighbors (int): Number of nearest neighbors.
        
    Returns:
        W (scipy.sparse.csr_matrix): Spatial weights matrix of shape (N, N).
    """
    N = coords.shape[0]
    knn = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean").fit(coords)
    dists, idx = knn.kneighbors(coords)

    with np.errstate(divide="ignore"):
        weight = (1.0 / dists)
    weight[dists == 0] = 0

    rows = np.repeat(np.arange(N), k_neighbors)
    cols = idx.reshape(-1)
    data = weight.reshape(-1)

    rows_sym = cols         
    cols_sym = rows
    data_sym = data        

    rows = np.concatenate((rows, rows_sym))
    cols = np.concatenate((cols, rows_sym)) 
    data = np.concatenate((data, data_sym))

    W = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    W.sum_duplicates()       
    W.setdiag(0)

    row_sums = np.asarray(W.sum(axis=1)).ravel()
    nz_rows = row_sums != 0
    inv_row = np.zeros_like(row_sums)
    inv_row[nz_rows] = 1.0 / row_sums[nz_rows]

    for i in np.where(nz_rows)[0]:
        start, end = W.indptr[i], W.indptr[i + 1]
        W.data[start:end] *= inv_row[i]

    W.eliminate_zeros()
    return W


# compute_moran
def compute_moran(x: np.ndarray, W: sp.csr_matrix) -> float:
    """
    Compute global Moran's I for vector x using the classical formula:
    
         I = (N / S0) * (sum_{i,j} w_{ij}(x_i - mean(x)) (x_j - mean(x)) / sum_i (x_i - mean(x))^2)
    
    Parameters:
        x (np.array): 1D expression vector for a gene, shape (N,).
        W (scipy.sparse.csr_matrix): Spatial weights matrix, shape (N, N), with zero diagonal.
    
    Returns:
        float: Moran's I value, or np.nan if variance is zero.
    """
    x = np.asarray(x).ravel()
    z = x - x.mean()
    denom = np.dot(z, z)
    if denom == 0:
        return np.nan
    num = z @ (W @ z)
    return num / denom    


# calc_border_flags
def calc_border_flags(coords, k, iqr_factor=1.5):
    """
    Identify border cells based on neighbor distances.

    Parameters:
        coords (ndarray): Spatial coordinates, shape (N, D).
        k (int): Number of neighbors to consider.
        iqr_factor (float): Multiplier for IQR when setting distance threshold.

    Returns:
        border_mask (ndarray of bool): True for border cells.
        knn_dists (ndarray): Distances to k nearest neighbors for each cell.
    """
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    mean_d = dists[:, 1:].mean(axis=1)
    q1, q3 = np.percentile(mean_d, [25, 75])
    threshold = q3 + iqr_factor * (q3 - q1)
    border_mask = mean_d > threshold
    return border_mask, dists[:, 1:]



#################### main function ####################
# assign_module_colors 
def assign_module_colors(adata, ggm_key='ggm', seed=1):
    """
    Create and store a consistent color mapping for gene modules.

    Depending on the number of modules, this function selects from
    Scanpy/Vega discrete palettes (up to 100 modules) or XKCD_COLORS
    (above 100 modules). If `seed` is non-zero, colors are shuffled
    reproducibly; if zero, the order is deterministic.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing module metadata under
        adata.uns['ggm_keys'][ggm_key].
    ggm_key : str, default 'ggm'
        Key in adata.uns['ggm_keys'] that defines module stats and
        where to store the resulting color dictionary.
    seed : int, default 1
        Random seed for reproducible shuffling. A value of 0 means
        no randomization.

    Returns
    -------
    dict
        Mapping from module IDs to hex color codes or named colors.
    """
    # Retrieve GGM metadata from adata.uns
    ggm_meta = adata.uns.get('ggm_keys', {}).get(ggm_key)
    if ggm_meta is None:
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    mod_info_key = ggm_meta.get('module_info')
    mod_col_val   = ggm_meta.get('module_colors')

    # Load module statistics and extract module IDs
    module_info = adata.uns.get(mod_info_key)
    if module_info is None:
        raise ValueError(f"Module Info not found in adata.uns['{mod_info_key}']")
    module_ids = module_info['module_id'].unique()

    n_all = len(module_ids)
    if n_all == 0:
        return {}
    n_modules = min(n_all, 806)

    # Initialize random number generator if needed
    rng = np.random.RandomState(seed) if seed != 0 else None
    
    # Select base color palette according to module count
    if n_modules <= 10:
        colors = vega_10[:n_modules]
    elif n_modules <= 20:
        colors = default_20[:n_modules]
    elif n_modules <= 40:
        # combine two 20-color palettes to reach up to 40
        colors = (default_20 + vega_20)[:n_modules]
    elif n_modules <= 60:
        # combine three 20-color palettes to reach up to 60
        tab20b = [mpl.colors.to_hex(c) for c in plt.get_cmap('tab20b').colors]
        colors = (default_20 + vega_20 + tab20b)[:n_modules]
    elif n_modules <= 100:
        tab20b  = [mpl.colors.to_hex(c) for c in plt.get_cmap('tab20b').colors]
        tab20c  = [mpl.colors.to_hex(c) for c in plt.get_cmap('tab20c').colors]
        set3    = [mpl.colors.to_hex(c) for c in plt.get_cmap('Set3').colors]
        pastel2 = [mpl.colors.to_hex(c) for c in plt.get_cmap('Pastel2').colors]
        palette_combo = default_20 + vega_20 + tab20b + tab20c + set3 + pastel2
        colors = palette_combo[:n_modules]
    else:
        # More than 100 modules: sample from filtered XKCD_COLORS by HSV order
        xkcd_colors = mcolors.XKCD_COLORS
        to_remove = ['gray','grey','black','white','light',
                     'lawngreen','silver','gainsboro','snow',
                     'mintcream','ivory','fuchsia','cyan']
        filtered = {
            name: col for name, col in xkcd_colors.items()
            if not any(key in name for key in to_remove)
        }
        sorted_hsv = sorted(
            ((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(col)[:3])), name))
            for name, col in filtered.items()
        )
        sorted_names = [name for hsv, name in sorted_hsv]
        if rng is not None:
            if len(sorted_names) >= n_modules:
                colors = list(rng.choice(sorted_names, size=n_modules, replace=False))
            else:
                base = sorted_names.copy()
                rem = n_modules - len(base)
                vir = plt.cm.viridis
                extra = [mpl.colors.to_hex(vir(i/(rem-1))) for i in range(rem)]
                colors = base + extra
        else:
            L = len(sorted_names)
            step = L / n_modules
            colors = [sorted_names[int(i * step)] for i in range(n_modules)]

    # Handle modules beyond 806 by repeating or sampling
    if n_all > 806:
        extra_count = n_all - 806
        if rng is not None:
            extra = list(rng.choice(colors, size=extra_count, replace=True))
        else:
            extra = [colors[i % len(colors)] for i in range(extra_count)]
        colors += extra

    # Shuffle full color list if RNG provided
    if rng is not None:
        perm = rng.permutation(len(colors))
        colors = [colors[i] for i in perm]

    # Build the module-to-color dictionary and store in adata.uns
    color_dict = {module_ids[i]: colors[i] for i in range(n_all)}
    if isinstance(mod_col_val, str):
        adata.uns.setdefault(mod_col_val, {}).update(color_dict)
    else:
        raise ValueError(f"Invalid module color key in adata.uns['{mod_col_val}']: {mod_col_val}")


# calculate_module_expression 
def calculate_module_expression(adata, 
                                ggm_obj, 
                                ggm_key = 'ggm',
                                top_genes=30, 
                                weighted=True,
                                calculate_moran=False, 
                                embedding_key='spatial',
                                k_neighbors=6,
                                add_go_anno=3,
                                set_module_colors=True):
    """
    Calculate and store module expression in adata based on input modules.
    
    Parameters:
        adata: AnnData object containing gene expression data.
        ggm_obj: GGM object containing module information or a DataFrame with columns 'module_id', 'gene', 'degree', and 'rank'.
        ggm_key: str, key for storing all information about the GGM object in adata.
                 Default 'ggm'. If not 'ggm', all information will be prefixed with ggm_key.
        top_genes: int, Number of top genes to use for module expression calculation.
        weighted: bool, Whether to calculate weighted average expression based on gene degree.
        calculate_moran: bool, if True, compute global Moran's I for each gene in module_info.
        embedding_key: str, key in adata.obsm for spatial coordinates.
        k_neighbors: int, number of nearest neighbors for constructing spatial weights.
        add_go_anno: int, default as 3. If the value is greater than 0 (and between 0 and 10), 
                     for each module, extract the top GO terms from ggm.go_enrichment and integrate them into module_df.
        set_module_colors: bool, if True, assign colors to modules and store in adata.uns['module_colors'].
    """
    # Make sure adata is an AnnData object and adata.X is a csr_matrix
    if isinstance(adata, AnnData):
        if adata.X is None or adata.var_names is None:
            raise ValueError("AnnData object must have X and var_names.")
        if not np.issubdtype(adata.X.dtype, np.number):
            raise ValueError("Expression data must be numeric.")
        x_matrix = adata.X
        if sp.issparse(x_matrix):
            x_matrix = x_matrix.tocsr()
        elif isinstance(x_matrix, np.matrix):
            print("Converting np.matrix to csr_matrix...")
            x_matrix = sp.csr_matrix(np.asarray(x_matrix))
        elif isinstance(x_matrix, np.ndarray):
            print("Converting np.ndarray to csr_matrix...")
            x_matrix = sp.csr_matrix(x_matrix)
    else:
        raise ValueError("adata must be an AnnData object.")

    # get module information
    if isinstance(ggm_obj, pd.DataFrame):
        module_df = ggm_obj.copy()
    else:
        if ggm_obj.modules is None:
            raise ValueError("No modules found in the GGM object. Please run `find_modules` first.")
        module_df = ggm_obj.modules.copy()
    
    if calculate_moran and embedding_key not in adata.obsm:
        raise ValueError(f"{embedding_key} coordinates not found in adata.obsm.")
     
    print(f"\nCalculating module expression using top {top_genes} genes...")
    
    if 'ggm_keys' not in adata.uns:
        adata.uns['ggm_keys'] = {}
    if ggm_key in adata.uns['ggm_keys']:
        print(f"\n'{ggm_key}' already exists in adata.uns['ggm_keys']. Overwriting all information about '{ggm_key}'...")
    
    # 1. Filter out genes with rank larger than top_genes
    module_df = module_df[module_df['rank'] <= top_genes].copy()

    # 2. Calculate the weights of each gene in the input module
    if weighted:
        print("\nCalculating gene weights based on degree...")
        module_df['weight']  = module_df.groupby('module_id')['degree'].transform(lambda x: x / x.sum())
    else:
        print("\nUsing unweighted gene expression...")
        module_df['weight'] = module_df.groupby('module_id')['degree'].transform(lambda x: 1 / x.size) 
    
    # 3. Filter the input modules to keep only genes that exist in adata
    genes_in_adata = adata.var_names
    module_df = module_df[module_df['gene'].isin(genes_in_adata)]
    module_df.index = range(module_df.shape[0])
    
    # 4. Set the keys for storing module information and expression
    if ggm_key == 'ggm':
        mod_info_key = 'module_info'
        mod_stats_key = 'module_stats'
        mod_color_key = 'module_colors'
        mod_filtering_key = 'module_filtering'
        expr_key = 'module_expression'
        expr_scaled_key = 'module_expression_scaled'
        col_prefix = ''
    else:
        mod_info_key = f"{ggm_key}_module_info"
        mod_stats_key = f"{ggm_key}_module_stats"
        mod_color_key = f"{ggm_key}_module_colors"
        mod_filtering_key = f"{ggm_key}_module_filtering"
        expr_key = f"{ggm_key}_module_expression"
        expr_scaled_key = f"{ggm_key}_module_expression_scaled"
        col_prefix = f"{ggm_key}_"
    
    module_df['module_id'] = col_prefix + module_df['module_id'].astype(str)
    
    # (optional) Add GO annotation to module information
    if add_go_anno:
        try:
            add_go_anno = int(add_go_anno)
        except Exception as e:
            raise ValueError("Parameter add_go_anno must be an integer between 0 and 10.") from e
        if add_go_anno < 0 or add_go_anno > 10:
            raise ValueError("Parameter add_go_anno must be between 0 and 10.")
        for i in range(1, add_go_anno+1):
            module_df[f"top_{i}_go_term"] = None
        
        # get GO enrichment information
        go_enrichment_df = None
        if isinstance(ggm_obj, pd.DataFrame):
            print("Warning: ggm_obj is a DataFrame; no GO enrichment information available.")
        else:
            if hasattr(ggm_obj, 'go_enrichment'):
                go_enrichment_df = ggm_obj.go_enrichment
            else:
                print("Warning: ggm_obj does not have go_enrichment attribute; skipping GO annotation integration.")
        
        if go_enrichment_df is not None:
            # add prefix to module_id column
            if col_prefix and 'module_id' in go_enrichment_df.columns:
                go_enrichment_df['module_id'] = col_prefix + go_enrichment_df['module_id'].astype(str)
            
            # find top GO terms for each module
            unique_modules = module_df['module_id'].unique()
            for mod in unique_modules:
                module_gene_set = set(module_df.loc[module_df['module_id'] == mod, 'gene'])
                mod_go_df = go_enrichment_df[go_enrichment_df['module_id'] == mod].sort_values(by='go_rank')
                for i in range(1, add_go_anno+1):
                    if i-1 < len(mod_go_df):
                        row = mod_go_df.iloc[i-1]
                        genes_str = row['genes_with_go_in_module']
                        if isinstance(genes_str, str):
                            go_genes = set(genes_str.split('/'))
                        else:
                            go_genes = set()
                        intersection = go_genes & module_gene_set
                        other_genes = module_gene_set - intersection
                        if intersection:
                            go_term_val = row['go_term']
                            module_df.loc[module_df['gene'].isin(intersection), f"top_{i}_go_term"] = go_term_val
                            if other_genes:
                                module_df.loc[module_df['gene'].isin(other_genes), f"top_{i}_go_term"] = ''
                        else:
                            module_df.loc[module_df['module_id'] == mod, f"top_{i}_go_term"] = ''
                    else:
                        module_df.loc[module_df['module_id'] == mod, f"top_{i}_go_term"] = ''

    # 5. Make a mapping from gene to index in adata and from module ID to index in the transformation matrix
    gene_to_index = {gene: i for i, gene in enumerate(adata.var_names)}
    module_ids = module_df['module_id'].unique()
    module_to_index = {module: i for i, module in enumerate(module_ids)}
    
    # Remove existing module-related columns in adata.obs
    for col in list(adata.obs.columns):
        if col.startswith(f'{col_prefix}M') and (col.endswith('_exp') or col.endswith('_exp_trim') or col.endswith('_anno') or col.endswith('_anno_smooth')):
            adata.obs.drop(columns=col, inplace=True)

    # 6. Construct a transformation matrix
    n_genes = len(adata.var_names)
    n_modules = len(module_ids)
    transformation_matrix = sp.lil_matrix((n_genes, n_modules), dtype=np.float32)
    
    for _, row in module_df.iterrows():
        gene_idx = gene_to_index[row['gene']]
        module_idx = module_to_index[row['module_id']]
        transformation_matrix[gene_idx, module_idx] = row['weight']
    
    transformation_matrix = transformation_matrix.tocsr()
    
    # 7. Multiply adata by the transformation matrix to obtain the weighted-average-expression matrix
    weighted_expression = x_matrix.dot(transformation_matrix)
    if sp.issparse(weighted_expression):
        weighted_expression = weighted_expression.toarray()
    
    # 8. Calculate the Moran's I for each gene in module_info
    if calculate_moran:
        coords = adata.obsm[embedding_key]
        W = construct_spatial_weights(coords, k_neighbors=k_neighbors)
        print("Calculating Moran's I for modules in ggm...")
        module_moran = {}
        for mod in module_ids:
            mod_expr = weighted_expression[:, module_to_index[mod]]
            I_mod = compute_moran(mod_expr, W)
            module_moran[mod] = I_mod
        print("Calculating Moran's I for genes in ggm...")
        moran_values = []
        for gene in module_df['gene']:
            i = gene_to_index[gene]
            if sp.issparse(x_matrix):
                x_gene = x_matrix[:, i].toarray().flatten()
            else:
                x_gene = x_matrix[:, i]
            I_gene = compute_moran(x_gene, W)
            moran_values.append(I_gene)
        module_df['module_moran_I'] = module_df['module_id'].map(module_moran)
        module_df['gene_moran_I'] = moran_values        

    # 9. Store module information in adata.uns
    print(f"Storing module information in adata.uns['{mod_info_key}']...")
    adata.uns[mod_info_key] = module_df.copy()
   
    # 10. Store the weighted-average-expression in both obsm and obs
    scaler = StandardScaler()
    adata.obsm[expr_key] = weighted_expression
    adata.obsm[expr_scaled_key] = scaler.fit_transform(weighted_expression)
    
    weighted_expression_df = pd.DataFrame(
        weighted_expression,
        index=adata.obs_names,
        columns=[f'{mod}_exp' for mod in module_ids]
    )
    obs_df = pd.concat([adata.obs, weighted_expression_df], axis=1)
    obs_df = obs_df.loc[:, ~obs_df.columns.duplicated(keep='last')]
    adata.obs = obs_df.copy()
    
    adata.uns['ggm_keys'][ggm_key] = {
        'module_info': mod_info_key,
        'module_stats': mod_stats_key,
        'module_colors': mod_color_key,
        'module_filtering': mod_filtering_key,
        'module_expression': expr_key,
        'module_expression_scaled': expr_scaled_key,
        'module_obs_prefix': col_prefix
    }
    # (Optional) Set module colors
    if set_module_colors:
        print(f"\nAssigning colors to {len(module_ids)} modules...")
        assign_module_colors(adata, ggm_key=ggm_key)
    print(f"\nTotal {n_modules} modules' average expression calculated and stored in adata.obs and adata.obsm")


# calculate_gmm_annotations
def calculate_gmm_annotations(adata, 
                              ggm_key='ggm',
                              modules_used=None,
                              modules_excluded=None,
                              calculate_moran=True,
                              embedding_key='spatial',
                              k_neighbors=6,
                              max_iter=200,
                              prob_threshold=0.99,
                              min_samples=10,
                              n_components=3,
                              enable_fallback=True,
                              random_state=42
                              ):
    """
    Gaussian Mixture Model annotation with additional module-level statistics.
    
    Statistics added to mod_stats_key include:
        - module_id: Module ID.
        - status: 'success' or 'failed'.
        - anno_one: Number of cells annotated as 1.
        - anno_zero: Number of cells expressed module but not annotated as 1.
        - skew: Skewness of the non-zero expression distribution for the module.
        - top1pct_ratio: Ratio of the average expression among the top 1% high-expressing cells
                        to the overall cells mean.
        - module_moran_I: Global Moran's I computed on the module expression (all cells) (if calculate_moran True).
        - positive_moran_I: Moran's I computed on module expression for cells annotated as 1 (if calculate_moran True).
        - negative_moran_I: Moran's I computed on module expression for cells annotated as 0 (if calculate_moran True).
        - n_components: Number of components in the GMM.
        - final_components: Number of components after fallback.
        - threshold: Threshold for calling a cell positive.
        - components: List of dictionaries with keys 'component', 'mean', 'var', 'weight'.
        - main_component: Index of the main component.
        - error_info: Error message if status is 'failed'.
        - top_go_terms: Top GO terms associated with the module.
    """

    # Retrieve keys from adata.uns['ggm_keys']
    ggm_keys = adata.uns.get('ggm_keys', {})
    if ggm_key not in ggm_keys:
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    mod_info_key = ggm_keys[ggm_key].get('module_info')
    mod_stats_key = ggm_keys[ggm_key].get('module_stats')
    expr_key = ggm_keys[ggm_key].get('module_expression')
    if expr_key not in adata.obsm:
        raise ValueError(f"{expr_key} not found in adata.obsm")
    if mod_info_key not in adata.uns:
        raise ValueError(f"{mod_info_key} not found in adata.uns")
    
    # Extract module expression matrix and module information
    module_expr_matrix = pd.DataFrame(adata.obsm[expr_key], index=adata.obs.index)
    unique_mods = adata.uns[mod_info_key]['module_id'].unique()
    if len(unique_mods) != module_expr_matrix.shape[1]:
        raise ValueError(f"module_info and module_expression dimensions for the ggm '{ggm_key}' do not match")
    else:
        module_expr_matrix.columns = unique_mods
    
    # Determine modules to use
    if modules_used is None and modules_excluded is None and mod_stats_key in adata.uns:
       adata.uns.pop(mod_stats_key, None)

    if modules_used is None:
        modules_used = list(unique_mods)
    if modules_excluded is not None:
        modules_used = [mid for mid in modules_used if mid not in modules_excluded]
    valid_modules = [mid for mid in modules_used if mid in module_expr_matrix.columns]
    if not valid_modules:
        raise ValueError(f"Ensure that the input module IDs exist in adata.uns['{mod_info_key}']")
    
    # Remove existing annotation columns
    for col in list(adata.obs.columns):
        if col.endswith('_anno') and any(col.startswith(mid) for mid in valid_modules):
            adata.obs.drop(columns=col, inplace=True)
        if col.endswith('_exp_trim') and any(col.startswith(mid) for mid in valid_modules):
            adata.obs.drop(columns=col, inplace=True)
        if col.endswith('_anno_smooth') and any(col.startswith(mid) for mid in valid_modules):
            adata.obs.drop(columns=col, inplace=True)    
    
    # Initialize annotation matrix (0/1) for modules
    anno_cols = valid_modules
    annotations = pd.DataFrame(np.zeros((adata.obs.shape[0], len(anno_cols)), dtype=int),
                               index=adata.obs.index, columns=anno_cols)
    stats_records = []
    
    # Pre-calculate expression ranking for tie-breaks
    expr_score = {}
    for mid in valid_modules:
        module_col = f"{mid}_exp"
        if module_col not in adata.obs.columns:
            raise KeyError(f"'{module_col}' not found in adata.obs.")
        rank_vals = adata.obs[module_col].rank(method='dense', ascending=False).astype(int)
        expr_score[mid] = rank_vals.values
    
    # If calculate_moran is True, construct spatial weights matrix W based on embedding_key.
    if calculate_moran:
        if embedding_key not in adata.obsm:
            raise ValueError(f"{embedding_key} not found in adata.obsm")
        coords = adata.obsm[embedding_key]
        W = construct_spatial_weights(coords, k_neighbors=k_neighbors)
    else:
        coords = None
        W = None
    
    # Build mapping from gene to index using adata.var_names
    gene_to_index = {gene: i for i, gene in enumerate(adata.var_names)}
    
    # Process each module for GMM annotation and extra statistics
    for module_id in valid_modules:
        stats = {
            'module_id': module_id,
            'status': 'success',
            'n_components': n_components,
            'final_components': n_components,
            'threshold': np.nan,
            'anno_one': 0,
            'anno_zero': 0,
            'components': [],
            'error_info': 'None',
            'module_moran_I': np.nan,
            'positive_moran_I': np.nan,
            'negative_moran_I': np.nan,
            'skew': np.nan,
            'top1pct_ratio': np.nan,
            'effect_size': np.nan
        }
        try:
            expr_values = module_expr_matrix[module_id].values
            non_zero_mask = expr_values != 0
            non_zero_expr = expr_values[non_zero_mask]
            if len(non_zero_expr) == 0:
                raise ValueError("all_zero_expression")
            if len(non_zero_expr) < min_samples:
                raise ValueError(f"insufficient_samples ({len(non_zero_expr)} < {min_samples})")
            if np.var(non_zero_expr) < 1e-6:
                raise ValueError("zero_variance")
            
            # Fit GMM on non-zero expression
            gmm = GaussianMixture(n_components=n_components, random_state=random_state, max_iter=max_iter)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                gmm.fit(non_zero_expr.reshape(-1, 1))
            means = gmm.means_.flatten()
            main_component = int(np.argmax(means))
            main_mean = means[main_component]
            probs = gmm.predict_proba(non_zero_expr.reshape(-1, 1))[:, main_component]
            anno_non_zero = (probs >= prob_threshold).astype(int)
            if np.sum(anno_non_zero) == 0:
                raise ValueError("no_positive_cells")
            positive_expr = non_zero_expr[anno_non_zero == 1]
            threshold = float(np.min(positive_expr))
            if n_components >= 3 and threshold > main_mean:
                raise ValueError(f"threshold {threshold:.2f} > μ ({main_mean:.2f})")
            stats.update({
                'threshold': threshold,
                'anno_one': int(np.sum(anno_non_zero)),
                'anno_zero': int(len(anno_non_zero) - np.sum(anno_non_zero)),
                'components': [
                    {'component': i,
                     'mean': float(gmm.means_[i][0]),
                     'var': float(gmm.covariances_[i][0][0]),
                     'weight': float(gmm.weights_[i])
                    } for i in range(n_components)
                ],
                'main_component': main_component
            })
            
            # Build full annotation vector for module: default 0, update non-zero positions
            module_annotation = np.zeros_like(expr_values, dtype=int)
            module_annotation[non_zero_mask] = anno_non_zero
            annotations[module_id] = module_annotation
            
            # Calculate additional statistics
            if calculate_moran:
                mod_I = compute_moran(expr_values, W)
                stats['module_moran_I'] = mod_I
                
                pos_expr_masked = np.where(module_annotation == 1, expr_values, 0)
                stats['positive_moran_I'] = compute_moran(pos_expr_masked, W)
                
                neg_expr_masked = np.where(module_annotation == 0, expr_values, 0)
                stats['negative_moran_I'] = compute_moran(neg_expr_masked, W)
            else:
                stats['module_moran_I'] = np.nan
                stats['positive_moran_I'] = np.nan
                stats['negative_moran_I'] = np.nan
            
            stats['skew'] = float(skew(non_zero_expr))
            
            if len(expr_values) > 0:
                top_n = max(1, int(len(expr_values) * 0.01))
                sorted_expr = np.sort(expr_values)
                top1_mean = np.mean(sorted_expr[-top_n:])
                overall_mean = np.mean(expr_values)
                stats['top1pct_ratio'] = top1_mean / overall_mean if overall_mean != 0 else np.nan
            else:
                stats['top1pct_ratio'] = np.nan
            
            if len(positive_expr) > 0:
                std_all = np.std(non_zero_expr)
                neg_expr = non_zero_expr[anno_non_zero == 0]
                stats['effect_size'] = float(np.mean(positive_expr) - np.mean(neg_expr)) / std_all if std_all != 0 else np.nan
            else:
                stats['effect_size'] = np.nan
            
        except Exception as e:
            stats.update({
                'status': 'failed',
                'error_info': str(e),
                'components': [],
                'threshold': np.nan,
                'anno_one': 0,
                'anno_zero': expr_values.size if 'expr_values' in locals() else 0,
                'module_moran_I': np.nan,
                'positive_moran_I': np.nan,
                'negative_moran_I': np.nan,
                'skew': np.nan,
                'top1pct_ratio': np.nan,
                'effect_size': np.nan
            })
            if enable_fallback and n_components > 2:
                try:
                    gmm = GaussianMixture(n_components=2, random_state=random_state, max_iter=max_iter)
                    gmm.fit(non_zero_expr.reshape(-1, 1))
                    means = gmm.means_.flatten()
                    main_component = int(np.argmax(means))
                    probs = gmm.predict_proba(non_zero_expr.reshape(-1, 1))[:, main_component]
                    anno_non_zero = (probs >= ((1 - (1 - prob_threshold) * 1e-2))).astype(int)
                    if np.sum(anno_non_zero) > 0:
                        positive_expr = non_zero_expr[anno_non_zero == 1]
                        threshold = float(np.min(positive_expr))
                        stats.update({
                            'status': 'success',
                            'final_components': 2,
                            'threshold': threshold,
                            'anno_one': int(np.sum(anno_non_zero)),
                            'anno_zero': int(len(anno_non_zero) - np.sum(anno_non_zero)),
                            'components': [
                                {'component': 0,
                                 'mean': float(gmm.means_[0][0]),
                                 'var': float(gmm.covariances_[0][0][0]),
                                 'weight': float(gmm.weights_[0])
                                },
                                {'component': 1,
                                 'mean': float(gmm.means_[1][0]),
                                 'var': float(gmm.covariances_[1][0][0]),
                                 'weight': float(gmm.weights_[1])
                                }
                            ],
                            'main_component': main_component
                        })
                        fallback_annotation = np.zeros_like(expr_values, dtype=int)
                        fallback_annotation[non_zero_mask] = anno_non_zero
                        annotations.loc[non_zero_mask, module_id] = anno_non_zero
                        if calculate_moran:
                            fallback_mod_I = compute_moran(expr_values, W)
                            stats['module_moran_I'] = fallback_mod_I
                            pos_expr_masked = np.where(fallback_annotation == 1, expr_values, 0)
                            stats['positive_moran_I'] = compute_moran(pos_expr_masked, W)
                            neg_expr_masked = np.where(fallback_annotation == 0, expr_values, 0)
                            stats['negative_moran_I'] = compute_moran(neg_expr_masked, W)
                        else:
                            stats['module_moran_I'] = np.nan
                            stats['positive_moran_I'] = np.nan
                            stats['negative_moran_I'] = np.nan
                        stats['skew'] = float(skew(non_zero_expr))
                        if len(non_zero_expr) > 0:
                            top_n = max(1, int(len(non_zero_expr) * 0.01))
                            sorted_expr = np.sort(non_zero_expr)
                            top1_mean = np.mean(sorted_expr[-top_n:])
                            overall_mean = np.mean(non_zero_expr)
                            stats['top1pct_ratio'] = top1_mean / overall_mean if overall_mean != 0 else np.nan
                        else:
                            stats['top1pct_ratio'] = np.nan
                        if len(positive_expr) > 0:
                            std_all = np.std(non_zero_expr)
                            neg_expr = non_zero_expr[anno_non_zero == 0]
                            stats['effect_size'] = float(np.mean(positive_expr) - np.mean(neg_expr)) / std_all if std_all != 0 else np.nan
                        else:
                            stats['effect_size'] = np.nan
                except Exception as fallback_e:
                    stats['error_info'] += f"; Fallback failed: {str(fallback_e)}"
        finally:
            if stats['status'] == 'success':
                print(f"{module_id} processed successfully, annotated cells: {stats['anno_one']}")
            else:
                print(f"{module_id} processed, failed: {stats['error_info']}")
            stats['components'] = str(stats['components'])
            stats_records.append(stats)
    
    # Transform annotations to categorical and store in adata.obs
    annotations.columns = [f"{col}_anno" for col in annotations.columns]
    for col in annotations.columns:
        orig_name = col.replace("_anno", "")
        annotations[col] = np.where(annotations[col] == 1, orig_name, None)
        annotations[col] = pd.Categorical(annotations[col])
    adata.obs = pd.concat([adata.obs, annotations], axis=1)
    
    # Add trimmed expression columns which are 0 if the annotation is None for this module
    trim_dict = {}
    for mod in valid_modules:
        exp_col = f"{mod}_exp"
        anno_col = f"{mod}_anno"
        trim_col = f"{mod}_exp_trim"
        if exp_col in adata.obs and anno_col in adata.obs:
            mask = (adata.obs[anno_col] == mod).astype(int)
            trim_dict[trim_col] = adata.obs[exp_col] * mask

    if trim_dict:
        trim_df = pd.DataFrame(trim_dict, index=adata.obs.index)
        adata.obs = pd.concat([adata.obs, trim_df], axis=1)

    # Add GO annotations to module_stats_key 
    stats_records_df = pd.DataFrame(stats_records)
    module_info_df = adata.uns[mod_info_key]
    go_cols = [col for col in module_info_df.columns if col.startswith("top_") and col.endswith("_go_term")]
    if len(go_cols) > 0:
        def concat_go_terms(mod_id):
            rows = module_info_df[module_info_df['module_id'] == mod_id]
            terms = []
            for col in go_cols:
                vals = rows[col].dropna().unique().tolist()
                if vals:
                    terms.extend(vals)
            if terms:
                return " || ".join(sorted(set(terms)))
            else:
                return ""
        stats_records_df["top_go_terms"] = stats_records_df["module_id"].apply(concat_go_terms)
        new_order = [
            'module_id', 'status', 'anno_one', 'anno_zero', 'top_go_terms', 'skew', 'top1pct_ratio', 
            'module_moran_I', 'positive_moran_I', 'negative_moran_I', 'effect_size',
            'n_components', 'final_components','threshold', 'components', 'main_component', 'error_info']
        stats_records_df = stats_records_df[new_order]
    else:
        new_order = [
            'module_id', 'status', 'anno_one', 'anno_zero', 'skew', 'top1pct_ratio',
            'module_moran_I', 'positive_moran_I', 'negative_moran_I', 'effect_size',
            'n_components', 'final_components','threshold', 'components', 'main_component', 'error_info']
        stats_records_df = stats_records_df[new_order]
    
    if mod_stats_key in adata.uns:
        existing_stats = adata.uns[mod_stats_key]
        for mid in stats_records_df['module_id'].unique():
            new_row = stats_records_df.loc[stats_records_df['module_id'] == mid].iloc[0]
            mask = existing_stats['module_id'] == mid
            if mask.any():
                num_rows = mask.sum()
                new_update_df = pd.DataFrame([new_row] * num_rows, index=existing_stats.loc[mask].index)
                existing_stats.loc[mask] = new_update_df
            else:
                existing_stats = pd.concat([existing_stats, pd.DataFrame([new_row])], ignore_index=True)
        existing_stats.dropna(how='all', inplace=True)
        adata.uns[mod_stats_key] = existing_stats
    else:
        adata.uns[mod_stats_key] = stats_records_df

        

# smooth_annotations
def smooth_annotations(
    adata,
    ggm_key='ggm',
    modules_used=None,
    modules_excluded=None,
    embedding_key='spatial',
    k_neighbors=24,
    min_drop_neighbors=1,
    min_add_neighbors='half',
    max_weight_ratio=1.5,
    border_iqr_factor=1.5,
    border_protect_fraction=0.3,
    verbose=True
):
    """
    Smooths module annotations by dropping isolated positives and adding supported negatives.

    Parameters:
        adata (AnnData): Must contain '<module>_anno' and '<module>_exp' in .obs.
        ggm_key (str): Key under adata.uns['ggm_keys'] containing module_stats.
        modules_used (list or None): Modules to process; None for all.
        modules_excluded (list or None): Modules to skip.
        embedding_key (str): Key in adata.obsm for coordinates.
        k_neighbors (int): Number of neighbors (excluding self).
        min_drop_neighbors (int): Minimum positive neighbors to retain a positive cell.
        min_add_neighbors ('half'|'none'|int): Neighbors needed to add negatives.
        max_weight_ratio (float): Cap for exp/threshold ratio.
        border_iqr_factor (float): IQR factor for border detection.
        border_protect_fraction (float): If fraction of positives on border exceeds this,
            border positives are protected from dropping.
        verbose (bool): Print before/after counts per module.
    """
    # Load module stats
    if ggm_key not in adata.uns.get('ggm_keys', {}):
        raise KeyError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    stats_key = adata.uns['ggm_keys'][ggm_key]['module_stats']
    stats_df = adata.uns[stats_key]

    # Build KNN and detect border cells
    coords = adata.obsm.get(embedding_key)
    if coords is None:
        raise KeyError(f"{embedding_key} not found in adata.obsm")
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, metric='euclidean').fit(coords)
    knn_dists_all, knn_idx_all = nbrs.kneighbors(coords)
    knn_idx = knn_idx_all[:, 1:]
    knn_dists = knn_dists_all[:, 1:]
    border_mask, _ = calc_border_flags(coords, k_neighbors, border_iqr_factor)

    # Determine modules to process
    all_mods = stats_df['module_id'].values
    mods = list(all_mods) if modules_used is None else [m for m in modules_used if m in all_mods]
    if modules_excluded:
        mods = [m for m in mods if m not in modules_excluded]
    
    # Remove existing smoothed annotation columns if they exist
    existing_columns = [f"{mid}_anno_smooth" for mid in mods if f"{mid}_anno_smooth" in adata.obs]
    if existing_columns:
        print(f"Removing existing smooth annotation columns: {existing_columns}")
        adata.obs.drop(columns=existing_columns, inplace=True)
    
    # Parse add threshold
    if isinstance(min_add_neighbors, str):
        if min_add_neighbors == 'half':
            add_thresh = max(1, k_neighbors // 2)
        elif min_add_neighbors == 'none':
            add_thresh = None
        else:
            raise ValueError("min_add_neighbors must be 'half', 'none', or int")
    else:
        add_thresh = int(min_add_neighbors)
        if add_thresh < 1:
            raise ValueError("min_add_neighbors must be >=1 when integer")

    # Prepare a dict to collect all smoothed annotation series
    smooth_dict = {}

    # Process each module
    for mod in mods:
        anno_col = f"{mod}_anno"
        exp_col = f"{mod}_exp"
        if anno_col not in adata.obs or exp_col not in adata.obs:
            if verbose:
                print(f"skip {mod}: missing anno/exp")
            continue

        a = (adata.obs[anno_col] == mod).astype(int).values
        E = adata.obs[exp_col].values

        # Compute weight and cap
        try:
            thr = stats_df.set_index('module_id').loc[mod, 'threshold']
        except KeyError:
            thr = np.percentile(E, 90)
        w = E / thr
        w = np.minimum(w, max_weight_ratio)

        # Determine border protection
        total_pos = a.sum()
        frac_border = (border_mask & (a == 1)).sum() / total_pos if total_pos > 0 else 0
        protect_border = frac_border > border_protect_fraction

        # Stage 1: drop isolated positives
        b1 = np.zeros_like(a)
        for i in range(len(a)):
            if a[i] != 1:
                continue
            neigh = knn_idx[i]
            support = np.sum(a[neigh] * w[neigh])
            required = 0.0 if (border_mask[i] and protect_border) else float(min_drop_neighbors)
            b1[i] = 1 if support >= required else 0

        # Stage 2: add supported negatives
        b2 = b1.copy()
        if add_thresh is not None:
            for i in range(len(a)):
                if b1[i] == 1:
                    continue
                cnt = np.sum(b1[knn_idx[i]])
                if cnt >= add_thresh:
                    b2[i] = 1

        # store the result in smooth_dict for batch concatenation later.
        out_col = f"{mod}_anno_smooth"
        smooth_dict[out_col] = pd.Categorical(np.where(b2, mod, None))

        if verbose:
            after = b2.sum()
            print(f"{mod} processed. remain cells: {after}")

    # Once all modules are processed, concatenate all smoothed columns at once
    if smooth_dict:
        smooth_df = pd.DataFrame(smooth_dict, index=adata.obs.index)
        adata.obs = pd.concat([adata.obs, smooth_df], axis=1)

    if verbose:
        print("\nAnnotation smoothing completed. Results stored in adata.obs.\n")


# annotate_with_ggm
def annotate_with_ggm(
    adata,
    ggm_obj,
    ggm_key='ggm',
    top_genes=30,
    weighted=True,
    calculate_gene_moran=False,
    calculate_module_moran=True,
    embedding_key='spatial',
    k_neighbors_for_moran=6,
    add_go_anno=3,
    max_iter=200,
    prob_threshold=0.99,
    min_samples=10,
    n_components=3,
    enable_fallback=True,
    random_state=42,
    k_neighbors_for_smooth=24,
    min_drop_neighbors=1,
    min_add_neighbors='half',
    max_weight_ratio=1.5,
    border_iqr_factor=1.5,
    border_protect_fraction=0.3,
    modules_used=None,
    modules_excluded=None,
    verbose=True
):
    """
    Execute the Annotate and Smooth pipeline for GGM analysis in one step:
      1. Compute module average expression using provided GGM information (via calculate_module_expression).
      2. Annotate cells based on module expression with a Gaussian Mixture Model (GMM) and calculate additional module-level statistics (via calculate_gmm_annotations).
      3. Perform spatial smoothing on the annotation results (via smooth_annotations).

    Parameters:
      --- For calculate_module_expression ---
      adata: AnnData object containing gene expression data.
      ggm_obj: GGM object or DataFrame containing module information.
      ggm_key: Key for storing GGM information in adata, default 'ggm'.
      top_genes: Number of top genes used for module expression calculation, default 30.
      weighted: Whether to compute weighted average expression based on gene degree, default True.
      calculate_gene_moran: Whether to compute Moran's I for genes during module expression calculation, default False.
      embedding_key: Key in adata.obsm that stores spatial coordinates, default 'spatial'.
      k_neighbors_for_moran: Number of neighbors used for constructing the spatial weight matrix, default 6.
      add_go_anno: Parameter for GO annotation integration; default 3 (extracts top 3 GO terms).

      --- For calculate_gmm_annotations ---
      ggm_key: Key for storing GGM information in adata, default 'ggm'. same as above.
      calculate_module_moran: Whether to compute Moran's I for modules during GMM annotation, default True.
      embedding_key: Key in adata.obsm that stores spatial coordinates, default 'spatial'. same as above.
      k_neighbors_for_moran: Number of neighbors used for constructing the spatial weight matrix, default 6. same as above.
      max_iter: Maximum iterations for the GMM, default 200.
      prob_threshold: Probability threshold for calling a cell positive, default 0.99.
      min_samples: Minimum number of non-zero samples required for GMM analysis, default 10.
      n_components: Number of components in the GMM, default 3.
      enable_fallback: Whether to fallback to a 2-component model if GMM fitting fails, default True.
      random_state: Random seed for reproducibility, default 42.
      (Note: Optional parameters like modules_used and modules_excluded are handled within the function.)

      --- For smooth_annotations ---
      ggm_key: Key for storing GGM information in adata, default 'ggm'. same as above.
      embedding_key: Key in adata.obsm that stores spatial coordinates, default 'spatial'. same as above.
      k_neighbors_for_smooth: Number of KNN neighbors used for smoothing annotations, default 24.
      min_drop_neighbors: Minimum number of annotated neighbors required to retain a positive annotation, default 1.
      min_add_neighbors: 'half'|'none'|int for smoothing addition, default 'half'.
      max_weight_ratio: Cap for expression ratio in smoothing, default 1.5.
      border_iqr_factor: IQR factor for border detection in smoothing, default 1.5.
      border_protect_fraction: Fraction threshold for protecting border cells in smoothing, default 0.3.
      modules_used: Modules to smooth; default None.
      modules_excluded: Modules to exclude from smoothing; default None.

    Returns:
      The updated AnnData object with module expression, cell annotations, and smoothed results stored in .obs.
    """
    # 1. Compute module average expression
    print("============ Calculating module average expression ============")
    calculate_module_expression(
        adata=adata,
        ggm_obj=ggm_obj,
        ggm_key=ggm_key,
        top_genes=top_genes,
        weighted=weighted,
        calculate_moran=calculate_gene_moran,
        embedding_key=embedding_key,
        k_neighbors=k_neighbors_for_moran,
        add_go_anno=add_go_anno
    )

    # 2. Annotate cells with GMM
    print("\n======== Annotating cells based on module expression ========")
    calculate_gmm_annotations(
        adata=adata,
        ggm_key=ggm_key,
        calculate_moran=calculate_module_moran,
        embedding_key=embedding_key,
        k_neighbors=k_neighbors_for_moran,
        max_iter=max_iter,
        prob_threshold=prob_threshold,
        min_samples=min_samples,
        n_components=n_components,
        enable_fallback=enable_fallback,
        random_state=random_state
    )

    # 3. Smooth annotations spatially
    print("\n=================== Smoothing annotations ===================")
    smooth_annotations(
        adata=adata,
        ggm_key=ggm_key,
        modules_used=modules_used,
        modules_excluded=modules_excluded,
        embedding_key=embedding_key,
        k_neighbors=k_neighbors_for_smooth,
        min_drop_neighbors=min_drop_neighbors,
        min_add_neighbors=min_add_neighbors,
        max_weight_ratio=max_weight_ratio,
        border_iqr_factor=border_iqr_factor,
        border_protect_fraction=border_protect_fraction,
        verbose=verbose
    )
    print("\n============= Finished annotating and smoothing =============")



# integrate_annotations
def integrate_annotations(
    adata,
    ggm_key="ggm",
    modules_used=None,
    modules_excluded=None,
    modules_preferred=None,
    result_anno="anno_density",
    embedding_key="spatial",
    k_neighbors=24,
    purity_adjustment=True,
    alpha=0.4,
    beta=0.3,
    gamma=0.3,
    delta=0.4,
    lambda_pair=0.3,
    lr=0.1,
    target_purity=0.8,
    w_floor=0.1,
    w_ceil=1.0,
    max_iter=100,
    energy_tol=1e-3,
    p0=0.1,
    tau=8.0,
    random_state=None,
):
    """
    Integrate module-wise annotations into a single, spatially smooth label
    using a first-order Potts Conditional Random Field (CRF).

    Parameters
    ----------
    adata : AnnData
        Must contain:
          • adata.obsm[embedding_key]: spatial coordinates
          • per-module columns 'Mx_anno' or 'Mx_anno_smooth', and 'Mx_exp'
    ggm_key : str
        Key in adata.uns['ggm_keys'] pointing to module stats.
    modules_used : list[str] or None
        Modules to include; None means use all.
    modules_excluded : list[str] or None
        Modules to remove from the set.
    modules_preferred : list[str] or None
        Preferred modules when multiple candidates exist for a cell.
    result_anno : str
        Name of the output column in adata.obs.
    embedding_key : str
        Key for spatial coordinates in adata.obsm.
    k_neighbors : int
        Number of nearest neighbours (including the cell).
    purity_adjustment : bool
        Whether to perform Leiden clustering each iteration to adjust module weights.
    alpha, beta, gamma, delta : float
        Weights of the unary terms:
          alpha - neighbor-vote weight (how much neighboring labels influence cost)
          beta  - expression-rank weight (how much gene expression rank influences cost)
          gamma - density weight (how much spatial cluster density influences cost)
          delta - low-purity penalty weight (penalizes modules with low global purity, only if purity_adjustment is True)
    lambda_pair : float
        Strength of the Potts pairwise smoothing term (penalizes label differences).
    lr : float
        Learning rate for dynamic module weights.
    target_purity : float
        Target purity for weight updates.
    w_floor, w_ceil : float
        Bounds for dynamic module weights.
    max_iter : int
        Maximum number of iterations.
    energy_tol : float
        Convergence threshold on relative energy change.
    p0 : float
        Initial perturbation probability (simulated annealing).
    tau : float
        Decay constant for the perturbation probability.
    random_state : int or None
        Seed for random number generator.

    Returns
    -------
    AnnData
        The same AnnData with adata.obs[result_anno] containing final labels.
    """
    # reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    rng = random.Random(random_state)

    # sanity checks
    if embedding_key not in adata.obsm:
        raise KeyError(f"{embedding_key} not found in adata.obsm")
    if ggm_key not in adata.uns["ggm_keys"]:
        raise KeyError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    
    # print parameter meanings and values
    print(
        f"Main parameters for integration:\n"
        f"  alpha = {alpha:.2f}  for neighbor-vote weight\n"
        f"  beta  = {beta:.2f}  for expression-rank weight\n"
        f"  gamma = {gamma:.2f}  for spatial-density weight\n"
        f"  delta = {delta:.2f}  for low-purity penalty weight\n"
        f"  lambda_pair = {lambda_pair:.2f}  for potts smoothing strength\n"
    )

    # load module list
    stats_key = adata.uns["ggm_keys"][ggm_key]["module_stats"]
    modules_df = adata.uns[stats_key]
    all_modules = list(pd.unique(modules_df["module_id"]))
    modules_used = modules_used or all_modules
    if modules_excluded:
        modules_used = [m for m in modules_used if m not in modules_excluded]

    # build spatial KNN graph
    X = adata.obsm[embedding_key]
    n_cells = adata.n_obs
    knn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X)
    dist, idx = knn.kneighbors(X)
    sigma = max(np.mean(dist[:, 1:]), 1e-6)
    W = np.exp(-dist**2 / sigma**2)
    lam = lambda_pair * W[:, 1:]

    # load annotations, compute ranks and density
    anno = {}
    rank_norm = {}
    dens_log = {}
    for m in modules_used:
        col = f"{m}_anno_smooth" if f"{m}_anno_smooth" in adata.obs else f"{m}_anno"
        if col not in adata.obs or f"{m}_exp" not in adata.obs:
            raise KeyError(f"Missing columns for module {m}")
        lab = (adata.obs[col] == m).astype(int).values
        anno[m] = lab

        r = rankdata(-adata.obs[f"{m}_exp"].values, method="dense")
        rank_norm[m] = (r - 1) / (n_cells - 1)

        pts = X[lab == 1]
        if len(pts) >= 3:
            dens_log[m] = -KernelDensity().fit(pts).score_samples(X)
        else:
            dens_log[m] = np.zeros(n_cells)

    # define unary energy
    w_mod = {m: 1.0 for m in modules_used}
    def unary(i, m):
        nb = idx[i, 1:]
        s = W[i, 1:].sum()
        vote = (W[i, 1:] * anno[m][nb]).sum() / s if s else 0.0
        vote *= w_mod[m]
        return (
            alpha * (1 - vote)
            + beta * rank_norm[m][i]
            + gamma * dens_log[m][i]
            + delta * (1 - w_mod[m])
        )

    # initial labeling
    label = np.empty(n_cells, dtype=object)
    for i in range(n_cells):
        cand = [m for m in modules_used if anno[m][i]] or modules_used
        if modules_preferred:
            pref = [m for m in cand if m in modules_preferred]
            cand = pref or cand
        label[i] = min(cand, key=lambda m: unary(i, m))

    # energy function
    def energy():
        u = sum(unary(i, label[i]) for i in range(n_cells))
        p = float((lam * (label[idx[:, 1:]] != label[:, None])).sum())
        return u + p

    prev_E = energy()
    print("Starting optimization...")
    print(f"Iteration:   0, Energy: {prev_E:.2f}")

    # optimization loop
    converged = False
    last_pr_len = 0
    for t in range(1, max_iter + 1):
        # optionally update module weights based on Leiden purity
        if purity_adjustment:
            g = nx.Graph()
            g.add_nodes_from(range(n_cells))
            for i in range(n_cells):
                for j in idx[i, 1:]:
                    g.add_edge(i, j)
            nx.set_node_attributes(g, {i: label[i] for i in range(n_cells)}, "label")
            part = leidenalg.find_partition(
                Graph.from_networkx(g),
                leidenalg.RBConfigurationVertexPartition,
                seed=random_state
            )

            purity = {m: [] for m in modules_used}
            for cluster in part:
                counts = {}
                for v in cluster:
                    counts[label[v]] = counts.get(label[v], 0) + 1
                cp = max(counts.values()) / len(cluster)
                for m, v in counts.items():
                    if m in modules_used:
                        purity[m].append(v / len(cluster) * cp)
            for m in modules_used:
                if purity[m]:
                    mp = float(np.mean(purity[m]))
                    w_mod[m] = np.clip(w_mod[m] * (1 + lr * (mp - target_purity)),
                                       w_floor, w_ceil)

        # ICM with annealed perturbation
        changed = 0
        p_t = p0 * np.exp(-t / tau)
        for i in range(n_cells):
            cand = [m for m in modules_used if anno[m][i]] or modules_used
            if modules_preferred:
                pref = [m for m in cand if m in modules_preferred]
                cand = pref or cand
            if label[i] not in cand:
                cand.append(label[i])

            if random.random() < p_t:
                new_lab = rng.choice(cand)
            else:
                pair_cost = lam[i] * (label[idx[i, 1:]] != np.array(cand)[:, None])
                total = [unary(i, m) + pair_cost[k].sum() for k, m in enumerate(cand)]
                new_lab = cand[int(np.argmin(total))]

            if new_lab != label[i]:
                label[i] = new_lab
                changed += 1

        curr_E = energy()
        msg = f"Iteration: {t:3}, Energy: {curr_E:.2f}, ΔE: {prev_E-curr_E:+.2f}, Changed: {changed}"
        sys.stdout.write("\r" + " " * last_pr_len + "\r")
        sys.stdout.write(msg)
        sys.stdout.flush()
        last_pr_len = len(msg)
        if abs(prev_E - curr_E) / max(abs(prev_E), 1.0) < energy_tol:
            print("\nConverged\n")
            converged = True
            break
        prev_E = curr_E

    if not converged:
        print("\nStopped after max_iter without convergence\n")

    adata.obs[result_anno] = label













############################################################################################################
# Old functions
# smooth_annotations_noadd
def smooth_annotations_noadd(adata,
                            ggm_key='ggm', 
                            modules_used=None,
                            modules_excluded=None, 
                            embedding_key='spatial', k_neighbors=24, min_annotated_neighbors=1):
    """
    Smooth spatial annotations by processing each module's annotation.
    
    Parameters:
      adata (anndata.AnnData): AnnData object containing spatial transcriptomics data.
      ggm_key (str): Key for the GGM object in adata.uns['ggm_keys'].
      modules_used (list): List of modules to smooth; if None, all modules are used.(default None)
      modules_excluded (list): List of modules to exclude from smoothing (default None).
      embedding_key (str): Key in adata.obsm for spatial coordinates (default 'spatial').
      k_neighbors (int): Number of KNN neighbors (default 24); may need adjustment based on technology and cell density.
      min_annotated_neighbors (int): Minimum number of neighbors with annotation 1 required to retain the annotation (default 1).
    """
    if ggm_key not in adata.uns['ggm_keys']:
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    
    module_stats_key = adata.uns['ggm_keys'][ggm_key]['module_stats']

    # Check input: ensure the embedding key exists in adata.obsm
    if embedding_key not in adata.obsm:
        raise ValueError(f"{embedding_key} not found in adata.obsm. Please ensure the coordinate exists.")
    
    # If modules_used is not provided, get all modules from adata.uns['module_stats']
    if modules_used is None:
        modules_used = adata.uns[module_stats_key]['module_id'].unique()
    # Exclude modules if the modules_excluded list is provided
    if modules_excluded is not None:
        modules_used = [mid for mid in modules_used if mid not in modules_excluded]
                        
    # Remove existing smoothed annotation columns if they exist
    existing_columns = [f"{mid}_anno_smooth" for mid in modules_used if f"{mid}_anno_smooth" in adata.obs]
    if existing_columns:
        print(f"Removing existing smooth annotation columns: {existing_columns}")
        adata.obs.drop(columns=existing_columns, inplace=True)

    # Extract spatial coordinates and the annotation columns to be smoothed
    embedding_coords = adata.obsm[embedding_key]
    module_annotations = adata.obs.loc[:, [f"{mid}_anno" for mid in modules_used]]
    # Reset the module id anno to 0/1 anno
    for col in module_annotations.columns:
        orig_name = col.replace("_anno", "")
        module_annotations[col] = (module_annotations[col] == orig_name).astype(int)

        n_cells, _ = module_annotations.shape

    # Compute KNN neighbors based on the embedding coordinates
    print(f"\nCalculating {k_neighbors} nearest neighbors for each cell based on {embedding_key} embedding...\n")
    k = k_neighbors + 1  # include self
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(embedding_coords)
    _, indices = nbrs.kneighbors(embedding_coords)  # indices: KNN indices for each cell

    # Initialize the smoothed annotation matrix
    smooth_annotations = np.zeros_like(module_annotations, dtype=int)

    # Smooth each module's annotation
    for mid in module_annotations.columns:
        module_values = module_annotations[mid].values  # current module's annotation values
        smooth_values = np.zeros(n_cells, dtype=int)  # initialize smoothed values

        for i in range(n_cells):
            if module_values[i] == 1:  # only process cells with annotation 1
                neighbor_values = module_values[indices[i, 1:]]  # KNN neighbor annotations (excluding self)
                if np.sum(neighbor_values) >= min_annotated_neighbors:  # if at least min_annotated_neighbors neighbors are 1
                    smooth_values[i] = 1

        # Store the smoothed values in the matrix
        smooth_annotations[:, module_annotations.columns.get_loc(mid)] = smooth_values
        print(f"{mid.replace('_anno', '')} processed. removed cells: {np.sum(module_values) - np.sum(smooth_values)}, remain cells: {np.sum(smooth_values)}")

    # Save the smoothed annotations to adata.obs
    smooth_annotations = pd.DataFrame(smooth_annotations,
                                      index=adata.obs_names,
                                      columns=[f"{mid}_smooth" for mid in module_annotations.columns])
    # Reset the 0/1 anno to module id or None
    for col in smooth_annotations.columns:
        orig_name = col.replace("_anno_smooth", "")
        smooth_annotations[col] = np.where(smooth_annotations[col] == 1, orig_name, None)
        smooth_annotations[col] = pd.Categorical(smooth_annotations[col])
    adata.obs = pd.concat([adata.obs, smooth_annotations], axis=1)

    print("\nAnnotation smoothing completed. Results stored in adata.obs.\n")

# integrate_annotations_noweight
def integrate_annotations_noweight(adata,
                                    ggm_key='ggm',
                                    cross_ggm = False,
                                    modules_used=None,
                                    modules_excluded=None,
                                    modules_preferred=None,
                                    result_anno='annotation',
                                    use_smooth=True,
                                    embedding_key='spatial',
                                    k_neighbors=24,
                                    neighbor_similarity_ratio=0.90
                                    ):            
    """
    Integrate cell annotations from multiple modules using the following logic:
      1) Optionally use smoothed annotations (controlled by use_smooth);
      2) Automatically compute k_neighbors nearest neighbors;
      3) For cells annotated by multiple modules:
            3.1 If There are modules_preferred and the cell has these modules in its potential annotation,
                the cell will be given priority to be annotated with these modules. 
            3.2 If the fraction of a cell's neighbors annotated with a particular module exceeds 
                the neighbor_similarity_ratio threshold, the cell is assigned that module.
            3.3 Otherwise, the final annotation is determined based on expression scores, where 
                higher expression scores confer higher priority.
    
    Parameters:
      adata (anndata.AnnData): AnnData object containing module annotations.
      ggm_key (str): Key for the GGM object in adata.uns['ggm_keys'].
      cross_ggm (bool): Whether to integrate annotations from multiple GGMs (default False).
                        When True, modules_used must be provided manually.
      modules_used (list): List of modules to integrate; if None, all modules of the GGM mentioned in ggm_key will be used.
      modules_excluded (list): List of modules to exclude from integration (default None).
      modules_preferred (list): List of preferred modules; if a cell has these modules in its potential annotation, 
                                the cell will be given priority to be annotated with these modules.
      result_anno (str): Column name for the integrated annotation (default 'annotation').
      use_smooth (bool): Whether to use smoothed annotations (default True).
      embedding_key (str): Key in adata.obsm for KNN coordinates (default 'spatial').
      k_neighbors (int): Number of KNN neighbors (default 24); may need adjustment based on technology and cell density.
      neighbor_similarity_ratio (float): A threshold representing the proportion of a cell's annotation that must match 
                                         its neighboring cells' annotations. 
                                         If the matching ratio exceeds this threshold, the cell will be directly annotated 
                                         with its neighbor's annotation. (Valid range: [0, 1]. default 0.90.)
    """
    if ggm_key not in adata.uns['ggm_keys']:
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    mod_stats_key = adata.uns['ggm_keys'][ggm_key]['module_stats']

    if cross_ggm and len(adata.uns['ggm_keys']) > 2: 
        if modules_used is None:
            raise ValueError("When cross_ggm is True, modules_used must be provided manually.")

    # Check input: ensure embedding_key exists in adata.obsm.
    if neighbor_similarity_ratio < 0 or neighbor_similarity_ratio > 1:
        raise ValueError("neighbor_similarity_ratio must be within [0, 1].")
    if neighbor_similarity_ratio == 0:
        print("When set to 0, the neighbor_similarity_ratio will not be used.")

    if  embedding_key not in adata.obsm:
        raise ValueError(f"{embedding_key} not found in adata.obsm. Please ensure the coordinate exists.")
    
    # Check if the integrated annotation column already exists; if so, remove it.
    if adata.obs.get(result_anno) is not None:
        print(f"NOTE: The '{result_anno}' already exists in adata.obs, which will be overwritten.")
        adata.obs.drop(columns=result_anno, inplace=True)
    
    # Check and extract annotation columns.
    # If modules_used is not provided, use all modules in adata.uns[mod_stats_key].
    if modules_used is None:
        modules_used = adata.uns[mod_stats_key]['module_id'].unique()
    # Exclude modules if the modules_excluded list is provided.
    if modules_excluded is not None:
        modules_used = [mid for mid in modules_used if mid not in modules_excluded]

    if use_smooth:
        # Identify modules missing smoothed annotation columns.
        missing_smooth = [mid for mid in modules_used if f"{mid}_anno_smooth" not in adata.obs]
        if missing_smooth:
            print(f"\nThese modules do not have 'Smooth anno': {missing_smooth}. Using 'anno' instead.")
        # Use smoothed columns if available; otherwise, use the original '_anno' columns.
        existing_columns = []
        for mid in modules_used:
            if f"{mid}_anno_smooth" in adata.obs:
                existing_columns.append(f"{mid}_anno_smooth")
            elif f"{mid}_anno" in adata.obs:
                existing_columns.append(f"{mid}_anno")
    else:
        existing_columns = [f"{mid}_anno" for mid in modules_used if f"{mid}_anno" in adata.obs]
    
    if len(existing_columns) != len(modules_used):
        raise ValueError("The annotation columns for the specified modules do not fully match those in adata.obs. Please check your input.")
    
    
    # 1) Compute nearest neighbors.
    print(f"\nCalculating {k_neighbors} nearest neighbors for each cell based on {embedding_key} embedding...\n")
    embedding_coords = adata.obsm[embedding_key]
    k = k_neighbors + 1  # include self
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(embedding_coords)
    _, indices = nbrs.kneighbors(embedding_coords)

    n_obs = adata.n_obs
    combined_annotation = [None] * n_obs  # Store final integrated annotation for each cell.

    # 2) Pre-calculate expression scores based on ranked expression.
    expr_score = {}
    anno_dict = {}
    for mid in modules_used:
        module_col = f"{mid}_exp"
        if module_col not in adata.obs.columns:
            raise KeyError(f"'{module_col}' not found in adata.obs.")        
        rank_vals = adata.obs[module_col].rank(method='dense', ascending=False).astype(int)
        expr_score[mid] = rank_vals.values
    
    # 3) Build annotation mask.
    for mid in modules_used:
        if f"{mid}_anno" in existing_columns:
            anno_col = f"{mid}_anno"
        else:
            anno_col = f"{mid}_anno_smooth"    
        anno_dict[mid] = (adata.obs[anno_col] == mid).astype(int).values
    
    unclear_cells = 0
    unclear_cells_neighbor = 0
    unclear_cells_rank = 0
    # 3) Integrate annotations for each cell.
    for i in range(n_obs):
        # Find which modules annotate the cell.
        annotated_modules = [p for p in modules_used if anno_dict[p][i]]
        if len(annotated_modules) > 1:
            unclear_cells += 1

        # If modules_preferred is provided, take the intersection (prioritize these modules).
        if modules_preferred is not None:
            intersection = [p for p in annotated_modules if p in modules_preferred]
            if intersection:
                annotated_modules = intersection
        
        if len(annotated_modules) == 0:
            combined_annotation[i] = None
            continue
        if len(annotated_modules) == 1:
            combined_annotation[i] = annotated_modules[0]
            continue
        
        # Multiple annotations => Step 1: neighbor voting.
        neighbor_idx = indices[i, 1:]  # exclude self
        n_nb = len(neighbor_idx)

        # Count annotations among neighbors for each module.
        neighbor_counts = {}
        for mid in annotated_modules:
            neighbor_anno = anno_dict[mid][neighbor_idx]
            neighbor_counts[mid] = np.sum(neighbor_anno)

        # Check if any module meets the neighbor majority fraction.
        annotated_modules_re = []
        for mid in annotated_modules:
            frac = neighbor_counts[mid] / n_nb
            if frac >= neighbor_similarity_ratio:
                annotated_modules_re.append(mid)
        
        if len(annotated_modules_re) == 1:
            combined_annotation[i] = annotated_modules_re[0]
            unclear_cells_neighbor += 1
            continue
        elif len(annotated_modules_re) > 1:
            annotated_modules = annotated_modules_re    

        # Step 2: If neighbor voting is indecisive, use expression score.
        best_mid = None
        best_score = n_obs + 1  # initialize with a large value 
        for mid in annotated_modules:
            sc = expr_score[mid][i]
            if sc < best_score:
                best_score = sc
                best_mid = mid
            
        combined_annotation[i] = best_mid
        unclear_cells_rank += 1
    
    # Print statistics about cells with multiple annotations.
    print(f"\n{unclear_cells} of total {n_obs} cells have multiple annotations. Among them, ")
    print(f"    {unclear_cells_neighbor} cells are resolved by neighbors.")
    print(f"    {unclear_cells_rank} cells are resolved by expression score.")

    # Update integrated annotation in adata.obs.
    annotation_df = pd.Series(combined_annotation, index=adata.obs.index, name=result_anno)
    adata.obs[result_anno] = annotation_df
    print(f"\nIntegrated annotation stored in adata.obs['{result_anno}'].\n")


# integrate_annotations_old
def integrate_annotations_old(adata, ggm_key='ggm',cross_ggm = False,
                              modules_used=None, modules_excluded=None,
                              result_anno = "annotation", use_smooth=True):
    """
    Integrate module annotations into a single cell annotation.
    
    Parameters:
        adata (anndata.AnnData): AnnData object containing module annotations.
        ggm_key (str): Key for the GGM object in adata.uns['ggm_keys'].
        cross_ggm (bool): Whether to integrate annotations from multiple GGMs (default False).
        modules_used (list): List of module IDs to integrate; if None, all modules are used.
        modules_excluded (list): List of module IDs to exclude from integration (default None).
        result_anno (str): Column name for the integrated annotation (default 'annotation').
        use_smooth (bool): Whether to use smoothed annotations (default True).
    """
    if ggm_key not in adata.uns['ggm_keys']:    
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    mod_stats_key = adata.uns['ggm_keys'][ggm_key]['module_stats']

    if cross_ggm and len(adata.uns['ggm_keys']) > 2:
        if modules_used is None:
            raise ValueError("When cross_ggm is True, modules_used must be provided manually.")

    # Check if the integrated annotation column already exists
    if adata.obs.get(result_anno) is not None:
        print(f"NOTE: The '{result_anno}' already exists in adata.obs, which will be overwritten.")
        adata.obs.drop(columns=result_anno, inplace=True)

    # Determine and extract annotation columns
    if modules_used is None:
        modules_used = adata.uns[mod_stats_key]['module_id'].unique()
    # Exclude modules if the modules_excluded list is provided
    if modules_excluded is not None:
        modules_used = [mid for mid in modules_used if mid not in modules_excluded]

    if use_smooth:
        # Identify modules missing smoothed annotation columns
        missing_smooth = [mid for mid in modules_used if f"{mid}_anno_smooth" not in adata.obs]
        if missing_smooth:
            print(f"\nThese modules do not have 'Smooth anno': {missing_smooth}. Using 'anno' instead.")
        # For modules with smoothed columns, use them; otherwise, use the original '_anno' columns
        existing_columns = []
        for mid in modules_used:
            if f"{mid}_anno_smooth" in adata.obs:
                existing_columns.append(f"{mid}_anno_smooth")
            elif f"{mid}_anno" in adata.obs:
                existing_columns.append(f"{mid}_anno")
    else:
        existing_columns = [f"{mid}_anno" for mid in modules_used if f"{mid}_anno" in adata.obs]
    
    if len(existing_columns) != len(modules_used):
        raise ValueError("The annotation columns for the specified modules do not fully match those in adata.obs. Please check your input.")
    
    
    # Extract module annotations
    module_annotations = adata.obs.loc[:, [mid for mid in existing_columns]]
    # Reset the module id anno to 0/1 anno
    for col in module_annotations.columns:
        orig_name_1 = col.replace("_anno", "")
        orig_name_2 = col.replace("_anno_smooth", "")
        if orig_name_1 in module_annotations.columns:
            module_annotations[col] = (module_annotations[col] == orig_name_1).astype(int)
        else:
            module_annotations[col] = (module_annotations[col] == orig_name_2).astype(int)
        
    # Compute the number of annotated cells for each module
    module_counts = module_annotations.sum(axis=0)

    # Sort modules by the number of annotated cells in ascending order
    sorted_modules = module_counts.sort_values().index

    # Initialize an empty Series to store the final cell annotations
    cell_annotations = pd.Series(index=module_annotations.index, dtype=object)

    # Annotate cells iteratively
    for module in sorted_modules:
        module_name = module.replace("_anno_smooth", "")
        module_name = module_name.replace("_anno", "")
        # Find cells with annotation value 1 for the current module
        cells_to_annotate = module_annotations.index[module_annotations[module] == 1]

        # Only annotate cells that have not been annotated yet
        cells_to_annotate = cells_to_annotate[cell_annotations[cells_to_annotate].isna()]

        # Set these cells to the current module name
        cell_annotations[cells_to_annotate] = module_name

    # Add the final cell annotations to adata.obs
    adata.obs[result_anno] = cell_annotations

    print(f"Cell annotation completed. Results stored in adata.obs['{result_anno}'].")

# def construct_spatial_weights(coords, k_neighbors=6):
#     """
#     Construct a spatial weights matrix using kNN and 1/d as weights.
#     The resulting W is NOT row-normalized.
#     Diagonal entries are set to 0.
    
#     Parameters:
#         coords (np.array): Spatial coordinates of cells, shape (N, d).
#         k_neighbors (int): Number of nearest neighbors.
        
#     Returns:
#         W (scipy.sparse.csr_matrix): Spatial weights matrix of shape (N, N).
#     """
#     N = coords.shape[0]
#     nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean').fit(coords)
#     distances, indices_knn = nbrs.kneighbors(coords)
    
#     # calculate weights
#     with np.errstate(divide='ignore', invalid='ignore'):
#         weights = 1 / distances
#     weights[distances == 0] = 0
    
#     # construct sparse matrix
#     row_idx = np.repeat(np.arange(N), k_neighbors)
#     col_idx = indices_knn.flatten()
#     data_w = weights.flatten()
#     W = sp.coo_matrix((data_w, (row_idx, col_idx)), shape=(N, N)).tocsr()
#     # set diagonal to zero
#     W.setdiag(0)
#     return W

# def compute_moran(x, W):
#     """
#     Compute global Moran's I for vector x using the classical formula:
    
#          I = (N / S0) * (sum_{i,j} w_{ij}(x_i - mean(x)) (x_j - mean(x)) / sum_i (x_i - mean(x))^2)
    
#     Parameters:
#         x (np.array): 1D expression vector for a gene, shape (N,).
#         W (scipy.sparse.csr_matrix): Spatial weights matrix, shape (N, N), with zero diagonal.
    
#     Returns:
#         float: Moran's I value, or np.nan if variance is zero.
#     """
#     N = x.shape[0]
#     x_bar = np.mean(x)
#     z = x - x_bar
#     denominator = np.sum(z ** 2)
#     if denominator == 0:
#         return np.nan
#     S0 = W.sum()
#     numerator = z.T.dot(W.dot(z))
#     return (N / S0) * (numerator / denominator)

# def calculate_gmm_annotations(adata, 
#                               ggm_key='ggm',
#                               modules_used = None,
#                               modules_excluded = None,
#                               max_iter=200,
#                               prob_threshold=0.99,
#                               min_samples=10,
#                               n_components=3,
#                               enable_fallback=True,
#                               random_state=42
#                               ):
#     """
#     Gaussian Mixture Model annotation (with threshold check for 3 components).
    
#     Parameters:
#       adata: AnnData object.
#       ggm_key: Key for the GGM object in adata.uns['ggm_keys'].
#       modules_used: List of module IDs.(default None)
#       modules_excluded: List of module IDs to exclude.(default None)
#       max_iter: Maximum iterations.
#       prob_threshold: Probability threshold for high expression.
#       min_samples: Minimum valid sample count.
#       n_components: Initial number of GMM components.
#       enable_fallback: Whether to enable fallback to fewer components.
#       random_state: Random seed.
      
#     Returns:
#       adata: Updated AnnData object with:
#         - obs: Integrated annotation data (columns prefixed with "Anno_").
#         - uns['module_stats']: Raw statistics records.
#     """
#     # Input validation
#     if ggm_key not in adata.uns['ggm_keys']:
#         raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")

#     mod_info_key = adata.uns['ggm_keys'][ggm_key]['module_info']
#     expr_key = adata.uns['ggm_keys'][ggm_key]['module_expression']
#     mod_stats_key = adata.uns['ggm_keys'][ggm_key]['module_stats']

#     if expr_key not in adata.obsm:
#         raise ValueError(f"{expr_key} not found in adata.obsm")
#     if mod_info_key not in adata.uns:
#         raise ValueError(f"{mod_info_key} not found in adata.uns")
    
#     module_expr_matrix = adata.obsm[expr_key]
#     module_expr_matrix = pd.DataFrame(module_expr_matrix, index=adata.obs.index)
#     if len(adata.uns[mod_info_key]['module_id'].unique()) != module_expr_matrix.shape[1]:
#         raise ValueError(f"module_info and module_expression dimensions for the ggm '{ggm_key}' do not match")
#     else:
#         module_expr_matrix.columns = adata.uns[mod_info_key]['module_id'].unique()

#     # Check module list
#     # If modules_used is not provided, use all modules in adata.uns[mod_info_key]
#     if modules_used is None:
#         modules_used = adata.uns[mod_info_key]['module_id'].unique()
#     # Exclude modules if the modules_excluded list is provided
#     if modules_excluded is not None:
#         modules_used = [mid for mid in modules_used if mid not in modules_excluded]

#     valid_modules = [mid for mid in modules_used if mid in module_expr_matrix.columns]
    
#     if not valid_modules:
#         raise ValueError(f"Ensure that the input module IDs exist in adata.uns['{mod_info_key}']")
    
#     existing_columns = [f"{mid}_anno" for mid in modules_used if f"{mid}_anno" in adata.obs]
#     if existing_columns:
#         print(f"Removing existing annotation columns: {existing_columns}")
#         adata.obs.drop(columns=existing_columns, inplace=True)

#     # Initialize annotation matrix
#     anno_cols = [f"{mid}" for mid in modules_used]
#     annotations = pd.DataFrame(
#         np.zeros((adata.obs.shape[0], len(anno_cols)), dtype=int),
#         index=adata.obs.index,
#         columns=anno_cols
#     )
#     stats_records = []
    
#     # Process each module
#     for module_id in valid_modules:
#         stats = {
#             'module_id': module_id,
#             'status': 'success',
#             'n_components': n_components,
#             'final_components': n_components,
#             'threshold': np.nan,
#             'anno_one': 0,
#             'anno_zero': 0,
#             'components': [],
#             'error_info': 'None'
#         }
        
#         try:
#             expr_values = module_expr_matrix[module_id].values
#             non_zero_mask = expr_values != 0
#             non_zero_expr = expr_values[non_zero_mask]
            
#             # Basic checks
#             if len(non_zero_expr) == 0:
#                 raise ValueError("all_zero_expression")
#             if len(non_zero_expr) < min_samples:
#                 raise ValueError(f"insufficient_samples ({len(non_zero_expr)}<{min_samples})")
#             if np.var(non_zero_expr) < 1e-6:
#                 raise ValueError("zero_variance")

#             # Fit GMM
#             gmm = GaussianMixture(
#                 n_components=n_components,
#                 random_state=random_state,
#                 max_iter=max_iter
#             )
#             with warnings.catch_warnings():
#                 warnings.filterwarnings("ignore")
#                 gmm.fit(non_zero_expr.reshape(-1, 1))

#             # Select high-expression component
#             means = gmm.means_.flatten()
#             main_component = np.argmax(means)
#             main_mean = means[main_component]
            
#             # Compute probabilities and generate annotation
#             probs = gmm.predict_proba(non_zero_expr.reshape(-1, 1))[:, main_component]
#             anno_non_zero = (probs >= prob_threshold).astype(int)
            
#             if np.sum(anno_non_zero) == 0:
#                 raise ValueError("no_positive_cells")
            
#             # Compute expression threshold
#             positive_expr = non_zero_expr[anno_non_zero == 1]
#             threshold = np.min(positive_expr)
            
#             # For 3-component model, check threshold validity
#             if n_components >= 3 and threshold > main_mean:
#                 raise ValueError(f"threshold {threshold:.2f} > μ ({main_mean:.2f})")
            
#             # Store module annotation
#             anno_col = f"{module_id}"
#             annotations.loc[non_zero_mask, anno_col] = anno_non_zero
            
#             # Update statistics
#             stats.update({
#                 'threshold': threshold,
#                 'anno_one': int(anno_non_zero.sum()),
#                 'anno_zero': int(len(anno_non_zero) - anno_non_zero.sum()),
#                 'components': [
#                     {   'component': i,
#                         'mean': float(gmm.means_[i][0]),
#                         'var': float(gmm.covariances_[i][0][0]),
#                         'weight': float(gmm.weights_[i])
#                     } 
#                     for i in range(n_components)
#                 ],
#                 'main_component': int(main_component)
#             })

#         except Exception as e:
#             stats.update({
#                 'status': 'failed',
#                 'error_info': str(e),
#                 'components': [],
#                 'threshold': np.nan,
#                 'anno_one': 0,
#                 'anno_zero': expr_values.size
#             })
            
#             # Fallback strategy
#             if enable_fallback and n_components > 2:
#                 try:
#                     # Try 2-component model (without threshold check)
#                     gmm = GaussianMixture(
#                         n_components=2,
#                         random_state=random_state,
#                         max_iter=max_iter
#                     )
#                     gmm.fit(non_zero_expr.reshape(-1, 1))
                    
#                     # Select high-expression component
#                     means = gmm.means_.flatten()
#                     main_component = np.argmax(means)
                    
#                     probs = gmm.predict_proba(non_zero_expr.reshape(-1, 1))[:, main_component]
#                     #anno_non_zero = (probs >= 0.9999).astype(int)
#                     #anno_non_zero = (probs >= prob_threshold).astype(int)
#                     anno_non_zero = (probs >= ((1 - (1 - prob_threshold) * 1e-2))).astype(int)
                    
#                     if anno_non_zero.sum() > 0:
#                         positive_expr = non_zero_expr[anno_non_zero == 1]
#                         threshold = np.min(positive_expr)
                        
#                         stats.update({
#                             'status': 'success',
#                             'final_components': 2,
#                             'threshold': threshold,
#                             'anno_one': int(anno_non_zero.sum()),
#                             'anno_zero': int(len(anno_non_zero) - anno_non_zero.sum()),
#                             'components': [
#                                 {   'component': 0,
#                                     'mean': float(gmm.means_[0][0]),
#                                     'var': float(gmm.covariances_[0][0][0]),
#                                     'weight': float(gmm.weights_[0])
#                                 },
#                                 {   'component': 1,
#                                     'mean': float(gmm.means_[1][0]),
#                                     'var': float(gmm.covariances_[1][0][0]),
#                                     'weight': float(gmm.weights_[1])
#                                 }
#                             ],
#                             'main_component': int(main_component)
#                         })
#                         annotations.loc[non_zero_mask, f"{module_id}"] = anno_non_zero
#                 except Exception as fallback_e:
#                     stats['error_info'] += f"; Fallback failed: {str(fallback_e)}"

#         finally:
#             if stats['status'] == 'success':
#                 print(f"{module_id} processed, {stats['status']}, anno cells : {stats['anno_one']}")
#             else:
#                 print(f"{module_id} processed, {stats['status']}")

#             if stats.get('components'):
#                 stats['components'] = str(stats['components'])
#             else:
#                 stats['components'] = 'None'
#             stats_records.append(stats)

#     # Store annotations in adata.obs        
#     annotations.columns = [f"{col}_anno" for col in annotations.columns]
    
#     # Reset the 0/1 anno to module id or None
#     for col in annotations.columns:
#         orig_name = col.replace("_anno", "")
#         annotations[col] = np.where(annotations[col] == 1, orig_name, None)
#         annotations[col] = pd.Categorical(annotations[col])

#     adata.obs = pd.concat([adata.obs, annotations], axis=1)
    
#     # Store statistics in adata.uns
#     stats_records_df = pd.DataFrame(stats_records)
#     stats_records_df = pd.DataFrame(stats_records)
#     if mod_stats_key in adata.uns:
#         existing_stats = adata.uns[mod_stats_key]
#         # For each module, update existing records with new data
#         for mid in stats_records_df['module_id'].unique():
#             new_row = stats_records_df.loc[stats_records_df['module_id'] == mid].iloc[0]
#             mask = existing_stats['module_id'] == mid
#             if mask.any():
#                 num_rows = mask.sum()
#                 new_update_df = pd.DataFrame([new_row] * num_rows, index=existing_stats.loc[mask].index)
#                 existing_stats.loc[mask] = new_update_df
#             else:
#                 existing_stats = pd.concat([existing_stats, pd.DataFrame([new_row])], ignore_index=True)
#         existing_stats.dropna(how='all', inplace=True)
#         adata.uns[mod_stats_key] = existing_stats
#     else:
#         adata.uns[mod_stats_key] = stats_records_df

