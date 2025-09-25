import numpy as np
import pandas as pd
import networkx as nx
import torch
import gc
import itertools
import scanpy as sc

def run_mcl_qc(SigEdges, expansion=2, inflation=1.7, add_self_loops='mean', 
               max_iter=1000, tol=1e-6, pruning_threshold=1e-5, 
               run_mode=2):
    """
    Quickly perform MCL-Hub clustering on input SigEdges (DataFrame with 'GeneA', 'GeneB', 'Pcor')
    until the candidate node integration stage and return the clustering result.
    
    Parameters:
        SigEdges: DataFrame with 'GeneA', 'GeneB', and 'Pcor' columns.
        expansion: Exponent for the expansion step (default 2).
        inflation: Exponent for the inflation step (typically >1, default 1.7).
        add_self_loops: Method for adding self-loops. Options: 'mean', 'min', 'max', 'dynamic', 'none' (default 'mean').
        max_iter: Maximum iterations (default 1000).
        tol: Convergence threshold (default 1e-6).
        pruning_threshold: Elements below this threshold are set to zero (default 1e-5).
        run_mode: 0 for CPU, non-zero for GPU (default 2, uses GPU).
        
    Returns:
        clusters: A list of sets, each set contains indices of genes (according to SigEdges order) in one cluster.
    """
    try:
        # 1. Extract unique genes and create a mapping from gene to index.
        genes = pd.unique(SigEdges[['GeneA', 'GeneB']].values.ravel())
        gene_index = {gene: i for i, gene in enumerate(genes)}
        n = len(genes)
        
        # Build a symmetric weighted adjacency matrix using 'Pcor' values.
        M = np.zeros((n, n), dtype=np.float32)
        for _, row in SigEdges.iterrows():
            i = gene_index[row['GeneA']]
            j = gene_index[row['GeneB']]
            M[i, j] = row['Pcor']
            M[j, i] = row['Pcor']
        
        # Construct the original graph (for candidate node integration) with nodes as indices.
        G_ori = nx.from_numpy_array(M)
        
        # 2. Add self-loops according to the specified method.
        if add_self_loops == 'mean':
            np.fill_diagonal(M, np.mean(M[M > 0]))
        elif add_self_loops == 'min':
            np.fill_diagonal(M, np.min(M[M > 0]))
        elif add_self_loops == 'max':
            np.fill_diagonal(M, np.max(M))
        elif add_self_loops == 'dynamic':
            for col_idx in range(M.shape[1]):
                nonzero_elements = M[:, col_idx][M[:, col_idx] > 0]
                if len(nonzero_elements) > 0:
                    M[col_idx, col_idx] = np.mean(nonzero_elements)
                else:
                    M[col_idx, col_idx] = np.min(M[M > 0])
        elif add_self_loops == 'none':
            pass
        else:
            raise ValueError("Invalid value for 'add_self_loops'. Choose from 'mean', 'min', 'max', 'dynamic', or 'none'.")
        
        # 3. Convert the matrix to a PyTorch tensor and set the computation device.
        device = 'cpu' if run_mode == 0 else 'cuda'
        M = torch.tensor(M, device=device)
        
        # Define a function for column normalization (each column sums to 1).
        def normalize_columns(matrix):
            col_sums = matrix.sum(dim=0)
            return matrix / col_sums
        
        # Initial normalization.
        M = normalize_columns(M)
        
        # 4. MCL iteration: expansion, inflation, normalization, pruning, and convergence check.
        for iteration in range(max_iter):
            M_prev = M.clone()
            M = torch.matrix_power(M, expansion)   # Expansion step
            M = M.pow(inflation)                     # Inflation step
            M = normalize_columns(M)
            M[M < pruning_threshold] = 0            # Pruning step
            M = normalize_columns(M)
            diff = torch.max(torch.abs(M - M_prev))
            if diff < tol:
                break
        
        # 5. Convert the final matrix back to NumPy and construct a graph for candidate integration.
        M_np = M.cpu().numpy()
        G = nx.from_numpy_array(M_np)
        del M, M_np  # Release memory
        
        # 6. Extract connected components as initial clusters.
        clusters = list(nx.connected_components(G))
        # Treat single-node clusters as candidate nodes and remove them from clusters.
        size1_clusters = [cluster for cluster in clusters if len(cluster) == 1]
        candidate_nodes = set()
        for cluster in size1_clusters:
            for node in cluster:
                candidate_nodes.add(node)
        clusters = [cluster for cluster in clusters if len(cluster) > 1]
        
        # 7. Helper function to compute the shortest path.
        def find_shortest_path(net, source, target):
            try:
                return nx.shortest_path(net, source=source, target=target)
            except nx.NetworkXNoPath:
                return None
        
        # 8. For each cluster (sorted in descending order by size), find zero-degree nodes 
        # and use candidate nodes to supplement the cluster.
        clusters_sorted = sorted(clusters, key=lambda x: len(x), reverse=True)
        for cluster in clusters_sorted:
            subgraph = G.subgraph(cluster)
            degrees = dict(subgraph.degree())
            if not degrees:
                continue
            attractor = max(degrees, key=degrees.get)  # Select the node with highest degree
            subgraph_ori = G_ori.subgraph(cluster)
            degree_zero_nodes = [node for node, deg in subgraph_ori.degree() if deg == 0]
            if degree_zero_nodes:
                net_temp = G_ori.subgraph(set(candidate_nodes) | cluster)
                for node in degree_zero_nodes:
                    path = find_shortest_path(net_temp, node, attractor)
                    if path:
                        for n in path:
                            if n in candidate_nodes:
                                candidate_nodes.remove(n)
                        cluster.update(path)
        
        # 9. For each cluster, move zero-degree nodes to the candidate set.
        for cluster in clusters:
            subgraph_ori = G_ori.subgraph(cluster)
            for node, deg in list(subgraph_ori.degree()):
                if deg == 0:
                    cluster.remove(node)
                    candidate_nodes.add(node)
        
        # 10. Map nodes to clusters for candidate assignment.
        node_to_cluster = {}
        for idx, cluster in enumerate(clusters):
            for node in cluster:
                node_to_cluster[node] = idx
        
        # 11. For each candidate node, assign it to the best matching cluster based on neighbor counts.
        for node in list(candidate_nodes):
            neighbors = set(G_ori.neighbors(node))
            if not neighbors:
                continue
            cluster_count = {}
            for nbr in neighbors:
                if nbr in node_to_cluster:
                    cid = node_to_cluster[nbr]
                    cluster_count[cid] = cluster_count.get(cid, 0) + 1
            total_neighbors = sum(cluster_count.values())
            for cid, count in cluster_count.items():
                if count >= total_neighbors / 2:
                    clusters[cid].add(node)
                    node_to_cluster[node] = cid
                    candidate_nodes.remove(node)
                    break
        
        # Return the clustering result: each cluster is a set of node indices.
        return clusters
    
    finally:
        gc.collect()
        if run_mode != 0:
            torch.cuda.empty_cache()

def build_original_graph(SigEdges, add_self_loops='mean'):
    """
    Constructs the original network graph from SigEdges for modularity computation.
    Nodes are represented as indices.
    """
    genes = pd.unique(SigEdges[['GeneA', 'GeneB']].values.ravel())
    gene_index = {gene: i for i, gene in enumerate(genes)}
    n = len(genes)
    M = np.zeros((n, n), dtype=np.float32)
    for _, row in SigEdges.iterrows():
        i = gene_index[row['GeneA']]
        j = gene_index[row['GeneB']]
        M[i, j] = row['Pcor']
        M[j, i] = row['Pcor']
    if add_self_loops == 'mean':
        np.fill_diagonal(M, np.mean(M[M > 0]))
    elif add_self_loops == 'min':
        np.fill_diagonal(M, np.min(M[M > 0]))
    elif add_self_loops == 'max':
        np.fill_diagonal(M, np.max(M))
    elif add_self_loops == 'dynamic':
        for col_idx in range(M.shape[1]):
            nonzero_elements = M[:, col_idx][M[:, col_idx] > 0]
            if len(nonzero_elements) > 0:
                M[col_idx, col_idx] = np.mean(nonzero_elements)
            else:
                M[col_idx, col_idx] = np.min(M[M > 0])
    elif add_self_loops == 'none':
        pass
    G_ori = nx.from_numpy_array(M)
    return G_ori

# find_best_inflation
def find_best_inflation(ggm, min_inflation=1.1, max_inflation=5,
                        coarse_step=0.1, mid_step=0.05, fine_step=0.01,
                        expansion=2, add_self_loops='mean', max_iter=1000,
                        tol=1e-6, pruning_threshold=1e-5, run_mode=2,
                        min_module_size=10,
                        phase=3, show_plot=False):
    """
    Optimize the inflation parameter for MCL clustering using SigEdges data by 
    performing a three-phase search and evaluating clustering quality using the 
    NetworkX modularity metric.

    Phases:
      - Phase 1 (Coarse Search): Starting from min_inflation, increment by coarse_step.
        If, within the first three evaluations, both the second and third modularity 
        values are lower than the first, then min_inflation is considered too high and 
        the process terminates. Otherwise, if modularity decreases in two consecutive 
        steps, the phase stops and the inflation with the highest modularity in this 
        phase (max_coarse) is selected.
      - Phase 2 (Mid-step Search): Search the range [max_coarse - coarse_step, max_coarse + coarse_step]
        with a step of mid_step to obtain an intermediate best value (max_mid).
      - Phase 3 (Fine Search): Search the range [max_mid - mid_step, max_mid + mid_step]
        with a step of fine_step to obtain the final optimal inflation value (max_fine).

    The 'phase' parameter controls execution:
      - phase = 1: Execute only Phase 1.
      - phase = 2: Execute Phases 1 and 2.
      - phase = 3: Execute all three phases.

    All evaluation points are recorded, and an optional scatter plot is produced (if show_plot=True)
    with the x-axis ticks set at mid_step intervals. The best point is marked with a red dot,
    and dashed lines are drawn at the best inflation and modularity values.

    Parameters:
      ggm: GGM object containing the SigEdges information.
      max_inflation: Maximum inflation value for the search.
      min_inflation: Minimum inflation value allowed (default 1.1).
      coarse_step: Step size for coarse search (default 0.1).
      mid_step: Step size for mid-step search (default 0.05).
      fine_step: Step size for fine search (default 0.01).
      expansion, add_self_loops, max_iter, tol, pruning_threshold, run_mode:
          Parameters for run_mcl_qc controlling the MCL clustering.
      phase: Which phase to execute up to (1, 2, or 3; default 3).
      show_plot: Whether to display a scatter plot (default False).

    Returns:
      (best_inflation, best_modularity)
    """
    print("\nFinding best inflation value for MCL-Hub clustering...\n")
    # Build the original graph for modularity computation.
    if ggm.SigEdges is None:
        raise ValueError("ggm object does not have 'SigEdges' information.")
    SigEdges = ggm.SigEdges
    G_ori = build_original_graph(SigEdges, add_self_loops=add_self_loops)
    
    # Cache to store computed modularity values to avoid redundant calculations.
    eval_cache = {}
    # Global lists to record evaluation points (used for plotting).
    global_inflations = []
    global_modularities = []
    
    def evaluate_inflation(inflation):
        if inflation in eval_cache:
            return eval_cache[inflation]
        clusters = run_mcl_qc(SigEdges, expansion=expansion, inflation=inflation, 
                                add_self_loops=add_self_loops, max_iter=max_iter, 
                                tol=tol, pruning_threshold=pruning_threshold, 
                                run_mode=run_mode)
        
        sorted_clusters = sorted(clusters, key=lambda comp: len(comp), reverse=True)
        filtered_clusters = [cluster for cluster in sorted_clusters if len(cluster) >= min_module_size]
        filtered_clustered_nodes = set().union(*filtered_clusters)
        G_sub = G_ori.subgraph(filtered_clustered_nodes)

        # Ensure all nodes are assigned (unassigned nodes become singleton communities).
        # all_nodes = set(G_ori.nodes())
        # clustered_nodes = set().union(*clusters)
        # missing_nodes = all_nodes - clustered_nodes
        # for node in missing_nodes:
        #     clusters.append({node})

        if len(filtered_clusters) <= 1:
            Q = 0.0
        else:
            #Q = nx.algorithms.community.modularity(G_ori, clusters, weight='weight')
            Q = nx.algorithms.community.modularity(G_sub, filtered_clusters, weight='weight',resolution=2)
        eval_cache[inflation] = Q
        return Q

    best_inflation = None
    best_modularity = -np.inf

    # -------------------- Phase 1: Coarse Search (Ascending) --------------------
    print("Phase 1: Coarse search")
    inflation_val = min_inflation
    coarse_results = []
    consecutive_decrease = 0
    prev_Q = None
    first_Q = None
    second_Q = None
    third_Q = None

    while inflation_val <= max_inflation:
        Q = evaluate_inflation(inflation_val)
        coarse_results.append((inflation_val, Q))
        global_inflations.append(inflation_val)
        global_modularities.append(Q)
        #print(f"Inflation = {inflation_val:.2f}, Modularity = {Q:.4f}")
        
        # Check the first three evaluations.
        if len(coarse_results) == 1:
            first_Q = Q
        elif len(coarse_results) == 2:
            second_Q = Q
        elif len(coarse_results) == 3:
            third_Q = Q
            if second_Q < first_Q and third_Q < first_Q:
                print("\nTerminating: min_inflation is set too high; initial modularity exceeds subsequent values. Please adjust min_inflation.")
                return None, None
        
        if prev_Q is not None:
            if Q < prev_Q:
                consecutive_decrease += 1
            else:
                consecutive_decrease = 0
        prev_Q = Q
        
        if consecutive_decrease >= 2:
            break
        inflation_val = round(inflation_val + coarse_step, 4)
    
    max_coarse = max(coarse_results, key=lambda x: x[1])[0]
    print(f"Coarse search best inflation: {max_coarse:.2f}")
    
    if phase == 1:
        best_inflation, best_modularity = max_coarse, evaluate_inflation(max_coarse)
        print(f"\nBest inflation (Phase 1): {best_inflation:.2f}, Modularity: {best_modularity:.4f}")
        if show_plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            x_min, x_max = min(global_inflations), max(global_inflations)
            ticks = np.arange(round(x_min,2), round(x_max + mid_step,2), mid_step)
            plt.xticks(ticks)
            plt.scatter(global_inflations, global_modularities, c='blue', alpha=0.6, edgecolors='k')
            plt.scatter([best_inflation], [best_modularity], c='red', s=100, edgecolors='k', label="Best")
            plt.axvline(x=best_inflation, linestyle='--', color='lightgrey')
            plt.axhline(y=best_modularity, linestyle='--', color='lightgrey')
            plt.xlabel("Inflation")
            plt.ylabel("Modularity")
            plt.title("Inflation vs. Modularity")
            plt.grid(False)
            plt.tight_layout()
            plt.legend()
            plt.show()
        return best_inflation, best_modularity
    
    # -------------------- Phase 2: Mid-step Search --------------------
    print("\nPhase 2: Mid-step search")
    lower_bound = max(min_inflation, max_coarse - coarse_step)
    upper_bound = min(max_inflation, max_coarse + coarse_step)
    mid_results = []
    inflation_vals_mid = np.arange(lower_bound, upper_bound + mid_step/2, mid_step)
    for inflation_val in inflation_vals_mid:
        inflation_val = round(inflation_val, 4)
        Q = evaluate_inflation(inflation_val)
        mid_results.append((inflation_val, Q))
        if inflation_val not in global_inflations:
            global_inflations.append(inflation_val)
            global_modularities.append(Q)
        #print(f"Inflation = {inflation_val:.2f}, Modularity = {Q:.4f}")
    
    max_mid = max(mid_results, key=lambda x: x[1])[0]
    print(f"Mid-step search best inflation: {max_mid:.2f}")
    
    if phase == 2:
        best_inflation, best_modularity = max_mid, evaluate_inflation(max_mid)
        print(f"\nBest inflation (Phase 2): {best_inflation:.2f}, Modularity: {best_modularity:.4f}")
        if show_plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            x_min, x_max = min(global_inflations), max(global_inflations)
            ticks = np.arange(round(x_min,2), round(x_max + mid_step,2), mid_step)
            plt.xticks(ticks)
            plt.scatter(global_inflations, global_modularities, c='blue', alpha=0.6, edgecolors='k')
            plt.scatter([best_inflation], [best_modularity], c='red', s=100, edgecolors='k', label="Best")
            plt.axvline(x=best_inflation, linestyle='--', color='lightgrey')
            plt.axhline(y=best_modularity, linestyle='--', color='lightgrey')
            plt.xlabel("Inflation")
            plt.ylabel("Modularity")
            plt.title("Inflation vs. Modularity")
            plt.grid(False)
            plt.tight_layout()
            plt.legend()
            plt.show()
        return best_inflation, best_modularity
    
    # -------------------- Phase 3: Fine Search --------------------
    print("\nPhase 3: Fine search")
    lower_bound = max(min_inflation, max_mid - mid_step)
    upper_bound = min(max_inflation, max_mid + mid_step)
    fine_results = []
    inflation_vals_fine = np.arange(lower_bound, upper_bound + fine_step/2, fine_step)
    for inflation_val in inflation_vals_fine:
        inflation_val = round(inflation_val, 4)
        Q = evaluate_inflation(inflation_val)
        fine_results.append((inflation_val, Q))
        if inflation_val not in global_inflations:
            global_inflations.append(inflation_val)
            global_modularities.append(Q)
        #print(f"Inflation = {inflation_val:.2f}, Modularity = {Q:.4f}")
    
    best_inflation, best_modularity = max(fine_results, key=lambda x: x[1])
    print(f"\nBest inflation (Phase 3): {best_inflation:.2f}, Modularity: {best_modularity:.4f}")
    
    # Plotting (Optional)
    if show_plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        x_min, x_max = min(global_inflations), max(global_inflations)
        ticks = np.arange(round(x_min,2), round(x_max + mid_step,2), mid_step)
        plt.xticks(ticks)
        plt.scatter(global_inflations, global_modularities, c='blue', alpha=0.6, edgecolors='k')
        plt.scatter([best_inflation], [best_modularity], c='red', s=100, edgecolors='k', label="Best")
        plt.axvline(x=best_inflation, linestyle='--', color='lightgrey')
        plt.axhline(y=best_modularity, linestyle='--', color='lightgrey')
        plt.xlabel("Inflation")
        plt.ylabel("Modularity")
        plt.title("Inflation vs. Modularity")
        plt.grid(False)
        plt.tight_layout()
        plt.legend()
        plt.show()
    
    return best_inflation, best_modularity




# classify_modules
def classify_modules(adata, 
                     ggm_key='ggm',
                     ref_anno=None,
                     ref_cluster_method='leiden', 
                     ref_cluster_resolution=1.0, 
                     skew_threshold=2.0,
                     top1pct_threshold=2.0,
                     Moran_I_threshold=0.2,
                     min_dominant_cluster_fraction=0.3,
                     anno_overlap_threshold=0.6):
    """
    Classify spatial specificity modules based on module statistics and annotation data.
    
    This function uses module-level statistics (stored in adata.uns[mod_stats_key]),
    module expression (adata.obsm['module_expression']), and annotation columns (e.g., "M01_anno")
    to determine which modules serve as robust markers for cell identity.
    
    Parameters:
      adata : AnnData object containing spatial transcriptomics data along with:
              - Module statistics in adata.uns[mod_stats_key]
              - Module expression matrix in adata.obsm['module_expression']
              - Module annotation columns in adata.obs (e.g., "M01_anno")
      ggm_key : str, key in adata.uns for the GGM object.
      ref_anno : str, key in adata.obs for reference cluster labels; if provided, this column is used.
      ref_cluster_method : str, clustering method to use if ref_anno is not provided (e.g., 'leiden' or 'louvain').
      ref_cluster_resolution : float, resolution parameter for clustering.
      skew_threshold : float, threshold for skewness to flag modules with ubiquitous expression.
      top1pct_threshold : float, threshold for the top 1% expression ratio to flag modules with ubiquitous expression.
      Moran_I_threshold : float, threshold for positive Moran's I to flag diffuse (weakly spatial) modules.
      min_dominant_cluster_fraction : float, the minimum fraction of a module's annotated cells that must be concentrated 
                                      in one reference cluster to avoid being flagged as mixed-regional.
      anno_overlap_threshold : float, the Jaccard index threshold above which two modules are considered redundant.
       
    The function updates adata.uns['module_filtering'] with a DataFrame containing:
      - module_id: Module identifier.
      - is_identity: Boolean flag indicating whether the module is suitable as a cell identity marker.
      - type_tag: Category tag, one of:
          * "cellular_activity_module"
          * "ubiquitous_module"
          * "diffuse_module"
          * "mixed_regional_module"
          * "redundant_module"
          * "cell_identity_module"
      - information: A brief explanation for exclusion/inclusion.
      
    Also, adata.uns[mod_stats_key] is updated with 'is_identity' and 'type_tag' for each module.
    
    """
    if ggm_key not in adata.uns['ggm_keys']:
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    mod_stats_key = adata.uns['ggm_keys'][ggm_key]['module_stats']
    mod_filtering_key = adata.uns['ggm_keys'][ggm_key]['module_filtering']
    exp_scaled_key = adata.uns['ggm_keys'][ggm_key]['module_expression_scaled']

    # Retrieve module statistics
    module_stats = adata.uns.get(mod_stats_key)
    if module_stats is None:
        raise ValueError("Module statistics not found in adata.uns[mod_stats_key]")
    mod_stats_df = module_stats if isinstance(module_stats, pd.DataFrame) else pd.DataFrame(module_stats)
    
    # Check if GO annotation info exists (i.e. 'top_go_terms' column)
    has_go_info = "top_go_terms" in mod_stats_df.columns

    # Get list of module IDs
    module_ids = mod_stats_df['module_id'].tolist()
    
    # (1) Obtain spatial clustering labels
    if ref_anno is None:
        if ref_cluster_method.lower() == 'leiden':
            ref_anno = 'tmp_leiden_for_filtering'
            sc.pp.neighbors(adata, use_rep=exp_scaled_key,
                            n_pcs=adata.obsm[exp_scaled_key].shape[1])
            sc.tl.leiden(adata, resolution=ref_cluster_resolution, key_added=ref_anno)
        elif ref_cluster_method.lower() == 'louvain':
            ref_anno = 'tmp_louvain_for_filtering'
            sc.pp.neighbors(adata, use_rep=exp_scaled_key,
                            n_pcs=adata.obsm[exp_scaled_key].shape[1])
            sc.tl.louvain(adata, resolution=ref_cluster_resolution, key_added=ref_anno)
        else:
            print(f"Unknown clustering method '{ref_cluster_method}'; skipping cluster assignment")
            ref_anno = None
    else:
        if ref_anno not in adata.obs.columns:
            raise ValueError(f"Cluster label column '{ref_anno}' not found in adata.obs")
        
    # (2) For each module, get the set of annotated cell indices and counts
    module_cells = {}
    module_cell_counts = {}
    for module_id in module_ids:
        col = f"{module_id}_anno"
        if col not in adata.obs.columns:
            module_cells[module_id] = set()
            module_cell_counts[module_id] = 0
        else:
            cells = adata.obs[~adata.obs[col].isna()].index
            module_cells[module_id] = set(cells)
            module_cell_counts[module_id] = len(cells)
    # Sort modules by number of annotated cells (descending)
    modules_sorted = sorted(module_ids, key=lambda m: module_cell_counts.get(m, 0), reverse=True)
    
    # Define GO keywords indicative of general cellular activity
    activity_keywords = [kw.lower() for kw in [
        "proliferation", "cell cycle", "cell division", "DNA replication", 
        "RNA processing", "translation", "metabolic process", "biosynthetic process", 
        "ribosome", "chromosome segregation", "spindle", "mitotic", "cell growth"
    ]]
    
    # Initialize result dictionaries
    results = []  # Each record: {module_id, is_identity, type_tag, reason, information}
    is_identity_dict = {}
    type_tag_dict = {}
    reason_dict = {}
    
    # (3) First pass: apply filters sequentially
    for mod in modules_sorted:
        # Retrieve module stats: skew, top1pct_ratio, and positive Moran's I
        if 'skew' in mod_stats_df.columns:
            skewness = float(mod_stats_df.loc[mod_stats_df['module_id'] == mod, 'skew'].iloc[0])
        else:
            skewness = None
        if 'top1pct_ratio' in mod_stats_df.columns:
            top1pct = float(mod_stats_df.loc[mod_stats_df['module_id'] == mod, 'top1pct_ratio'].iloc[0])
        else:
            top1pct = None
        if 'positive_moran_I' in mod_stats_df.columns:
            moranI = float(mod_stats_df.loc[mod_stats_df['module_id'] == mod, 'positive_moran_I'].iloc[0])
        else:
            moranI = None
        
        # Default: module is considered a cell identity marker
        is_identity = True
        type_tag = 'cell_identity_module'
        reason = "Passed all filters"
        
        # (a) Exclude modules with GO terms indicating general cellular activity.
        is_activity = False
        if has_go_info:
            go_terms_str = mod_stats_df.loc[mod_stats_df['module_id'] == mod, 'top_go_terms']
            if not go_terms_str.empty:
                go_terms_str = go_terms_str.iloc[0]
                if pd.notnull(go_terms_str):
                    terms = [t.strip().lower() for t in go_terms_str.split("||")]
                    for term in terms:
                        if any(kw in term for kw in activity_keywords):
                            go_terms_str = term
                            is_activity = True
                            break
        if is_activity:
            is_identity = False
            type_tag = 'cellular_activity_module'
            reason = f'Excluded: GO enrichment indicates cellular activity ({go_terms_str})'
            is_identity_dict[mod] = is_identity
            type_tag_dict[mod] = type_tag
            reason_dict[mod] = reason
            results.append({
                'module_id': mod,
                'is_identity': is_identity,
                'type_tag': type_tag,
                'information': reason
            })
            continue
        
        # (b) Exclude modules with ubiquitous expression (low skew and low top1pct_ratio)
        if skewness is not None and top1pct is not None:
            if skewness < skew_threshold and top1pct < top1pct_threshold:
                is_identity = False
                type_tag = 'ubiquitous_module'
                reason = f'Excluded: Skew ({skewness:.2f}) and top1pct_ratio ({top1pct:.2f}) below thresholds'
        if not is_identity:
            is_identity_dict[mod] = is_identity
            type_tag_dict[mod] = type_tag
            reason_dict[mod] = reason
            results.append({
                'module_id': mod,
                'is_identity': is_identity,
                'type_tag': type_tag,
                'information': reason
            })
            continue
        
        # (c) Exclude modules with diffuse spatial patterns (weak spatial autocorrelation)
        if moranI is not None:
            if moranI < Moran_I_threshold and moranI > 0:
                is_identity = False
                type_tag = 'diffuse_module'
                reason = f'Excluded: Positive Moran’s I ({moranI:.2f}) below threshold'
        if not is_identity:
            is_identity_dict[mod] = is_identity
            type_tag_dict[mod] = type_tag
            reason_dict[mod] = reason
            results.append({
                'module_id': mod,
                'is_identity': is_identity,
                'type_tag': type_tag,
                'information': reason
            })
            continue
        
        # (d) Exclude modules that are not concentrated in a single spatial domain.
        if ref_anno is not None and module_cell_counts.get(mod, 0) > 0:
            cells = module_cells.get(mod, set())
            clusters = adata.obs.loc[list(cells), ref_anno] if ref_anno in adata.obs.columns else None
            if clusters is not None:
                cluster_counts = clusters.value_counts()
                if len(cluster_counts) > 1:
                    total = cluster_counts.sum()
                    dominant_fraction = cluster_counts.max() / total
                    if dominant_fraction < min_dominant_cluster_fraction:
                        is_identity = False
                        type_tag = 'mixed_regional_module'
                        reason = f'Excluded: Dominant cluster fraction ({dominant_fraction:.2f}) below {min_dominant_cluster_fraction}'
        if not is_identity:
            is_identity_dict[mod] = is_identity
            type_tag_dict[mod] = type_tag
            reason_dict[mod] = reason
            results.append({
                'module_id': mod,
                'is_identity': is_identity,
                'type_tag': type_tag,
                'information': reason
            })
            continue
        
        # If the module passes all filters, mark it as a cell identity module.
        is_identity = True
        type_tag = 'cell_identity_module'
        reason = "Included: Passed all filters"
        is_identity_dict[mod] = is_identity
        type_tag_dict[mod] = type_tag
        reason_dict[mod] = reason
        results.append({
            'module_id': mod,
            'is_identity': is_identity,
            'type_tag': type_tag,
            'information': reason
        })
    
    # (4) Second pass: Identify redundant modules by pairwise comparison.
    # For modules with high annotation overlap (Jaccard index >= anno_overlap_threshold),
    # the module with the lower effect_size is considered less discriminative and is marked as redundant.
    identity_modules = [mod for mod in modules_sorted if is_identity_dict.get(mod, False)]
    identity_modules = sorted(identity_modules)
    marked_as_similar = set()
    for i, modA in enumerate(identity_modules):
        for modB in identity_modules[i+1:]:
            if modA in marked_as_similar or modB in marked_as_similar:
                continue
            cellsA = module_cells.get(modA, set())
            cellsB = module_cells.get(modB, set())
            if not cellsA or not cellsB:
                continue
            inter = len(cellsA & cellsB)
            union = len(cellsA | cellsB)
            if union == 0:
                continue
            jaccard = inter / union
            if jaccard >= anno_overlap_threshold:
                # Use effect_size first to decide which module to keep.
                keep_mod = modA
                drop_mod = modB
                effect_available = False
                try:
                    effA = float(mod_stats_df.loc[mod_stats_df['module_id'] == modA, 'effect_size'].iloc[0])
                    effB = float(mod_stats_df.loc[mod_stats_df['module_id'] == modB, 'effect_size'].iloc[0])
                    if not (np.isnan(effA) or np.isnan(effB)):
                        effect_available = True
                except Exception:
                    effA = effB = None
                if effect_available:
                    # The module with the smaller effect_size (lower discriminative power) is dropped.
                    if effA < effB:
                        keep_mod, drop_mod = modB, modA
                    else:
                        keep_mod, drop_mod = modA, modB
                    info_str = (f'Excluded: Jaccard index {jaccard:.2f} >= {anno_overlap_threshold} and ' 
                                f'lower effect_size ({min(effA, effB):.2f})')
                else:
                    # Fallback: use sum of skew and top1pct_ratio as a score.
                    scoreA = 0
                    scoreB = 0
                    if modA in type_tag_dict and modB in type_tag_dict:
                        scoreA = (float(mod_stats_df.loc[mod_stats_df['module_id'] == modA, 'skew'].iloc[0]) 
                                  if 'skew' in mod_stats_df.columns else 0) + \
                                 (float(mod_stats_df.loc[mod_stats_df['module_id'] == modA, 'top1pct_ratio'].iloc[0]) 
                                  if 'top1pct_ratio' in mod_stats_df.columns else 0)
                        scoreB = (float(mod_stats_df.loc[mod_stats_df['module_id'] == modB, 'skew'].iloc[0]) 
                                  if 'skew' in mod_stats_df.columns else 0) + \
                                 (float(mod_stats_df.loc[mod_stats_df['module_id'] == modB, 'top1pct_ratio'].iloc[0]) 
                                  if 'top1pct_ratio' in mod_stats_df.columns else 0)
                    if scoreB > scoreA:
                        keep_mod, drop_mod = modB, modA
                    info_str = f'Excluded: Jaccard index {jaccard:.2f} >= {anno_overlap_threshold}; fallback score used'
                is_identity_dict[drop_mod] = False
                type_tag_dict[drop_mod] = 'redundant_module'
                reason_dict[drop_mod] = f'Overlap with module {keep_mod}'
                marked_as_similar.add(drop_mod)
                for rec in results:
                    if rec['module_id'] == drop_mod:
                        rec['is_identity'] = False
                        rec['type_tag'] = 'redundant_module'
                        rec['information'] = info_str
                        break
                else:
                    results.append({
                        'module_id': drop_mod,
                        'is_identity': False,
                        'type_tag': 'redundant_module',
                        'information': info_str
                    })
    # (5) Assemble final results DataFrame
    result_df = pd.DataFrame(results)
    if 'module_id' in result_df.columns:
        result_df = result_df.sort_values(by='module_id').reset_index(drop=True)
    adata.uns[mod_filtering_key] = result_df
    
    # Update adata.uns[mod_stats_key] with is_identity and type_tag
    if 'module_id' in mod_stats_df.columns:
        mod_stats_df['is_identity'] = mod_stats_df['module_id'].map(is_identity_dict).fillna(False)
        mod_stats_df['type_tag'] = mod_stats_df['module_id'].map(type_tag_dict).fillna('filtered_module')
        adata.uns[mod_stats_key] = mod_stats_df
    else:
        new_cols = []
        for idx in mod_stats_df.index:
            mod_id = str(idx)
            if idx in is_identity_dict:
                new_cols.append((is_identity_dict[idx], type_tag_dict.get(idx, 'filtered_module')))
            elif mod_id in is_identity_dict:
                new_cols.append((is_identity_dict[mod_id], type_tag_dict.get(mod_id, 'filtered_module')))
            else:
                new_cols.append((False, 'filtered_module'))
        mod_stats_df['is_identity'] = [col[0] for col in new_cols]
        mod_stats_df['type_tag'] = [col[1] for col in new_cols]
        adata.uns[mod_stats_key] = mod_stats_df



# calculate_module_overlap
def calculate_module_overlap(adata, ggm_key='ggm', cross_ggm=False,
                             modules_used=None, 
                             modules_excluded=None,
                             use_smooth=True):
    """
    Compute the overlap ratios between modules based on annotation columns in adata.obs.
    
    The returned DataFrame has the following seven columns:
      - module_a: ID of module A
      - module_b: ID of module B
      - overlap_ratio_a: Overlap count divided by the cell count of module A
      - overlap_ratio_b: Overlap count divided by the cell count of module B
      - overlap_ratio_union: Overlap count divided by the cell count of the union of module A and module B
      - count_a: Number of cells annotated by module A
      - count_b: Number of cells annotated by module B
    
    Pairs with zero overlap are omitted. The final DataFrame is sorted by module_a (ascending)
    and then by overlap_ratio_a (descending).

    Parameters:
      adata (AnnData): AnnData object. The annotation columns should exist in adata.obs with names 
                       "{module_id}_anno" or "{module_id}_anno_smooth".
      ggm_key (str): Key for the GGM object in adata.uns
      cross_ggm (bool): Whether to calculate overlaps across multiple GGMs (default False).
      modules_used (list): List of module IDs to compute overlaps.
      modules_excluded (list): List of module IDs to exclude from overlap calculation.
      use_smooth (bool): Whether to use smoothed annotation columns (default True).

    Returns:
      pd.DataFrame: DataFrame containing the overlap statistics.
    """
    if ggm_key not in adata.uns['ggm_keys']:
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    mod_stats_key = adata.uns['ggm_keys'][ggm_key]['module_stats']

    if cross_ggm and len(adata.uns['ggm_keys']) > 2:
        if modules_used is None:
            raise ValueError("When cross_ggm is True, modules_used must be provided manually.")

    # If modules_used is not provided, get modules from uns['module_stats']
    if modules_used is None:
        modules_used = adata.uns[mod_stats_key]['module_id'].unique()
    # Exclude modules if the modules_excluded list is provided    
    if modules_excluded is not None:
        modules_used = [mid for mid in modules_used if mid not in modules_excluded]

    # Determine which annotation column to use for each module.
    if use_smooth:
        missing_smooth = [mid for mid in modules_used if f"{mid}_anno_smooth" not in adata.obs]
        if missing_smooth:
            print(f"\nThese modules do not have 'smoothed anno': {missing_smooth}. Using 'anno' instead.")
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
    
    # Build a dictionary mapping module IDs to their annotation (0/1) arrays.
    anno_dict = {}
    for mid in modules_used:
        if f"{mid}_anno" in existing_columns:
            anno_col = f"{mid}_anno"
        else:
            anno_col = f"{mid}_anno_smooth"    
        anno_dict[mid] = (adata.obs[anno_col] == mid).astype(int).values
    
    overlap_records = []
    # Iterate over all pairs of modules.
    for mod_a, mod_b in itertools.combinations(modules_used, 2):
        a = anno_dict[mod_a]
        b = anno_dict[mod_b]
        count_a = np.sum(a)
        count_b = np.sum(b)
        overlap = np.sum((a == 1) & (b == 1))
        if overlap == 0:
            continue
        
        ratio_a = overlap / count_a if count_a > 0 else np.nan
        ratio_b = overlap / count_b if count_b > 0 else np.nan
        union_count = count_a + count_b - overlap
        ratio_union = overlap / union_count if union_count > 0 else np.nan
        
        overlap_records.append({
            "module_a": mod_a,
            "module_b": mod_b,
            "overlap_ratio_a": ratio_a,
            "overlap_ratio_b": ratio_b,
            "overlap_ratio_union": ratio_union,
            "count_a": count_a,
            "count_b": count_b
        })
    
    df = pd.DataFrame(overlap_records)
    df.sort_values(by=["module_a", "overlap_ratio_a"], ascending=[True, False], inplace=True)
    df.index = range(df.shape[0])
    return df

