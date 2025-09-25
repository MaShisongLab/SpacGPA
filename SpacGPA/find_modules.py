
import torch
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
import mygene
import subprocess
import tempfile
import sys
import time
import gc
import os

from .enrich_analysis import ensure_gene_symbol_table
# Set the base directory and Ref directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Ref_for_Enrichment")


# run_mcl
import torch
import numpy as np
import pandas as pd
import networkx as nx
import sys
import mygene
import time

def run_mcl(SigEdges, expansion=2, inflation=1.7, add_self_loops='mean', 
            max_iter=1000, tol=1e-6, pruning_threshold=1e-5, 
            run_mode=2,
            min_module_size = 10, topology_filtering = True, 
            convert_to_symbols=False, species=None, species_taxonomy_id=None):
    """
    Perform Markov Clustering (MCL) on a graph constructed from a DataFrame and output
    modules sorted by size with re-numbered module IDs using PyTorch for acceleration.

    Parameters:
        SigEdges: DataFrame containing 'GeneA', 'GeneB', and 'Pcor' columns.
        expansion: Exponent for the expansion step (default is 2).
        inflation: Exponent for the inflation step (usually >1).
        add_self_loops: Method for adding self-loops to the adjacency matrix: 'min', 'mean', 'max', 'dynamic', or 'none'.
        max_iter: Maximum number of iterations.
        tol: Convergence threshold; iteration stops if max change is below this value.
        pruning_threshold: Threshold below which matrix elements are set to zero to speed up convergence.
        run_mode: 0 for cpu, 1 for hybrid, 2 for gpu.
        min_module_size: The minimum number of genes required for a module to be retained.
        topology_filtering: Whether to apply topology filtering for each module.
        convert_to_symbols: Whether to convert gene IDs to gene symbols.
        species: The species for gene ID conversion.default is 'human'.can be 'human','mouse'.

    Returns:
        module_df: DataFrame with two columns: 'Module' (module ID) and 'Gene' (gene name).
    """
    try:
        # Set the device (GPU or CPU)
        if run_mode == 0:
            device = 'cpu'
        else:
            device = 'cuda'

        # 1. Extract all unique genes and create mappings between gene names and indices.
        genes = pd.unique(SigEdges[['GeneA', 'GeneB']].values.ravel())
        gene_index = {gene: i for i, gene in enumerate(genes)}
        index_to_gene = {i: gene for gene, i in gene_index.items()}
        n = len(genes)
        
        # Build a symmetric adjacency matrix using Pcor as weights.
        M = np.zeros((n, n), dtype=np.float32)
        for _, row in SigEdges.iterrows():
            i = gene_index[row['GeneA']]
            j = gene_index[row['GeneB']]
            M[i, j] = row['Pcor']
            M[j, i] = row['Pcor']

        # Convert the matrix to a original networkx graph.
        G_ori = nx.from_numpy_array(M)

        # Add self-loops. 
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
            raise ValueError("Invalid value for 'add_self_loops'. Please choose from 'mean', 'min','max', 'dynamic', or 'none'.")

        # Convert the matrix to a PyTorch tensor and move it to the specified device
        M = torch.tensor(M, device=device)

        # 2. Define a function to normalize columns (each column sums to 1).
        def normalize_columns(matrix):
            col_sums = matrix.sum(dim=0)
            #col_sums[col_sums == 0] = 1  # Prevent division by zero
            return matrix / col_sums

        # Perform initial column normalization
        M = normalize_columns(M)

        # 3. MCL iteration: expansion, inflation, normalization, and pruning.
        for iteration in range(max_iter):
            M_prev = M.clone()
            
            # Expansion: raise the matrix to the power of 'expansion'.
            M = torch.matrix_power(M, expansion)
            
            # Inflation: raise each element of the matrix to the power of 'inflation'.
            M = M.pow(inflation)
            
            # Column normalization
            M = normalize_columns(M)
            
            # Prune: set elements below the threshold to zero.
            M[M < pruning_threshold] = 0
            
            # Re-normalize the matrix
            M = normalize_columns(M)

            
            # Check for convergence
            diff = torch.max(torch.abs(M - M_prev))
            re_e = (M>0).sum().item()
            sys.stdout.write(f"\rIteration: {iteration + 1}, "
                            f"Max change: {diff.item():.8f} ")
            sys.stdout.flush()
            if diff < tol:
                print(f"\nConverged at iteration {iteration + 1}.")        
                break      
        del M_prev, diff, re_e
        
        # 4. Build a graph from the final matrix and extract connected components (modules).
        M_np = M.cpu().numpy()  # Convert the matrix back to NumPy for NetworkX compatibility
        G = nx.from_numpy_array(M_np)
        del M, M_np

        # Extract connected components (modules)
        clusters = list(nx.connected_components(G))
        
        # Remove all size-1 clusters and store them as candidate nodes.
        size_1_clusters = [cluster for cluster in clusters if len(cluster) == 1]
        candidate_nodes = [node for cluster in size_1_clusters for node in cluster]
        clusters = [cluster for cluster in clusters if len(cluster) > 1]

        # Define a function to find the shortest path from a source node to a target node.
        def find_shortest_path_to_connected_component(net, source_node, target_node):
            """
            Find the shortest path from source_node to any of the target_nodes in G_ori.
            """
            try:
                return nx.shortest_path(net, source=source_node, target=target_node)
            except nx.NetworkXNoPath:
                return None
        
        # Find shortest paths for degree-0 nodes to the attractor node of each cluster 
        for cluster in sorted(clusters, key=lambda x: len(x), reverse=True):
            atttractor = max(dict(G.subgraph(cluster).degree()))
            subgraph = G_ori.subgraph(cluster)
            degree_zero_nodes = [node for node, degree in subgraph.degree() if degree == 0]
            if degree_zero_nodes:
                # Split the current cluster into two groups: degree > 0 and degree == 0
                net_temp = G_ori.subgraph(set(candidate_nodes) | cluster)
                # Use the candidate nodes to try and fill the degree-0 nodes
                for node in degree_zero_nodes:
                    # Find the shortest path from the degree-0 node to any of the non-zero-degree nodes
                    path = find_shortest_path_to_connected_component(net_temp, node, atttractor)
                    if path:
                        # Add all nodes in the path to the current cluster and remove them from the candidate list
                        for n in path:
                            if n in candidate_nodes:
                                candidate_nodes.remove(n)
                        cluster.update(path)  # Add path nodes to the current cluster

        # Remove degree-0 nodes from the clusters and add them to the candidate node list
        candidate_nodes = set(candidate_nodes)
        for cluster in clusters:
            subgraph = G_ori.subgraph(cluster)
            for node, deg in subgraph.degree():
                if deg == 0:
                    cluster.remove(node)
                    candidate_nodes.add(node)

        # Create a mapping from node to cluster index for easy lookup
        node_to_cluster = {}
        for idx, cluster in enumerate(clusters):
            for node in cluster:
                node_to_cluster[node] = idx

        # Find most matching cluster for each candidate node.
        for node in candidate_nodes:
            neighbors = set(G_ori.neighbors(node))
            if not neighbors:
                continue
            cluster_count = {}
            for nbr in neighbors:
                if nbr in node_to_cluster:
                    cid = node_to_cluster[nbr]
                    cluster_count[cid] = cluster_count.get(cid, 0) + 1
            total_neighbors_in_clusters = sum(cluster_count.values())
            for cid, count in cluster_count.items():
                if count >= total_neighbors_in_clusters / 2:
                    clusters[cid].add(node)
                    node_to_cluster[node] = cid
                    break
        
        # 5. Sort modules by the number of genes (descending) and re-number them starting from 0.
        sorted_clusters = sorted(clusters, key=lambda comp: len(comp), reverse=True)
        filtered_clusters = [cluster for cluster in sorted_clusters if len(cluster) >= min_module_size]
        num_modules = len(filtered_clusters)
        module_names = [f"M{str(i+1).zfill(len(str(num_modules)))}" for i in range(num_modules)]
        
        # 6. Construct the result: convert node indices to gene names and assign new module IDs.
        rows = []
        for module_id, cluster in enumerate(filtered_clusters):
            # Create a subgraph for the current module.
            subgraph = G_ori.subgraph(cluster)
            # Calculate the degree of each gene in the subgraph (intra-module connections).
            degrees = dict(subgraph.degree())
            # Sort genes by degree in descending order.
            sorted_genes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
            # Assign ranking based on sorted order.
            for rank, node in enumerate(sorted_genes, start=1):
                gene_name = index_to_gene[node]
                degree = degrees[node]
                rows.append([module_names[module_id], gene_name, degree, rank])            

        # Create the final DataFrame.
        module_df = pd.DataFrame(rows, columns=['module_id', 'gene', 'degree', 'rank'])
        
        # Filtering linear and radial topologies if requested.
        if topology_filtering:
            grouped = module_df.groupby('module_id')
            filtered_modules = []
            removed_modules = [] 
            for module_id, group in grouped:
                if (group['degree'] > 2).any() and (group['degree'] > 1).sum() > 1:
                    filtered_modules.append(group)
                else:
                    removed_modules.append(module_id)
            
            print(f"\n{len(removed_modules)} modules were removed due to their linear or radial topology structure out of {len(grouped)} modules.")
            module_df = pd.concat(filtered_modules)
            module_df = module_df.reset_index(drop=True)
            unique_modules = module_df['module_id'].unique()
            module_id_mapping = {module_id: f'M{str(i+1).zfill(len(str(len(unique_modules))))}' for i, module_id in enumerate(unique_modules)}
            module_df['module_id'] = module_df['module_id'].map(module_id_mapping)
            del grouped, filtered_modules, removed_modules, unique_modules, module_id_mapping

        # 7. Convert Ensembl IDs to gene symbols if requested.
        if convert_to_symbols:
            print("\nConverting Ensembl IDs to gene symbols...")
            gene_symbl_file = f"{DATA_DIR}/{species}.gene.symbl.txt.gz"
            ensembl_to_symbol = ensure_gene_symbol_table(species, species_taxonomy_id, gene_symbl_file)
            ensembl_to_symbol = {str(k): str(v) for k, v in ensembl_to_symbol.items()}
            # Add the 'Symbol' column to the DataFrame.
            module_df['symbol'] = module_df['gene'].map(ensembl_to_symbol)
            module_df['symbol'] = module_df['symbol'].fillna(module_df['gene']).astype(str)
        
        module_df['module_id'] = module_df['module_id'].apply(lambda x: "M" + str(int(x[1:])))
        return module_df
    
    finally:
        del rows, sorted_clusters, filtered_clusters, num_modules, module_names, index_to_gene, module_df, G, G_ori
        gc.collect()
        torch.cuda.empty_cache()


# run_louvain 
def run_louvain(SigEdges, 
                resolution=1.0, randomize=None, random_state=None,
                min_module_size=10, topology_filtering=True, 
                convert_to_symbols=False, species=None, species_taxonomy_id=None):
    """
    Perform community detection using the Louvain method on a graph constructed from a DataFrame
    and output modules sorted by size with re-numbered module IDs.

    Parameters:
        SigEdges: DataFrame containing 'GeneA', 'GeneB', and 'Pcor' columns.
        resolution: double, A parameter controlling the 'resolution' in the Louvain algorithm.
                    Higher values lead to more (and smaller) communities.
                    Default is 1.0.
        randomize:boolean, optional
                    Will randomize the node evaluation order and the community evaluation order to get different partitions at each call
        random_state:int, RandomState instance or None, optional (default=None)
                     If int, random_state is the seed used by the random number generator; 
                     If RandomState instance, random_state is the random number generator; 
                     If None, the random number generator is the RandomState instance used by np.random.
        min_module_size: The minimum number of genes required for a module to be retained.
        topology_filtering: Whether to apply topology filtering for each module.
        convert_to_symbols: Whether to convert gene IDs to gene symbols.
        species: The species for gene ID conversion, default is 'human'.

    Returns:
        module_df: DataFrame with columns: 'Module', 'Gene', 'Degree', and 'Rank'.
    """
    try:
        # 1. Extract all unique genes and create mappings between gene names and indices.
        genes = pd.unique(SigEdges[['GeneA', 'GeneB']].values.ravel())
        gene_index = {gene: i for i, gene in enumerate(genes)}
        index_to_gene = {i: gene for gene, i in gene_index.items()}
        n = len(genes)
        
        # 2. Build the symmetric adjacency matrix using 'Pcor' as weights.
        M = np.zeros((n, n))
        for _, row in SigEdges.iterrows():
            i = gene_index[row['GeneA']]
            j = gene_index[row['GeneB']]
            M[i, j] = row['Pcor']
            M[j, i] = row['Pcor']
        
        # 3. Construct a graph from the adjacency matrix.
        G = nx.from_numpy_array(M)
        
        # 4. Apply Louvain community detection with the given resolution.
        partition = community_louvain.best_partition(G, weight='weight', resolution=resolution,
                                                     randomize=randomize, random_state=random_state)
        
        # 5. Convert partition to a list of sets, where each set represents a community
        modules = {}
        for node, module_id in partition.items():
            if module_id not in modules:
                modules[module_id] = set()
            modules[module_id].add(node)

        # Convert format where each set of nodes is represented as a list of sets
        clusters = list(modules.values())

        # 6. Sort modules by their size (descending) and re-number them from 0.
        sorted_clusters = sorted(clusters, key=lambda comp: len(comp), reverse=True)
        filtered_clusters = [cluster for cluster in sorted_clusters if len(cluster) >= min_module_size]
        num_modules = len(filtered_clusters)
        module_names = [f"M{str(i+1).zfill(len(str(num_modules)))}" for i in range(num_modules)]
        
        # 7. Construct the result: create rows for module_id, gene, and ranking.
        rows = []
        for module_id, cluster in enumerate(filtered_clusters):
            # Create a subgraph for the current module.
            subgraph = G.subgraph(cluster)
            # Calculate the degree of each gene in the subgraph (intra-module connections).
            degrees = dict(subgraph.degree())
            # Sort genes by degree in descending order.
            sorted_genes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
            # Assign ranking based on sorted order.
            for rank, node in enumerate(sorted_genes, start=1):
                gene_name = index_to_gene[node]
                degree = degrees[node]
                rows.append([module_names[module_id], gene_name, degree, rank])            

        # Create the final DataFrame.
        module_df = pd.DataFrame(rows, columns=['module_id', 'gene', 'degree', 'rank'])
        
        # Filtering linear and radial topologies if requested.
        if topology_filtering:
            grouped = module_df.groupby('module_id')
            filtered_modules = []
            removed_modules = [] 
            for module_id, group in grouped:
                if (group['degree'] > 2).any() and (group['degree'] > 1).sum() > 1:
                    filtered_modules.append(group)
                else:
                    removed_modules.append(module_id)
            
            print(f"\n{len(removed_modules)} modules were removed due to their linear or radial topology structure out of {len(grouped)} modules.")
            module_df = pd.concat(filtered_modules)
            module_df = module_df.reset_index(drop=True)
            unique_modules = module_df['module_id'].unique()
            module_id_mapping = {module_id: f'M{str(i+1).zfill(len(str(len(unique_modules))))}' for i, module_id in enumerate(unique_modules)}
            module_df['module_id'] = module_df['module_id'].map(module_id_mapping)
            del grouped, filtered_modules, removed_modules, unique_modules, module_id_mapping

        # 8. Convert Ensembl IDs to gene symbols if requested.
        if convert_to_symbols:
            print("\nConverting Ensembl IDs to gene symbols...")
            gene_symbl_file = f"{DATA_DIR}/{species}.gene.symbl.txt.gz"
            ensembl_to_symbol = ensure_gene_symbol_table(species, species_taxonomy_id, gene_symbl_file)
            ensembl_to_symbol = {str(k): str(v) for k, v in ensembl_to_symbol.items()}
            # Add the 'Symbol' column to the DataFrame.
            module_df['symbol'] = module_df['gene'].map(ensembl_to_symbol)
            module_df['symbol'] = module_df['symbol'].fillna(module_df['gene']).astype(str)

        module_df['module_id'] = module_df['module_id'].apply(lambda x: "M" + str(int(x[1:])))
        return module_df
    
    finally:
        del rows, sorted_clusters, filtered_clusters, num_modules, module_names, index_to_gene, module_df, G
        gc.collect()


# run_mcl_original
def run_mcl_original(SigEdges, inflation=1.7, scheme=7, threads=1,
                     min_module_size=10, topology_filtering=False,  
                     convert_to_symbols=False, species=None, species_taxonomy_id=None):
    """
    Perform Markov Clustering (MCL) on a graph constructed from a DataFrame and output
    modules sorted by size with re-numbered module IDs using the original MCL software.

    Parameters:
        SigEdges: DataFrame containing 'GeneA', 'GeneB', and 'Pcor' columns.
        inflation: The MCL inflation parameter.
        scheme: The MCL scheme parameter.
        threads: The number of threads to use for MCL.       
            **see details of the above three parameters in MCL manuals.
        min_module_size: The minimum number of genes required for a module to be retained.
        topology_filtering: Whether to apply topology filtering for each module.
        convert_to_symbols: Whether to convert gene IDs to gene symbols.
        species: The species for gene ID conversion, default is 'human'.

    Returns:
        module_df: DataFrame with two columns: 'Module' (module ID) and 'Gene' (gene name).
    """
    try:
        # 1. Extract all unique genes and create mappings between gene names and indices.
        genes = pd.unique(SigEdges[['GeneA', 'GeneB']].values.ravel())
        gene_index = {gene: i for i, gene in enumerate(genes)}
        index_to_gene = {i: gene for gene, i in gene_index.items()}
        n = len(genes)
        
        # 2. Build the symmetric adjacency matrix using 'Pcor' as weights.
        M = np.zeros((n, n))
        for _, row in SigEdges.iterrows():
            i = gene_index[row['GeneA']]
            j = gene_index[row['GeneB']]
            M[i, j] = row['Pcor']
            M[j, i] = row['Pcor']
        
        # 3. Construct a graph from the adjacency matrix.
        G = nx.from_numpy_array(M)

        # 4. Save the DataFrame to a tab-separated file
        temp_file_prefix = tempfile.mktemp(prefix="mcl_")
        edges_file = f"{temp_file_prefix}_edges.txt"
        clusters_file = f"{temp_file_prefix}_clusters.txt"
        SigEdges = SigEdges[['GeneA', 'GeneB', 'Pcor']]
        SigEdges.to_csv(edges_file, sep='\t', index=False, header=False)

        # 5. run MCL command on loacl
        cmd = f"mcl {edges_file} -I {inflation} -scheme {scheme} -o {clusters_file} -te {threads} --abc"
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error running MCL: {result.stderr}")
                return []
        except subprocess.CalledProcessError as e:
            print(f"Error running MCL: {e.stderr}")
            return []

        clusters_ori = []
        try:
            with open(clusters_file) as f:
                for line in f:
                    cluster = set(line.strip().split())
                    clusters_ori.append(cluster)
        except FileNotFoundError:
            print("Error: mcl_clusters file not found.")
            return []

        # 6. Maps genes in each cluster to their corresponding index based on the gene_index mapping.
        clusters = []
        for cluster in clusters_ori:
            # Convert each gene in the cluster to its corresponding index using the gene_index dictionary
            mapped_cluster = {gene_index[gene] for gene in cluster if gene in gene_index}
            clusters.append(mapped_cluster)
        del clusters_ori, mapped_cluster

        # 7. Sort modules by their size (descending) and re-number them from 0.
        sorted_clusters = sorted(clusters, key=lambda comp: len(comp), reverse=True)
        filtered_clusters = [cluster for cluster in sorted_clusters if len(cluster) >= min_module_size]
        num_modules = len(filtered_clusters)
        module_names = [f"M{str(i+1).zfill(len(str(num_modules)))}" for i in range(num_modules)]

        # 8. Construct the result: create rows for module_id, gene, and ranking.
        rows = []
        for module_id, cluster in enumerate(filtered_clusters):
            # Create a subgraph for the current module.
            subgraph = G.subgraph(cluster)
            # Calculate the degree of each gene in the subgraph (intra-module connections).
            degrees = dict(subgraph.degree())
            # Sort genes by degree in descending order.
            sorted_genes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
            # Assign ranking based on sorted order.
            for rank, node in enumerate(sorted_genes, start=1):
                gene_name = index_to_gene[node]
                degree = degrees[node]
                rows.append([module_names[module_id], gene_name, degree, rank])            

        # Create the final DataFrame.
        module_df = pd.DataFrame(rows, columns=['module_id', 'gene', 'degree', 'rank'])
        
        # Filtering linear and radial topologies if requested.
        if topology_filtering:
            grouped = module_df.groupby('module_id')
            filtered_modules = []
            removed_modules = [] 
            for module_id, group in grouped:
                if (group['degree'] > 2).any() and (group['degree'] > 1).sum() > 1:
                    filtered_modules.append(group)
                else:
                    removed_modules.append(module_id)
            
            print(f"\n{len(removed_modules)} modules were removed due to their linear or radial topology structure out of {len(grouped)} modules.")
            module_df = pd.concat(filtered_modules)
            module_df = module_df.reset_index(drop=True)
            unique_modules = module_df['module_id'].unique()
            module_id_mapping = {module_id: f'M{str(i+1).zfill(len(str(len(unique_modules))))}' for i, module_id in enumerate(unique_modules)}
            module_df['module_id'] = module_df['module_id'].map(module_id_mapping)
            del grouped, filtered_modules, removed_modules, unique_modules, module_id_mapping

        # 9. Convert Ensembl IDs to gene symbols if requested.
        if convert_to_symbols:
            print("\nConverting Ensembl IDs to gene symbols...")
            gene_symbl_file = f"{DATA_DIR}/{species}.gene.symbl.txt.gz"
            ensembl_to_symbol = ensure_gene_symbol_table(species, species_taxonomy_id, gene_symbl_file)
            ensembl_to_symbol = {str(k): str(v) for k, v in ensembl_to_symbol.items()}
            # Add the 'Symbol' column to the DataFrame.
            module_df['symbol'] = module_df['gene'].map(ensembl_to_symbol)
            module_df['symbol'] = module_df['symbol'].fillna(module_df['gene']).astype(str)

        module_df['module_id'] = module_df['module_id'].apply(lambda x: "M" + str(int(x[1:])))
        return module_df
    
    finally:
        if os.path.exists(edges_file):
            os.remove(edges_file)
        if os.path.exists(clusters_file):
            os.remove(clusters_file)
        del rows, sorted_clusters, filtered_clusters, num_modules, module_names, index_to_gene, module_df, G
        gc.collect()
   