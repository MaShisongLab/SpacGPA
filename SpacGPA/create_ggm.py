import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
import time
import gc

from .calculate_pcors import calculate_pcors_pytorch
from .calculate_pcors import set_selects, estimate_rounds
from .find_modules import run_mcl, run_louvain, run_mcl_original
from .module_show import get_module_edges as get_edges
from .module_show import get_module_anno as get_anno
from .enrich_analysis import go_enrichment_analysis as run_go
from .enrich_analysis import mp_enrichment_analysis as run_mp
from .par_optimization import find_best_inflation
class FDRResults:

    def __init__(self):
        """
        Class to store FDR results.
        """   

        self.permutation_fraction = None 
        self.permutation_matrix = None
        self.permutation_gene_name = None
        self.permutation_coexpressed_cell_num = None
        self.permutation_pcor_all = None
        self.permutation_pcor_sampling_num = None
        self.permutation_rho_all = None
        self.permutation_SigEdges = None
        self.FDR_threshold = None
        self.summary = None


class create_ggm:   
    def __init__(self, x, round_num=None, selected_num=None, target_sampling_time=100, 
                 gene_name=None, sample_name=None, project_name='na',
                 cut_off_pcor=0.03, cut_off_coex_cell=10,  
                 use_chunking=True, chunk_size=5000, 
                 seed=98, run_mode=2, double_precision=False, stop_threshold=0,
                 FDR_control=False, FDR_threshold=0.01, auto_adjust=False,
                 auto_find_modules=False):
        """
        Instruction:  
        Class to create a ggm object.      
        Please Normalize The Expression Matrix Before Running; 

        Parameters: 
        x : an expression matrix or AnnData object; spot(or cell) in row, gene in column.
        round_num : Manually set the total number of iterations. The default as None, estimated by the gene number,
                     selected number and target sampling count.
        selected_num : Manually set the number of genes selected in each iteration to calculate the partial correlation 
                       coefficient. The default is None and it is recommended to be set between 500 and 2000 based on the
                       number of input genes.
                       (See the recommended setting in the annotation of the set_selects function)
        target_sampling_time : The total expected number of times each gene pair is collected in all iterations。
                                The default is 100. The larger the setting, the greater the total number of iterations.
        gene_name : an array of gene names. only used when x is a matrix.
        sample_name : an array of spot(or cell) ids. only used when x is a matrix.
        project_name : optional, default as 'na'. set a name for the ggm object.
        cut_off_pcor : optional, default as 0.03. 
        cut_off_coex_cell : optional, default as 10.
        use_chunking : optional, default as True. Whether to enable chunk-based computations.
        chunk_size : optional, default as 5000. Controls memory usage for large datasets.   
        seed : optional, default as 98.
        run_mode : optional, default as 2.
                   0 - All computations on CPU.
                   1 - Preprocessing on CPU, partial correlation calculations on GPU.
                   2 - All computations on GPU.
        double_precision : optional, default as False. Whether to use double precision for calculations.
        stop_threshold: threshold for stopping the loop if the sum of `valid_elements_count` over 100 iterations is less than this value.
        FDR_control: optional, default as False. Whether to perform FDR control automatically.
        FDR_threshold: optional, default as 0.01. The FDR threshold for filtering significant edges. only used when FDR_control is True.
        auto_adjust: optional, default as False. Whether to adjust the cutoff based on FDR results. only used when FDR_control is True.
        auto_find_modules: optional, default as False. Whether to automatically find modules based on the significant edges.
                           Note: This parameter is only supported with methods='mcl-hub'.
        """

        self.matrix = None
        self.round_num = round_num
        self.selected_num = selected_num
        self.target_sampling_time = target_sampling_time
        self.gene_num = None
        self.gene_name = gene_name
        self.samples_num = None
        self.sample_name = sample_name
        self.project_name = project_name

        self.cut_off_pcor = cut_off_pcor
        self.cut_off_coex_cell = cut_off_coex_cell
        self.seed_used = seed
        self.run_mode = run_mode 
        self.double_precision = double_precision 
        self.use_chunking = use_chunking
        self.chunk_size = chunk_size
        self.stop_threshold = stop_threshold
        self.FDR_control = FDR_control
        self.FDR_threshold = FDR_threshold
        self.auto_adjust = auto_adjust
        self.auto_find_modules = auto_find_modules

        self.coexpressed_cell_num = None
        self.pcor_all = None
        self.pcor_sampling_num = None
        self.rho_all = None
        self.SigEdges = None
        self.fdr = None
        self.modules = None
        self.modules_summary = None
        self.go_enrichment = None
        self.mp_enrichment = None

        # Validate and extract input
        x, gene_name, sample_name = self._validate_and_extract_input(x, gene_name, sample_name)
        self.matrix = sp.csr_matrix(x) 
        self.gene_name = gene_name
        self.sample_name = sample_name 
        
        self._initialize_variables(x)

        selected_num_default = set_selects(self.gene_num)
        if selected_num is None:
            selected_num = selected_num_default
            self.selected_num = selected_num_default
        
        round_num_default = estimate_rounds(self.gene_num, selected_num, target_sampling_time)
        if round_num is None:
            round_num = round_num_default
            self.round_num = round_num_default 

        if run_mode == 0:
            print("\nRunning all calculations on CPU.")
        elif run_mode == 1:
            print("\nRunning preprocessing on CPU and computing on GPU (if available).")
        elif run_mode == 2:
            print("\nRunning all calculations on GPU (if available).")
        else:
            raise ValueError("Invalid run_mode. Use 0 for CPU, 1 for hybrid, or 2 for GPU.")

        # Determine dtype based on parameters
        if double_precision:
            print("Using double precision (float64) for all calculations.")
        else:
            print("Using single precision (float32) for all calculations.")

        # Adjust chunk_size if not using chunking
        if  use_chunking:
            print(f"Using chunk size of {chunk_size} for efficient computation")  

        # Calculate partial correlations
        self.round_num, self.coexpressed_cell_num, self.pcor_all, self.pcor_sampling_num, self.rho_all, self.SigEdges = calculate_pcors_pytorch(
            x, 
            round_num=round_num, 
            selected_num=selected_num,
            gene_name=self.gene_name, 
            project_name=self.project_name, 
            cut_off_pcor=cut_off_pcor, 
            cut_off_coex_cell=cut_off_coex_cell, 
            seed=seed,
            run_mode=run_mode, 
            double_precision=double_precision, 
            use_chunking=use_chunking,
            chunk_size=chunk_size,
            stop_threshold=stop_threshold)
        
        print("Found", self.SigEdges.shape[0], "significant co-expressed gene pairs with partial correlation >=", cut_off_pcor)
        
        if FDR_control:
            self.fdr_control(permutation_fraction=1.0, FDR_threshold=self.FDR_threshold)
            fdr_summary = self.fdr.summary
            fdr_filtered = fdr_summary.loc[fdr_summary['FDR'] <= self.FDR_threshold]
            if fdr_filtered.shape[0] > 0:
                min_pcor_row = fdr_filtered.iloc[0]
                min_pcor = min_pcor_row['Pcor']
                if auto_adjust:
                    self.adjust_cutoff(pcor_threshold=round(min_pcor, 3),
                                        coex_cell_threshold=self.cut_off_coex_cell) 
                    
        if auto_find_modules:
            best_inflation, _ = find_best_inflation(self, 
                                                    min_inflation=1.1, max_inflation=10,
                                                    coarse_step=0.1, mid_step=0.05, fine_step=0.01,
                                                    expansion=2, add_self_loops='mean', max_iter=1000,
                                                    tol=1e-6, pruning_threshold=1e-5, run_mode=self.run_mode,
                                                    phase=3, show_plot=False)
            self.find_modules(methods='mcl-hub', 
                              expansion=2, inflation=best_inflation, add_self_loops='mean', 
                              max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                              min_module_size=10, topology_filtering=True, convert_to_symbols=False)         
        torch.cuda.empty_cache() 
        print("\nTask completed. Resources released.")
    

    def __repr__(self):
        # Create a string representation of the object
        s = f"View of ggm object: {self.project_name}\n"
        s += "MetaInfo:\n"
        s += f"  Gene Number: {self.gene_num}\n"
        s += f"  Sample Number: {self.samples_num}\n"
        s += f"  Pcor Threshold: {self.cut_off_pcor}\n"
        s += "\nResults:\n"
        if self.SigEdges is not None:
            s += f"  SigEdges: DataFrame with {self.SigEdges.shape[0]} significant gene pairs\n"
        else:
            s += "  SigEdges: None\n"
        if self.modules is not None:
            unique_mods = self.modules['module_id'].unique() if 'module_id' in self.modules.columns else None
            s += f"  modules: {len(unique_mods)} modules with {self.modules.shape[0]} genes\n"
        else:
            s += "  modules: None\n"
        if self.modules_summary is not None:
            s += f"  modules_summary: DataFrame with {self.modules_summary.shape[0]} rows\n"
        else:
            s += "  modules_summary: None\n"
        if self.fdr is not None:
            s += "  FDR: Exists\n"
        else:
            s += "  FDR: None\n"
        return s
    
    def _validate_and_extract_input(self, x, gene_name, sample_name):
        print("Please Normalize The Expression Matrix Before Running!")
        print("Loading Data.")
        if isinstance(x, AnnData):
            if x.X is None or x.var_names is None:
                raise ValueError("AnnData object must have X and var_names.")
            if not np.issubdtype(x.X.dtype, np.number):
                raise ValueError("Expression data must be numeric.")
            x_matrix = x.X
            if sp.issparse(x_matrix):
                x_matrix = x_matrix.tocsr()
            elif isinstance(x_matrix, np.matrix):
                print("Converting np.matrix to csr_matrix...")
                x_matrix = sp.csr_matrix(np.asarray(x_matrix))
            elif isinstance(x_matrix, np.ndarray):
                print("Converting np.ndarray to csr_matrix...")
                x_matrix = sp.csr_matrix(x_matrix)
            gene_name = np.array(x.var_names)
            sample_name = np.array(x.obs_names)

        elif isinstance(x, np.matrix):
            x_matrix = sp.csr_matrix(np.asarray(x))
        elif isinstance(x, np.ndarray): 
            x_matrix = sp.csr_matrix(x)
        elif sp.issparse(x):
            x_matrix = x.tocsr()    
        else:
            raise ValueError("x must be a 2D cell x gene matrix(accepted formats include scipy sparse CSR matrix, numpy ndarray) or an AnnData object.")
        
        if x_matrix.ndim != 2:
            raise ValueError("x must be a 2D cell x gene matrix.(accepted formats include scipy sparse CSR matrix, numpy ndarray)")
        
        if gene_name is None or len(gene_name) != x_matrix.shape[1]:
            raise ValueError("Length of gene_name must match the number of columns in x.")
        
        if sample_name is None or len(sample_name) != x_matrix.shape[0]:
            raise ValueError("Length of sample_name must match the number of rows in x.")
        
        return x_matrix, gene_name, sample_name

    def _initialize_variables(self, x):
        self.samples_num = x.shape[0]
        self.gene_num = x.shape[1]
    
    
    def fdr_control(self, permutation_fraction=1.0, FDR_threshold=0.01):
        """
        Perform FDR control by permuting gene columns and calculating the necessary statistics.

        Parameters:
        - permutation_fraction: Fraction of genes to permute.
        - FDR_threshold: The FDR threshold that determines the significance of the Pcors Threshold.

        Returns:
        - fdr: The FDR results object.
        """
        self.FDR_threshold = FDR_threshold
        run_mode = self.run_mode
        if run_mode == 0:
            device = 'cpu'
        else:   
            device = 'cuda' if run_mode != 0 else 'cpu'
        
        print("\nPerforming FDR control...")
        
        # Create result storage
        print("Randomly redistribute the expression distribution of input genes...")
        fdr = FDRResults()
        fdr.permutation_fraction = permutation_fraction
        fdr.FDR_threshold = FDR_threshold
        new_gene_name = self.gene_name.copy()

        # Randomly select columns to permute
        gene_num = self.gene_num
        samples_num = self.samples_num
        permutation_num = round(gene_num * permutation_fraction)
        
        np.random.seed()
        perm_cols = np.random.choice(gene_num, size=permutation_num, replace=False)
        perm_cols_set = set(perm_cols)

        # Convert to CSC format for modification
        permutation_x = self.matrix.tocsc()
        indices = permutation_x.indices
        data = permutation_x.data
        indptr = permutation_x.indptr

        # Process each column
        for c in range(gene_num):
            c_start = indptr[c]
            c_end = indptr[c + 1]
            row_idx = indices[c_start:c_end]
            col_vals = data[c_start:c_end]

            # If the column needs to be permuted
            if c in perm_cols_set:
                # Permute the column and update the gene name
                new_gene_name[c] = 'P_' + new_gene_name[c]
                permutation_x.data[c_start:c_end] = np.random.choice(col_vals, size=len(row_idx), replace=False)
                permutation_x.indices[c_start:c_end] = np.random.choice(samples_num, size=len(row_idx), replace=False)
                permutation_x.indices[c_start:c_end] = np.sort(permutation_x.indices[c_start:c_end])

        # Convert back to CSR format
        permutation_x_csr = permutation_x.tocsr()

        # Store results
        fdr.permutation_matrix = permutation_x_csr
        fdr.permutation_gene_name = new_gene_name

        # Calculate partial correlations and other statistics
        print("\nCalculate correlation between genes after redistribution...")
        round_num = self.round_num
        cut_off_pcor = self.cut_off_pcor
        cut_off_coex_cell = self.cut_off_coex_cell
        selected_num = self.selected_num

        seed = int(time.time() * 1000) % 4294967296
        
        _, fdr.permutation_coexpressed_cell_num, fdr.permutation_pcor_all, fdr.permutation_pcor_sampling_num, fdr.permutation_rho_all, fdr.permutation_SigEdges = calculate_pcors_pytorch(
            permutation_x_csr, 
            round_num=round_num, 
            selected_num=selected_num, 
            gene_name=new_gene_name, 
            project_name=self.project_name, 
            cut_off_pcor=cut_off_pcor, 
            cut_off_coex_cell=cut_off_coex_cell, 
            seed=seed, 
            run_mode=self.run_mode,
            double_precision=self.double_precision,
            use_chunking=self.use_chunking,
            chunk_size=self.chunk_size,
            stop_threshold=0
        )

        del permutation_x_csr, new_gene_name
        gc.collect()

        # Calculate the statistics (num_ori_sig, num_permutated_sig, pct1, pct2)
        fdr_stat = torch.zeros((91, 5), dtype=torch.float32, device=device)
        idx_np = torch.ones((self.gene_num, self.gene_num), dtype=torch.bool, device=device)
        idx_p = torch.ones_like(idx_np, dtype=torch.bool, device=device)
        
        # Precompute permutation matrix mask
        idx_np[:, perm_cols] = False
        idx_np[perm_cols, :] = False
        idx_p = ~idx_np
        idx_p = idx_p.cpu().numpy()
        del idx_np
        gc.collect()
        torch.cuda.empty_cache()

        # Precompute number of permutations and the remaining edges
        num_np = (self.gene_num - permutation_num) * (self.gene_num - permutation_num - 1) / 2
        num_permutated = self.gene_num * (self.gene_num - 1) / 2 - num_np

        # Convert pcor_all and coexpressed_cell_num to tensors once before the loop
        print("\nSummarizing the FDR Statistics...")
        pcor_all_tensor = torch.tensor(self.pcor_all, device=device, dtype=torch.float32)
        coexpressed_cell_num_tensor = torch.tensor(self.coexpressed_cell_num, device=device, dtype=torch.int32)

        # First loop: Calculate num_ori_sig and store
        for i in range(10, 101):
            # Calculate idx_ori for original condition
            idx_ori = (pcor_all_tensor >= i / 1000) & (coexpressed_cell_num_tensor >= self.cut_off_coex_cell)
            idx_ori = idx_ori.cpu().numpy() 
            # Calculate num_ori_sig
            num_ori_sig = np.sum(idx_ori).item()            
            # Store the result in fdr_stat for later use
            fdr_stat[i - 10, 0] = i / 1000
            fdr_stat[i - 10, 1] = num_ori_sig

            del idx_ori, num_ori_sig
            gc.collect()
            torch.cuda.empty_cache()

        del pcor_all_tensor, coexpressed_cell_num_tensor
        gc.collect()
        torch.cuda.empty_cache()

        # Convert permutation_pcor_all and permutation_coexpressed_cell_num to tensors before the second loop
        perm_pcor_all_tensor = torch.tensor(fdr.permutation_pcor_all, device=device, dtype=torch.float32)
        perm_coexpressed_cell_num_tensor = torch.tensor(fdr.permutation_coexpressed_cell_num, device=device, dtype=torch.int32)

        # Second loop: Calculate num_permutated_sig and store
        for i in range(10, 101):
            # Calculate idx_permutated for permuted condition
            idx_permutated = (perm_pcor_all_tensor >= i / 1000) & (perm_coexpressed_cell_num_tensor >= self.cut_off_coex_cell)
            idx_permutated = idx_permutated.cpu().numpy()
            # Calculate num_permutated_sig
            num_permutated_sig = np.sum(idx_permutated & idx_p).item()
            # Store the result in fdr_stat for later use
            fdr_stat[i - 10, 3] = num_permutated_sig

            del idx_permutated, num_permutated_sig
            gc.collect()
            torch.cuda.empty_cache()

        del perm_pcor_all_tensor, perm_coexpressed_cell_num_tensor
        gc.collect()
        torch.cuda.empty_cache()

        # Third loop: Calculate pct1 and pct2
        for i in range(10, 101):
            # Calculate pct1 and pct2 using the stored values
            num_ori_sig = fdr_stat[i - 10, 1]
            num_permutated_sig = fdr_stat[i - 10, 3]
            pct1 = num_permutated_sig / num_permutated
            pct2 = self.gene_num * (self.gene_num - 1) / 2 * pct1 / num_ori_sig
            if pct2 > 1:
                pct2 = 1

            # Store pct2 and pct1 in fdr_stat
            fdr_stat[i - 10, 2] = pct2
            fdr_stat[i - 10, 4] = pct1

        # Convert to DataFrame for easier inspection
        fdr_summary = pd.DataFrame(fdr_stat.cpu().numpy(), columns=['Pcor', 'SigEdgeNum', 'FDR', 'SigPermutatedEdgeNum', 'SigPermutatedEdgeProportion'])
        fdr_summary['SigEdgeNum'] = fdr_summary['SigEdgeNum'].astype(int)
        fdr_summary['SigPermutatedEdgeNum'] = fdr_summary['SigPermutatedEdgeNum'].astype(int)
        fdr.summary = fdr_summary
        self.fdr = fdr
        print("FDR control completed.")
        print(f"Current Pcor threshold: {self.cut_off_pcor:.3f}")
        fdr_filtered = fdr_summary.loc[fdr_summary['FDR'] <= self.FDR_threshold]
        if fdr_filtered.shape[0] == 0:
            print(f"No Pcors threshold found with FDR <= {self.FDR_threshold}.")
        else:
            min_pcor_row = fdr_filtered.iloc[0]
            min_pcor = min_pcor_row['Pcor']
            print(f"Minimum Pcor threshold with FDR <= {FDR_threshold}: {min_pcor:.3f}")


    def adjust_cutoff(self, pcor_threshold=0.03, coex_cell_threshold=10):
        """
        Adjust the Pcor and coexpressed cell number thresholds based on FDR results.

        Parameters:
            pcor_threshold: The new Pcor threshold.
            oex_cell_threshold: The new coexpressed cell number threshold.

        """   
        print("Adjusting Threshold of Pcor and coexpressed cell number...")
        cut_off_pcor = pcor_threshold
        cut_off_coex_cell = coex_cell_threshold        
        cellnum = np.diagonal(self.coexpressed_cell_num)
        
        old_pcor = self.cut_off_pcor
        old_coex_cell = self.cut_off_coex_cell
        self.cut_off_pcor = cut_off_pcor
        self.cut_off_coex_cell = cut_off_coex_cell

        if old_pcor == cut_off_pcor and old_coex_cell == cut_off_coex_cell:
            print("No changes made for cutoff values.")
        if old_pcor != cut_off_pcor:
            print(f"Ajusted Pcor Threshold: {old_pcor} -> {cut_off_pcor}")    
        if old_coex_cell != cut_off_coex_cell:
            print(f"Ajusted Coexpressed Cell Number Threshold: {old_coex_cell} -> {cut_off_coex_cell}")  

        idx = np.where((self.pcor_all >= cut_off_pcor)
                    & (self.pcor_all < 1)
                    & (self.coexpressed_cell_num >= cut_off_coex_cell))
        e1 = list(self.gene_name[idx[0]])
        e1n = list(cellnum[idx[0]].astype(int))
        e2 = list(self.gene_name[idx[1]])
        e2n = list(cellnum[idx[1]].astype(int))
        e3 = list(self.pcor_all[idx])
        e4 = list(self.pcor_sampling_num[idx].astype(int))
        e5 = list(self.rho_all[idx])
        e6 = list(self.coexpressed_cell_num[idx].astype(int))
        e7 = [self.project_name] * len(e1)
        self.SigEdges = pd.DataFrame({'GeneA': e1, 'GeneB': e2, 'Pcor': e3, 'SamplingTime': e4,
                                        'r': e5, 'Cell_num_A': e1n, 'Cell_num_B': e2n,
                                        'Cell_num_coexpressed': e6, 'Project': e7})
        print("Found", self.SigEdges.shape[0], "significant co-expressed gene pairs with partial correlation >=", cut_off_pcor)


    def find_modules(self, methods='mcl-hub', 
                    expansion=2, inflation=1.7, add_self_loops='mean', 
                    max_iter=1000, tol=1e-6, pruning_threshold=1e-5,
                    resolution=1.0, randomize=None, random_state=None,
                    scheme=7, threads=1,
                    min_module_size=10, topology_filtering=True,
                    convert_to_symbols=False, species=None, species_taxonomy_id=None):
        """
        Find modules using the specified method.

        Parameters:
        - methods: The method to use for module detection. Options are 'mcl-hub', 'louvain' or 'mcl'.
        - mcl-hub parameters:
            - expansion: The mcl expansion parameter.
            - inflation: The mcl inflation parameter.
            - add_self_loops: Method for adding self-loops to the adjacency matrix: 'min', 'mean', 'max', 'dynamic', or 'none'.            
            - max_iter: The maximum number of iterations for mcl.
            - tol: The convergence threshold for mcl.
            - pruning_threshold: The mcl pruning threshold.
        - louvain parameters:
            - resolution: The resolution parameter for Louvain.
            - randomize: Will randomize the node evaluation order and the community evaluation order 
                         to get different partitions at each call
            - random_state:int, RandomState instance or None, optional (default=None)
                           If int, random_state is the seed used by the random number generator; 
                           If RandomState instance, random_state is the random number generator; 
                           If None, the random number generator is the RandomState instance used by np.random.
            (see more details from community.best_partition function in python-louvain package)
        - mcl parameters:
            - inflation: The mcl inflation parameter.
            - scheme: The mcl scheme parameter.
            - threads: The number of threads to use for mcl.
        - min_module_size: The minimum number of genes required for a module to be retained.
        - topology_filtering: Whether to apply topology filtering for each module.
        - convert_to_symbols: Whether to convert gene IDs to gene symbols.
        - species: The species for gene ID conversion.

        """
        if methods == 'mcl-hub':
            print("\nFind modules using MCL-Hub...")
            print(f"Current Pcor: {self.cut_off_pcor}")
            print(f"Total significantly co-expressed gene pairs: {len(self.SigEdges)}")
            inflation = inflation
            expansion = expansion
            max_iter = max_iter
            tol = tol
            pruning_threshold = pruning_threshold
            module_df = run_mcl(self.SigEdges, run_mode=self.run_mode,
                                inflation=inflation, expansion=expansion, add_self_loops=add_self_loops,
                                max_iter=max_iter, tol=tol, pruning_threshold=pruning_threshold,
                                min_module_size=min_module_size, topology_filtering=topology_filtering,
                                convert_to_symbols=convert_to_symbols, species=species, species_taxonomy_id=species_taxonomy_id)
            self.modules = module_df.copy()
        elif methods == 'louvain':
            print("\nFind modules using Louvain...")
            print(f"Current Pcor: {self.cut_off_pcor}")
            print(f"Total significantly co-expressed gene pairs: {len(self.SigEdges)}")
            resolution = resolution
            module_df = run_louvain(self.SigEdges, 
                                    resolution=resolution, random_state=random_state, randomize=randomize,
                                    min_module_size=min_module_size, topology_filtering=topology_filtering,
                                    convert_to_symbols=convert_to_symbols, species=species, species_taxonomy_id=species_taxonomy_id)
            self.modules = module_df.copy()                          
        elif methods == 'mcl':
            print("\nFind modules using MCL ...")
            print(f"Current Pcor: {self.cut_off_pcor}")
            print(f"Total significantly co-expressed gene pairs: {len(self.SigEdges)}")
            module_df = run_mcl_original(self.SigEdges,
                                        inflation=inflation, scheme=scheme, threads=threads,
                                        min_module_size=min_module_size, topology_filtering=topology_filtering,
                                        convert_to_symbols=convert_to_symbols, species=species, species_taxonomy_id=species_taxonomy_id)
            self.modules = module_df.copy()   
        else:
            raise ValueError("Invalid method. Use 'mcl-hub', 'louvain', or 'mcl'.")
        
        if convert_to_symbols:
            grouped = module_df.groupby('module_id')
            self.modules_summary = grouped.agg(
                size =('gene','size'),
                num_genes_degree_ge_2=('degree', lambda x: (x >= 2).sum()),
                all_symbols=('symbol',', '.join),
                all_genes=('gene', ', '.join)
            ).reset_index()
        else:
            grouped = module_df.groupby('module_id')
            self.modules_summary = grouped.agg(
                size =('gene','size'),
                num_genes_degree_ge_2=('degree', lambda x: (x >= 2).sum()),
                all_genes=('gene', ', '.join)
            ).reset_index()

        self.modules_summary['module_num'] = self.modules_summary['module_id'].apply(lambda x: int(x.lstrip('M')))
        self.modules_summary = self.modules_summary.sort_values(by='module_num').reset_index(drop=True)
        self.modules_summary.drop(columns='module_num', inplace=True)   
    

    def get_module_edges(self, module_id):
        module_edges = get_edges(self, module_id)
        return module_edges


    def get_module_anno(self, module_id, add_enrich_info=True, top_n=None, term_id=None):
        module_anno = get_anno(self, module_id, add_enrich_info, top_n, term_id)
        return module_anno


    def go_enrichment_analysis(self,
               species="mouse",
               padjust_method="BH",
               pvalue_cutoff=0.05
                ):
        run_go(self, species, padjust_method, pvalue_cutoff)
    

    def mp_enrichment_analysis(self,
               species="mouse",
               padjust_method="BH",
               pvalue_cutoff=0.05
                ):
        run_mp(self, species, padjust_method, pvalue_cutoff)
    
    