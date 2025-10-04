import torch
import numpy as np
import pandas as pd
import time
import math
import sys
import gc


def csr_to_torch_sparse_tensor(csr_mat, device, dtype=torch.float32):
    """
    Convert a scipy sparse CSR matrix to a PyTorch sparse tensor.

    Parameters:
        csr_mat: scipy.sparse.csr_matrix
            Input sparse matrix.
        device: torch.device
            Target device for the sparse tensor.
        dtype: torch.dtype
            Desired precision for the sparse tensor values.

    Returns:
        torch.sparse.Tensor
            PyTorch sparse tensor representation of the input matrix.
    """
    try:
        # Convert CSR matrix to COO format
        coo = csr_mat.tocoo()
        
        # Create the indices and values tensors
        indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=device)
        values = torch.tensor(coo.data, dtype=dtype, device=device)
        shape = coo.shape
        del coo

        # Return the sparse tensor
        sparse_tensor = torch.sparse_coo_tensor(indices, values, shape, device=device)
        del indices, values, shape
        return sparse_tensor
    
    finally:
        # Ensure that no tensors are left on GPU memory
        del sparse_tensor
        torch.cuda.empty_cache()
        gc.collect()


def calculate_coex_in_chunks(x, chunk_size, device, dtype=torch.float32):
    """
    Compute the co-expression matrix (coex) of a large matrix x in chunks.

    Parameters:
        x: scipy.sparse.csr_matrix
            Input sparse matrix (cells x genes).
        chunk_size: int
            Number of rows to process in each chunk.
        device: torch.device
            Device to use for computation (CPU or GPU).
        dtype: torch.dtype
            Desired precision for computation.

    Returns:
        coex: torch.Tensor
            Co-expression matrix (genes x genes), stored as int32 to save memory.
        cellnum: torch.Tensor
            Number of cells expressing each gene.
    """
    nrow, ncol = x.shape
    coex = torch.zeros((ncol, ncol), dtype=torch.int32, device=device)  # Store co-expression counts
    cellnum = torch.zeros(ncol, dtype=torch.int32, device=device)  # Store the number of cells expressing each gene

    try:
        for start_row in range(0, nrow, chunk_size):
            end_row = min(start_row + chunk_size, nrow)

            # Convert chunk to PyTorch sparse tensor
            chunk_csr = x[start_row:end_row]
            chunk_sparse = csr_to_torch_sparse_tensor(chunk_csr, device, dtype=dtype)

            # Convert sparse tensor to dense binary matrix
            chunk_dense = (chunk_sparse.to_dense() > 0).int()

            # Update cell counts for each gene
            cellnum += torch.sum(chunk_dense, dim=0).int()

            # Calculate co-expression contributions
            coex += torch.matmul(chunk_dense.T.float(), chunk_dense.float()).int()

            # Release memory for intermediate tensors
            del chunk_csr, chunk_sparse, chunk_dense
            gc.collect()
            torch.cuda.empty_cache()

        # Return result inside try block
        return coex, cellnum

    finally:
        # Ensure all tensors are deleted and memory is cleared
        del coex, cellnum
        gc.collect()
        torch.cuda.empty_cache()


def calculate_covariance_in_chunks(x, chunk_size, device, dtype=torch.float32):
    """
    Compute the covariance matrix of a large matrix `x` in chunks with specified precision.

    Parameters:
        x: scipy.sparse.csr_matrix
            Input matrix (cells x genes).
        chunk_size: int
            Number of rows to process in each chunk.
        device: torch.device
            Device to use for computation (CPU or GPU).
        dtype: torch.dtype
            Desired precision for computation (torch.float32 or torch.float64).

    Returns:
        cov_all: torch.Tensor
            Covariance matrix (genes x genes).
    """
    nrow, ncol = x.shape
    cov_all = torch.zeros((ncol, ncol), dtype=dtype, device=device)
    row_sums = torch.zeros(ncol, dtype=dtype, device=device)
    row_counts = 0

    try:
        for start_row in range(0, nrow, chunk_size):
            end_row = min(start_row + chunk_size, nrow)
            chunk_csr = x[start_row:end_row]
            chunk_sparse = csr_to_torch_sparse_tensor(chunk_csr, device, dtype=dtype)
            chunk = chunk_sparse.to_dense()

            # Update cumulative sums and counts
            row_sums += torch.sum(chunk, dim=0)
            row_counts += chunk.size(0)

            # Update covariance contributions
            cov_all += torch.matmul(chunk.T, chunk)

            # Release memory for intermediate tensors
            del chunk_csr, chunk_sparse, chunk
            gc.collect()
            torch.cuda.empty_cache()

        # Normalize row_sums and calculate means
        row_means = row_sums / row_counts

        # Update cov_all in chunks to save memory
        for start_col in range(0, ncol, chunk_size):
            end_col = min(start_col + chunk_size, ncol)
            cov_all[:, start_col:end_col] /= (row_counts - 1)
            # Use torch.ger to ensure dimensions match
            outer_chunk = torch.ger(row_means, row_means[start_col:end_col])
            cov_all[:, start_col:end_col] -= outer_chunk
            del outer_chunk
            gc.collect()
            torch.cuda.empty_cache()

        # Ensure that results are properly finalized
        cov_all /= (row_counts - 1)

        # Return the result inside try block
        return cov_all

    finally:
        # Ensure all tensors are deleted and memory is cleared
        del row_sums, row_counts, row_means
        gc.collect()
        torch.cuda.empty_cache()



def calculate_pcors_pytorch(x, round_num, selected_num, gene_name, project_name, cut_off_pcor, cut_off_coex_cell,seed, 
                            run_mode=1, double_precision=False, use_chunking=True, chunk_size=1000, stop_threshold=0):
    """
    Optimized version of calculate_pcors using PyTorch for improved performance on GPU or CPU.

    Parameters:
        x: np.ndarray
            Input expression matrix (cells x genes).
        round_num: int
            Number of iterations to perform.
        selected_num: int
            Number of genes to sample in each iteration.
        gene_name: list
            List of gene names corresponding to columns of x.
        project_name: str
            Name of this ggm project.
        cut_off_pcor: float
            Cut-off for partial correlations.
        cut_off_coex_cell: int
            Minimum number of co-expressed cells.
        seed: int
            Random seed for reproducibility.
        run_mode: int
            Mode of execution:
            0 - Compute entirely on CPU.
            1 - Preprocessing on CPU, partial correlations on GPU.
            2 - Compute entirely on GPU.
        double_precision: bool
            Use double precision for computations.
        use_chunking: bool
            Whether to enable chunked computation. If False, chunk_size will be set to the total number of rows in `x`.
        chunk_size: int
            Size of chunks for memory-efficient computation.
        stop_threshold: int
            Threshold for stopping the loop if the sum of `valid_elements_count` over 100 iterations is less than this value.

    Returns:
        coexpressed_cell_num: np.ndarray
            Number of co-expressed cells per gene pair.
        pcor_all: np.ndarray
            Matrix of partial correlations.
        pcor_sampling_num: np.ndarray
            Number of times each gene pair is sampled.
        rho: np.ndarray
            Matrix of Pearson correlations.
        SigEdges: pd.DataFrame
            DataFrame of significant edges.
    """
    # Define devices
    device_p1 = torch.device("cpu")
    device_p2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    if run_mode == 0:
        device_pre = device_p1
        device = device_p1
    elif run_mode == 1:
        device_pre = device_p1
        device = device_p2
    elif run_mode == 2:
        device_pre = device_p2
        device = device_p2
    else:
        raise ValueError("Invalid run_mode. Use 0 for CPU, 1 for hybrid, or 2 for GPU.")
    
    # Determine dtype based on parameters
    if double_precision:
        chosen_dtype = torch.float64
    else:
        chosen_dtype = torch.float32
    
    # Adjust chunk_size if not using chunking
    if not use_chunking:
        chunk_size = x.shape[0]

    try:
        # Step 1: Preprocessing
        print("\nComputing the number of cells co-expressing each gene pair...")
        coex, cellnum = calculate_coex_in_chunks(x, chunk_size, device_pre)
        coex = coex.cpu() if coex.device.type != "cpu" else coex
        cellnum = cellnum.cpu() if cellnum.device.type != "cpu" else cellnum
        print("Computing covariance matrix...")
        cov_all = calculate_covariance_in_chunks(x, chunk_size, device_pre, dtype=chosen_dtype)
        print("Computing Pearson correlation matrix...")
        std_dev = torch.sqrt(torch.diag(cov_all))
        outer_std = std_dev.unsqueeze(1) * std_dev.unsqueeze(0)
        rho = cov_all / outer_std
        rho = rho.cpu() if rho.device.type != "cpu" else rho  
        del std_dev, outer_std
        gc.collect()
        torch.cuda.empty_cache()

        if run_mode == 1 and device.type == "cuda":
            cov_all = cov_all.to(device)
        
        # Step 2: Initialize matrices for partial correlation computation
        ncol = x.shape[1]
        pcor_all = torch.ones((ncol, ncol), dtype=chosen_dtype, device=device)
        pcor_sampling_num = torch.zeros((ncol, ncol), dtype=torch.int32, device=device)
        time_trend = torch.zeros(100, device=device)
        valid_elements_history = []

        print(f"\nCalculating partial correlations in {round_num} iterations.")
        print(f"Number of genes randomly selected in each iteration: {selected_num}")
        torch.manual_seed(seed)

        # Step 3: Iteratively compute partial correlations
        t0 = time.time()
        for i in range(round_num):
            loop_start_t = time.time()
            j = torch.randperm(ncol, device="cpu")[:selected_num]  # Random sampling on CPU
            j = j.to(device)
            cov_x = cov_all[j][:, j]

            try:
                ix = torch.linalg.inv(cov_x)
            except RuntimeError:
                print(f"Iteration {i + 1}: Submatrix is singular, skipping inversion.")
                continue

            d = torch.diag(torch.sqrt(torch.diag(ix)))
            d_inv = torch.linalg.inv(d)
            pc = -d_inv @ ix @ d_inv
            pc += torch.eye(selected_num, device=device) * 2

            indices = torch.arange(selected_num, device=device)
            row_all, col_all = torch.meshgrid(indices, indices, indexing='ij')
            mask = (j[row_all] > j[col_all])
            row_idx = row_all[mask]
            col_idx = col_all[mask]
            r = j[row_idx]
            s = j[col_idx]

            current_pc_values = pc[row_idx, col_idx]
            pcor_sampling_num[r, s] += 1
            
            updated_mask = torch.abs(current_pc_values) < torch.abs(pcor_all[r, s])
            updated_elements_count = updated_mask.sum().item()
            
            valid_elements_mask = (torch.abs(current_pc_values) < cut_off_pcor) & (torch.abs(pcor_all[r, s]) >= cut_off_pcor)
            valid_elements_count = valid_elements_mask.sum().item() 

            pcor_all[r, s] = torch.where(
                #torch.abs(current_pc_values) < torch.abs(pcor_all[r, s]),
                updated_mask,
                current_pc_values,
                pcor_all[r, s]
            )
            
            loop_time = time.time() - loop_start_t
            time_trend[i % 100] = loop_time
            average_loop_time = time_trend[:min(i + 1, 100)].mean().item()
            time_left = (round_num - i - 1) * average_loop_time / 60
            elapsed_min = (time.time() - t0) / 60.0
            sys.stdout.write(f"\rIteration: {i + 1}/{round_num}, "
                  f"Avg loop time: {average_loop_time:.4f} s, "
                  f"Elapsed time: {elapsed_min:.2f} min, "
                  f"Estimated time left: {time_left:.2f} min.                  "
                  )
            sys.stdout.flush()

            valid_elements_history.append(valid_elements_count)
            if len(valid_elements_history) > 100:
                valid_elements_history.pop(0)  # Remove the oldest value
            if sum(valid_elements_history) < stop_threshold:
                round_num = i + 1
                print(f"\nStopping early at iteration {i + 1} due to fewer gene pairs were removed.")
                break
            
        del ix, d, d_inv, indices, row_all, col_all, mask, pc, i, j, row_idx, col_idx, r, s 
        del current_pc_values, cov_all, cov_x
        gc.collect()
        torch.cuda.empty_cache()

        print("\nAll iterations completed.")

        # Step 4: Process and collect results
        pcor_all[pcor_sampling_num == 0] = 0
        coex = coex.cpu() if coex.device.type != "cpu" else coex
        pcor_all = pcor_all.cpu() if pcor_all.device.type != "cpu" else pcor_all
        pcor_sampling_num = pcor_sampling_num.cpu()
        rho = rho.cpu() if rho.device.type != "cpu" else rho  

        gc.collect()
        torch.cuda.empty_cache()

        # Identify significant edges  
        idx = torch.where((pcor_all >= cut_off_pcor) & (pcor_all < 1) & (coex >= cut_off_coex_cell))
        e1 = [gene_name[i] for i in idx[0].tolist()]
        e1n = [cellnum[i].item() for i in idx[0].tolist()]
        e2 = [gene_name[i] for i in idx[1].tolist()]
        e2n = [cellnum[i].item() for i in idx[1].tolist()]
        e3 = pcor_all[idx].tolist()
        e4 = pcor_sampling_num[idx].tolist()
        e5 = [rho[i, j].item() for i, j in zip(idx[0].tolist(), idx[1].tolist())]
        e6 = [coex[i, j].item() for i, j in zip(idx[0].tolist(), idx[1].tolist())]
        e7 = [project_name] * len(e1)

        SigEdges = pd.DataFrame({
            'GeneA': e1,
            'GeneB': e2,
            'Pcor': e3,
            'SamplingTime': e4,
            'r': e5,
            'Cell_num_A': e1n,
            'Cell_num_B': e2n,
            'Cell_num_coexpressed': e6,
            'Project': e7,
        })
        del idx, e1, e1n, e2, e2n, e3, e4, e5, e6, e7
        gc.collect()
        torch.cuda.empty_cache()
        
        return round_num, coex.numpy(), pcor_all.numpy(), pcor_sampling_num.numpy(), rho.numpy(), SigEdges

    finally:
        del pcor_all, pcor_sampling_num, coex, rho, cellnum
        gc.collect()
        torch.cuda.empty_cache()



def set_selects(gene_num):
    """
    Automatically determine selected_num based on the total number of genes (gene_num)

    Parameters:
        gene_num (int): Total number of genes in the input dataset (must be greater than 0).
        
    Returns:
        selected_num (int): The number of genes selected in each iteration to calculate the partial correlation coefficient
                         
    Conditions (in strict order):
        1. If gene_num is less than 500, use all genes and set target_sampling_count to 1.
        2. If gene_num is in the range [500, 2500), set selected_num = 500 and use the user-specified target_sampling_count.
        3. If gene_num is in the range [2500, 10000), set selected_num to ceil(gene_num / 5) and use the user-specified target_sampling_count.
        4. If gene_num is greater than 10000, set selected_num = 2000 and use the user-specified target_sampling_count.
    """
    if gene_num <= 0:
        raise ValueError("gene_num must be greater than 0.")

    # Condition 1: When gene_num is less than 500
    elif gene_num < 500:
        selected_num = gene_num
    # Condition 2: When gene_num is between 500 and 2500
    elif gene_num < 2500:
        selected_num = 500
    # Condition 3: When gene_num is between 2500 and 10000 (inclusive)
    elif gene_num < 10000:
        selected_num = math.ceil(gene_num / 5)
    # Condition 4: When gene_num is greater than 10000
    else:
        selected_num = 2000

    return selected_num


def estimate_rounds(gene_num, selected_num, target_sampling_count):
    """
    Instruction:
    Estimate round_num based on total sample size, selected sample size and target sampling number

    Parameters:
    gene_num (int): total number of genes in the dataset
    selected_num (int): number of genes selected in each sampling
    target_sampling_count (int): total expected number of times each gene pair is collected in all iterations。       

    Returns:
    int: estimated round_num
    """
    if selected_num > gene_num:
        raise ValueError("selected_num cannot be greater than gene_num")
    
    p_single_gene = selected_num / gene_num
    p_two_genes = p_single_gene ** 2
    round_num = target_sampling_count / p_two_genes
    round_num = math.ceil(round_num)

    return round_num






############################################################################################################
# Old functions
# set_selects_v1
# def set_selects_v1(gene_num):
#     """
#     Automatically determine selected_num based on the total number of genes (gene_num)

#     Parameters:
#         gene_num (int): Total number of genes in the input dataset (must be greater than 0).
        
#     Returns:
#         selected_num (int): The number of genes selected in each iteration to calculate the partial correlation coefficient
                         
#     Conditions (in strict order):
#         1. If gene_num is less than 500, use all genes and set target_sampling_count to 1.
#         2. If gene_num is in the range [500, 5000), set selected_num = 500 and use the user-specified target_sampling_count.
#         3. If gene_num is in the range [5000, 20000] (inclusive), set selected_num to ceil(gene_num / 10) and use the user-specified target_sampling_count.
#         4. If gene_num is greater than 20000, set selected_num = 2000 and use the user-specified target_sampling_count.
#     """
#     if gene_num <= 0:
#         raise ValueError("gene_num must be greater than 0.")

#     # Condition 1: When gene_num is less than 500
#     elif gene_num < 500:
#         selected_num = gene_num
#     # Condition 2: When gene_num is between 500 and 5000
#     elif gene_num < 5000:
#         selected_num = 500
#     # Condition 3: When gene_num is between 5000 and 20000 (inclusive)
#     elif gene_num <= 20000:
#         selected_num = math.ceil(gene_num / 10)
#     # Condition 4: When gene_num is greater than 20000
#     else:
#         selected_num = 2000

#     return selected_num
