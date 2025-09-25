
import numpy as np
import pandas as pd
import os
import sys
import gzip
import time
import mygene
from scipy.stats import hypergeom

# Set the base directory and Ref directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Ref_for_Enrichment")

# ensure_gene_symbol_table 
def ensure_gene_symbol_table(species, species_taxonomy_id, gene_symbl_file) -> dict:
    """
    Ensure a local Ensembl->symbol lookup exists for the given species.
    1) If file exists, load and return mapping.
    2) Else, fetch ALL genes (with Ensembl IDs) for the species from MyGene, save to gzip TSV, then return.
       File format (no header): <ensembl_gene_id>\t<symbol>
    """
    # Make sure the species name is set
    if species is None :
        raise ValueError("Please provide a common name for the species you are searching for.")
    if species_taxonomy_id is None and species not in ["human", "mouse", "rat", "fruitfly", "nematode", "zebrafish", "thale-cress", "frog", "pig"]:
        raise ValueError(f"Please provide the NCBI taxonomy ID of {species} by setting species_taxonomy_id.")

    # 1) cache first
    if os.path.exists(gene_symbl_file):
        df = pd.read_csv(
            gene_symbl_file, sep="\t", header=None, names=["ensembl", "symbol"],
            dtype=str, comment="#", quoting=3
        ).dropna(subset=["ensembl"]).drop_duplicates(subset=["ensembl"], keep="first")
        return df.set_index("ensembl")["symbol"].to_dict()

    print(f"\n[MyGene] Building ALL-genes Ensembl→symbol table for species='{species}' ...")
    time.sleep(1)

    import mygene
    mg = mygene.MyGeneInfo()
    mapping = {}

    # fold hits into mapping (first-win; ensembl can be dict or list)
    def add_hits(hits):
        for doc in hits:
            if not isinstance(doc, dict):
                continue
            sym = doc.get("symbol")
            ens = doc.get("ensembl")
            if isinstance(ens, dict):
                gid = ens.get("gene")
                if isinstance(gid, str) and gid not in mapping:
                    mapping[gid] = sym if isinstance(sym, str) and sym else gid
            elif isinstance(ens, list):
                for e in ens:
                    if isinstance(e, dict):
                        gid = e.get("gene")
                        if isinstance(gid, str) and gid not in mapping:
                            mapping[gid] = sym if isinstance(sym, str) and sym else gid

    def pull_all(q_string: str):
        # try fetch_all (streaming)
        try:
            if species_taxonomy_id is not None:
                cursor = mg.query(q=q_string, species=species_taxonomy_id,
                                fields="symbol,ensembl.gene",
                                size=1000, fetch_all=True)
            elif species_taxonomy_id is None:
                cursor = mg.query(q=q_string, species=species,
                                fields="symbol,ensembl.gene",
                                size=1000, fetch_all=True)
            got = False
            for chunk in cursor:
                if isinstance(chunk, dict) and "hits" in chunk:
                    hits = chunk["hits"]
                elif isinstance(chunk, list):
                    hits = chunk
                else:
                    hits = [chunk]
                if hits:
                    got = True
                    add_hits(hits)
            if got:
                return
        except Exception:
            pass
        # fallback: paginated
        try:
            from_ = 0
            size = 1000
            while True:
                page = mg.query(q=q_string, species=species,
                                fields="symbol,ensembl.gene",
                                size=size, from_=from_)
                hits = page.get("hits", [])
                if not hits:
                    break
                add_hits(hits)
                from_ += size
                if from_ > 2_000_000:  # hard cap
                    break
        except Exception:
            pass

    # 2) build mapping
    pull_all("_exists_:ensembl.gene")
    if not mapping:
        pull_all("ensembl.gene:*")

    if not mapping:
        raise RuntimeError(f"[MyGene] Failed to build ALL-genes mapping for species='{species}'.")

    # 3) save cache (two columns, no header)
    os.makedirs(os.path.dirname(gene_symbl_file), exist_ok=True)
    with gzip.open(gene_symbl_file, "wt", encoding="utf-8") as gz:
        for gid, sym in mapping.items():
            gz.write(f"{gid}\t{sym}\n")
    print(f"[MyGene] Saved mapping to: {gene_symbl_file}  (n={len(mapping)})")

    return mapping


# go_enrichment_analysis
def go_enrichment_analysis(self,
                           species="mouse",
                           padjust_method="BH",
                           pvalue_cutoff=0.05
                           ):
    """
    Perform GO enrichment analysis using module information from a ggm object.

    Parameters:
        self: An object containing module information (ggm.modules) and background genes (ggm.gene_name).
        species: Species for the GO annotation file. Default is "mouse". 
                 * make sure the species is consistent with the GO annotation file before using this function.
                 * one can create a new GO annotation file for a different species named "species.GO.annotation.txt.gz" and "species.gene.symbl.txt.gz".
        padjust_method: Method for adjusting p-values. choose from "BH" or "Bonferroni". Default is "BH".
        pvalue_cutoff: P-value cutoff for selecting significant GO terms. Default is 0.05.
    """    
    # Check if the GO annotation files exist
    go_name_file = f"{DATA_DIR}/GO.names.txt.gz"
    go_annotation_file = f"{DATA_DIR}/{species}.GO.annotation.txt.gz"
    gene_symbl_file = f"{DATA_DIR}/{species}.gene.symbl.txt.gz"
    if os.path.exists(go_annotation_file) and os.path.exists(go_name_file) and os.path.exists(gene_symbl_file):
        pass
    else:
        raise ValueError("GO annotation files not found.")
    
    # Define a function for adjusting p-values
    def p_adjust(pvalues, method="BH"):
        """
        Adjust p-values with Benjamini-Hochberg (BH) or Bonferroni.
        - Returns adjusted p-values aligned to the original order.
        - Preserves NaNs.
        """
        p = np.asarray(pvalues, dtype=float)
        out = np.full_like(p, np.nan, dtype=float)

        # operate only on finite values
        mask = np.isfinite(p)
        if not np.any(mask):
            return out

        pv = p[mask]
        m = pv.size

        method = method.upper()
        if method == "BH":
            # 1) sort ascending
            order = np.argsort(pv)
            pv_sorted = pv[order]

            # 2) raw BH factor
            ranks = np.arange(1, m + 1, dtype=float)
            adj = pv_sorted * m / ranks

            # 3) step-up (enforce monotonicity from right to left)
            adj = np.minimum.accumulate(adj[::-1])[::-1]
            adj = np.clip(adj, 0.0, 1.0)

            # 4) place back to original positions of the finite subset
            adj_back = np.empty_like(adj)
            adj_back[order] = adj
            out[mask] = adj_back

        elif method == "Bonferroni":
            out[mask] = np.clip(pv * m, 0.0, 1.0)
        else:
            raise ValueError("Unsupported method. Choose 'BH' or 'Bonferroni'.")

        return out
    
    print(f"\nReading GO term information for |{species}|...")
    # Extract module information from ggm.modules
    modules = self.modules[['gene', 'module_id']].drop_duplicates()
    bk_genes = self.gene_name.tolist()  # Background genes
    total_gene_num = len(bk_genes)

    # Process GO annotation file
    allgo = pd.read_csv(go_annotation_file, sep="\t", header=None).drop_duplicates()
    allgo = allgo[allgo[0].isin(bk_genes)]  # Filter by background genes
    bk_go_count = allgo[1].value_counts()
    bk_go_count = bk_go_count[bk_go_count < 2500]  # Filter GO terms with < 2500 annotations
    allgo = allgo[allgo[1].isin(bk_go_count.index)]

    # Read in GO names
    go_name = pd.read_csv(go_name_file, sep="\t", header=None, comment="#", quoting=3)
    go_name.set_index(0, inplace=True)

    # Read in gene symbols
    gene_table = pd.read_csv(gene_symbl_file, sep="\t", header=None, comment="#", quoting=3)
    gene_symbl = gene_table.set_index(0)[1].to_dict()
    all_genes = list(set(modules['gene'].tolist() + bk_genes))
    missing_genes = set(all_genes) - set(gene_symbl.keys())
    gene_symbl.update({gene: gene for gene in missing_genes})

    # GO enrichment analysis
    all_table = []  # List to collect all results
    module_name = modules['module_id'].unique().tolist()
    print(f"\nStart GO enrichment analysis ...")
    for i in module_name:
        selected_genes = modules[modules['module_id'] == i]['gene'].tolist()
        module_size_val = len(selected_genes)
        
        # Get the GO terms associated with the selected genes
        module_go = allgo[allgo[0].isin(selected_genes)]
        module_go_count = module_go[1].value_counts()

        go_ids = module_go_count.index.tolist()
        if len(go_ids) > 0:
            # Pre-compute in_c, in_g for all GO terms
            in_c_list = [module_go_count[j] for j in go_ids]  # GO count in module
            in_g_list = [bk_go_count[j] for j in go_ids]  # GO count in background

            # Batch compute p-values for all GO terms
            p_values = hypergeom.sf(np.array(in_c_list) - 1, total_gene_num, np.array(in_g_list), module_size_val)
            # Adjust p-values using BH method
            p_values_adjusted = p_adjust(p_values, method=padjust_method)
            num_significant = (p_values_adjusted <= pvalue_cutoff).sum()
            sys.stdout.write(f"\rFound {num_significant} significant enriched GO terms in {i}       ")
            sys.stdout.flush()
            del num_significant 

            if len(p_values_adjusted[p_values_adjusted <= pvalue_cutoff]) > 0:    
                # Filter out GO terms with pValueAdjusted greater than 0.05
                valid_go_ids = [go_ids[idx] for idx in range(len(p_values_adjusted)) if p_values_adjusted[idx] <= pvalue_cutoff]
                valid_p_values = [p_values[idx] for idx in range(len(p_values_adjusted)) if p_values_adjusted[idx] <= pvalue_cutoff]
                valid_p_values_adjusted = [p_values_adjusted[idx] for idx in range(len(p_values_adjusted)) if p_values_adjusted[idx] <= pvalue_cutoff]
                valid_in_c = [in_c_list[idx] for idx in range(len(in_c_list)) if p_values_adjusted[idx] <= pvalue_cutoff]
                valid_in_g = [in_g_list[idx] for idx in range(len(in_g_list)) if p_values_adjusted[idx] <= pvalue_cutoff]

                # Prepare the new rows for the valid GO terms
                new_rows = []
                for idx, go_id in enumerate(valid_go_ids):
                    genes_with_go = "/".join(sorted(module_go[module_go[1] == go_id][0].tolist()))
                    symbols_with_go = "/".join([gene_symbl.get(g, g) for g in genes_with_go.split("/")])
                    new_row = {
                        "module_id": i,
                        "module_size": module_size_val,
                        "go_rank": 1,  # Rank will be assigned later
                        "go_id": go_id,
                        "go_category": go_name.loc[go_id, 1],
                        "go_term": go_name.loc[go_id, 2],
                        "module_go_count": valid_in_c[idx],
                        "genome_go_count": valid_in_g[idx],
                        "total_gene_number": total_gene_num,
                        "genes_with_go_in_module": genes_with_go,
                        "symbols_with_go_in_module": symbols_with_go,
                        "pValue": valid_p_values[idx],
                        "pValueAdjusted": valid_p_values_adjusted[idx]
                    }
                    new_rows.append(new_row)

                # Create the module table from the new rows
                module_table = pd.DataFrame(new_rows)
                module_table = module_table.sort_values(by="pValue")
                module_table["go_rank"] = range(1, len(module_table) + 1)

                # Append the current module table to all_table
                all_table.append(module_table)
    
    # Merge all module tables into a single DataFrame
    if all_table:
        all_table = pd.concat(all_table, ignore_index=True)
        all_table.reset_index(drop=True, inplace=True)
        if all_table['genes_with_go_in_module'].equals(all_table['symbols_with_go_in_module']):
            all_table = all_table.drop(columns=['symbols_with_go_in_module'])
        self.go_enrichment = all_table
        print(f"\nGO enrichment analysis completed. Found {len(all_table)} significant enriched GO terms total.")
    else:
        print("No significant GO term found.")



# mp_enrichment_analysis
def mp_enrichment_analysis(self,
                           species="mouse",
                           padjust_method="BH",
                           pvalue_cutoff=0.05):
    """
    Perform MP enrichment analysis using module information from a ggm object.

    Parameters:
        self: An object containing module information (ggm.modules) and background genes (ggm.gene_name).
        species: Species for the MP annotation file. Default is "mouse". 
        padjust_method: Method for adjusting p-values. Choose from "BH" or "Bonferroni". Default is "BH".
        pvalue_cutoff: P-value cutoff for selecting significant MP terms. Default is 0.05.
    """

    # Check if the MP annotation files exist
    mp_name_file = f"{DATA_DIR}/MP.names.txt.gz"
    mp_annotation_file = f"{DATA_DIR}/{species}.MP.annotation.txt.gz"
    gene_symbl_file = f"{DATA_DIR}/{species}.gene.symbl.txt.gz"
    if os.path.exists(mp_annotation_file) and os.path.exists(mp_name_file) and os.path.exists(gene_symbl_file):
        pass
    else:
        raise ValueError("MP annotation files not found.")
    
    # Define a function for adjusting p-values
    def p_adjust(pvalues, method="BH"):
        """
        Adjust p-values with Benjamini-Hochberg (BH) or Bonferroni.
        - Returns adjusted p-values aligned to the original order.
        - Preserves NaNs.
        """
        p = np.asarray(pvalues, dtype=float)
        out = np.full_like(p, np.nan, dtype=float)

        # operate only on finite values
        mask = np.isfinite(p)
        if not np.any(mask):
            return out

        pv = p[mask]
        m = pv.size

        method = method.upper()
        if method == "BH":
            # 1) sort ascending
            order = np.argsort(pv)
            pv_sorted = pv[order]

            # 2) raw BH factor
            ranks = np.arange(1, m + 1, dtype=float)
            adj = pv_sorted * m / ranks

            # 3) step-up (enforce monotonicity from right to left)
            adj = np.minimum.accumulate(adj[::-1])[::-1]
            adj = np.clip(adj, 0.0, 1.0)

            # 4) place back to original positions of the finite subset
            adj_back = np.empty_like(adj)
            adj_back[order] = adj
            out[mask] = adj_back

        elif method == "Bonferroni":
            out[mask] = np.clip(pv * m, 0.0, 1.0)
        else:
            raise ValueError("Unsupported method. Choose 'BH' or 'Bonferroni'.")

        return out
    
    print(f"\nReading MP term information for |{species}|...") 
    # Extract cluster information from ggm.modules
    clusters = self.modules[['gene', 'module_id']].drop_duplicates()
    bk_genes = self.gene_name.tolist()  # Background genes
    total_gene_num = len(bk_genes)

    # Process MP annotation file
    allmp = pd.read_csv(mp_annotation_file, sep="\t", header=None).drop_duplicates()
    allmp = allmp[allmp[0].isin(bk_genes)]  # Filter by background genes
    bk_mp_count = allmp[1].value_counts()
    bk_mp_count = bk_mp_count[bk_mp_count < 2500]  # Filter MP terms with < 2500 annotations
    allmp = allmp[allmp[1].isin(bk_mp_count.index)]

    # Read in MP names
    mp_name = pd.read_csv(mp_name_file, sep="\t", header=None, comment="#", quoting=3)
    mp_name.set_index(0, inplace=True)

    # Read in gene symbols
    gene_table = pd.read_csv(gene_symbl_file, sep="\t", header=None, comment="#", quoting=3)
    gene_symbl = gene_table.set_index(0)[1].to_dict()
    all_genes = list(set(clusters['gene'].tolist() + bk_genes))
    missing_genes = set(all_genes) - set(gene_symbl.keys())
    gene_symbl.update({gene: gene for gene in missing_genes})

    # MP enrichment analysis
    all_table = []  # List to collect all results
    module_name = clusters['module_id'].unique().tolist()
    print(f"\nStart MP enrichment analysis ...")
    
    for i in module_name:
        selected_genes = clusters[clusters['module_id'] == i]['gene'].tolist()
        module_size_val = len(selected_genes)

        # Get the MP terms associated with the selected genes
        module_mp = allmp[allmp[0].isin(selected_genes)]
        module_mp_count = module_mp[1].value_counts()

        mp_ids = module_mp_count.index.tolist()

        if len(mp_ids) > 0:
            # Pre-compute in_c, in_g for all MP terms
            in_c_list = [module_mp_count[j] for j in mp_ids]  # MP count in module
            in_g_list = [bk_mp_count[j] for j in mp_ids]  # MP count in background

            # Batch compute p-values for all MP terms
            p_values = hypergeom.sf(np.array(in_c_list) - 1, total_gene_num, np.array(in_g_list), module_size_val)
            # Adjust p-values using selected method
            p_values_adjusted = p_adjust(p_values, method=padjust_method)
            num_significant = (p_values_adjusted <= pvalue_cutoff).sum()
            sys.stdout.write(f"\rFound {num_significant} significant enriched MP terms in {i}       ")
            sys.stdout.flush()

            if len(p_values_adjusted[p_values_adjusted <= pvalue_cutoff]) > 0:    
                # Filter out MP terms with pValueAdjusted greater than 0.05
                valid_mp_ids = [mp_ids[idx] for idx in range(len(p_values_adjusted)) if p_values_adjusted[idx] <= pvalue_cutoff]
                valid_p_values = [p_values[idx] for idx in range(len(p_values_adjusted)) if p_values_adjusted[idx] <= pvalue_cutoff]
                valid_p_values_adjusted = [p_values_adjusted[idx] for idx in range(len(p_values_adjusted)) if p_values_adjusted[idx] <= pvalue_cutoff]
                valid_in_c = [in_c_list[idx] for idx in range(len(in_c_list)) if p_values_adjusted[idx] <= pvalue_cutoff]
                valid_in_g = [in_g_list[idx] for idx in range(len(in_g_list)) if p_values_adjusted[idx] <= pvalue_cutoff]

                # Prepare the new rows for the valid MP terms
                new_rows = []
                for idx, mp_id in enumerate(valid_mp_ids):
                    genes_with_mp = "/".join(sorted(module_mp[module_mp[1] == mp_id][0].tolist()))
                    symbols_with_mp = "/".join([gene_symbl.get(g, g) for g in genes_with_mp.split("/")])
                    new_row = {
                        "module_id": i,
                        "module_size": module_size_val,
                        "mp_rank": 1,  # Rank will be assigned later
                        "mp_id": mp_id,
                        "mp_term": mp_name.loc[mp_id, 1],
                        "mp_description": mp_name.loc[mp_id, 2],
                        "module_mp_count": valid_in_c[idx],
                        "genome_mp_count": valid_in_g[idx],
                        "total_gene_number": total_gene_num,
                        "genes_with_mp_in_module": genes_with_mp,
                        "symbols_with_mp_in_module": symbols_with_mp,
                        "pValue": valid_p_values[idx],
                        "pValueAdjusted": valid_p_values_adjusted[idx]
                    }
                    new_rows.append(new_row)

                # Create the module table from the new rows
                module_table = pd.DataFrame(new_rows)
                module_table = module_table.sort_values(by="pValue")
                module_table["mp_rank"] = range(1, len(module_table) + 1)

                # Append the current module table to all_table
                all_table.append(module_table)

    # Merge all module tables into a single DataFrame
    if all_table:
        all_table = pd.concat(all_table, ignore_index=True)
        all_table.reset_index(drop=True, inplace=True)
        if all_table['genes_with_mp_in_module'].equals(all_table['symbols_with_mp_in_module']):
            all_table = all_table.drop(columns=['symbols_with_mp_in_module'])
        self.mp_enrichment = all_table
        print(f"\nMP enrichment analysis completed. Found {len(all_table)} significant enriched MP terms total.")
    else:
        print("No significant MP term found.")


# get_GO_annoinfo
def get_GO_annoinfo(species_name=None,
                    species_taxonomy_id=None):
    """
    Download GO annotation and gene symbol mapping for the given species from MyGene.info and save them into compressed files.
    If higher quality GO annotation information is desired, it may be necessary to construct the relevant files manually.
    
    Parameters:
        species_name: set a common name for the species, which will be used to name the output files 
                      and as an anchor for the enrichment analysis function.
                      (e.g. species_name="mouse" for Mus musculus)
            - These common names can be used instead of passing species_taxonomy_id:
                'human'	        for Homo sapiens	
                'mouse'	        for Mus musculus	
                'rat'	        for Rattus norvegicus	
                'fruitfly'	    for Drosophila melanogaster	
                'nematode'	    for Caenorhabditis elegans	
                'zebrafish'	    for Danio rerio	GRCz10 
                'thale-cress'   for Arabidopsis thaliana
                'frog'	        for Xenopus tropicalis	
                'pig'	        for Sus scrofa
                (see https://docs.mygene.info/projects/mygene-py/en/ver_2.3.0/index.html for details.)
            - For other species, you must pass the NCBI taxonomy ID to parameters 'species_taxonomy_id'.

        species_taxonomy_id: NCBI taxonomy ID for the species. 
                             (e.g. species_name='rice', species_taxonomy_id='39947' for Oryza sativa Japonica Group) 
                             (see https://www.ncbi.nlm.nih.gov/taxonomy for details.)

    """
    # Make sure the species name is set
    if species_name is None :
        raise ValueError("Please provide a common name for the species you are searching for.")
    if species_taxonomy_id is None and species_name not in ["human", "mouse", "rat", "fruitfly", "nematode", "zebrafish", "thale-cress", "frog", "pig"]:
        raise ValueError(f"Please provide the NCBI taxonomy ID of {species_name}.")

    # Make sure the GO names file exists
    go_names_file = f"{DATA_DIR}/GO.names.txt.gz"
    out_annotation_file = f"{DATA_DIR}/{species_name}.GO.annotation.txt.gz"
    out_gene_symbl_file = f"{DATA_DIR}/{species_name}.gene.symbl.txt.gz"

    if not os.path.exists(go_names_file):
        raise ValueError("GO names file not found.")
    
    # If the GO Anno file already exist, rebuild them
    if os.path.exists(out_annotation_file):
        print(f"\nNOTE! The GO annotation file for |{species_name}| will be rebuilt")
        os.remove(out_annotation_file)

    # Make sure the gene symbol file exists, if not, construct it
    _ = ensure_gene_symbol_table(species=species_name,
                                species_taxonomy_id=species_taxonomy_id,
                                gene_symbl_file=out_gene_symbl_file)

    # Set up MyGeneInfo and query for genes with GO annotations
    mg = mygene.MyGeneInfo()
    query = "go:*"
    fields = "ensembl,symbol,go"
    print(f"\nQuerying MyGene.info for {species_name} genes with GO annotations ...")
    time.sleep(1)
    if species_taxonomy_id is not None:
        res0 = mg.query(query, species=species_taxonomy_id, fields=fields, size=0)
    elif species_taxonomy_id is None:
        res0 = mg.query(query, species=species_name, fields=fields, size=0)
    total = res0.get("total", 0)
    print(f"Total genes with GO annotation (reported): {total}")
    
    # Fetch all genes with GO annotations
    print(f"\nFetching all genes with GO annotation ...")
    if species_taxonomy_id is not None:
        res = mg.query(q=query, species=species_taxonomy_id, fields=fields, fetch_all=True)
    elif species_taxonomy_id is None:
        res = mg.query(q=query, species=species_name, fields=fields, fetch_all=True)
    hits = []
    for gene in res:
        hits.append(gene)    
    
    # Collect (gene, GO term) pairs for Ensembl and Symbol separately
    ensembl_annotation_pairs = set()
    symbol_annotation_pairs = set()

    for hit in hits:
        # Get the Ensembl gene ID
        ensembl_info = hit.get("ensembl")
        ensembl_ids = []
        if isinstance(ensembl_info, list):
            for e in ensembl_info:
                if isinstance(e, dict):
                    gid = e.get("gene")
                    if isinstance(gid, str):
                        ensembl_ids.append(gid)
        elif isinstance(ensembl_info, dict):
            gid = ensembl_info.get("gene")
            if isinstance(gid, str):
                ensembl_ids.append(gid)
        # Get the gene symbol
        symbol = hit.get("symbol")

        # Process GO annotations
        go_data = hit.get("go")
        if not (go_data and isinstance(go_data, dict)):
            continue
        for cat in ["BP", "MF", "CC"]:
            terms = go_data.get(cat)
            if not terms:
                continue
            if isinstance(terms, dict):
                terms = [terms]
            if not isinstance(terms, list):
                continue
            for term in terms:
                go_term = term.get("id") if isinstance(term, dict) else None
                if not go_term:
                    continue
                for gid in ensembl_ids:
                    ensembl_annotation_pairs.add((gid, go_term))
                if isinstance(symbol, str) and symbol:
                    symbol_annotation_pairs.add((symbol, go_term))

    # Read valid GO terms from GO.names.txt.gz    
    valid_go_terms = set()
    with gzip.open(go_names_file, "rt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                valid_go_terms.add(parts[0])

    # Filter out annotation pairs with GO terms not present in GO.names.txt.gz
    ensembl_annotation_pairs_final = {(g, go) for (g, go) in ensembl_annotation_pairs if go in valid_go_terms}
    symbol_annotation_pairs_final  = {(s, go) for (s, go) in symbol_annotation_pairs  if go in valid_go_terms}

    # Write annotation file: first write Ensembl-based annotations, then Symbol-based annotations
    with gzip.open(out_annotation_file, "wt") as f:
        for gene_id, go_term in sorted(ensembl_annotation_pairs_final):
            f.write(f"{gene_id}\t{go_term}\n")
        for sym, go_term in sorted(symbol_annotation_pairs_final):
            f.write(f"{sym}\t{go_term}\n")
    
    print(f"Files written:\n  {out_annotation_file}\n  {out_gene_symbl_file}")