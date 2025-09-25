import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import statsmodels.api as sm
import itertools
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr, kendalltau
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from statsmodels.stats.multitest import multipletests
from typing import List, Optional

# get_module_edges 
def get_module_edges(self, module_id):
    """
    Extract edges within a module.
    Parameters:
        module_id: The ID of the module to extract.
    Returns:
        module_edges: The edges within the module.
    """
    module_list = self.modules['module_id'].unique()
    if module_id not in module_list:
        raise ValueError("Module ID not found.")
    if self.modules is None:
        raise ValueError("Please run find_modules first.")
    genes_in_module = self.modules.loc[self.modules["module_id"] == module_id, "gene"].unique()

    mask = self.SigEdges["GeneA"].isin(genes_in_module) & self.SigEdges["GeneB"].isin(genes_in_module)
    module_edges = self.SigEdges[mask].copy()
    module_edges.index = range(len(module_edges))
    module_edges.insert(0, "module_id", module_id)
    
    return module_edges

# get_module_anno 
def get_module_anno(self, module_id, add_enrich_info=True, top_n=None, term_id=None):
    """
    Get the annotation information of a specific module.
    Parameters:
        self: GGM object
        module_id: str, a module id in the modules_summary
        add_enrich_info: bool, whether to add GO and MP enrichment information to the module annotation
        top_n: int, the top n GO or MP terms to add to the module annotation, default as None
            use when add_enrich_info is True and too many GO or MP terms are enriched in the module
        term_id: a list of GO or MP term ids to add to the module annotation, default as None
            use for specific GO or MP terms to add to the module annotation
    """
    if term_id is not None and top_n is not None:
        raise ValueError("term_id and top_n cannot be specified at the same time.")
    
    if self.modules is None:
        raise ValueError("No modules found. run find_modules function first.")
    
    if module_id not in self.modules['module_id'].values:
        raise ValueError(f"{module_id} not found in modules.")

    module_anno = self.modules[self.modules['module_id'] == module_id].copy()
    if add_enrich_info:
        if self.go_enrichment is not None:
            go_df = self.go_enrichment[self.go_enrichment['module_id'] == module_id].copy()
            if go_df.empty:
                print(f"No significant enrichment GO term found for {module_id}.")
            else:
                if top_n is not None:
                    go_df = go_df.head(top_n)
                for _, row in go_df.iterrows():
                    go_id = row['go_id']
                    go_term = row['go_term']
                    gene_list = row['genes_with_go_in_module'].split("/")
                    module_anno[go_id] = module_anno['gene'].apply(lambda g: go_term if g in gene_list else None)

        if self.mp_enrichment is not None:
            mp_df = self.mp_enrichment[self.mp_enrichment['module_id'] == module_id].copy()
            if mp_df.empty:
                print(f"No significant enrichment MP term found for {module_id}.")
            else:
                if top_n is not None:
                    mp_df = mp_df.head(top_n)
                for _, row in mp_df.iterrows():
                    mp_id = row['mp_id']
                    mp_term = row['mp_term']
                    gene_list = row['genes_with_mp_in_module'].split("/")
                    module_anno[mp_id] = module_anno['gene'].apply(lambda g: mp_term if g in gene_list else None)
    if term_id is not None:
        save_id = np.concatenate((go_df['go_id'].values, mp_df['mp_id'].values))
        remove_id = [x for x in save_id if x not in term_id]
        keep_id = [x for x in term_id if x in save_id]
        wrong_id = [x for x in term_id if x not in save_id]
        if len(keep_id) == 0:
            print(f"Make sure the term_id listed are in the enrichment information.")
        else:    
            if len(wrong_id) > 0:
                print(f"The term_id {wrong_id} not in the enrichment information.")
            module_anno = module_anno.drop(columns=remove_id)  

    return module_anno

# module_network_plot
def module_network_plot(
    nodes_edges: pd.DataFrame,
    nodes_anno: pd.DataFrame,
    show_nodes = 30,
    highlight_anno: Optional[str] = None,
    highlight_genes: Optional[List[str]] = None,
    label_show: str = 'all',
    use_symbol: bool = True,
    
    seed: int = 42,
    weight_power: float = 1.0,
    layout: str = 'spring',
    layout_k: float = 1.0,
    layout_iterations: int = 50,
    margin: float = 0.1,
    
    line_style: str = '-',
    line_width: float = 1.0,
    line_alpha: float = 0.4,
    line_color: str = 'lightskyblue',
    
    node_border_width: float = 0,
    node_color: str = 'darkgray',
    node_size: int = 100,
    node_alpha: float = 0.8,
    highlight_node_color: str = 'salmon',
    highlight_node_size: int = 100,
    highlight_node_alpha: float = 0.8,
    
    label_color: str = 'black',
    label_font_size: int = 10,
    label_font_weight: str = 'normal',
    label_alpha: float = 1.0,
    highlight_label_color: str = 'black',
    highlight_label_font_size: int = 10,
    highlight_label_font_weight: str = 'bold',
    highlight_label_alpha: float = 1.0,
    
    plot: bool = True,
    save_plot_as: Optional[str] = None,
    save_network_as: Optional[str] = None
) -> nx.Graph:
    """
    Construct and optionally render a gene co-expression network.

    Builds a NetworkX graph from pairwise partial correlations and gene annotations,
    computes node positions via a chosen layout, and renders the network with
    fully customizable styling. Optionally exports the plot to PDF/PNG and saves
    the graph in GraphML format for Cytoscape.

    Parameters
    ----------
    nodes_edges : pd.DataFrame
        DataFrame with columns ['GeneA', 'GeneB', 'Pcor'], defining edges and base weights.
    nodes_anno : pd.DataFrame
        DataFrame with at least ['gene', 'rank']; may include 'symbol' and other annotation
        columns for highlighting.
    show_nodes : int or 'all', default=30
        Number of top-ranked genes to include, or 'all' to include every gene.
    highlight_anno : str or None, default=None
        Column in nodes_anno whose non-null entries mark highlighted nodes.
        Ignored if highlight_genes is provided.
    highlight_genes : list of str or None, default=None
        List of gene identifiers or symbols to highlight. Matches first against
        nodes_anno['symbol'] if present, then nodes_anno['gene']. Overrides highlight_anno.
    label_show : {'all', 'highlight', 'none'}, default='all'
        Which nodes to label: all nodes, only highlighted nodes, or none.
    use_symbol : bool, default=True
        If True and 'symbol' exists in nodes_anno, use it for labels instead of gene IDs.
    seed : int, default=42
        Random seed for reproducible layout.
    weight_power : float, default=1.0
        Exponent applied to Pcor when computing edge weight: weight = Pcor ** weight_power.
    layout : {'spring', 'circular', 'kamada_kawai', 'spectral'}, default='spring'
        Layout algorithm for node positioning.
    layout_k : float, default=1.0
        Optimal node distance for spring layout; smaller values yield a more compact graph.
    layout_iterations : int, default=50
        Number of iterations for the spring layout algorithm.
    margin : float, default=0.1
        Fractional margin around the figure to prevent clipping.
    line_style : str, default='-'
        Line style for edges.
    line_width : float, default=1.0
        Line width for edges.
    line_alpha : float, default=0.4
        Opacity of edges (0.0 to 1.0).
    line_color : str, default='lightskyblue'
        Color of edges.
    node_border_width : float, default=0
        Width of node borders.
    node_color : str, default='darkgray'
        Color of non-highlighted nodes.
    node_size : int, default=100
        Size of non-highlighted nodes.
    node_alpha : float, default=0.8
        Opacity of non-highlighted nodes.
    highlight_node_color : str, default='salmon'
        Color of highlighted nodes.
    highlight_node_size : int, default=100
        Size of highlighted nodes.
    highlight_node_alpha : float, default=0.8
        Opacity of highlighted nodes.
    label_color : str, default='black'
        Color of non-highlighted node labels.
    label_font_size : int, default=10
        Font size for non-highlighted node labels.
    label_font_weight : str, default='normal'
        Font weight for non-highlighted node labels.
    label_alpha : float, default=1.0
        Opacity of non-highlighted node labels.
    highlight_label_color : str, default='black'
        Color of highlighted node labels.
    highlight_label_font_size : int, default=10
        Font size for highlighted node labels.
    highlight_label_font_weight : str, default='bold'
        Font weight for highlighted node labels.
    highlight_label_alpha : float, default=1.0
        Opacity of highlighted node labels.
    plot : bool, default=True
        If True, display the network plot.
    save_plot_as : str or None, default=None
        File path ending in .pdf or .png to save the rendered plot.
    save_network_as : str or None, default=None
        File path to export the graph in GraphML format.

    Returns
    -------
    networkx.Graph
        The constructed co-expression network graph.
    """
    # Prepare node labels
    has_symbol = use_symbol and 'symbol' in nodes_anno.columns
    label_map = {
        row['gene']: row['symbol'] if has_symbol and pd.notnull(row['symbol']) else row['gene']
        for _, row in nodes_anno.iterrows()
    }

    # Select nodes
    if show_nodes != 'all':
        top_n = min(int(show_nodes), len(nodes_anno))
        genes = nodes_anno.sort_values('rank').iloc[:top_n]['gene'].tolist()
    else:
        genes = nodes_anno['gene'].tolist()

    # Determine highlight set
    if highlight_genes is not None:
        symbol_to_gene = {}
        if has_symbol:
            symbol_to_gene = {
                sym: gene for gene, sym in zip(nodes_anno['gene'], nodes_anno['symbol'])
                if pd.notnull(sym)
            }
        highlight_set = set()
        for g in highlight_genes:
            if has_symbol and g in symbol_to_gene:
                highlight_set.add(symbol_to_gene[g])
            elif g in genes:
                highlight_set.add(g)
    else:
        highlight_set = set()
        if highlight_anno and highlight_anno in nodes_anno.columns:
            highlight_set = set(
                nodes_anno.loc[nodes_anno[highlight_anno].notnull(), 'gene']
            )

    # Build graph
    G = nx.Graph()
    for gene in genes:
        G.add_node(gene, highlight=(gene in highlight_set))
    sub = nodes_edges[
        nodes_edges['GeneA'].isin(genes) & nodes_edges['GeneB'].isin(genes)
    ]
    for _, row in sub.iterrows():
        w = row.get('Pcor', 1.0) ** weight_power
        G.add_edge(row['GeneA'], row['GeneB'], weight=w)

    # Export if requested
    if save_network_as:
        nx.write_graphml(G, save_network_as)

    # Layout
    if layout == 'spring':
        pos = nx.spring_layout(G, weight='weight', k=layout_k,
                               iterations=layout_iterations, seed=seed)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G, weight='weight')
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, weight='weight', k=layout_k,
                               iterations=layout_iterations, seed=seed)

    # Draw & save
    if plot or save_plot_as:
        fig = plt.figure(figsize=(8, 8))
        nx.draw_networkx_edges(
            G, pos, style=line_style, width=line_width,
            alpha=line_alpha, edge_color=line_color
        )
        normal = [n for n, d in G.nodes(data=True) if not d['highlight']]
        highl = [n for n, d in G.nodes(data=True) if d['highlight']]
        nx.draw_networkx_nodes(
            G, pos, nodelist=normal, node_size=node_size,
            node_color=node_color, edgecolors='k',
            linewidths=node_border_width, alpha=node_alpha
        )
        if highlight_set:
            nx.draw_networkx_nodes(
                G, pos, nodelist=highl, node_size=highlight_node_size,
                node_color=highlight_node_color, edgecolors='k',
                linewidths=node_border_width, alpha=highlight_node_alpha
            )
        labels_normal = {}
        labels_highlight = {}
        if label_show == 'all':
            labels_normal = {n: label_map[n] for n in normal}
            labels_highlight = {n: label_map[n] for n in highl}
        elif label_show == 'highlight':
            labels_highlight = {n: label_map[n] for n in highl}
        if labels_normal:
            nx.draw_networkx_labels(
                G, pos, labels=labels_normal,
                font_size=label_font_size,
                font_weight=label_font_weight,
                font_color=label_color,
                alpha=label_alpha
            )
        if labels_highlight:
            nx.draw_networkx_labels(
                G, pos, labels=labels_highlight,
                font_size=highlight_label_font_size,
                font_weight=highlight_label_font_weight,
                font_color=highlight_label_color,
                alpha=highlight_label_alpha
            )
        plt.margins(margin)
        plt.axis('off')
        plt.tight_layout(pad=margin)
        if save_plot_as:
            ext = save_plot_as.split('.')[-1].lower()
            if ext not in ['pdf', 'png']:
                raise ValueError("save_plot_as must end with .pdf or .png")
            dpi = 300 if ext == 'png' else None
            fig.savefig(save_plot_as, dpi=dpi, bbox_inches='tight')
        if plot:
            plt.show()

    return G



def module_go_enrichment_plot(
    ggm,
    *,
    top_n_modules: int = 5,
    selected_modules = None,
    module_colors = None,
    go_per_module: int = 2,
    genes_per_go: int = 5,
    bar_height: float = 0.5,
    row_gap: float = 1.0,
    text_size: int = 12,
    label_fontsize: int = 15,
    tick_fontsize: int = 15,
    module_col_width: float = 0.05,
    module_col_alpha: float = 1.0,
    bar_alpha: float = 0.6,
    min_rows: int = 2,
    bottom_gap: float = 0.8,
    fig_width: float = None,
    fig_height: float = None,
    save_plot_as: str = None,
) -> None:
    """
    Plot GO enrichment bars for gene co-expression modules).

    Parameters
    ----------
    ggm : object
        Object containing attributes: 'go_enrichment' (GO results) and modules' (gene co-expression modules).
    top_n_modules : int, default 5
        Number of modules to draw if *selected_modules* is None.
    selected_modules : list[str] | None, default None
        Explicit list of module IDs to plot. If None, the first
        *top_n_modules* (sorted numerically by ID) are used.
    module_colors : dict[str, str] | None, default None
        Mapping {module_id: hex_color}. If None, a tab20 colormap is used.
    go_per_module : int, default 2
        Maximum number of GO terms to display per module.
    genes_per_go : int, default 5
        Maximum number of genes to list under each GO term.
    bar_height : float, default 0.5
        Height of each horizontal bar.
    row_gap : float, default 1.0
        Vertical distance between consecutive GO rows.
    text_size : int, default 12
        Font size for GO terms and gene lists.
    label_fontsize : int, default 15
        Font size for the X-axis label.
    tick_fontsize : int, default 15
        Font size for X-axis ticks.
    module_col_width : float, default 0.05
        Fractional width of the left color column.
    module_col_alpha : float, default 1.0
        Opacity of the module color blocks.
    bar_alpha : float, default 0.6
        Opacity of the bars.
    min_rows : int, default 2
        Minimum number of GO rows required to draw the figure.
    bottom_gap : float, default 0.8
        Additional space below the lowest row, expressed as a
        multiple of *row_gap*.
    fig_width : float or None
        Figure width in inches; defaults to 11 if None.
    fig_height : float or None
        Figure height in inches; computed from data if None.
    save_plot_as : str or None
        Path ending in .pdf or .png to save the figure.
    Returns
    -------
    None
        Opens a matplotlib figure window.
    """
     # prepare data
    go_df = ggm.go_enrichment.rename(
        columns={"module_id": "module", "go_term": "Description", "pValueAdjusted": "padj"}
    ).copy()
    mod_df = ggm.modules.rename(columns={"module_id": "module"}).copy()

    # determine modules
    if selected_modules:
        module_list = [m for m in selected_modules if m in go_df["module"].unique()]
    else:
        module_list = sorted(go_df["module"].unique(), key=lambda x: int(x[1:]))[:top_n_modules]

    # assign colors
    if module_colors is None:
        cmap = plt.get_cmap("tab20")
        module_colors = {m: cmap(i % cmap.N) for i, m in enumerate(module_list)}
    for i, m in enumerate(module_list):
        module_colors.setdefault(m, plt.get_cmap("tab20")(i % 20))

    # gather rows
    rows = []
    for m in module_list:
        subset = go_df[go_df["module"] == m].nsmallest(go_per_module, "padj")
        if subset.empty:
            continue
        for _, rec in subset.iterrows():
            ens = [g for g in rec["genes_with_go_in_module"].split("/") if g]
            genes = (
                mod_df[(mod_df["module"] == m) & (mod_df["gene"].isin(ens))]
                .sort_values("rank")
                .head(genes_per_go)["symbol"]
                .tolist()
            )
            rows.append(
                {
                    "module": m,
                    "Description": rec["Description"],
                    "gene_str": "/".join(genes),
                    "neglog10q": -np.log10(rec["padj"]),
                }
            )

    plot_df = pd.DataFrame(rows)
    if len(plot_df) < min_rows:
        raise ValueError(f"Only {len(plot_df)} GO rows (< {min_rows}); aborting.")

    plot_df["module"] = pd.Categorical(plot_df["module"], categories=module_list)
    plot_df = (plot_df.sort_values(["module", "neglog10q"],ascending=[True, False]).reset_index(drop=True))
    plot_df["ypos"] = plot_df.index * row_gap

    # layout
    default_fig_width = 11
    data_fig_height = row_gap * len(plot_df) * 1.1

    width = fig_width if fig_width is not None else default_fig_width
    height = fig_height if fig_height is not None else data_fig_height

    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(1, 2, width_ratios=[module_col_width, 1 - module_col_width], wspace=0)


    low_lim = -bottom_gap * row_gap
    high_lim = plot_df["ypos"].max() + row_gap / 2

    # module color bar column
    ax_left = fig.add_subplot(gs[0, 0])
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(low_lim, high_lim)
    ax_left.axis("off")
    for m in module_list:
        ypos = plot_df[plot_df["module"] == m]["ypos"]
        if ypos.empty:
            continue
        ymin, ymax = ypos.min() - bar_height / 2, ypos.max() + bar_height / 2
        ax_left.add_patch(
            Rectangle((0.05, ymin), 0.9, ymax - ymin, facecolor=module_colors[m], alpha=module_col_alpha, lw=0)
        )
        ax_left.text(0.5, (ymin + ymax) / 2, m, rotation=90, ha="center", va="center", fontsize=text_size)

    # enrichment bars
    ax = fig.add_subplot(gs[0, 1])
    ax.set_ylim(low_lim, high_lim)
    ax.barh(
        plot_df["ypos"],
        plot_df["neglog10q"],
        left=1.0,
        height=bar_height,
        color=[module_colors[m] for m in plot_df["module"]],
        alpha=bar_alpha,
    )
    for _, row in plot_df.iterrows():
        y = row["ypos"]
        col = module_colors[row["module"]]
        ax.text(1.1, y, row["Description"], ha="left", va="center", fontsize=text_size)
        ax.text(
            1.1,
            y - bar_height * 0.7,
            row["gene_str"],
            ha="left",
            va="top",
            fontsize=text_size,
            style="italic",
            color=col,
        )

    ax.set_yticks([])
    ax.set_xlabel(r"$-\log_{10}\left(\mathit{P}\mathrm{_{adj}}\right)$",
                  fontsize=label_fontsize)
    ax.set_xlim(0, 1.0 + plot_df["neglog10q"].max() * 1.05)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.spines[["right", "top"]].set_visible(False)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    if save_plot_as:
        ext = save_plot_as.split(".")[-1].lower()
        if ext not in ["pdf", "png"]:
            raise ValueError("save_plot_as must end with .pdf or .png")
        dpi = 300 if ext == "png" else None
        fig.savefig(save_plot_as, dpi=dpi, bbox_inches="tight")

    plt.tight_layout()
    plt.show()



# module_mp_enrichment_plot
def module_mp_enrichment_plot(
    ggm,
    *,
    top_n_modules: int = 5,
    selected_modules=None,
    module_colors=None,
    mp_per_module: int = 2,
    genes_per_mp: int = 5,
    bar_height: float = 0.5,
    row_gap: float = 1.0,
    text_size: int = 12,
    label_fontsize: int = 15,
    tick_fontsize: int = 15,
    module_col_width: float = 0.05,
    module_col_alpha: float = 1.0,
    bar_alpha: float = 0.6,
    min_rows: int = 2,
    bottom_gap: float = 0.8,
    fig_width: float = None,
    fig_height: float = None,
    save_plot_as: str = None,
) -> None:
    """
    Plot MP-enrichment bars for gene co-expression modules.

    Parameters
    ----------
    ggm : object
        Must have attributes `mp_enrichment` and `modules` (both DataFrames).
    top_n_modules : int
        Number of modules to draw if selected_modules is None.
    selected_modules : list or None
        Explicit list of module IDs to plot.
    module_colors : dict or None
        Mapping module -> color; generated if None.
    mp_per_module : int
        Maximum MP terms per module.
    genes_per_mp : int
        Number of genes listed under each MP term.
    bar_height : float
        Height of each horizontal bar.
    row_gap : float
        Vertical spacing between rows.
    text_size : int
        Font size for term labels and gene lists.
    label_fontsize : int
        Font size for the x-axis label.
    tick_fontsize : int
        Font size for x-axis ticks.
    module_col_width : float
        Fractional width of the left color column.
    module_col_alpha : float
        Opacity of the module color blocks.
    bar_alpha : float
        Opacity of the bars.
    min_rows : int
        Minimum number of rows required to plot.
    bottom_gap : float
        Extra space below the lowest row (multiple of row_gap).
    fig_width : float or None
        Figure width in inches; defaults to 11.
    fig_height : float or None
        Figure height in inches; computed from data if None.
    save_plot_as : str or None
        Path ending in .pdf or .png to save the figure.
    """
    # prepare data
    mp_df = ggm.mp_enrichment.rename(
        columns={"module_id": "module", "mp_term": "Description", "pValueAdjusted": "padj"}
    ).copy()
    mod_df = ggm.modules.rename(columns={"module_id": "module"}).copy()

    # select modules
    if selected_modules:
        module_list = [m for m in selected_modules if m in mp_df["module"].unique()]
    else:
        module_list = sorted(mp_df["module"].unique(), key=lambda x: int(x[1:]))[:top_n_modules]

    # assign colors
    if module_colors is None:
        cmap = plt.get_cmap("tab20")
        module_colors = {m: cmap(i % cmap.N) for i, m in enumerate(module_list)}
    for i, m in enumerate(module_list):
        module_colors.setdefault(m, plt.get_cmap("tab20")(i % 20))

    # gather rows
    rows = []
    for m in module_list:
        subset = mp_df[mp_df["module"] == m].nsmallest(mp_per_module, "padj")
        if subset.empty:
            continue
        for _, rec in subset.iterrows():
            ens = [g for g in rec["genes_with_mp_in_module"].split("/") if g]
            genes = (
                mod_df[(mod_df["module"] == m) & (mod_df["gene"].isin(ens))]
                .sort_values("rank")
                .head(genes_per_mp)["symbol"]
                .tolist()
            )
            rows.append({
                "module": m,
                "Description": rec["Description"],
                "gene_str": "/".join(genes),
                "neglog10p": -np.log10(rec["padj"]),
            })

    plot_df = pd.DataFrame(rows)
    if len(plot_df) < min_rows:
        raise ValueError(f"Only {len(plot_df)} MP rows (< {min_rows}); aborting.")

    plot_df["module"] = pd.Categorical(plot_df["module"], categories=module_list)
    plot_df = (plot_df.sort_values(["module", "neglog10p"],ascending=[True, False]).reset_index(drop=True))
    plot_df["ypos"] = plot_df.index * row_gap

    # figure size
    default_w = 11
    default_h = row_gap * len(plot_df) * 1.1
    w = fig_width if fig_width is not None else default_w
    h = fig_height if fig_height is not None else default_h

    fig = plt.figure(figsize=(w, h))
    gs = fig.add_gridspec(1, 2,
                         width_ratios=[module_col_width, 1 - module_col_width],
                         wspace=0)

    low_lim = -bottom_gap * row_gap
    high_lim = plot_df["ypos"].max() + row_gap / 2

    # module color column
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_xlim(0, 1)
    ax0.set_ylim(low_lim, high_lim)
    ax0.axis("off")
    for m in module_list:
        ys = plot_df[plot_df["module"] == m]["ypos"]
        if ys.empty:
            continue
        ymin, ymax = ys.min() - bar_height/2, ys.max() + bar_height/2
        ax0.add_patch(Rectangle((0.05, ymin), 0.9, ymax - ymin,
                                 facecolor=module_colors[m],
                                 alpha=module_col_alpha, lw=0))
        ax0.text(0.5, (ymin+ymax)/2, m,
                 rotation=90, ha="center", va="center", fontsize=text_size)

    # enrichment bars
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_ylim(low_lim, high_lim)
    ax1.barh(plot_df["ypos"], plot_df["neglog10p"],
             left=1.0, height=bar_height,
             color=[module_colors[m] for m in plot_df["module"]],
             alpha=bar_alpha)
    for _, row in plot_df.iterrows():
        y = row["ypos"]
        col = module_colors[row["module"]]
        ax1.text(1.1, y, row["Description"], ha="left", va="center", fontsize=text_size)
        ax1.text(1.1, y - bar_height*0.7, row["gene_str"],
                 ha="left", va="top",
                 fontsize=text_size, style="italic", color=col)

    ax1.set_yticks([])
    ax1.set_xlabel(r"$-\log_{10}(\mathit{P}\mathrm{_{adj}})$",
                   fontsize=label_fontsize)
    ax1.set_xlim(0, 1.0 + plot_df["neglog10p"].max()*1.05)
    ax1.tick_params(axis="x", labelsize=tick_fontsize)
    ax1.spines[["right","top"]].set_visible(False)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    if save_plot_as:
        ext = save_plot_as.split(".")[-1].lower()
        if ext not in ["pdf","png"]:
            raise ValueError("save_plot_as must end with .pdf or .png")
        dpi = 300 if ext=="png" else None
        fig.savefig(save_plot_as, dpi=dpi, bbox_inches="tight")

    plt.tight_layout()
    plt.show()



# module_similarity_plot
def module_similarity_plot(
    adata,
    ggm_key='ggm',
    modules_used=None,
    modules_excluded=None,
    use_smooth=True,
    corr_method='pearson',
    linkage_method='average',
    return_summary=True,
    plot_heatmap=True,
    show_full_heatmap = True,
    heatmap_metric='correlation',
    fig_height=15,
    fig_width=16,
    dendrogram_height=0.15,
    dendrogram_space=0.08,
    axis_fontsize=12,
    axis_labelsize=15,
    legend_fontsize=12,
    legend_labelsize=15,
    cmap_name='bwr',
    save_plot_as=None
):
    """
    Compute pairwise module similarity and optionally visualize as a clustered heatmap.

    This function calculates correlation and Jaccard index between modules based on
    expression profiles and annotations stored in an AnnData object. It can produce
    either a half-matrix (upper triangle only) or a full-matrix heatmap with
    hierarchical clustering dendrograms.

    Parameters
    ----------
    adata : AnnData
        Annotated data containing module expression in .obsm and annotations in .obs.
    ggm_key : str, default 'ggm'
        Key to access module stats and expression arrays in adata.uns['ggm_keys'].
    modules_used : list of str or None
        Module IDs to include; all modules if None.
    modules_excluded : list of str or None
        Module IDs to exclude; none if None.
    use_smooth : bool, default True
        Use smoothed annotations if available.
    corr_method : {'pearson','spearman','kendall'}, default 'pearson'
        Correlation method for expression profiles.
    linkage_method : str, default 'average'
        Linkage method for hierarchical clustering.
    return_summary : bool, default True
        Return DataFrame of module-pair metrics if True.
    plot_heatmap : bool, default True
        Display the heatmap and dendrogram plot if True.
    show_full_heatmap : bool, default True
        Plot full symmetric matrix if True; otherwise upper triangle only.
    heatmap_metric : {'correlation','jaccard'}, default 'correlation'
        Metric to visualize in the heatmap.
    fig_height, fig_width : float
        Figure dimensions in inches.
    dendrogram_height : float
        Fraction of figure allocated to each dendrogram.
    dendrogram_space : float
        Spacing between dendrogram and heatmap axes.
    axis_fontsize, axis_labelsize : int
        Font sizes for tick labels and axis labels.
    legend_fontsize, legend_labelsize : int
        Font sizes for colorbar and legend text.
    cmap_name : str
        Name of diverging colormap.
    save_plot_as : str or None
        File path to save the figure; supports '.pdf' or '.png'.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns ['module_a','module_b','correlation','jaccard_index']
        if return_summary is True and plotting is disabled; otherwise None.
    """
    # validate colormap and metric
    allowed_cmaps = [
        'PiYG','PRGn','BrBG','PuOr','RdGy','RdBu',
        'RdYlBu','RdYlGn','Spectral','coolwarm','bwr','seismic'
    ] + [c + '_r' for c in [
        'PiYG','PRGn','BrBG','PuOr','RdGy','RdBu',
        'RdYlBu','RdYlGn','Spectral','coolwarm','bwr','seismic'
    ]]
    if cmap_name not in allowed_cmaps:
        raise ValueError(f"cmap_name must be one of {allowed_cmaps}")
    if heatmap_metric not in ['correlation','jaccard']:
        raise ValueError("heatmap_metric must be 'correlation' or 'jaccard'")

    # prepare module expression matrix
    ggm_keys = adata.uns.get('ggm_keys', {})
    if ggm_key not in ggm_keys:
        raise ValueError(f"{ggm_key} missing in adata.uns['ggm_keys']")
    stat_key = ggm_keys[ggm_key]['module_stats']
    expr_key = ggm_keys[ggm_key]['module_expression']
    stats_df = adata.uns[stat_key]
    module_expr = pd.DataFrame(
        adata.obsm[expr_key], index=adata.obs_names,
        columns=stats_df['module_id']
    )

    # filter modules
    modules = list(module_expr.columns) if modules_used is None else [
        m for m in modules_used if m in module_expr
    ]
    if modules_excluded:
        modules = [m for m in modules if m not in modules_excluded]
    if len(modules) < 2:
        raise ValueError("At least two modules are required")
    module_expr = module_expr[modules]

    # compute correlation matrix
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        corr_df = module_expr.astype(float).corr(method=corr_method)
    corr_mat = corr_df.values

    # compute Jaccard index matrix
    if use_smooth:
        anno_dict = {
            m: (adata.obs.get(f"{m}_anno_smooth", adata.obs[f"{m}_anno"]) == m).astype(int).values
            for m in modules
        }
    else:
        anno_dict = {
            m: (adata.obs[f"{m}_anno"] == m).astype(int).values
            for m in modules
        }
    records = []
    jacc_mat = pd.DataFrame(np.nan, index=modules, columns=modules)
    for i, j in itertools.combinations(range(len(modules)), 2):
        a = anno_dict[modules[i]]
        b = anno_dict[modules[j]]
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        jacc = inter / union if union > 0 else np.nan
        records.append({
            'module_a': modules[i],
            'module_b': modules[j],
            'correlation': float(corr_mat[i, j]),
            'jaccard_index': jacc
        })
        jacc_mat.iat[i, j] = jacc_mat.iat[j, i] = jacc
    result_df = pd.DataFrame(records)

    # return summary if no plotting
    if not plot_heatmap and save_plot_as is None:
        return result_df if return_summary else None

    # hierarchical clustering
    Z = linkage(squareform(1 - corr_mat, checks=False), method=linkage_method)
    ordered = [modules[i] for i in leaves_list(Z)]
    n = len(ordered)

    # select heatmap data
    if heatmap_metric == 'correlation':
        data_df = corr_df.loc[ordered, ordered]
        vmin, vmax, cbar_label = np.nanmin(data_df.values), 1, 'Correlation'
    else:
        data_df = jacc_mat.loc[ordered, ordered].copy()
        np.fill_diagonal(data_df.values, 1.0)
        vmin, vmax, cbar_label = 0, 1, 'Jaccard index'
    mask = None if show_full_heatmap else np.tril(np.ones_like(data_df, bool), -1)

    # prepare colormap slice
    full_cmap = plt.get_cmap(cmap_name, 200)
    sub_cmap = ListedColormap(
        full_cmap(np.arange(200))[int(((vmin+1)/2 if heatmap_metric=='correlation' and vmin<0 else 0.5)*199):],
        name=f'{cmap_name}_slice'
    )

    # plotting
    if not show_full_heatmap:
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(2, 1,
                              height_ratios=(dendrogram_height, 1-dendrogram_height),
                              hspace=dendrogram_space)
        ax_dend = fig.add_subplot(gs[0, 0])
        ax_heat = fig.add_subplot(gs[1, 0])

        dendrogram(Z, ax=ax_dend, labels=ordered,
                   orientation='top', no_labels=True,
                   link_color_func=lambda *a, **k: 'black')
        ax_dend.axis('off')

        sns.heatmap(data_df.values, mask=mask, cmap=sub_cmap,
                    vmin=vmin, vmax=vmax,
                    xticklabels=ordered, yticklabels=ordered,
                    square=False, cbar=False, ax=ax_heat)

        # adjust axes for half-matrix
        ax_heat.xaxis.tick_top()
        ax_heat.xaxis.set_label_position('top')
        ax_heat.yaxis.tick_right()
        ax_heat.yaxis.set_label_position('right')
        ticks = np.arange(n) + 0.5
        ax_heat.set_xticks(ticks)
        ax_heat.set_yticks(ticks)
        ax_heat.set_xticklabels(ordered, rotation=90, ha='center', fontsize=axis_fontsize)
        ax_heat.set_yticklabels(ordered, rotation=0, fontsize=axis_fontsize)
        ax_heat.set_ylabel('Module', fontsize=axis_labelsize)

        # colorbar for half-matrix
        cax = fig.add_axes([0.14, 0.14, 0.02, 0.2])
        sm = plt.cm.ScalarMappable(cmap=sub_cmap,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array(data_df.values)
        cb = fig.colorbar(sm, cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=legend_fontsize)
        cb.set_label(cbar_label, fontsize=legend_labelsize)

    else:
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(2, 2,
                              width_ratios=(dendrogram_height, 1-dendrogram_height),
                              height_ratios=(dendrogram_height, 1-dendrogram_height),
                              wspace=dendrogram_space, hspace=0)
        ax_row = fig.add_subplot(gs[1, 0])
        ax_col = fig.add_subplot(gs[0, 1])
        ax_heat = fig.add_subplot(gs[1, 1])

        dendrogram(Z, ax=ax_col, labels=ordered,
                   orientation='top', no_labels=True,
                   link_color_func=lambda *a, **k: 'black')
        ax_col.axis('off')

        dendrogram(Z, ax=ax_row, labels=ordered,
                   orientation='left', no_labels=True,
                   link_color_func=lambda *a, **k: 'black')
        ax_row.axis('off')
        ax_row.invert_yaxis()

        sns.heatmap(data_df.values, mask=mask, cmap=sub_cmap,
                    vmin=vmin, vmax=vmax,
                    xticklabels=ordered, yticklabels=ordered,
                    square=False, cbar=False, ax=ax_heat)

        # adjust axes for full-matrix
        ticks = np.arange(n) + 0.5
        ax_heat.xaxis.tick_bottom()
        ax_heat.set_xticks(ticks)
        ax_heat.set_xticklabels(ordered, rotation=45,
                                ha='right', va='top',
                                fontsize=axis_fontsize,
                                rotation_mode='anchor')
        ax_heat.yaxis.tick_left()
        ax_heat.set_yticks(ticks)
        ax_heat.set_yticklabels(ordered, fontsize=axis_fontsize)
        ax_heat.tick_params(axis='both', pad=1)
        ax_heat.set_xlabel('Module', fontsize=axis_labelsize)
        ax_heat.set_ylabel('')

        # colorbar for full-matrix
        cax = fig.add_axes([0.92, 0.3, 0.02, 0.2])
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax),
                                   cmap=sub_cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation='vertical')
        cb.ax.set_ylabel(cbar_label, fontsize=legend_labelsize)
        cb.ax.tick_params(labelsize=legend_fontsize)

    # save figure if requested
    if save_plot_as:
        ext = save_plot_as.split('.')[-1]
        if ext not in ['pdf', 'png']:
            raise ValueError("save_plot_as must end with .pdf or .png")
        dpi = 300 if ext == 'png' else None
        fig.savefig(save_plot_as, dpi=dpi, bbox_inches='tight')

    if plot_heatmap:
        plt.show()

    return result_df if return_summary else None


# module_dot_plot
def module_dot_plot(
    adata,
    ggm_key: str = 'ggm',
    modules_used: List[str] = None,
    modules_excluded: List[str] = None,
    groupby: str = None,
    show_dendrogram: bool = True,
    dendrogram_height: float = 0.15,
    dendrogram_space: float = 0.1,
    scale: bool = False,
    corr_method: str = 'pearson',
    linkage_method: str = 'average',
    fig_height: float = 8,
    fig_width: float = 10,
    dot_max_size: float = 200,
    cmap: str = 'Reds',
    axis_fontsize: int = 12,
    axis_labelsize: int = 15,
    legend_fontsize: int = 12,
    legend_labelsize: int = 15,
    save_plot_as: str = None,
    return_df: bool = False
):
    """
    Generate a dot plot of module expression across defined cell groups or cluster with optional dendrograms.

    This function clusters cell groups and modules based on correlation of expression,
    computes mean expression and percentage of expressing cells per module-group pair,
    and visualizes the results. dot size indicates percentage expressing; color indicates mean expression.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing expression in .X, metadata in .obs,
        module statistics in .uns, and module expressions in .obsm.
    ggm_key : str
        Key in adata.uns['ggm_keys'] that references module stats and expression.
    modules_used : List[str], optional
        Modules to include. If None, all available modules are used.
    modules_excluded : List[str], optional
        Modules to exclude from analysis.
    groupby : str
        Column name in adata.obs for grouping cells.
    show_dendrogram : bool, default True
        Whether to display hierarchical clustering dendrograms.
    dendrogram_height : float, default 0.15
        Fraction of figure height allocated to each dendrogram.
    dendrogram_space : float, default 0.1
        Horizontal gap between the left dendrogram and the y-axis tick labels when
        show_dendrogram is True (0 → flush; larger → wider gap).    
    scale : bool, default False
        Scale mean expression values for each module to [0,1].
    corr_method : str, default 'pearson'
        Correlation method for distance computation.
    linkage_method : str, default 'average'
        Linkage method for hierarchical clustering.
    fig_height : float, default 10
        Figure height in inches.
    fig_width : float, default 8
        Figure width in inches.
    dot_max_size : float, default 200
        Maximum bubble size for 100% expression.
    cmap : str, default 'Reds'
        Colormap for mean expression.
    axis_fontsize : int, default 12
        Font size for axis ticks.
    axis_labelsize : int, default 15
        Font size for axis labels.
    legend_fontsize : int, default 12
        Font size for legend text.
    legend_labelsize : int, default 15
        Font size for legend title.
    save_plot_as : str, optional
        Path to save figure (formats: 'pdf', 'png').
    return_df : bool, default False
        If True, return DataFrame of computed metrics.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns ['group','module','mean_expr','pct_expr'] if return_df is True; else None.
    """
    # Check if groupby is provided
    if groupby is None:
        raise ValueError("groupby must be specified")
    if groupby not in adata.obs:
        raise ValueError(f"{groupby} not found in adata.obs")
    keys = adata.uns.get('ggm_keys', {})
    if ggm_key not in keys:
        raise ValueError(f"{ggm_key} not found in adata.uns['ggm_keys']")
    
    # 1) Cluster cell groups
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    expr = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    grp_means = expr.groupby(adata.obs[groupby]).mean()
    grp_corr = grp_means.T.corr(method=corr_method)
    Zg = linkage(squareform(1 - grp_corr.values), method=linkage_method)
    ordered_groups = list(grp_corr.index[leaves_list(Zg)])

    # 2) Cluster modules
    stats_key = keys[ggm_key]['module_stats']
    expr_key = keys[ggm_key]['module_expression']
    stats_df = adata.uns[stats_key]
    mod_expr = pd.DataFrame(
        adata.obsm[expr_key], index=adata.obs_names,
        columns=stats_df['module_id']
    )
    modules = list(mod_expr.columns) if modules_used is None else [m for m in modules_used if m in mod_expr]
    if modules_excluded:
        modules = [m for m in modules if m not in modules_excluded]
    if len(modules) < 2:
        raise ValueError('At least two modules are required')
    mod_expr = mod_expr[modules]
    mod_corr = mod_expr.corr(method=corr_method)
    Zm = linkage(squareform(1 - mod_corr.values), method=linkage_method)
    ordered_modules = [modules[i] for i in leaves_list(Zm)]

    # 3) Compute mean expression and percent expressing
    trim_cols = [f"{m}_exp_trim" for m in ordered_modules]
    df_trim = adata.obs[[groupby] + trim_cols]
    grp = df_trim.groupby(groupby)
    mean_expr = grp[trim_cols].mean().loc[ordered_groups, trim_cols]
    pct_expr = (
        grp[trim_cols]
        .apply(lambda x: (x > 0).sum() / len(x) * 100)
        .loc[ordered_groups, trim_cols]
    )
    if scale:
        mn, mx = mean_expr.min(), mean_expr.max()
        rng = (mx - mn).replace(0, 1)
        mean_expr = (mean_expr - mn) / rng

    plot_df = pd.DataFrame({
        'group': np.repeat(ordered_groups, len(ordered_modules)),
        'module': ordered_modules * len(ordered_groups),
        'mean_expr': mean_expr.values.flatten(),
        'pct_expr': pct_expr.values.flatten(),
    })
    g2i = {g: i for i, g in enumerate(ordered_groups)}
    m2i = {m: i for i, m in enumerate(ordered_modules)}
    xs = plot_df['module'].map(m2i)
    ys = plot_df['group'].map(g2i)

    # 4) Create figure and axes
    fig = plt.figure(figsize=(fig_width, fig_height))
    if show_dendrogram:
        gs = fig.add_gridspec(
            2, 2,
            width_ratios=(dendrogram_height, 1 - dendrogram_height),
            height_ratios=(dendrogram_height, 1 - dendrogram_height),
            hspace=0, wspace=dendrogram_space
        )
        ax_row = fig.add_subplot(gs[1, 0])
        ax_col = fig.add_subplot(gs[0, 1])
        ax_sc = fig.add_subplot(gs[1, 1])
        dendrogram(Zg, ax=ax_row, labels=ordered_groups[::-1], orientation='left', no_labels=True,
                   link_color_func=lambda *args, **kwargs: 'black')
        ax_row.axis('off')
        dendrogram(Zm, ax=ax_col, labels=ordered_modules, orientation='top', no_labels=True,
                   link_color_func=lambda *args, **kwargs: 'black')
        ax_col.axis('off')
    else:
        ax_sc = fig.add_subplot(1, 1, 1)

    # 5) Plot bubbles
    sc = ax_sc.scatter(
        xs, ys,
        s=plot_df['pct_expr'] / 100 * dot_max_size,
        c=plot_df['mean_expr'], cmap=cmap, edgecolors='none'
    )
    ax_sc.set_xlim(-0.5, len(ordered_modules) - 0.5)
    ax_sc.set_ylim(-0.5, len(ordered_groups) - 0.5)
    ax_sc.set_xticks(range(len(ordered_modules)))
    ax_sc.set_xticklabels(ordered_modules, rotation=45, 
                          ha='right', va='top', rotation_mode='anchor', 
                          fontsize=axis_fontsize)
    ax_sc.set_yticks(range(len(ordered_groups)))
    ax_sc.set_yticklabels(ordered_groups, fontsize=axis_fontsize)
    ax_sc.tick_params(axis='both', pad=1) 
    ax_sc.set_xlabel('Module', fontsize=axis_labelsize)
    if not show_dendrogram:
        ax_sc.set_ylabel(groupby, fontsize=axis_labelsize)

    pos = ax_sc.get_position()          
    pad = 0.02                         
    cb_x = pos.x1 + pad
    lg_x = pos.x1 + pad

    # 6) Add colorbar
    cax = fig.add_axes([cb_x, 0.4, 0.02, 0.2])
    cbar = fig.colorbar(sc, cax=cax, orientation='vertical')
    cbar.ax.set_title('Average Expression', fontsize=legend_labelsize, pad=10, loc='left')
    cbar.ax.tick_params(labelsize=legend_fontsize)

    # 7) Add size legend
    lax = fig.add_axes([lg_x, 0.0, 0.1, 0.30])
    lax.axis('off')
    sizes = [25, 50, 75, 100]
    handles = [
        Line2D([0], [0], linestyle='', marker='o', color='black',
               markersize=np.sqrt(s / 100 * dot_max_size))
        for s in sizes
    ]
    labels = [f"{s}%" for s in sizes]
    lax.legend(handles, labels, loc='upper left', frameon=False,
               handletextpad=0.5, labelspacing=0.8, borderaxespad=0)
    lax.text(0, 1.0, 'Percent Expressed', transform=lax.transAxes,
             va='bottom', ha='left', fontsize=legend_labelsize)

    # 8) Save figure if requested
    if save_plot_as:
        fmt = save_plot_as.split('.')[-1]
        if fmt not in ['pdf', 'png']:
            raise ValueError("Supported formats are 'pdf' and 'png'.")
        save_kwargs = {'bbox_inches': 'tight'}
        if fmt == 'png':
            save_kwargs['dpi'] = 300
        fig.savefig(save_plot_as, format=fmt, **save_kwargs)

    if return_df:
        return plot_df
    return



# module_degree_vs_moran_plot
def module_degree_vs_moran_plot(
    data,                       
    *,
    ggm_key="ggm",              
    module_id,                 
    highlight_genes=None,       
    show_highlight_genes=True,  
    show_stats=True,            
    corr_method="pearson",      
    adjust_method="fdr_bh",     
    show_regression=True,       
    show_module_moran=True,           
    nodes_size=30,              
    nodes_color="#A6CEE3",      
    highlight_color="#E6550D", 
    axis_label_size=13,
    tick_label_size=11,
    text_size=11,
    line_width=1.2,
    fig_width=5,
    fig_height=5,
    save_plot_as=None,          # 'name.png' or 'name.pdf'; None disables saving
):
    """
    Plot the relationship between gene connectivity (degree) and spatial
    autocorrelation (gene_moran_I) within a specified module.

    The function accepts either a standalone DataFrame or an AnnData object
    containing the module-level DataFrame in `adata.uns`.  When an AnnData
    object is supplied, `ggm_key` is used to locate the DataFrame via
    `adata.uns['ggm_keys'][ggm_key]['module_info']`.

    Parameters
    ----------
    data : pandas.DataFrame | AnnData
        DataFrame must contain the columns
        'module_id', 'degree', 'gene_moran_I', and 'module_moran_I'.
        AnnData must hold the same DataFrame at the location described above.
    ggm_key : str, optional
        Key for locating the module-information DataFrame in AnnData (default: 'ggm').
    module_id : str
        Module identifier to be plotted.
    highlight_genes : list[str] | None, optional
        Genes to highlight on the scatter plot (default: None).
    show_highlight_genes : bool, optional
        If True, print gene symbols for highlighted points (default: True).
    show_stats : bool, optional
        If True, compute and display the correlation coefficient and adjusted
        p-value (default: True).
    corr_method : {'pearson', 'spearman', 'kendall'}, optional
        Method for computing correlation (default: 'pearson').
    adjust_method : str, optional
        Multiple-testing correction method passed to
        `statsmodels.stats.multitest.multipletests` (default: 'fdr_bh').
    show_regression : bool, optional
        Draw an OLS regression line with 95 % confidence interval (default: True).
    show_module_moran : bool, optional
        Draw a dashed horizontal line at the module Moran's I value (default: True).
    nodes_size : int or float, optional
        Marker size for all scatter points (default: 30).
    nodes_color : str, optional
        Color for non-highlighted points (default: '#A6CEE3').
    highlight_color : str, optional
        Color for highlighted points (default: '#E6550D').
    axis_label_size : int, optional
        Font size for axis labels (default: 13).
    tick_label_size : int, optional
        Font size for tick labels (default: 11).
    text_size : int, optional
        Font size for all text annotations (default: 11).
    line_width : float, optional
        Line width for axes, regression line, and cutoff line (default: 1.2).
    fig_width, fig_height : float, optional
        Figure dimensions in inches (default: 5 X 5).
    save_plot_as : str | None, optional
        File name for saving the figure (.png or .pdf).  If None, the plot is
        not written to disk (default: None).

    Returns
    -------
    None
        The function shows the plot directly and optionally saves it to disk.
    """
    # ---------- locate DataFrame ----------
    if isinstance(data, pd.DataFrame):
        df = data
        required = {"module_id", "degree", "gene_moran_I", "module_moran_I"}
        if not required.issubset(df.columns):
            missing = ", ".join(sorted(required - set(df.columns)))
            raise ValueError(f"DataFrame is missing required column(s): {missing}")
    elif hasattr(data, "uns"):
        ggm_keys = data.uns.get("ggm_keys", {})
        if ggm_key not in ggm_keys:
            raise ValueError(f"'{ggm_key}' not found in adata.uns['ggm_keys']")
        mod_key = ggm_keys[ggm_key].get("module_info")
        if mod_key is None or mod_key not in data.uns:
            raise ValueError(f"module_info '{mod_key}' not present in adata.uns")
        df = data.uns[mod_key]
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Located module_info is not a pandas DataFrame")
    else:
        raise TypeError("Argument 'data' must be a DataFrame or AnnData-like object")

    # ---------- subset by module ----------
    sub = df[df["module_id"] == module_id].copy()
    if sub.empty:
        raise ValueError(f"Module '{module_id}' not found in the DataFrame")
    for col in ("degree", "gene_moran_I"):
        if col not in sub.columns:
            raise KeyError(f"Required column '{col}' is missing in the DataFrame")

    # ---------- correlation ----------
    if show_stats:
        method = corr_method.lower()
        if method == "pearson":
            r, p = pearsonr(sub["degree"], sub["gene_moran_I"])
        elif method == "spearman":
            r, p = spearmanr(sub["degree"], sub["gene_moran_I"])
        elif method == "kendall":
            r, p = kendalltau(sub["degree"], sub["gene_moran_I"])
        else:
            raise ValueError("corr_method must be 'pearson', 'spearman', or 'kendall'")
        p_adj = multipletests([p], method=adjust_method)[1][0]

    # ---------- regression ----------
    if show_regression:
        X = sm.add_constant(sub["degree"])
        model = sm.OLS(sub["gene_moran_I"], X).fit()
        x_pred = np.linspace(sub["degree"].min(), sub["degree"].max(), 200)
        pred = model.get_prediction(sm.add_constant(x_pred)).summary_frame(alpha=0.05)

    # ---------- plotting ----------
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.set_facecolor("#FFFFFF")
    for spine in ax.spines.values():
        spine.set_linewidth(line_width)

    ax.scatter(
        sub["degree"], sub["gene_moran_I"],
        s=nodes_size, c=nodes_color,
        edgecolors="k", linewidths=line_width * 0.35,
        alpha=0.9, zorder=2
    )

    if show_regression:
        ax.plot(x_pred, pred["mean"], color="#1F78B4",
                linewidth=line_width, zorder=3)
        ax.fill_between(
            x_pred, pred["mean_ci_lower"], pred["mean_ci_upper"],
            color="#1F78B4", alpha=0.25, zorder=2
        )

    if show_module_moran:
        if "module_moran_I" not in sub.columns:
            raise KeyError("Column 'module_moran_I' is missing in the DataFrame")
        cutoff = sub["module_moran_I"].iloc[0]
        ax.axhline(cutoff, color="k", linestyle="--",
                   linewidth=line_width, zorder=1.5)
        x_min = sub["degree"].min()
        x_rng = sub["degree"].max() - x_min
        y_rng = ax.get_ylim()[1] - ax.get_ylim()[0]
        ax.text(
            x_min - 0.02 * x_rng, cutoff - 0.03 * y_rng,
            f"Moran'I of {module_id}: {cutoff:.3f}",
            ha="left", va="top", fontsize=text_size
        )

    if highlight_genes:
        mask_symbol = sub["symbol"].isin(highlight_genes) if "symbol" in sub else False
        mask_gene = sub["gene"].isin(highlight_genes)
        high = sub[mask_symbol | mask_gene]
        ax.scatter(
            high["degree"], high["gene_moran_I"],
            s=nodes_size, c=highlight_color,
            edgecolors="k", linewidths=line_width * 0.5, zorder=4
        )
        if show_highlight_genes and not high.empty:
            y_rng = ax.get_ylim()[1] - ax.get_ylim()[0]
            offset = y_rng * 0.02
            for _, row in high.iterrows():
                ax.text(
                    row["degree"], row["gene_moran_I"] + offset,
                    row.get("symbol", row["gene"]),
                    fontsize=text_size, ha="center", va="bottom",
                    color=highlight_color, zorder=5
                )

    if show_stats:
        ax.text(
            0.95, 0.05,
            rf"$r = {r:.3f},\; \mathit{{P}}_{{\mathrm{{adj}}}} = {p_adj:.2e}$",
            ha="right", va="bottom",
            transform=ax.transAxes,
            fontsize=text_size,
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", alpha=0.75, lw=0)
        )

    ax.set_xlabel("Degree", fontsize=axis_label_size)
    ax.set_ylabel("Moran'I", fontsize=axis_label_size)
    ax.tick_params(labelsize=tick_label_size)
    plt.tight_layout()

    if save_plot_as:
        ext = save_plot_as.split(".")[-1].lower()
        if ext not in ("pdf", "png"):
            raise ValueError("save_plot_as must end with '.pdf' or '.png'")
        dpi = 300 if ext == "png" else None
        fig.savefig(save_plot_as, dpi=dpi, bbox_inches="tight")

    plt.show()
