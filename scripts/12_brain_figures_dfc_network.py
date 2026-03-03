#!/usr/bin/env python3
"""
12_brain_figures_dfc_network.py — Dynamic Functional Connectivity Network Diagrams

This script visualizes the Dynamic Functional Connectivity (dFC) matrices as
a network graph (connectome) projected onto the brain. It extracts the centroid
coordinates of the Schaefer 2018 parcels and plots the strongest edges (top 0.5%)
between these nodes using Nilearn's plot_connectome on a glass brain.
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from nilearn import datasets, plotting

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

warnings.filterwarnings("ignore")

def clean_label_name(label_bytes):
    """Convert Schaefer label byte strings to readable names."""
    try:
        label_str = label_bytes.decode('utf-8')
    except AttributeError:
        label_str = str(label_bytes)
        
    parts = label_str.split('_')
    if len(parts) >= 3:
        hemi = parts[1]
        net = parts[2]
        idx = parts[3] if len(parts) > 3 else ""
        net_mapped = config.YEO7_NETWORK_FULL_NAMES.get(net, net)
        return f"{hemi} {net_mapped} {idx}".strip()
    return label_str

def main():
    dfc_results_path = config.DFC_DIR / "mean_fc_per_stage.npz"
    if not dfc_results_path.exists():
        print(f"File not found: {dfc_results_path}")
        return
        
    figures_dir = config.FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("Fetching Schaefer 2018 atlas (200 parcels, 2mm)...")
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
    schaefer_img = nib.load(schaefer.maps)
    
    # Extract coordinates for each of the 200 parcels
    print("Computing parcellation center of mass coordinates...")
    node_coords = plotting.find_parcellation_cut_coords(labels_img=schaefer_img)
    
    # Read mean FC data
    dfc_data = np.load(str(dfc_results_path), allow_pickle=True)
    
    # Define colors for the 7 Yeo networks to color the nodes
    network_colors = {
        'Vis': '#781286',        # Purple
        'SomMot': '#4682B4',     # Steel Blue
        'DorsAttn': '#00760E',   # Green
        'SalVentAttn': '#C43AFA',# Violet
        'Limbic': '#DCF8A4',     # Light Green/Yellow
        'Cont': '#E69422',       # Orange
        'Default': '#CD3E4E',    # Red
    }
    
    node_colors = []
    for label_bytes in schaefer.labels[1:]:
        try:
            label_str = label_bytes.decode('utf-8')
        except AttributeError:
            label_str = str(label_bytes)
        parts = label_str.split('_')
        if len(parts) >= 3:
            net = parts[2]
            node_colors.append(network_colors.get(net, 'black'))
        else:
            node_colors.append('black')
            
    for stage in ["Wake", "N1", "N2"]:
        key = f"fc_{stage}"
        if key not in dfc_data:
            print(f"Skipping {stage}, no data found in npz.")
            continue
            
        print(f"Processing DFC Connectome for {stage}...")
        fc_matrix = dfc_data[key]
        
        # Zero out the diagonal for plotting
        np.fill_diagonal(fc_matrix, 0)
        
        # 1. Find the top 10 strongest edges (by absolute connectivity)
        # Using upper triangle to avoid duplicates
        upper_tri = np.triu(np.abs(fc_matrix), k=1)
        flat_indices = np.argsort(upper_tri, axis=None)[::-1]
        
        # We need the readable names of the 200 parcels
        labels = [clean_label_name(lbl) for lbl in schaefer.labels[1:]]
        
        top_edges = []
        for idx in flat_indices[:10]:
            r, c = np.unravel_index(idx, upper_tri.shape)
            val = fc_matrix[r, c]
            name1 = labels[r]
            name2 = labels[c]
            
            # Shorten names for table
            if len(name1) > 22: name1 = name1[:19] + "..."
            if len(name2) > 22: name2 = name2[:19] + "..."
                
            top_edges.append((val, name1, name2))
            
        # 2. Setup Figure with GridSpec for Legend
        fig = plt.figure(figsize=(24, 10), dpi=300)
        fig.suptitle(f"Dynamic FC Connectome Network ({stage}) - Top Strongest Edges", 
                     fontsize=24, fontweight='bold', y=0.98)
                     
        grid = plt.GridSpec(1, 2, width_ratios=[3, 1])
        ax_conn = fig.add_subplot(grid[0])
                     
        # Use plot_connectome on a glass brain
        # display_mode='lzry' gives Left, Zoomed-In, Right, and 3D perspectives
        # edge_threshold='99.5%' keeps only the top 0.5% of edges to avoid a hairball
        display = plotting.plot_connectome(
            adjacency_matrix=fc_matrix,
            node_coords=node_coords,
            node_color=node_colors,
            node_size=50,
            edge_threshold='99.5%',
            edge_cmap='coolwarm',
            display_mode='lzry',
            figure=fig,
            axes=ax_conn,
            colorbar=True
        )
        
        # 3. Add Top Edges Legend
        ax_leg = fig.add_subplot(grid[1])
        ax_leg.axis('off')
        
        legend_text = "Top 10 Strongest FC Edges:\n"
        legend_text += "-" * 55 + "\n"
        for i, (val, n1, n2) in enumerate(top_edges, 1):
            sign = "+" if val > 0 else "-"
            # Format: 1. Node A <--> Node B (r = +0.85)
            line = f"{i:2d}. {n1} <--> {n2}"
            # Pad to align correlations
            pad = max(42 - len(line), 1)
            line += " " * pad + f"(r = {sign}{abs(val):.2f})\n\n"
            legend_text += line
            
        ax_leg.text(0.05, 0.5, legend_text, fontsize=14, 
                    family='monospace', va='center', ha='left',
                    bbox=dict(boxstyle='round,pad=1.5', facecolor='#f8f9fa', alpha=0.9, edgecolor='#ced4da'))
        
        # Clean up output paths
        out_png = figures_dir / f"highdef_connectome_dfc_{stage.lower()}.png"
        out_pdf = figures_dir / f"highdef_connectome_dfc_{stage.lower()}.pdf"
        
        plt.savefig(out_png, bbox_inches='tight', facecolor='white')
        plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  -> Saved {out_png}")
        print(f"  -> Saved {out_pdf}")

if __name__ == "__main__":
    main()
