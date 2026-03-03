#!/usr/bin/env python3
"""
11_brain_figures_dfc.py — High-definition brain surface diagrams for DFC.

This script takes the stage-wise mean Dynamic Functional Connectivity (dFC) matrices,
calculates the 'node strength' (average connectivity) for each of the 200 Schaefer 
parcels, and plots them onto a high-definition fsaverage surface.
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from nilearn import datasets, surface, plotting

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
    
    print("Fetching high-definition fsaverage surface (163k vertices/hemi)...")
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    
    print("Fetching Schaefer 2018 atlas (200 parcels, 2mm)...")
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
    schaefer_img = nib.load(schaefer.maps)
    schaefer_data = schaefer_img.get_fdata()
    
    # Read mean FC data
    dfc_data = np.load(str(dfc_results_path), allow_pickle=True)
    
    for stage in ["Wake", "N1", "N2"]:
        key = f"fc_{stage}"
        if key not in dfc_data:
            print(f"Skipping {stage}, no data found in npz.")
            continue
            
        print(f"Processing DFC Node Strength for {stage}...")
        fc_matrix = dfc_data[key]
        
        # Calculate node strength (mean of absolute functional connectivity to all other nodes)
        np.fill_diagonal(fc_matrix, 0)
        node_strength = np.mean(np.abs(fc_matrix), axis=1)
        
        # We need a new data array to hold the mapped values
        new_stat_data = np.zeros_like(schaefer_data, dtype=np.float32)
        highlighted_regions = []
        
        # Map node strength back onto the Schaefer parcels
        for i, label_bytes in enumerate(schaefer.labels[1:], start=1):
            region_mask = (schaefer_data == i)
            # node_strength array is 0-indexed, atlas regions are 1-indexed
            val = node_strength[i - 1] 
            
            new_stat_data[region_mask] = val
            readable_name = clean_label_name(label_bytes)
            highlighted_regions.append((val, readable_name, i))
            
        if not highlighted_regions:
            print(f"  No regions mapped for {stage}.")
            continue
            
        # Sort regions by value to rank the top ones for labelling
        highlighted_regions.sort(key=lambda x: x[0], reverse=True)
        top_regions = highlighted_regions[:10]  # Show top 10 regions
            
        new_stat_img = nib.Nifti1Image(new_stat_data, schaefer_img.affine, schaefer_img.header)
        
        # Project to fsaverage pial surface
        texture_lh = surface.vol_to_surf(new_stat_img, fsaverage.pial_left, interpolation='nearest_most_frequent')
        texture_rh = surface.vol_to_surf(new_stat_img, fsaverage.pial_right, interpolation='nearest_most_frequent')
        
        # Figure layout
        fig = plt.figure(figsize=(24, 12), dpi=300)
        fig.suptitle(f"Dynamic FC Node Strength ({stage}) - Parcels Extrapolated on >91k fsaverage", 
                     fontsize=24, fontweight='bold', y=0.95)
        
        grid = plt.GridSpec(2, 3, width_ratios=[1, 1, 1.2])
        
        # Data is strictly positive since we took absolute mean, adjust colormap bounds
        vmax = np.max([np.max(texture_lh), np.max(texture_rh)])
        vmin = np.min([np.min(texture_lh[texture_lh > 0]), np.min(texture_rh[texture_rh > 0])])
        if vmax <= vmin: vmax = vmin + 0.1
        
        # Plot Left Hemi
        ax_lh_lat = fig.add_subplot(grid[0, 0], projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage.infl_left, texture_lh, hemi='left', view='lateral',
            colorbar=False, vmax=vmax, bg_map=fsaverage.sulc_left,
            axes=ax_lh_lat, symmetric_cbar=False, cmap='YlOrRd'
        )
        ax_lh_lat.set_title("Left Hemisphere - Lateral", fontsize=16)
        
        ax_lh_med = fig.add_subplot(grid[1, 0], projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage.infl_left, texture_lh, hemi='left', view='medial',
            colorbar=False, vmax=vmax, bg_map=fsaverage.sulc_left,
            axes=ax_lh_med, symmetric_cbar=False, cmap='YlOrRd'
        )
        ax_lh_med.set_title("Left Hemisphere - Medial", fontsize=16)
        
        # Plot Right Hemi
        ax_rh_lat = fig.add_subplot(grid[0, 1], projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage.infl_right, texture_rh, hemi='right', view='lateral',
            colorbar=False, vmax=vmax, bg_map=fsaverage.sulc_right,
            axes=ax_rh_lat, symmetric_cbar=False, cmap='YlOrRd'
        )
        ax_rh_lat.set_title("Right Hemisphere - Lateral", fontsize=16)
        
        ax_rh_med = fig.add_subplot(grid[1, 1], projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage.infl_right, texture_rh, hemi='right', view='medial',
            colorbar=True, vmax=vmax, bg_map=fsaverage.sulc_right,
            axes=ax_rh_med, symmetric_cbar=False, cmap='YlOrRd'
        )
        ax_rh_med.set_title("Right Hemisphere - Medial", fontsize=16)
        
        # Legend
        ax_leg = fig.add_subplot(grid[:, 2])
        ax_leg.axis('off')
        
        legend_text = "Top 10 Regions by Node Strength (Schaefer 2018):\n"
        legend_text += "-" * 55 + "\n"
        for idx, (val, name, _) in enumerate(top_regions, 1):
            if len(name) > 40: name = name[:37] + "..."
            legend_text += f"{idx:2d}. {name: <40} (mean |FC| = {val:.2f})\n\n"
            
        ax_leg.text(0.1, 0.5, legend_text, fontsize=16, 
                    family='monospace', va='center', ha='left',
                    bbox=dict(boxstyle='round,pad=1.5', facecolor='#f8f9fa', alpha=0.9, edgecolor='#ced4da'))
                    
        plt.subplots_adjust(wspace=0.1, hspace=0.1, right=0.95, left=0.05, top=0.9, bottom=0.1)
        
        # Save PNG and PDF
        out_png = figures_dir / f"highdef_surface_dfc_{stage.lower()}.png"
        out_pdf = figures_dir / f"highdef_surface_dfc_{stage.lower()}.pdf"
        
        plt.savefig(out_png, bbox_inches='tight', facecolor='white')
        plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  -> Saved {out_png}")
        print(f"  -> Saved {out_pdf}")

if __name__ == "__main__":
    main()
