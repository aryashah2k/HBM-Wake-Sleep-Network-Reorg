#!/usr/bin/env python3
"""
10_brain_figures_thalamocortical.py — High-definition brain surface diagrams for Thalamocortical coupling.

This script takes the stage-wise thalamocortical coupling results, maps the 
coupling strength (z-scored correlations) to the Yeo 7 networks on the 
Schaefer 2018 atlas, and plots them onto a high-definition fsaverage surface.
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import pandas as pd

from nilearn import datasets, surface, plotting, image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

warnings.filterwarnings("ignore")

def main():
    tc_results_path = config.THALAMOCORTICAL_DIR / "coupling_values.csv"
    if not tc_results_path.exists():
        print(f"File not found: {tc_results_path}")
        return
        
    figures_dir = config.FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("Fetching high-definition fsaverage surface (163k vertices/hemi)...")
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    
    print("Fetching Schaefer 2018 atlas (200 parcels, 2mm)...")
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
    schaefer_img = nib.load(schaefer.maps)
    schaefer_data = schaefer_img.get_fdata()
    
    # Read Thalamocortical coupling data
    df = pd.read_csv(tc_results_path)
    
    # Group by stage and network to get mean coupling values
    mean_coupling = df.groupby(['stage', 'network'])['coupling_z'].mean().reset_index()
    
    for stage in ["Wake", "N1", "N2"]:
        print(f"Processing Thalamocortical Coupling for {stage}...")
        stage_data = mean_coupling[mean_coupling['stage'] == stage]
        
        if stage_data.empty:
            continue
            
        # Create a dictionary mapping Network name to mean coupling
        coupling_dict = dict(zip(stage_data['network'], stage_data['coupling_z']))
        
        # We need a new data array to hold the mapped values
        new_stat_data = np.zeros_like(schaefer_data, dtype=np.float32)
        
        highlighted_regions = []
        
        for i, label_bytes in enumerate(schaefer.labels[1:], start=1):
            try:
                label_str = label_bytes.decode('utf-8')
            except AttributeError:
                label_str = str(label_bytes)
                
            region_mask = (schaefer_data == i)
            parts = label_str.split('_')
            
            if len(parts) >= 3:
                net = parts[2]
                if net in coupling_dict:
                    val = coupling_dict[net]
                    new_stat_data[region_mask] = val
                    if (val, net, i) not in highlighted_regions:
                        highlighted_regions.append((val, net, i))
                        
        if not highlighted_regions:
            print(f"  No regions mapped for {stage}.")
            continue
            
        # Deduplicate the highlighted regions by network name since all parcels in a network get the same val
        unique_networks = {}
        for val, net, i in highlighted_regions:
            unique_networks[net] = val
            
        top_networks = sorted(unique_networks.items(), key=lambda item: abs(item[1]), reverse=True)
            
        new_stat_img = nib.Nifti1Image(new_stat_data, schaefer_img.affine, schaefer_img.header)
        
        # Project to fsaverage pial surface
        texture_lh = surface.vol_to_surf(new_stat_img, fsaverage.pial_left, interpolation='nearest_most_frequent')
        texture_rh = surface.vol_to_surf(new_stat_img, fsaverage.pial_right, interpolation='nearest_most_frequent')
        
        # Figure layout
        fig = plt.figure(figsize=(24, 12), dpi=300)
        fig.suptitle(f"Thalamocortical Coupling ({stage}) - Yeo 7 Networks on >91k fsaverage", 
                     fontsize=24, fontweight='bold', y=0.95)
        
        grid = plt.GridSpec(2, 3, width_ratios=[1, 1, 1.2])
        
        # Fixed limits for better comparison across stages
        vmax = max(np.max(np.abs(texture_lh)), np.max(np.abs(texture_rh)))
        if vmax < 0.1: vmax = 0.5
        
        # Plot Left Hemi
        ax_lh_lat = fig.add_subplot(grid[0, 0], projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage.infl_left, texture_lh, hemi='left', view='lateral',
            colorbar=False, vmax=vmax, bg_map=fsaverage.sulc_left,
            axes=ax_lh_lat, symmetric_cbar=True, cmap='coolwarm'
        )
        ax_lh_lat.set_title("Left Hemisphere - Lateral", fontsize=16)
        
        ax_lh_med = fig.add_subplot(grid[1, 0], projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage.infl_left, texture_lh, hemi='left', view='medial',
            colorbar=False, vmax=vmax, bg_map=fsaverage.sulc_left,
            axes=ax_lh_med, symmetric_cbar=True, cmap='coolwarm'
        )
        ax_lh_med.set_title("Left Hemisphere - Medial", fontsize=16)
        
        # Plot Right Hemi
        ax_rh_lat = fig.add_subplot(grid[0, 1], projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage.infl_right, texture_rh, hemi='right', view='lateral',
            colorbar=False, vmax=vmax, bg_map=fsaverage.sulc_right,
            axes=ax_rh_lat, symmetric_cbar=True, cmap='coolwarm'
        )
        ax_rh_lat.set_title("Right Hemisphere - Lateral", fontsize=16)
        
        ax_rh_med = fig.add_subplot(grid[1, 1], projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage.infl_right, texture_rh, hemi='right', view='medial',
            colorbar=True, vmax=vmax, bg_map=fsaverage.sulc_right,
            axes=ax_rh_med, symmetric_cbar=True, cmap='coolwarm'
        )
        ax_rh_med.set_title("Right Hemisphere - Medial", fontsize=16)
        
        # Legend
        ax_leg = fig.add_subplot(grid[:, 2])
        ax_leg.axis('off')
        
        legend_text = "Thalamocortical Coupling by Yeo Network:\n"
        legend_text += "-" * 50 + "\n"
        for idx, (net, tval) in enumerate(top_networks, 1):
            sign = "+" if tval > 0 else "-"
            full_name = config.YEO7_NETWORK_FULL_NAMES.get(net, net)
            legend_text += f"{idx:2d}. {full_name: <30} (z = {sign}{abs(tval):.2f})\n\n"
            
        ax_leg.text(0.1, 0.5, legend_text, fontsize=16, 
                    family='monospace', va='center', ha='left',
                    bbox=dict(boxstyle='round,pad=1.5', facecolor='#f8f9fa', alpha=0.9, edgecolor='#ced4da'))
                    
        plt.subplots_adjust(wspace=0.1, hspace=0.1, right=0.95, left=0.05, top=0.9, bottom=0.1)
        
        # Save PNG and PDF
        out_png = figures_dir / f"highdef_surface_tc_{stage.lower()}.png"
        out_pdf = figures_dir / f"highdef_surface_tc_{stage.lower()}.pdf"
        
        plt.savefig(out_png, bbox_inches='tight', facecolor='white')
        plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  -> Saved {out_png}")
        print(f"  -> Saved {out_pdf}")

if __name__ == "__main__":
    main()
