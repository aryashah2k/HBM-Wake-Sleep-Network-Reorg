#!/usr/bin/env python3
"""
09_brain_figures.py — High-definition brain surface diagrams for GLM results.

This script takes the group-level GLM results, maps the significant clusters 
onto the Schaefer 2018 (200 parcels, 7 networks) atlas (extrapolating to the 
complete regions as requested), and plots them onto a high-definition 
fsaverage surface (163,842 vertices per hemisphere, satisfying >91k requirement).

Proper labelling is generated as a text legend alongside the brain plots.
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
    glm_group_dir = config.GLM_DIR / "group"
    figures_dir = config.FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("Fetching high-definition fsaverage surface (163k vertices/hemi)...")
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    
    print("Fetching Schaefer 2018 atlas (200 parcels, 2mm)...")
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
    schaefer_img = nib.load(schaefer.maps)
    schaefer_data = schaefer_img.get_fdata()
    
    contrasts = [
        "Wake_gt_N2", 
        "N2_gt_Wake", 
        "Wake_gt_N1", 
        "N1_gt_N2", 
        "Linear_Wake_to_N2"
    ]
    
    for contrast in contrasts:
        thresholded_path = glm_group_dir / f"{contrast}_thresholded.nii.gz"
        if not thresholded_path.exists():
            continue
            
        print(f"Processing {contrast}...")
        stat_img = nib.load(str(thresholded_path))
        stat_data = stat_img.get_fdata()
        
        # Ensure the Schaefer atlas matches the exact shape and affine of our GLM output
        resampled_schaefer_img = image.resample_to_img(schaefer_img, stat_img, interpolation='nearest')
        resampled_schaefer_data = resampled_schaefer_img.get_fdata()
        
        new_stat_data = np.zeros_like(resampled_schaefer_data, dtype=np.float32)
        highlighted_regions = []
        
        # 1. Extrapolate significant clusters to their complete Schaefer parcels
        # skip the first label ('Background')
        for i, label in enumerate(schaefer.labels[1:], start=1):
            region_mask = (resampled_schaefer_data == i)
            overlap = stat_data[region_mask]
            non_zero = overlap[overlap != 0]
            
            # If > 10 voxels in this parcel overlap with a significant cluster,
            # extrapolate the mean t-statistic of the overlap to the whole parcel.
            if len(non_zero) >= 10:
                mean_t = np.mean(non_zero)
                new_stat_data[region_mask] = mean_t
                readable_name = clean_label_name(label)
                highlighted_regions.append((mean_t, readable_name, i))
        
        if not highlighted_regions:
            print(f"  No significant overlap with parcellation for {contrast}.")
            continue
            
        # Sort regions by absolute t-value to rank the top ones for labelling
        highlighted_regions.sort(key=lambda x: abs(x[0]), reverse=True)
        top_regions = highlighted_regions[:10]  # Show top 10 regions
        
        new_stat_img = nib.Nifti1Image(new_stat_data, resampled_schaefer_img.affine, resampled_schaefer_img.header)
        
        # 2. Project the parcellated map to high-res fsaverage surface
        # Use pial surface for best visual representation of cortical folds
        texture_lh = surface.vol_to_surf(new_stat_img, fsaverage.pial_left, interpolation='nearest_most_frequent')
        texture_rh = surface.vol_to_surf(new_stat_img, fsaverage.pial_right, interpolation='nearest_most_frequent')
        
        # Calculate centroids on the surface for the top regions
        lh_verts, _ = surface.load_surf_mesh(fsaverage.infl_left)
        rh_verts, _ = surface.load_surf_mesh(fsaverage.infl_right)
        
        label_positions = {'left': [], 'right': []}
        
        for idx, (tval, name, region_val) in enumerate(top_regions, 1):
            # Create a mask for just this one parcel to find its surface map
            single_region_data = np.zeros_like(resampled_schaefer_data)
            single_region_data[resampled_schaefer_data == region_val] = 1.0
            single_region_img = nib.Nifti1Image(single_region_data, resampled_schaefer_img.affine, resampled_schaefer_img.header)
            
            # Check which hemisphere it's mostly in
            if name.startswith('LH '):
                surf_map = surface.vol_to_surf(single_region_img, fsaverage.pial_left, interpolation='nearest_most_frequent')
                verts = lh_verts
                hemi = 'left'
            else:
                surf_map = surface.vol_to_surf(single_region_img, fsaverage.pial_right, interpolation='nearest_most_frequent')
                verts = rh_verts
                hemi = 'right'
                
            # Find vertices belonging to this region
            region_verts_idx = np.where(surf_map > 0)[0]
            if len(region_verts_idx) > 0:
                # Get median coordinate of these vertices to be robust to strange shapes
                centroid = np.median(verts[region_verts_idx], axis=0)
                
                # Push the label slightly outwards along the normal vector from origin for visibility
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid + (centroid / norm) * 5.0
                    
                label_positions[hemi].append((idx, centroid))
        
        # 3. Create high-def Figure layout
        fig = plt.figure(figsize=(24, 12), dpi=300)
        fig.suptitle(f"{contrast.replace('_', ' ')} - Parcels Extrapolated on >91k fsaverage", 
                     fontsize=24, fontweight='bold', y=0.95)
        
        grid = plt.GridSpec(2, 3, width_ratios=[1, 1, 1.2])
        
        # Determine strict colormap bounds based on data
        vmax = max(np.max(np.abs(texture_lh)), np.max(np.abs(texture_rh)))
        if vmax == 0: vmax = 1.0
        
        def plot_labels(ax, hemi):
            for idx, pt in label_positions[hemi]:
                # Plot the label in 3D space
                ax.text(pt[0], pt[1], pt[2], str(idx), 
                        color='white', fontsize=12, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='circle,pad=0.2', fc='black', ec='white', alpha=0.8))
        
        # Plot Left Hemi
        ax_lh_lat = fig.add_subplot(grid[0, 0], projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage.infl_left, texture_lh, hemi='left', view='lateral',
            colorbar=False, vmax=vmax, bg_map=fsaverage.sulc_left,
            axes=ax_lh_lat, symmetric_cbar=True
        )
        ax_lh_lat.set_title("Left Hemisphere - Lateral", fontsize=16)
        
        ax_lh_med = fig.add_subplot(grid[1, 0], projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage.infl_left, texture_lh, hemi='left', view='medial',
            colorbar=False, vmax=vmax, bg_map=fsaverage.sulc_left,
            axes=ax_lh_med, symmetric_cbar=True
        )
        ax_lh_med.set_title("Left Hemisphere - Medial", fontsize=16)
        
        # Plot Right Hemi
        ax_rh_lat = fig.add_subplot(grid[0, 1], projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage.infl_right, texture_rh, hemi='right', view='lateral',
            colorbar=False, vmax=vmax, bg_map=fsaverage.sulc_right,
            axes=ax_rh_lat, symmetric_cbar=True
        )
        ax_rh_lat.set_title("Right Hemisphere - Lateral", fontsize=16)
        
        ax_rh_med = fig.add_subplot(grid[1, 1], projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage.infl_right, texture_rh, hemi='right', view='medial',
            colorbar=True, vmax=vmax, bg_map=fsaverage.sulc_right,
            axes=ax_rh_med, symmetric_cbar=True
        )
        ax_rh_med.set_title("Right Hemisphere - Medial", fontsize=16)
        
        # 4. Add Region Legend Annotations
        ax_leg = fig.add_subplot(grid[:, 2])
        ax_leg.axis('off')
        
        legend_text = "Top 10 Extrapolated Regions (Schaefer 2018):\n"
        legend_text += "-" * 55 + "\n"
        for idx, (tval, name, _) in enumerate(top_regions, 1):
            sign = "+" if tval > 0 else "-"
            # Shorten very long names slightly if needed
            if len(name) > 40: name = name[:37] + "..."
            legend_text += f"{idx:2d}. {name: <40} (t = {sign}{abs(tval):.2f})\n\n"
            
        # Draw legend centered vertically to avoid overlap with boundaries
        ax_leg.text(0.1, 0.5, legend_text, fontsize=16, 
                    family='monospace', va='center', ha='left',
                    bbox=dict(boxstyle='round,pad=1.5', facecolor='#f8f9fa', alpha=0.9, edgecolor='#ced4da'))
                    
        out_path_png = figures_dir / f"highdef_surface_{contrast}.png"
        out_path_pdf = figures_dir / f"highdef_surface_{contrast}.pdf"
        plt.subplots_adjust(wspace=0.1, hspace=0.1, right=0.95, left=0.05, top=0.9, bottom=0.1)
        plt.savefig(out_path_png, bbox_inches='tight', facecolor='white')
        plt.savefig(out_path_pdf, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  -> Saved {out_path_png}")
        print(f"  -> Saved {out_path_pdf}")

if __name__ == "__main__":
    main()
