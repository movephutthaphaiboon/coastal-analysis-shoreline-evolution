import rasterio
import numpy as np
import os
import re
from glob import glob
import matplotlib.pyplot as plt
from rasterio.windows import from_bounds

data_dir = "../data"
output_dir = "../outputs_ndvi"
os.makedirs(output_dir, exist_ok=True)

boxes = {
    'NW': (152700, 512000, 154900, 513450),
    'SW': (153650, 508300, 155950, 510400),
    'SE': (155850, 508900, 156350, 510900),
}

def get_year(filename):
    match = re.search(r"(19|20)\d{2}", filename)
    return match.group(0) if match else "unknown"


def get_season(filename):
    match = re.search(r"\d{4}-(\d{2})-\d{2}", filename)
    if not match:
        return "unknown"

    month = int(match.group(1))
    if 1 <= month <= 3:
        return "early"
    elif 4 <= month <= 9:
        return "mid"
    elif 10 <= month <= 12:
        return "late"
    return "unknown"


#get all TIFF files
tiff_files = glob(os.path.join(data_dir, "*.tif"))

#organize files by year (only late season)
images_by_year = {}

for tiff_path in tiff_files:
    file_name = os.path.basename(tiff_path)
    year = get_year(file_name)
    season = get_season(file_name)

    #only process late season images
    if season == "late" and year != "unknown":
        images_by_year[year] = tiff_path

#process each year's late season image
years_sorted = sorted(images_by_year.keys())

#create 3x3 grid of full area images
n_years = len(years_sorted)
n_cols = 3
n_rows = (n_years + n_cols - 1) // n_cols  # Ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
axes = axes.flatten() if n_years > 1 else [axes]

vmin, vmax = -1, 1

for idx, year in enumerate(years_sorted):
    tiff_path = images_by_year[year]

    with rasterio.open(tiff_path) as src:
        red = src.read(3)
        nir = src.read(4)
        ndvi = (nir.astype(float) - red.astype(float)) / (nir + red + 1e-10)

        im = axes[idx].imshow(ndvi, cmap='RdYlGn', vmin=vmin, vmax=vmax)
        axes[idx].set_title(f"{year}", fontsize=14)
        axes[idx].axis('off')

#hide empty subplots
for idx in range(n_years, len(axes)):
    axes[idx].axis('off')

#add shared colorbar
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='NDVI')

fig.suptitle("Late Season NDVI - Full Area", fontsize=18)

output_path = os.path.join(output_dir, "all_years_full_grid.png")
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

#process individual box images for each year
for year in years_sorted:
    tiff_path = images_by_year[year]

    with rasterio.open(tiff_path) as src:
        for box_name, coords in boxes.items():
            minx, miny, maxx, maxy = coords

            #create window from bounds
            window = from_bounds(minx, miny, maxx, maxy, src.transform)

            #read the bands for this window
            red = src.read(3, window=window)
            nir = src.read(4, window=window)

            #calculate NDVI
            ndvi = (nir.astype(float) - red.astype(float)) / (nir + red + 1e-10)

            #create individual figure for this box
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(ndvi, cmap='RdYlGn', vmin=vmin, vmax=vmax)
            ax.set_title(f"{box_name}")
            ax.axis('off')

            #add colorbar
            plt.colorbar(im, ax=ax, label='NDVI', fraction=0.046, pad=0.04)

            #save individual box image
            output_path = os.path.join(output_dir, f"{year}_{box_name}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()

#create 3x3 grids for each box across all years
for box_name, coords in boxes.items():
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_years > 1 else [axes]

    for idx, year in enumerate(years_sorted):
        tiff_path = images_by_year[year]

        with rasterio.open(tiff_path) as src:
            minx, miny, maxx, maxy = coords
            window = from_bounds(minx, miny, maxx, maxy, src.transform)

            red = src.read(3, window=window)
            nir = src.read(4, window=window)
            ndvi = (nir.astype(float) - red.astype(float)) / (nir + red + 1e-10)

            im = axes[idx].imshow(ndvi, cmap='RdYlGn', vmin=vmin, vmax=vmax)
            axes[idx].set_title(f"{year}", fontsize=14)
            axes[idx].axis('off')

    #hide empty subplots
    for idx in range(n_years, len(axes)):
        axes[idx].axis('off')

    #add shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='NDVI')

    fig.suptitle(f"Late Season NDVI - {box_name}", fontsize=18)

    output_path = os.path.join(output_dir, f"all_years_{box_name}_grid.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()