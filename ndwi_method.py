import os
import glob
import re
import numpy as np
import rasterio
from shapely.geometry import LineString, box as shapely_box
import geopandas as gpd
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage import measure
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

#configuration
data_dir = "C:/Users/Josephine/PycharmProjects/CoastalRemoteSensing/data"
output_dir = "C:/Users/Josephine/PycharmProjects/CoastalRemoteSensing/new_outputs"

boxes = {
    'NW': (152700, 512000, 154900, 513450),
    'SW': (153650, 508300, 155950, 510400),
    'SE': (155850, 508900, 156350, 510900),
}
os.makedirs(output_dir, exist_ok=True)

#functions
def compute_ndwi(tiff_path):
    with rasterio.open(tiff_path) as src:
        img = src.read().astype('float32')
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        #create mask for invalid pixels
        if nodata is not None:
            mask = np.any(img == nodata, axis=0)
        else:
            mask = np.all(img <= 0, axis=0)

    #extract bands
    green = img[1]
    nir = img[3]

    #compute NDWI
    ndwi = (green - nir) / (green + nir + 1e-8)
    ndwi[mask] = np.nan

    return ndwi, transform, crs

def extract_coastline(ndwi, transform, crs, min_length=20):
    #get valid pixels
    valid = np.isfinite(ndwi)
    if not np.any(valid):
        return gpd.GeoDataFrame()

    #threshold to separate water from land
    threshold = threshold_otsu(ndwi[valid])
    water_mask = ndwi > threshold

    #clean up small noise
    water_mask = remove_small_objects(water_mask, min_size=100)
    water_mask = remove_small_holes(water_mask, area_threshold=50)

    #extract contours at water-land boundary
    contours = measure.find_contours(water_mask.astype(float), level=0.5)

    #convert contours to geographic coordinates
    geometries = []
    for contour in contours:
        if len(contour) < 3:
            continue

        #convert from pixel to map coordinates
        coords = []
        for row, col in contour:
            x, y = transform * (col, row)
            coords.append((x, y))

        #create LineString and filter by length
        if len(coords) >= 2:
            line = LineString(coords)
            if line.length > min_length:
                geometries.append(line)

    if not geometries:
        return gpd.GeoDataFrame()

    #create GeoDataFrame and simplify
    gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=2.0, preserve_topology=True)

    #validate coordinates are reasonable
    bounds = gdf.total_bounds
    if bounds[0] < -1e6 or bounds[0] > 1e6 or bounds[2] < -1e6 or bounds[2] > 1e6:
        print(f"Warning: Suspicious coordinates detected: {bounds}")

    return gdf

def clip_to_box(gdf, box):
    if gdf.empty:
        return gdf
    bbox_poly = shapely_box(*box)
    return gdf[gdf.intersects(bbox_poly)].copy()

def calculate_land_area_in_box(ndwi, transform, box):
    #crop NDWI to box
    from rasterio.windows import from_bounds
    minx, miny, maxx, maxy = box

    window = from_bounds(minx, miny, maxx, maxy, transform=transform)
    row_start = max(int(window.row_off), 0)
    row_stop = min(int(window.row_off + window.height), ndwi.shape[0])
    col_start = max(int(window.col_off), 0)
    col_stop = min(int(window.col_off + window.width), ndwi.shape[1])

    if row_stop <= row_start or col_stop <= col_start:
        return np.nan

    cropped_ndwi = ndwi[row_start:row_stop, col_start:col_stop]

    #get valid pixels
    valid = np.isfinite(cropped_ndwi)
    if not np.any(valid):
        return np.nan

    #threshold to identify water
    threshold = threshold_otsu(cropped_ndwi[valid])
    land_mask = cropped_ndwi <= threshold
    land_mask = land_mask & valid

    #calculate area
    pixel_area = abs(transform.a * transform.e)
    land_area = np.sum(land_mask) * pixel_area

    return land_area

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

#plotting fucnitons
def plot_coastline_on_image(tiff_path, gdf, output_dir):
    name = os.path.splitext(os.path.basename(tiff_path))[0]
    year = get_year(tiff_path)
    season = get_season(tiff_path)

    with rasterio.open(tiff_path) as src:
        #read RGB bands
        red = src.read(1)
        green = src.read(2)
        blue = src.read(3)

        #stack for RGB display
        rgb = np.dstack([red, green, blue])

        #normalize to 0-1 range for display
        rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)

        #get extent
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    #create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(rgb, extent=extent)

    #plot coastline in red
    if not gdf.empty:
        gdf.plot(ax=ax, color='red', linewidth=0.5, alpha=0.9, label='Coastline')
        ax.legend(loc='upper right', fontsize=12)

    ax.set_title(f"Coastlines - {season} {year}", fontsize=16)
    ax.axis('off')
    out_path = os.path.join(output_dir, f"coastline_overlay_{name}.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

    return out_path

def plot_seasonal_coastlines_full(gdf_list, output_dir):
    seasons = {"early": [], "mid": [], "late": []}

    #group by season
    for gdf, name in gdf_list:
        if gdf.empty:
            continue
        season = get_season(name)
        year = get_year(name)
        if season in seasons:
            seasons[season].append((gdf, year))

    #plot each season
    for season, items in seasons.items():
        if not items:
            print(f"No data for {season} season")
            continue

        #sort by year and assign colors
        items_sorted = sorted(items, key=lambda x: x[1])
        print(f"  {season}: {len(items_sorted)} years found - {[y for _, y in items_sorted]}")

        #calculate bounds from all geometries and check for anomalies
        all_bounds = []
        valid_items = []
        for gdf, year in items_sorted:
            bounds = gdf.total_bounds

            #check for reasonable coordinate ranges (Dutch RD system: ~0-300000)
            if bounds[0] < 0 or bounds[0] > 300000 or bounds[2] < 0 or bounds[2] > 300000:
                print(f"Skipping {year} - invalid coordinates: {bounds}")
                continue

            all_bounds.append(bounds)
            valid_items.append((gdf, year))

        if not valid_items:
            print(f"No valid coastlines for {season} season after filtering")
            continue

        items_sorted = valid_items
        all_bounds = np.array(all_bounds)
        minx = all_bounds[:, 0].min()
        miny = all_bounds[:, 1].min()
        maxx = all_bounds[:, 2].max()
        maxy = all_bounds[:, 3].max()


        #add 5% margin
        x_margin = (maxx - minx) * 0.05
        y_margin = (maxy - miny) * 0.05

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(minx - x_margin, maxx + x_margin)
        ax.set_ylim(miny - y_margin, maxy + y_margin)
        ax.set_aspect('equal')
        ax.set_title(f"Coastlines - {season.capitalize()} Season (All Years)", fontsize=16)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.3)

        cmap = plt.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(len(items_sorted))]

        #plot each year
        legend_elements = []
        for idx, (gdf, year) in enumerate(items_sorted):
            gdf.plot(ax=ax, linewidth=1.0, color=colors[idx], alpha=0.8)
            legend_elements.append(Line2D([0], [0], color=colors[idx], lw=2, label=year))

        ax.legend(handles=legend_elements, title="Year", loc='best')

        out_path = os.path.join(output_dir, f"seasonal_full_{season}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {out_path}")


def plot_seasonal_coastlines_per_box(gdf_list, boxes, output_dir):
    seasons = ["early", "mid", "late"]

    for box_name, box in boxes.items():
        minx, miny, maxx, maxy = box

        #group by season for this box
        seasonal = {s: [] for s in seasons}
        for gdf, name in gdf_list:
            if gdf.empty:
                continue

            season = get_season(name)
            if season not in seasons:
                continue

            #clip to box
            gdf_box = clip_to_box(gdf, box)
            if not gdf_box.empty:
                year = get_year(name)
                seasonal[season].append((gdf_box, year))

        #plot each season for this box
        for season, items in seasonal.items():
            if not items:
                continue

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            ax.set_aspect('equal')
            ax.set_title(f"Box {box_name} - {season.capitalize()} Season", fontsize=14)

            #remove axes, ticks, and grid
            ax.set_axis_off()

            #sort and color by year
            items_sorted = sorted(items, key=lambda x: x[1])
            cmap = plt.get_cmap('tab10')
            colors = [cmap(i % 10) for i in range(len(items_sorted))]

            legend_elements = []
            for idx, (gdf_item, year) in enumerate(items_sorted):
                gdf_item.plot(ax=ax, linewidth=1, color=colors[idx], alpha=0.9)
                legend_elements.append(Line2D([0], [0], color=colors[idx], lw=2, label=str(year)))

            #legend
            fig.subplots_adjust(right=0.80)
            ax.legend(handles=legend_elements,
                      loc="upper left",
                      bbox_to_anchor=(1.02, 1),
                      borderaxespad=0.)

            out_path = os.path.join(output_dir, f"seasonal_box{box_name}_{season}.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()

def analyze_land_area_per_box(tif_files, boxes, output_dir):

    for box_name, box in boxes.items():

        #store results: {year: {season: area}}
        results = {}

        for tiff in tif_files:
            name = os.path.splitext(os.path.basename(tiff))[0]
            year = get_year(name)
            season = get_season(name)

            if season == "unknown":
                continue

            try:
                #compute NDWI
                ndwi, transform, crs = compute_ndwi(tiff)

                #calculate land area in box
                land_area = calculate_land_area_in_box(ndwi, transform, box)

                if np.isnan(land_area):
                    continue

                #store result
                if year not in results:
                    results[year] = {}
                results[year][season] = land_area

            except Exception as e:
                print(f"Error processing {name}: {e}")
                continue

        if not results:
            print(f"No valid results for Box {box_name}")
            continue

        #convert to DataFrame for easier plotting
        df_data = []
        for year, seasons in results.items():
            for season, area in seasons.items():
                df_data.append({
                    'year': int(year),
                    'season': season,
                    'land_area_m2': area
                })

        df = pd.DataFrame(df_data).sort_values(['year', 'season'])

        #save to CSV
        csv_path = os.path.join(output_dir, f"land_area_box{box_name}.csv")
        df.to_csv(csv_path, index=False)

        #plot land area over time (by season)
        fig, ax = plt.subplots(figsize=(16, 6))

        season_order = ['early', 'mid', 'late']
        colors = {'early': 'blue', 'mid': 'green', 'late': 'red'}
        markers = {'early': 'o', 'mid': 's', 'late': '^'}

        for season in season_order:
            season_data = df[df['season'] == season].sort_values('year')
            if not season_data.empty:
                ax.plot(season_data['year'], season_data['land_area_m2'],
                        marker=markers[season], color=colors[season],
                        linewidth=2, markersize=8, label=season.capitalize(), alpha=0.8)

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Land Area (m²)', fontsize=12)
        ax.set_title(f'Box {box_name} - Land Area Over Time', fontsize=14)
        ax.legend(title='Season')
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(output_dir, f"land_area_box{box_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        #print summary statistics
        print(f"\n  Land Area Summary (m²):")
        for season in season_order:
            season_data = df[df['season'] == season]
            if not season_data.empty:
                mean_area = season_data['land_area_m2'].mean()
                min_area = season_data['land_area_m2'].min()
                max_area = season_data['land_area_m2'].max()
                change = max_area - min_area
                print(
                    f"    {season.capitalize():5s}: mean={mean_area:.2f}, min={min_area:.2f}, max={max_area:.2f}, change={change:.2f}")


def main():
    #find all GeoTIFF files
    tif_files = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
    if not tif_files:
        print("No TIF files found in data directory")
        return

    #process each image
    gdf_list = []
    for tiff in tqdm(tif_files, desc="Processing"):
        name = os.path.splitext(os.path.basename(tiff))[0]

        try:
            #compute NDWI
            ndwi, transform, crs = compute_ndwi(tiff)

            #extract coastline
            gdf = extract_coastline(ndwi, transform, crs)

            if not gdf.empty:
                gdf_list.append((gdf, name))

                #create overlay image with coastline
                plot_coastline_on_image(tiff, gdf, OUTPUT_DIR)
            else:
                print(f"No coastline detected for {name}")

        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue

    #generate plots
    plot_seasonal_coastlines_full(gdf_list, output_dir)
    plot_seasonal_coastlines_per_box(gdf_list, boxes, output_dir)

    #analyze land area changes
    analyze_land_area_per_box(tif_files, boxes, output_dir)

    print("Done! All outputs saved to:", output_dir)


if __name__ == "__main__":
    main()
