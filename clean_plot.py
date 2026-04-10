import os

# 1. Set the OS variables completely BEFORE any geospatial imports
os.environ["PROJ_DATA"] = "/home/yelnat/miniconda3/envs/main/share/proj"
os.environ["PROJ_LIB"] = "/home/yelnat/miniconda3/envs/main/share/proj"

import pyproj

# 2. Explicitly force pyproj's internal engine to use the Conda path
pyproj.datadir.set_data_dir("/home/yelnat/miniconda3/envs/main/share/proj")

import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ... [rest of the script remains exactly the same]
# 1. File Paths
true_file = 'databases/spitz/true.geojson'
reconstructed_files = [
    'databases/spitz/LAMA.geojson', 
    'databases/spitz/even_less/spitz_prob100.0_uniform_even_less.json.geojson', 
    'databases/spitz/remin/spitz_prob100.0_uniform_classic.json.geojson'
]

# Load base data
true_gdf = gpd.read_file(true_file)

def generate_map(overlay_gdf=None, filename_out="output.pdf"):
    # 1. Rectangle figure size to prevent stretching
    fig = plt.figure(figsize=(8, 6))
    
    # Center the projection precisely on North Germany to eliminate distortion
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.TransverseMercator(central_longitude=9.0, central_latitude=52.5))

    # --- FLAT UI BACKGROUND ---
    # Very subtle, flat colors for land and water
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#f4f6f8', zorder=1)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#e2e8ed', zorder=1)
    
    # Soft, minimal borders
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.5, edgecolor='#d1d8e0', zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5, edgecolor='#d1d8e0', zorder=2)
    # ----------------------------

    # Plot True Points (Larger radius, Flat UI)
    true_gdf.plot(
        ax=ax, 
        transform=ccrs.PlateCarree(), 
        color='#005f9e', # Deep flat blue
        markersize=150,  # Generously larger radius
        linewidth=0,     # NO border stroke
        alpha=0.85,      # Slight transparency for a modern look
        label='True Points',
        zorder=5
    )

    # Plot Reconstructed Points (Flat UI)
    if overlay_gdf is not None:
        overlay_gdf.plot(
            ax=ax, 
            transform=ccrs.PlateCarree(), 
            color='#ff8c00', # Vibrant flat orange
            markersize=60,   # Noticeably smaller than True Points
            linewidth=0,     # NO border stroke
            alpha=0.8, 
            label='Reconstructed',
            zorder=6
        )

    # --- NORTH GERMANY BOUNDING BOX ---
    # [Lon Min, Lon Max, Lat Min, Lat Max]
    # Tightly frames the area from the Netherlands border past Hannover
    ax.set_extent([6.0, 12.0, 51.0, 54.0], crs=ccrs.PlateCarree())

    # Flat UI Legend
    plt.legend(
        loc='upper right', 
        framealpha=1.0,      # Solid background
        edgecolor='none',    # No border on the legend box
        facecolor='#ffffff', 
        fancybox=False       # Sharp corners
    )

    plt.savefig(filename_out, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
# ==========================================
# PLOT 1: True Only
# ==========================================
print("Generating Plot 1 (True Only)...")
generate_map(filename_out="plot_1_true_only.pdf")

# ==========================================
# PLOTS 2, 3, 4: Overlays
# ==========================================
for i, filename in enumerate(reconstructed_files, start=2):
    print(f"Generating Plot {i} ({filename} overlay)...")
    recon_gdf = gpd.read_file(filename)
    
    base_name = os.path.basename(filename) 
    safe_name = base_name.replace('.geojson', '').replace('.json', '')
    
    generate_map(overlay_gdf=recon_gdf, filename_out=f"plot_{i}_{safe_name}_overlay.pdf")

print("All Cartopy PDFs generated successfully!")
