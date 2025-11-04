import pandas as pd
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import os
import numpy as np
from matplotlib.patches import Patch

output_dir = "C:/Users/Josephine/PycharmProjects/CoastalRemoteSensing/outputs_wind"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("../data/etmgeg_258.txt", comment="#")
df.columns = df.columns.str.strip()
df = df[['YYYYMMDD', 'DDVEC', 'FG']]

#convert types
df['YYYYMMDD'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d', errors='coerce')
df['DDVEC'] = pd.to_numeric(df['DDVEC'], errors='coerce')
df['FG'] = pd.to_numeric(df['FG'], errors='coerce')
df = df.dropna(subset=['YYYYMMDD', 'DDVEC', 'FG'])

#time components
df['year'] = df['YYYYMMDD'].dt.year
df['month'] = df['YYYYMMDD'].dt.month
df['FG'] = df['FG'] / 10  # convert to m/s
df = df[(df['year'] >= 2017) & (df['year'] <= 2025)]

#periods
def period_label(month):
    if month <= 4:
        return 'early'
    elif month <= 8:
        return 'mid'
    else:
        return 'late'


df['period'] = df['month'].apply(period_label)

#speed bins
bin_width = 2
max_speed = np.ceil(df['FG'].max())
speed_bins = np.arange(0, max_speed + bin_width, bin_width)

#colormap for legend
cmap = plt.get_cmap('viridis')
n_bins = len(speed_bins) - 1
bin_colors = [cmap(i / n_bins) for i in range(n_bins)]
legend_labels = [f"{speed_bins[i]}-{speed_bins[i + 1]}" for i in range(n_bins)]

#max radial
max_radial = 18

#create single 2x2 plot
fig, axes = plt.subplots(2, 2, subplot_kw=dict(projection='windrose'), figsize=(14, 14))
axes = axes.flatten()

#plot 1: full dataset (2017-2025)
ax = axes[0]
ax.bar(
    df['DDVEC'],
    df['FG'],
    bins=speed_bins,
    normed=True,
    opening=0.8,
    edgecolor='white'
)
ax.set_title("Full Period (2017-2025)", fontsize=14, fontweight='bold')
ax.set_yticks(np.linspace(0, max_radial, 5))
ax.set_yticklabels([f"{int(v)}" for v in np.linspace(0, max_radial, 5)])

#plot 2-4: each period across all years
periods = ['early', 'mid', 'late']
period_names = ['Early Season (Jan-Apr)', 'Mid Season (May-Aug)', 'Late Season (Sep-Dec)']

for i, (period, period_name) in enumerate(zip(periods, period_names), start=1):
    ax = axes[i]
    df_period = df[df['period'] == period]

    if df_period.empty:
        ax.set_title(f"{period_name}\n(No data)")
        continue

    ax.bar(
        df_period['DDVEC'],
        df_period['FG'],
        bins=speed_bins,
        normed=True,
        opening=0.8,
        edgecolor='white'
    )
    ax.set_title(period_name, fontsize=14, fontweight='bold')
    ax.set_yticks(np.linspace(0, max_radial, 5))
    ax.set_yticklabels([f"{int(v)}" for v in np.linspace(0, max_radial, 5)])

#legend
patches = [Patch(facecolor=bin_colors[i], edgecolor='white', label=legend_labels[i]) for i in range(n_bins)]
fig.legend(handles=patches, title="Wind speed (m/s)", loc='center right',
           bbox_to_anchor=(1.08, 0.5), fontsize=11)

fig.suptitle("Wind Roses 2017-2025", fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 0.95, 0.96])
plt.savefig(f"{output_dir}/windrose_2017_2025_combined.png", dpi=300, bbox_inches='tight')
plt.close()

#statistics function
def mean_wind_direction(directions, speeds):
    #convert degrees to radians
    directions_rad = np.deg2rad(directions)

    #wind vector components
    u = speeds * np.sin(directions_rad)
    v = speeds * np.cos(directions_rad)

    #mean components
    u_mean = np.mean(u)
    v_mean = np.mean(v)

    #convert back to direction degrees
    mean_dir = np.rad2deg(np.arctan2(u_mean, v_mean))

    #convert to 0–360° meteorological convention
    if mean_dir < 0:
        mean_dir += 360

    return mean_dir


#compute stats
stats = []

for year in sorted(df['year'].unique()):
    for period in ['early', 'mid', 'late']:
        subset = df[(df['year'] == year) & (df['period'] == period)]

        if subset.empty:
            continue

        avg_speed = subset['FG'].mean()
        max_speed = subset['FG'].max()
        avg_dir = mean_wind_direction(subset['DDVEC'], subset['FG'])

        stats.append([year, period, avg_speed, max_speed, avg_dir])

#convert to DataFrame
stats_df = pd.DataFrame(stats, columns=['year', 'period', 'avg_speed_mps', 'max_speed_mps', 'avg_direction_deg'])

#save to CSV
stats_csv_path = os.path.join(output_dir, "wind_stats_per_period.csv")
stats_df.to_csv(stats_csv_path, index=False)

