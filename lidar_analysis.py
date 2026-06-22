import laspy
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ============================================================
# STEP 1: Load LiDAR Data
# ============================================================
print("Loading LiDAR file...")
las = laspy.read("points.laz")

x = np.array(las.x)
y = np.array(las.y)
z = np.array(las.z)
classification = np.array(las.classification)

print(f"Total points: {len(x):,}")

# ============================================================
# STEP 2: Build CHM (Canopy Height Model)
# ============================================================
resolution = 1.0
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

cols = int((x_max - x_min) / resolution) + 1
rows = int((y_max - y_min) / resolution) + 1

# Digital Terrain Model (ground)
ground_mask = classification == 2
dtm = np.full((rows, cols), np.nan)
for xi, yi, zi in zip(x[ground_mask], y[ground_mask], z[ground_mask]):
    col = int((xi - x_min) / resolution)
    row = int((yi - y_min) / resolution)
    if np.isnan(dtm[row, col]) or zi < dtm[row, col]:
        dtm[row, col] = zi

# Digital Surface Model (highest point)
dsm = np.full((rows, cols), np.nan)
for xi, yi, zi in zip(x, y, z):
    col = int((xi - x_min) / resolution)
    row = int((yi - y_min) / resolution)
    if np.isnan(dsm[row, col]) or zi > dsm[row, col]:
        dsm[row, col] = zi

# CHM = DSM - DTM
chm = dsm - dtm
chm[chm < 0] = 0
chm = np.nan_to_num(chm, nan=0)

print(f"CHM built. Max canopy height: {chm.max():.2f} m")

# ============================================================
# STEP 3: Extract Tree Metrics
# ============================================================
print("Extracting tree metrics...")

# Smooth CHM slightly to reduce noise
chm_smooth = ndimage.gaussian_filter(chm, sigma=1)

# Find local maxima (tree tops) — any cell higher than neighbours
tree_mask = chm_smooth > 1.0  # trees taller than 1 metre
labeled, num_trees = ndimage.label(tree_mask)

print(f"Number of tree clusters detected: {num_trees}")

# Extract metrics per tree cluster
tree_heights = []
tree_areas = []

for i in range(1, num_trees + 1):
    cluster = chm_smooth[labeled == i]
    if len(cluster) >= 2:  # ignore single-pixel noise
        tree_heights.append(cluster.max())
        tree_areas.append(len(cluster))  # area in square metres

tree_heights = np.array(tree_heights)
tree_areas = np.array(tree_areas)

print(f"Valid trees analysed: {len(tree_heights)}")
print(f"Mean tree height: {tree_heights.mean():.2f} m")
print(f"Max tree height:  {tree_heights.max():.2f} m")
print(f"Mean crown area:  {tree_areas.mean():.2f} m²")

# Tree density (trees per hectare)
total_area_ha = (rows * cols * resolution * resolution) / 10000
density = len(tree_heights) / total_area_ha
print(f"Tree density: {density:.1f} trees/hectare")

# ============================================================
# STEP 4: Regression Model — Predict Tree Height from Crown Area
# ============================================================
print("\nRunning regression model...")

X = tree_areas.reshape(-1, 1)
y_target = tree_heights

X_train, X_test, y_train, y_test = train_test_split(
    X, y_target, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score:  {r2:.3f}")
print(f"MAE:       {mae:.3f} metres")
print(f"Coefficient: {model.coef_[0]:.4f} (height per m² crown area)")

# ============================================================
# STEP 5: Visualisations
# ============================================================
print("\nGenerating visualisation plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Forest LiDAR Analysis — Tree Metrics', fontsize=14)

# Plot 1: CHM Heatmap
im1 = axes[0, 0].imshow(chm_smooth, cmap='YlGn', origin='lower',
                          vmin=0, vmax=np.percentile(chm_smooth, 95))
plt.colorbar(im1, ax=axes[0, 0], label='Height (m)')
axes[0, 0].set_title('Canopy Height Model (CHM)')
axes[0, 0].set_xlabel('X grid cells')
axes[0, 0].set_ylabel('Y grid cells')

# Plot 2: Tree Height Distribution
axes[0, 1].hist(tree_heights, bins=30, color='forestgreen', edgecolor='white')
axes[0, 1].set_title('Tree Height Distribution')
axes[0, 1].set_xlabel('Height (metres)')
axes[0, 1].set_ylabel('Number of Trees')
axes[0, 1].axvline(tree_heights.mean(), color='red',
                    linestyle='--', label=f'Mean: {tree_heights.mean():.2f}m')
axes[0, 1].legend()

# Plot 3: Crown Area vs Tree Height (scatter + regression)
axes[1, 0].scatter(tree_areas, tree_heights, alpha=0.3,
                    color='steelblue', s=10, label='Trees')
x_line = np.linspace(tree_areas.min(), tree_areas.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)
axes[1, 0].plot(x_line, y_line, color='red',
                 linewidth=2, label=f'Regression (R²={r2:.2f})')
axes[1, 0].set_title('Crown Area vs Tree Height')
axes[1, 0].set_xlabel('Crown Area (m²)')
axes[1, 0].set_ylabel('Tree Height (m)')
axes[1, 0].legend()

# Plot 4: Predicted vs Actual Heights
axes[1, 1].scatter(y_test, y_pred, alpha=0.5, color='darkorange', s=10)
axes[1, 1].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[1, 1].set_title(f'Predicted vs Actual Height\nMAE={mae:.2f}m')
axes[1, 1].set_xlabel('Actual Height (m)')
axes[1, 1].set_ylabel('Predicted Height (m)')

plt.tight_layout()
plt.savefig('tree_metrics_analysis.png', dpi=150)
plt.show()
print("Saved: tree_metrics_analysis.png")