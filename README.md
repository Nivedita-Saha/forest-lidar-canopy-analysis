# Forest LiDAR Canopy Analysis

Analysis of airborne LiDAR point cloud data to model forest canopy structure and predict tree height using machine learning.

## Dataset
- **Source:** OpenTopography — Monitoring Tree Water Stress, CA 2025 (NCALM)
- **Format:** LAZ point cloud (~893,000 points)
- **License:** CC BY 4.0

## Methods
1. **Point Cloud Parsing** — loaded LAZ file using `laspy`, extracted X, Y, Z coordinates and classification labels
2. **Digital Terrain Model (DTM)** — built ground surface from classified ground points (class 2)
3. **Digital Surface Model (DSM)** — extracted highest return per grid cell across all points
4. **Canopy Height Model (CHM)** — computed CHM = DSM − DTM at 1m resolution
5. **Tree Detection** — identified tree clusters using Gaussian smoothing and connected component labelling (`scipy.ndimage`)
6. **Regression Model** — trained Linear Regression (`scikit-learn`) to predict tree height from crown area

## Results
| Metric | Value |
|--------|-------|
| Total points processed | 892,925 |
| Ground points | 698,206 |
| Vegetation points | 194,719 |
| Mean tree height | 1.79 m |
| Max canopy height | 23.48 m |
| Tree density | per hectare |
| Regression MAE | 0.41 m |

## Visualisations
| Output | Description |
|--------|-------------|
| `point_cloud_topdown.png` | Top-down view of full point cloud coloured by elevation |
| `canopy_height_model.png` | CHM heatmap showing canopy structure |
| `point_cloud_3d.png` | 3D scatter plot of sampled point cloud |
| `tree_metrics_analysis.png` | 4-panel plot: CHM, height distribution, regression, predicted vs actual |

## Libraries Used
`laspy` · `numpy` · `matplotlib` · `scipy` · `scikit-learn`

## How to Run
```bash
# Clone the repo
git clone https://github.com/Nivedita-Saha/forest-lidar-canopy-analysis.git
cd forest-lidar-canopy-analysis

# Install dependencies
pip install 'laspy[lazrs]' numpy matplotlib scipy scikit-learn

# Run analysis
python lidar_analysis.py
```

## Author
Nivedita Saha — MSc AI & Data Science (Distinction), Keele University 2025