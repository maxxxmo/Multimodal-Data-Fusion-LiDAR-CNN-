import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import yaml

def visualize_sample(h5_path, file_id, config_path):
    # 1. Chargement de la config pour les dimensions
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    pc_range = config['dataset']['pc_range'] # [x_min, y_min, z_min, x_max, y_max, z_max]
    H, W = config['dataset']['grid_size']

    with h5py.File(h5_path, 'r') as f:
        group = f[str(file_id)]
        pseudo_image = group['pseudo_image'][:]  # (3, H, W)
        gt_boxes = group['gt_boxes_3d'][:]       # (N, 7) [x, y, z, w, l, h, yaw]

    # 2. Préparation du fond
    plt.figure(figsize=(10, 10))
    # On affiche le canal intensité/densité
    plt.imshow(pseudo_image[1], origin='lower', cmap='gray')

    # 3. Calcul des résolutions pour projeter les boîtes
    res_x = (pc_range[3] - pc_range[0]) / W
    res_y = (pc_range[4] - pc_range[1]) / H

    # 4. Dessin des boîtes 3D
    for box in gt_boxes:
        x, y, _, w, l, _, yaw = box
        
        # Projection centre monde -> grille (u, v)
        u = (x - pc_range[0]) / res_x
        v = (y - pc_range[1]) / res_y
        
        # Dimensions en pixels
        w_px = w / res_x
        l_px = l / res_y
        
        # Dessin du rectangle avec rotation
        # Note: matplotlib utilise l'angle en degrés, inversé selon le repère
        rect = patches.Rectangle(
            (u - w_px/2, v - l_px/2), w_px, l_px,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        # Application de la rotation autour du centre (u, v)
        t = patches.Affine2D().rotate_around(u, v, -yaw) + plt.gca().transData
        rect.set_transform(t)
        
        plt.gca().add_patch(rect)
        plt.plot(u, v, 'r+', markersize=10) # Point central

    plt.title(f"Visualisation 3D BBox sur pseudo-image - {file_id}")
    plt.xlabel("X (Width)")
    plt.ylabel("Y (Height)")
    plt.show()

# Utilisation
visualize_sample(
    "./data/kitti_lidar/test/h_data/dataset_unified.h5", 
    "0018_000103",
    "./src/late_fusion/LiDAR/config.yaml"
)