import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import yaml

def visualize_pseudo_image_with_bboxes(h5_path, file_id, config_path):
    """
    Visualise la pseudo-image (BEV) et superpose les BBoxes projetées en LiDAR.
    """
    key = f"0000_{file_id}"
    
    # 1. Chargement des données depuis le cache H5
    with h5py.File(h5_path, 'r') as f:
        if key not in f:
            print(f"Erreur : Clé '{key}' introuvable.")
            return
        group = f[key]
        pseudo_image = group['pseudo_image'][:] 
        labels = group['labels'][:] 
        
    # 2. Chargement de la config pour la résolution
    with open(config_path, 'r') as conf_f:
        config = yaml.safe_load(conf_f)
    pc_range = config['dataset']['pc_range']
    grid_size = config['dataset']['grid_size']
    
    # Résolution (mètres par pixel)
    res_x = (pc_range[3] - pc_range[0]) / grid_size[0]
    res_y = (pc_range[4] - pc_range[1]) / grid_size[1]
    
    # 3. Préparation de l'affichage
    bev_map = pseudo_image[1] # Canal de densité
    plt.figure(figsize=(10, 10))
    # 'origin=lower' aligne le 0,0 en bas à gauche
    plt.imshow(bev_map, cmap='viridis', origin='lower') 
    ax = plt.gca()
    
    # 4. Superposition des boîtes
    for box in labels:
        # box format: [x, y, z, l, w, h, yaw]
        x_m, y_m, z_m, l_m, w_m, h_m, yaw_rad = box
        
        # Conversion coordonnées monde (mètres) -> coordonnées image (pixels)
        # u est l'index de ligne (axe X), v est l'index de colonne (axe Y)
        u = (x_m - pc_range[0]) / res_x
        v = (y_m - pc_range[1]) / res_y
        
        l_pix = l_m / res_x
        w_pix = w_m / res_y
        
        # Création du rectangle
        # Dans matplotlib, le point d'ancrage est le coin bas-gauche
        rect = patches.Rectangle(
            (v - w_pix/2, u - l_pix/2), 
            w_pix, l_pix, 
            linewidth=0.75, edgecolor='red', facecolor='none',linestyle='--'
        )
        
        # Rotation autour du centre de la boîte
        # Le signe du yaw peut dépendre de ton système de coordonnées, 
        # teste avec -np.degrees si ça ne tourne pas dans le bon sens.
        t = patches.Affine2D().rotate_deg_around(v, u, -np.degrees(yaw_rad))
        rect.set_transform(t + ax.transData)
        ax.add_patch(rect)
    
    plt.title(f"Visualisation LiDAR BEV - {key}")
    plt.xlabel("Axe Y (Latéral)")
    plt.ylabel("Axe X (Longitudinal)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

# Utilisation
h5_cache = './data/kitti_lidar/train/h_data/dataset_cache.h5'
config_file = './src/late_fusion/LiDAR/config.yaml'
visualize_pseudo_image_with_bboxes(h5_cache, '000050', config_file)