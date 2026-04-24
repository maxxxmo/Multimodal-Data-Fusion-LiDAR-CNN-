import h5py
import matplotlib.pyplot as plt
import cv2
import numpy as np

def visualize_sample(h5_path, file_id):
    with h5py.File(h5_path, 'r') as f:
        group = f[str(file_id)]
        pseudo_image = group['pseudo_image'][:]  # (3, H, W)
        bev_label = group['bev_label'][:]        # (H, W)
        gt_boxes = group['gt_boxes_3d'][:]       # (N, 7)

    # 1. Préparation de l'affichage (Prendre Z-max pour le fond)
    plt.figure(figsize=(12, 5))
    

    plt.title("Density pseudo image 2D (BEV)")  
    # On affiche le masque en transparence sur la densité (canal 1)
    img_norm = cv2.normalize(pseudo_image[1], None, 0, 255, cv2.NORM_MINMAX)
    pseudo_image[1] = img_norm.astype(np.uint8)
    plt.imshow(pseudo_image[1], origin='lower')
    plt.imshow(bev_label, origin='lower', alpha=0.3, cmap='jet')
    
    plt.show()

# Utilisation
visualize_sample("./data/kitti_lidar/test/h_data/dataset_unified.h5", "0018_000103")