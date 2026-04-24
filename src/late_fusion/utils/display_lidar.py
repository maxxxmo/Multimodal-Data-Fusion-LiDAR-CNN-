import h5py
import matplotlib.pyplot as plt
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
from src.late_fusion.utils.calibration import KittiCalibration


def visualize_pseudo_image(h5_path, file_id):
    with h5py.File(h5_path, 'r') as f:
        group = f[str(file_id)]
        pseudo_image = group['pseudo_image'][:]  # (3, H, W)
        bev_label = group['bev_label'][:]        # (H, W)
        gt_boxes = group['gt_boxes_3d'][:]       # (N, 7)

    plt.figure(figsize=(12, 5))
    plt.title("Density pseudo image 2D (BEV)")  
    img_norm = cv2.normalize(pseudo_image[1], None, 0, 255, cv2.NORM_MINMAX)
    pseudo_image[1] = img_norm.astype(np.uint8)
    plt.imshow(pseudo_image[1], origin='lower')
    plt.imshow(bev_label, origin='lower', alpha=0.3, cmap='jet')
    plt.show()
    
    
    


def get_color_map(values, z_min=-2.0, z_max=0.5):
    """Génère un tableau de couleurs RGB."""
    norm_values = np.clip((values - z_min) / (z_max - z_min), 0, 1)
    cmap = plt.get_cmap('viridis')
    return cmap(norm_values)[:, :3]

def visualize_lidar_and_bboxes(points, annotations, point_colors=None):
    """Fonction utilitaire de rendu (logique isolée)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if point_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    geometries = [pcd]
    for x, y, z, l, w, h, yaw in annotations:
        rot_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,            0,           1]
        ])
        bbox = o3d.geometry.OrientedBoundingBox([x, y, z], rot_matrix, [l, w, h])
        bbox.color = (1, 0, 0)
        geometries.append(bbox)
        
    o3d.visualization.draw_geometries(geometries)

def debug_alignment(file_id, data_dir, config_path):
    """Logique de chargement et préparation des données."""
    data_dir = Path(data_dir)
    lidar_path = data_dir / "velodyne" / f"{file_id}.bin"
    label_path = data_dir / "labels" / f"{file_id}.txt"
    calib = KittiCalibration(data_dir / "calib" / f"{file_id}.txt")
    
    # Chargement points
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    point_colors = get_color_map(points[:, 2])
    
    # Parsing labels
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            p = line.split()
            if p[0] == 'DontCare': continue
            h, w, l = float(p[8]), float(p[9]), float(p[10])
            loc_cam = np.array([[float(p[11]), float(p[12]), float(p[13])]])
            ry = float(p[14])
            
            loc_lidar = calib.project_rect_to_velo(loc_cam).flatten()
            loc_lidar[2] += h / 2 
            annotations.append([*loc_lidar, l, w, h, -ry - np.pi / 2])

    # Appel de la fonction de visualisation isolée
    visualize_lidar_and_bboxes(points[:, :3], annotations, point_colors)
