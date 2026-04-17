import open3d as o3d
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from src.late_fusion.utils.calibration import KittiCalibration

def visualize_point_cloud(point_cloud):
    """Display a raw LiDAR point cloud using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    # Intensity coloration (optiional)
    if point_cloud.shape[1] > 3:
        intensity = point_cloud[:, 3]

        intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
        colors = np.zeros((len(intensity), 3))
        colors[:, 0] = intensity_norm 
        colors[:, 2] = 1.0 - intensity_norm 
        pcd.colors = o3d.utility.Vector3dVector(colors)

    print("close the window to continue.")
    o3d.visualization.draw_geometries([pcd])
    
    
def visualize_point_cloud_with_boxes(point_cloud, detected_boxes):
    """Visualize LiDAR point cloud with detected bounding boxes."""
    # pointcloud: (N, 4) -> x, y, z, intensity
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    pcd.paint_uniform_color([0.2, 0.2, 0.2]) # Gris foncé pour le fond

    # detected bboxes : List of dicts with 'box' (7 params) and 'score'
    vis_elements = [pcd]
    
    for det in detected_boxes:
        box_params = det['box'] # [x, y, z, l, w, h, yaw]
        score = det['score']
        
        
        center = box_params[0:3]
        dims = box_params[3:6] # l, w, h
        yaw = box_params[6]
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
        
        # WARNING: KITTI l,w,h is sometimes in a different order than Open3D (w,h,l). 
        # Adjust the order if the boxes appear deformed.
        obb = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, dims)
        color = [1.0 - score, score, 0.0] 
        obb.color = color
        
        vis_elements.append(obb)

    print(f"Interactive visualization. {len(detected_boxes)} boxes displayed. Use the mouse to rotate/zoom. Close to continue.")
    o3d.visualization.draw_geometries(vis_elements)
    
    

def get_projected_bbox_coords(calib_path, predictions, score_threshold=0.5):
    """This function takes raw LiDAR predictions (in LiDAR coordinates) and projects the 3D bounding boxes onto the image plane using the KITTI calibration."""
    calib = KittiCalibration(calib_path)
    if torch.is_tensor(predictions):
        predictions = predictions.permute(1, 2, 0).reshape(-1, 8)
        predictions = predictions.detach().cpu().numpy()
    mask = predictions[:, 0] > score_threshold
    valid_boxes = predictions[mask]
    
    projected_data = []

    for box in valid_boxes:
        score, x, y, z, l, w, h, yaw = box

        # 3. Créer les 8 coins en 3D (Coordonnées locales LiDAR)
        # Ordre classique : 4 bas (avant-gauche, avant-droit, arrière-droit, arrière-gauche)
        #                  puis 4 haut (idem)
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        z_corners = [0, 0, 0, 0, h, h, h, h] # Souvent z est à la base dans KITTI
        
        # Si ton modèle prédit le CENTRE vertical (z_center) :
        # z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
        # Si ton modèle prédit la BASE (z_bottom) :
        # z_corners = [0, 0, 0, 0, h, h, h, h]

        corners_3d = np.vstack([x_corners, y_corners, z_corners])

        # 4. Appliquer la rotation (Yaw) et la translation (x, y, z)
        # Rotation autour de l'axe Z (Vertical en LiDAR)
        R = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,            0,           1]
        ])
        corners_3d = R @ corners_3d
        corners_3d[0, :] += x
        corners_3d[1, :] += y
        corners_3d[2, :] += z

        # 5. Transformation LiDAR -> Caméra Rectifiée
        pts_3d_rect = calib.transform_velo_to_rect(corners_3d.T)
        
        # Vérifier si l'objet est devant la caméra (Z > 0 en repère caméra)
        if np.any(pts_3d_rect[:, 2] <= 0):
            continue

        # 6. Projection Caméra Rectifiée -> Pixels Image
        pts_2d = calib.project_rect_to_image(pts_3d_rect) # Résultat en [8, 2] (u, v)

        projected_data.append({
            'score': score,
            'corners_2d': pts_2d # Coordonnées (u, v) pour les 8 sommets
        })

    return projected_data



def visualize_boxes_on_image(image_path, projected_boxes, output_path=None):
    """
    Prend l'image d'origine et dessine les cubes 3D projetés.
    projected_boxes: Sortie de ta fonction get_projected_bbox_coords (liste de dicts)
    """
    # 1. Charger l'image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Erreur : Impossible de charger l'image {image_path}")
        return

    # 2. Parcourir chaque boîte détectée
    for det in projected_boxes:
        corners_2d = det['corners_2d'].astype(np.int32)
        score = det['score']
        
        # Définir une couleur (Vert si score élevé, Rouge sinon)
        color = (0, int(255 * score), int(255 * (1 - score))) # BGR

        # 3. Dessiner les 12 arêtes du cube
        # On relie les 4 coins de la base, les 4 du haut, et les 4 verticales
        for i in range(4):
            # Arêtes de la base (0,1,2,3)
            cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[(i+1)%4]), color, 2)
            # Arêtes du haut (4,5,6,7)
            cv2.line(img, tuple(corners_2d[i+4]), tuple(corners_2d[((i+1)%4)+4]), color, 2)
            # Arêtes verticales
            cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[i+4]), color, 2)

        # Optionnel : Afficher le score au-dessus de la boîte
        cv2.putText(img, f"{score:.2f}", tuple(corners_2d[4]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 4. Affichage
    plt.figure(figsize=(15, 5))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.show()