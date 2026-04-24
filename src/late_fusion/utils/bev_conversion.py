import cv2
import numpy as np

def convert_to_pixels(corners, pc_range, grid_size):
    """
    corners: (N, 2) array [x, y] en mètres.
    grid_size: [grid_x, grid_y]
    Retourne des points [colonne, ligne] pour OpenCV.
    """
    # 1. Décalage (x_rel, y_rel)
    shifted = corners - np.array([pc_range[0], pc_range[1]])
    
    # 2. Résolution (m/pixel)
    res_x = (pc_range[3] - pc_range[0]) / grid_size[0]
    res_y = (pc_range[4] - pc_range[1]) / grid_size[1]
    
    # 3. Conversion
    u = shifted[:, 0] / res_x  # Colonne
    v = shifted[:, 1] / res_y  # Ligne
    
    # Retourner sous forme [colonne, ligne] (i.e., [u, v])
    return np.stack([u, v], axis=1)

def boxes_to_bev_map(gt_boxes, grid_size, pc_range):
    # Important : [y, x] => [hauteur, largeur]
    bev_map = np.zeros((grid_size[1], grid_size[0]), dtype=np.uint8)
    
    for box in gt_boxes:
        x, y, z, l, w, h, yaw = box[0:7]
        
        corners = get_box_corners(x, y, w, l, yaw)
        
        # u_v_corners sont en [colonne, ligne]
        u_v_corners = convert_to_pixels(corners, pc_range, grid_size)
        
        # cv2.fillConvexPoly attend des points [x, y] -> [colonne, ligne]
        # C'est exactement ce que convert_to_pixels renvoie.
        cv2.fillConvexPoly(bev_map, u_v_corners.astype(np.int32), 1)
        
    return bev_map

def get_box_corners(x, y, w, l, yaw):
    """
    Calcule les 4 coins d'une box orientée.
    yaw: angle en radians (sens anti-horaire)
    """
    # 1. Demi-dimensions
    dx = l / 2
    dy = w / 2
    
    # 2. Coins dans le repère local de la box (non tourné)
    # Ordre : (x, y) -> haut-gauche, haut-droite, bas-droite, bas-gauche
    corners = np.array([
        [ dx,  dy],
        [-dx,  dy],
        [-dx, -dy],
        [ dx, -dy]
    ])
    
    # 3. Matrice de rotation
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    rotation_matrix = np.array([
        [cos_y, -sin_y],
        [sin_y,  cos_y]
    ])
    
    # 4. Appliquer la rotation et translater au centre (x, y)
    rotated_corners = corners @ rotation_matrix.T + np.array([x, y])
    
    return rotated_corners
