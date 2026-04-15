import open3d as o3d
import numpy as np

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