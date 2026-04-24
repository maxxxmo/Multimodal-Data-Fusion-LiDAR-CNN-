from src.late_fusion.utils.display_lidar import debug_alignment


if __name__ == "__main__":
    debug_alignment("0014_000088", "./data/kitti_lidar/val", "./src/late_fusion/LiDAR/config.yaml")