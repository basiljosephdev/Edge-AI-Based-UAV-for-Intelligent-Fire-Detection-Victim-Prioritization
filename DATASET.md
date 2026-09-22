# Dataset Documentation

This project combines multiple publicly available datasets for fire detection, human detection, pose estimation, and action recognition from aerial/drone imagery.

## Dataset Overview

| Module             | Purpose                             | Dataset             | Approx. Size                 | Download Link                                                                                                 |
| ------------------ | ----------------------------------- | ------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Fire Detection     | Detect fire and smoke               | DFire               | ~21,500 images               | https://github.com/gaia-solutions-on-demand/DFireDataset                                                      |
| Fire Detection     | Detect fire and smoke               | FLAME               | ~48,000 RGB/IR aerial images | https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset |
| Person Detection   | Detect humans in drone imagery      | VisDrone            | ~10,000 images               | https://github.com/VisDrone/VisDrone-Dataset                                                                  |
| Person Detection   | General object and person detection | COCO 2017           | ~118,000 training images     | https://cocodataset.org                                                                                       |
| Pose Estimation    | Human keypoint detection            | COCO Keypoints      | Included in COCO             | https://cocodataset.org/#keypoints-2020                                                                       |
| Pose Estimation    | Human posture estimation            | MPII Human Pose     | ~25,000 images               | https://human-pose.mpi-inf.mpg.de                                                                             |
| Action Recognition | Walking, running, standing, etc.    | Okutama-Action      | ~77 video sequences          | https://okutama-action.org                                                                                    |
| Testing            | Real-world validation               | Custom Drone Videos | Project-specific             | Collected by project team                                                                                     |

---

## Dataset Descriptions

### DFire
A fire and smoke detection dataset containing more than 21,000 annotated images. Includes fire-only, smoke-only, combined fire/smoke, and negative samples. Annotations are provided in YOLO format.

### FLAME
(Fire Luminosity Airborne-based Machine Learning Evaluation)
An aerial wildfire dataset collected using UAVs. Contains RGB and thermal imagery for wildfire detection and segmentation research.

### VisDrone
A large-scale benchmark collected using drones. Contains pedestrian, vehicle, and object annotations under varying altitudes, viewpoints, and weather conditions.

### COCO
(Common Objects in Context)
A standard computer vision dataset widely used for object detection, segmentation, and keypoint estimation.

### MPII Human Pose
A benchmark dataset for articulated human pose estimation containing diverse human activities and body postures.

### Okutama-Action
A drone-based action recognition dataset containing multiple concurrent human actions captured from aerial viewpoints.

---

## Download Instructions

1. Download each dataset from the provided source.
2. Extract all archives into their respective folders.
3. Preserve original annotation formats whenever possible.
4. Convert annotations only if required by the training pipeline.

---

## Citation and References

If you use these datasets, please cite their original authors and publications.

### DFire
Venâncio, P. V. A. B., Lisboa, A. C., Barbosa, A. V.
*An Automatic Fire Detection System Based on Deep Convolutional Neural Networks for Low-Power Resource-Constrained Devices.*

### FLAME
Chen, X., Hopkins, B., Wang, H., et al.
*Wildland Fire Detection and Monitoring Using a Drone-Collected RGB/IR Image Dataset.* IEEE Access, 2022.

### VisDrone
Zhu, P., Wen, L., Du, D., et al.
*Vision Meets Drones: A Challenge.*

### COCO
Lin, T.Y., Maire, M., Belongie, S., et al.
*Microsoft COCO: Common Objects in Context.*

### MPII Human Pose
Andriluka, M., Pishchulin, L., Gehler, P., Schiele, B.
*2D Human Pose Estimation: New Benchmark and State of the Art Analysis.*

### Okutama-Action
Barekatain, M., Martí, M., Shih, H.F., et al.
*Okutama-Action: An Aerial View Video Dataset for Concurrent Human Action Detection.*
