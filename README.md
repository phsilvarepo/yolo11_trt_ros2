## YOLO11 TRT ROS2 Package

This repository contains a package that utilizes the YOLOv11 architecture to perform object detection on the ROS2 framework. The TensorRT SDK is employed to enable faster inference on the images. This package subscribes to the /camera/image_raw (sensor_msgs/Image) topic and publishes to the /yolo11/image_annotated topic a (sensor_msgs/Image) message with the detected bounding boxes.

ros2 run yolo11_trt_node yolo11_trt_node
