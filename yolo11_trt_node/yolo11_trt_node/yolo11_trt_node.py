#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class Yolo11TRTNode(Node):
    def __init__(self):
        super().__init__('yolo11_trt_node')

        self.declare_parameter('engine_path', '/home/unparallel/ros2_ws/src/yolo11_trt_node/yolo11_trt_node/models/yolov11_trt.engine')
        self.engine_path = self.get_parameter('engine_path').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher(Image, '/yolo11/image_annotated', 10)

        self.load_engine()

    def load_engine(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(None, "")  # Initialize plugins[web:2]
        with open(self.engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context_trt = self.engine.create_execution_context()

        self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            binding_name = self.engine.get_tensor_name(i)  # Use the updated API
            shape = self.engine.get_tensor_shape(binding_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding_name))
            device_mem = cuda.mem_alloc(trt.volume(shape) * np.dtype(dtype).itemsize)
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'binding': binding_name, 'device_mem': device_mem, 'dtype': dtype, 'shape': shape})
            else:
                self.outputs.append({'binding': binding_name, 'device_mem': device_mem, 'dtype': dtype, 'shape': shape})


        self.get_logger().info("YOLOv11 TensorRT engine loaded.")


    def preprocess(self, img):
        # Adapt to your model’s input resolution
        resized = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_norm, (2, 0, 1))
        input_tensor = np.expand_dims(img_transposed, axis=0).astype(np.float32)
        return np.ascontiguousarray(input_tensor)


    def image_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        input_tensor = self.preprocess(cv_img)

        # Copy input to device
        cuda.memcpy_htod(self.inputs[0]['device_mem'], input_tensor)

        # Execute model
        self.context_trt.execute_v2(self.bindings)

        # Retrieve output
        output_shape = self.engine.get_tensor_shape(self.outputs[0]['binding'])
        output_data = np.empty(trt.volume(output_shape), dtype=np.float32)

        cuda.memcpy_dtoh(output_data, self.outputs[0]['device_mem'])

        # Reshape: (1, 84, 8400) -> (8400, 84)
        pred = output_data.reshape(1, 84, 8400)[0].T

        detections = []

        # Suppose your model input size
        inp_w, inp_h = 640, 640
        img_h, img_w, _ = cv_img.shape

        for det in pred:
            bx, by, bw, bh = det[:4]
            class_scores = det[4:]

            class_id = np.argmax(class_scores)
            score = class_scores[class_id]

            if score > 0.5:
                # Convert from normalized coords to original image size
                x1 = int((bx - bw/2) * img_w / inp_w)
                y1 = int((by - bh/2) * img_h / inp_h)
                x2 = int((bx + bw/2) * img_w / inp_w)
                y2 = int((by + bh/2) * img_h / inp_h)

                detections.append([x1, y1, x2, y2, score, class_id])
        
        detections = self.nms(detections, iou_threshold=0.5)
        annotated = self.draw_boxes(cv_img, detections)

        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        self.publisher.publish(out_msg)

    def draw_boxes(self, image, detections):
        for det in detections:  # detections format: [x1, y1, x2, y2, conf, class_id]
            x1, y1, x2, y2, conf, cls = det
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{cls}: {conf:.2f}"
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        return image

    def nms(self, detections, iou_threshold=0.5):
        """
        Apply Non-Maximum Suppression to avoid overlapping boxes.
        
        detections: list of [x1, y1, x2, y2, score, class_id]
        iou_threshold: float, IoU threshold to suppress boxes
        """
        if len(detections) == 0:
            return []

        # Convert to array for easier computation
        dets = np.array(detections)
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        class_ids = dets[:, 5]

        keep_boxes = []

        # Process each class separately
        unique_classes = np.unique(class_ids)
        for cls in unique_classes:
            idxs = np.where(class_ids == cls)[0]
            cls_boxes = dets[idxs]

            # Sort by confidence score descending
            order = np.argsort(cls_boxes[:, 4])[::-1]
            cls_boxes = cls_boxes[order]

            while len(cls_boxes) > 0:
                # Pick the box with highest score
                box = cls_boxes[0]
                keep_boxes.append(box.tolist())

                if len(cls_boxes) == 1:
                    break

                # Compute IoU of the remaining boxes with the first box
                xx1 = np.maximum(box[0], cls_boxes[1:, 0])
                yy1 = np.maximum(box[1], cls_boxes[1:, 1])
                xx2 = np.minimum(box[2], cls_boxes[1:, 2])
                yy2 = np.minimum(box[3], cls_boxes[1:, 3])

                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)
                inter = w * h

                area_box = (box[2] - box[0]) * (box[3] - box[1])
                area_cls_boxes = (cls_boxes[1:, 2] - cls_boxes[1:, 0]) * (cls_boxes[1:, 3] - cls_boxes[1:, 1])
                iou = inter / (area_box + area_cls_boxes - inter)

                # Keep boxes with IoU less than threshold
                cls_boxes = cls_boxes[1:][iou < iou_threshold]

        return keep_boxes

def main(args=None):
    rclpy.init(args=args)
    node = Yolo11TRTNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
