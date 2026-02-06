"""
YOLO Object Detection Library Documentation

A simple Python library for real-time object detection using YOLOv8 and OpenCV.
Supports webcam, video files, or static images. Designed for ease with minimal code.
Features three top-level functions for quick use and a flexible class for advanced control.

Installation:
    pip install ultralytics opencv-python

Features:
- Model selection: Pretrained ('n', 's', 'm', 'l', 'x') or custom (.pt files).
- Targets: Specific class (e.g., "dog", 16) or None for all 80 COCO classes.
- Callbacks: No-arg or with info dict {'class_name', 'confidence', 'bbox', 'frame'}.
- Sources: Webcam (0), video files, or static images.
- Modes: Live camera (with/without preview), headless image processing.
- Outputs: Console logs, saved images, or returned frames for further processing.

Quick Start Examples:
    # Live webcam, specific target, auto-printing (no callback needed)
    import detector
    detector.run_live_detection("person", "n")  # Press 'q' to quit (or Ctrl+C if headless)

    # Live webcam with callback, headless for servers
    def alert(): print("Detected!")
    detector.detectviacamerafeed("dog", alert, "s", headless=True)  # Ctrl+C to stop

    # Static image, no save, headless
    def image_alert(info): print(f"{info['class_name']} found!")
    detector.detect_in_image("input.jpg", "car", image_alert, "m")  # Returns annotated frame
"""

import cv2
from ultralytics import YOLO
from contextlib import contextmanager
import inspect  # For smart callback handling
import os  # For file checks


class ObjectDetector:
    """
    A class for performing real-time object detection using YOLOv8. Easy to use with context manager.

    Args:
        model_path (str, optional): Path to custom YOLO model (e.g., "path/to/custom.pt"). Overrides model_size.
        model_size (str): Pretrained model size: 'n' (nano/fast), 's' (small), 'm' (medium), 'l' (large), 'x' (extra-large). Default: 'n'.
        source (int or str): Video source (0 for webcam, or video file path). Default: 0.
        confidence_threshold (float): Minimum detection confidence (0.0 to 1.0). Default: 0.25.
        on_object_detected (callable, optional): Callback function for detections. Takes either no args or a dict with keys:
            - class_name (str): Detected class (e.g., "person").
            - confidence (float): Confidence score (0.0 to 1.0).
            - bbox (tuple): Bounding box (x1, y1, x2, y2).
            - frame (numpy.ndarray): Frame with drawn boxes.
        target_class (int, str, or None): Class to detect (e.g., "person", 0 for person, None for all classes). Default: None.

    Raises:
        ValueError: If model_size is invalid, source can't be opened, or target_class is unknown.

    Example:
        with ObjectDetector(model_size='m', target_class="dog") as det:
            det.run(quit_key='x', headless=True)  # Custom quit key, no preview
    """

    def __init__(self, model_path=None, model_size='n', source=0, confidence_threshold=0.25, on_object_detected=None, target_class=None):
        # Resolve model path from size if not provided
        if model_path is None:
            size_map = {'n': 'yolov8n.pt', 's': 'yolov8s.pt', 'm': 'yolov8m.pt', 'l': 'yolov8l.pt', 'x': 'yolov8x.pt'}
            if model_size not in size_map:
                raise ValueError(f"Invalid model_size '{model_size}'. Choose: n, s, m, l, x")
            model_path = size_map[model_size]
        
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(source)
        self.model_names = self.model.names
        self.confidence_threshold = confidence_threshold
        self.on_object_detected = on_object_detected
        self.target_class_id = self._resolve_class_id(target_class)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")

    def __enter__(self):
        """Context manager entry: Returns self."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: Auto-cleanup."""
        self.close()

    def _resolve_class_id(self, target_class):
        """Resolve class name to ID or return None."""
        if target_class is None:
            return None
        if isinstance(target_class, int):
            return target_class
        if isinstance(target_class, str):
            for cls_id, name in self.model_names.items():
                if name.lower() == target_class.lower():
                    return int(cls_id)
            raise ValueError(f"Unknown class: {target_class}. Try: person, car, dog, etc.")
        raise ValueError("target_class must be int, str, or None")

    def _draw_detections(self, frame, results):
        """Draw boxes and trigger callback for matches."""
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    conf = box.conf[0]
                    if conf < self.confidence_threshold:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    class_name = self.model_names[cls]
                    label = f"{class_name} {conf:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if self.target_class_id is None or cls == self.target_class_id:
                        if self.on_object_detected:
                            self.on_object_detected({
                                'class_name': class_name,
                                'confidence': conf,
                                'bbox': (x1, y1, x2, y2),
                                'frame': frame
                            })
        return frame

    def run(self, window_name="Detection", quit_key='q', headless=False):
        """
        Start real-time loop. Draws boxes on frames and triggers callback.

        Args:
            window_name (str): Name of the display window (ignored if headless). Default: "Detection".
            quit_key (str): Single character to press to quit (e.g., 'q', 'x'). Ignored if headless; use Ctrl+C. Default: 'q'.
            headless (bool): If True, no preview window—runs silently (ideal for servers). Default: False.

        Notes:
            - If headless=False, press quit_key to stop and close window.
            - If headless=True, runs indefinitely; stop with Ctrl+C in terminal.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame. Exiting...")
                break

            results = self.model(frame)
            annotated_frame = self._draw_detections(frame, results)

            if not headless:
                cv2.imshow(window_name, annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord(quit_key):
                    break
            else:
                # Headless: Small delay for CPU efficiency; Ctrl+C to stop
                if cv2.waitKey(1) & 0xFF == ord(quit_key):
                    break  # Still checks for 'q' if piped, but mainly for Ctrl+C

        if not headless:
            cv2.destroyAllWindows()

    def detect_single_frame(self, frame):
        """
        Detect on a single frame (no loop). Used internally by detect_in_image.

        Args:
            frame (numpy.ndarray): Input image frame (BGR format).

        Returns:
            numpy.ndarray: Frame with drawn bounding boxes.
        """
        results = self.model(frame)
        return self._draw_detections(frame, results)

    def close(self):
        """Cleanup: Release camera and close windows (if any)."""
        self.cap.release()
        cv2.destroyAllWindows()


def detect_objects(target_class="person", on_detected=None, source=0, confidence_threshold=0.25, model_path=None, model_size='n', headless=False):
    """
    Quick-start function for live detection with minimal setup.

    Args:
        target_class (str/int/None): Class to detect (e.g., "person", 0, None for all). Default: "person".
        on_detected (callable): Callback function (no args or takes info dict). Default: None.
        source (int/str): Video source (0 for webcam, or path to video file). Default: 0.
        confidence_threshold (float): Minimum detection confidence (0.0 to 1.0). Default: 0.25.
        model_path (str, optional): Path to custom YOLO model. Overrides model_size. Default: None.
        model_size (str): Pretrained model size ('n', 's', 'm', 'l', 'x'). Default: 'n'.
        headless (bool): If True, no preview window—runs silently (Ctrl+C to stop). Default: False.

    Example:
        detect_objects(target="dog", on_detected=lambda info: print(f"🐕 {info['confidence']:.1f}"), headless=True)
    """
    with ObjectDetector(model_path, model_size, source, confidence_threshold, on_detected, target_class) as detector:
        detector.run(headless=headless)


def detectviacamerafeed(target_class, on_detected, model_name_or_size='n', headless=False):
    """
    Ultra-simple webcam detection with a callback. Just pass target, callback, and model.

    Args:
        target_class (str/int/None): What to detect (e.g., "dog", 16, None for all). Required.
        on_detected (callable): Function called on detection (no args or takes info dict). Required.
        model_name_or_size (str): Model file path (e.g., "yolov8n.pt") or size ('n', 's', 'm', 'l', 'x'). Default: 'n'.
        headless (bool): If True, no preview window—runs silently (Ctrl+C to stop). Default: False.

    Example:
        def main(): print("hello")
        detectviacamerafeed("person", main, "n", headless=True)  # Server-friendly
    """
    # Smart wrapper: Check if callback expects args
    def wrapper(info):
        if inspect.signature(on_detected).parameters:
            on_detected(info)
        else:
            on_detected()
    
    # Resolve model_path
    if model_name_or_size in ['n', 's', 'm', 'l', 'x']:
        model_path = None
        model_size = model_name_or_size
    else:
        model_path = model_name_or_size
        model_size = 'n'
    
    detect_objects(target_class, wrapper, model_size=model_size, model_path=model_path, headless=headless)


def detect_in_image(image_path, target_class, on_detected, model_name_or_size='n', output_path=None, confidence_threshold=0.25):
    """
    Headless static image detection: Load image, detect, callback, and optional save.

    Args:
        image_path (str): Path to input image (e.g., "input.jpg"). Required.
        target_class (str/int/None): What to detect (e.g., "dog", 16, None for all). Required.
        on_detected (callable): Function called on detection (no args or takes info dict). Required.
        model_name_or_size (str): Model file path or size ('n', 's', 'm', 'l', 'x'). Default: 'n'.
        output_path (str, optional): Path to save annotated image (e.g., "output.jpg"). Default: None (no save).
        confidence_threshold (float): Minimum detection confidence (0.0 to 1.0). Default: 0.25.

    Returns:
        numpy.ndarray: Annotated frame with bounding boxes.

    Example:
        def alert(info): print(f"{info['class_name']} found!")
        detect_in_image("my_photo.jpg", "dog", alert, output_path="out.jpg")
    """
    # Check if image exists
    if not os.path.exists(image_path):
        raise ValueError(f"Image not found: {image_path}")
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Smart wrapper for callback
    def wrapper(info):
        if inspect.signature(on_detected).parameters:
            on_detected(info)
        else:
            on_detected()
    
    # Resolve model
    if model_name_or_size in ['n', 's', 'm', 'l', 'x']:
        model_path = None
        model_size = model_name_or_size
    else:
        model_path = model_name_or_size
        model_size = 'n'
    
    # Temp detector for single frame
    temp_detector = ObjectDetector(model_path, model_size, source=0, confidence_threshold=confidence_threshold,
                                   on_object_detected=wrapper, target_class=target_class)
    temp_detector.cap.release()  # No capture needed
    
    # Run detection
    results = temp_detector.model(frame)
    annotated_frame = temp_detector._draw_detections(frame, results)
    
    # Save if requested
    if output_path:
        cv2.imwrite(output_path, annotated_frame)
        print(f"Annotated image saved: {output_path}")
    
    return annotated_frame


def run_live_detection(target_class="person", model_name_or_size='n', source=0, confidence_threshold=0.25, headless=False):
    """
    Easiest live camera detection: Runs webcam, draws boxes, auto-prints to console.
    No callback required. Use target_class=None to detect all 80 COCO classes.

    Args:
        target_class (str/int/None): What to detect (e.g., "dog", 16, None for all). Default: "person".
        model_name_or_size (str): Model file path or size ('n', 's', 'm', 'l', 'x'). Default: 'n'.
        source (int/str): Video source (0 for webcam, or video file path). Default: 0.
        confidence_threshold (float): Minimum detection confidence (0.0 to 1.0). Default: 0.25.
        headless (bool): If True, no preview window—runs silently (Ctrl+C to stop). Default: False.

    Example:
        run_live_detection(None, "n", headless=True)  # Detects everything, server mode
    """
    def auto_print(info):
        print(f"Live: {info['class_name']} detected with {info['confidence']:.2f} confidence at {info['bbox']}")
    
    # Resolve model
    if model_name_or_size in ['n', 's', 'm', 'l', 'x']:
        model_path = None
        model_size = model_name_or_size
    else:
        model_path = model_name_or_size
        model_size = 'n'
    
    detect_objects(target_class, auto_print, source=source, confidence_threshold=confidence_threshold,
                   model_size=model_size, model_path=model_path, headless=headless)