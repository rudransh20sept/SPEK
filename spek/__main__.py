"""
Entry point for SPEK when run as a script:
    python -m spek --source 0 --model s --target person
"""

import argparse
from .detector import run_live_detection, detect_in_image

def main():
    parser = argparse.ArgumentParser(description="SPEK - Simple Python Extraction Kit (YOLOv8 Object Detection)")
    parser.add_argument("--source", type=str, default="0", help="Video source: 0 for webcam, or path to video file")
    parser.add_argument("--model", type=str, default="n", help="YOLO model size or custom .pt path (n, s, m, l, x)")
    parser.add_argument("--target", type=str, default="person", help="Target class to detect (e.g. 'person', 'dog', None for all)")
    parser.add_argument("--headless", action="store_true", help="Run without preview window (server mode)")
    parser.add_argument("--image", type=str, default=None, help="Path to static image for detection instead of video")

    args = parser.parse_args()

    if args.image:
        def callback(info):
            print(f"Detected {info['class_name']} ({info['confidence']:.2f}) at {info['bbox']}")
        detect_in_image(args.image, args.target, callback, args.model)
    else:
        run_live_detection(target_class=args.target, model_name_or_size=args.model,
                           source=0 if args.source == "0" else args.source,
                           headless=args.headless)

if __name__ == "__main__":
    main()
