
# SPEK - Simple Python Extraction Kit

SPEK is a simple Python library for real-time object detection using YOLOv8 and OpenCV.  
It supports webcam, video, or static image detection and is designed for ease of use.



![Logo](https://i.postimg.cc/QM0H7jNt/LOGO-COPY.png)


## Features

- Live detection from webcam or video files
- Static image detection
- Filter detection by specific class (e.g., person, dog, car)
- Headless mode (no preview window) for servers
- Callbacks on detection for custom behavior
- Quick-start helper functions


## Installation



```python
pip install spek
```
    
## Usage/Examples

1 Live webcam detection (default) 
```python
import spek
# model sizes are n, s, m, l, x
# Detect "person" using YOLOv8s model on webcam
# target_class="person" targets a person
spek.run_live_detection(target_class="person", model_name_or_size="s")

```

2 Detect a specific target class and run a function
```python

from spek import detect_objects

# Define the callback function
def your_function(info):
    print("Detected person")

# Run detection for "person"
detect_objects(
    target_class="person",
    on_detected=your_function,
    source=0,
    model_size="s"
)
```
or 
```python
from spek import detect_objects

# Define the callback function
def your_function(info):
    print("Detected person")

# Run detection for "person"
detect_objects(target_class="person", on_detected=your_function, source=0, model_size="s")
```


3 Detect objects from a video file
```python
from spek import detect_objects

# Detect objects from a local video file
detect_objects(source="video.mp4", model_size="m")
 ```

4 Detect objects from a static image
```python
from spek import detect_objects

# Detect objects in a single image
detect_objects(source="image.jpg", model_size="l")
```


5 Run in headless mode (no display, useful for servers)
```python
from spek import detect_objects

# Process webcam feed without displaying window
detect_objects(source=0, model_size="s", headless=True)
```

6 CLI usage
After installation, you can run live detection from the command line:
```python
python -m spek --source 0 --target person --model s
```


7 Static image via CLI
```python
python -m spek --image input.jpg --target dog --model m
```
## Authors

- [@Rudransh Joshi](https://rudransh.kafalfpc.com/)


## License

[MIT](https://choosealicense.com/licenses/mit/)

