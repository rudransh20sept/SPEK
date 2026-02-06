from spek import detect_objects

# Define the callback function
def your_function(info):
    print("Detected person")

# Run detection for "person"
detect_objects(target_class="person", on_detected=your_function, source=0, model_size="s")
