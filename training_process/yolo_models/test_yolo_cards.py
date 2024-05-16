import cv2
from ultralytics import YOLO
# from ultralytics import plot_results

# Load the trained model
model = YOLO('card_detector_model.pt')

# Load an image
img = cv2.imread('../my_data/white_bg/4_of_clubs.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Predict the objects in the image
results = model.predict(img_rgb, conf=0.5)

print(results)

# Print the results
for detection in results:
    print(f"Detected a {detection['name']} with confidence {detection['confidence']} at {detection['bbox']}")

# If you want to visualize the detections, you can do so using the following code:
# plot_results(results, img_rgb)
