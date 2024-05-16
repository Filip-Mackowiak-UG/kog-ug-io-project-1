# from ultralytics import YOLO
#
# # Initialize model
# model = YOLO('yolov8n.pt')
#
# # Train model on custom dataset
# model.train(data='huge_data/cards_kaggle_yolo/data.yaml')

# if the padded data is not enough consider different colours and distortion for BG

from ultralytics import YOLO

# Initialize model (smaller)
model = YOLO('models/yolov8s.pt')

# Train model on custom dataset
model.train(data='huge_data/cards_kaggle_yolo_padded_v5/data.yaml', epochs=60, augment=True)

model.save("card_detector_model.pt")
