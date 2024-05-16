import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"No image found at {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def predict_and_annotate(img_rgb, model, conf=0.3):
    results = model.predict(img_rgb, conf=conf)
    annotator = Annotator(img_rgb)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
    return annotator.result()


def save_image(img, path):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    model = YOLO('yolo_v5.pt')
    # img = load_image('my_data/testing/ace_of_hearts_pinkbg.jpg')
    # img = load_image('my_data/mouse_pad/6_of_hearts.jpg')
    # img = load_image('my_data/white_bg/4_of_clubs.jpg')
    # img = load_image('my_data/white_bg/4_of_clubs_cropped.jpg')
    # img = load_image('./001.jpg')
    # img = load_image('my_data/floor/6_of_hearts.jpg')
    # img = load_image('my_data/floor/ace_of_hearts.jpg')
    # img = load_image('my_data/floor/3_cards.jpg')
    # img = load_image('my_data/black_bg/jack_of_clubs.jpg')
    img = load_image('my_data/camera/ace_of_hearts.png')

    # img = cv2.resize(img, (288, 288))
    annotated_img = predict_and_annotate(img, model)
    save_image(annotated_img, 'other/annotator.png')


if __name__ == "__main__":
    main()
