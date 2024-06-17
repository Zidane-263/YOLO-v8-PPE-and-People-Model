import os
import cv2
import torch
import argparse
from pathlib import Path
from ultralytics import YOLO  

def draw_boxes(img, results, class_names, label_prefix, color):
    for result in results:
        x1, y1, x2, y2 = map(int, result[:4])
        conf = result[4]
        cls = int(result[5])
        class_name = class_names[cls]
        label = f'{label_prefix} {class_name}: {conf:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def process_images(input_dir, output_dir, person_model_path, ppe_model_path):
    if not os.path.exists(input_dir):
        print(f'Input directory {input_dir} does not exist.')
        return
    if not os.path.isfile(person_model_path):
        print(f'Person detection model {person_model_path} does not exist.')
        return
    if not os.path.isfile(ppe_model_path):
        print(f'PPE detection model {ppe_model_path} does not exist.')
        return

    person_model = YOLO(person_model_path)
    ppe_model = YOLO(ppe_model_path)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    person_class_names = person_model.names
    ppe_class_names = ppe_model.names

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f'Failed to load image {img_path}')
            continue

        try:
            person_results = person_model(img)[0].boxes.data.cpu().numpy()
            ppe_results = ppe_model(img)[0].boxes.data.cpu().numpy()
        except Exception as e:
            print(f'Error processing image {img_path}: {e}')
            continue

        draw_boxes(img, person_results, person_class_names, 'Person', (255, 0, 0))
        draw_boxes(img, ppe_results, ppe_class_names, 'PPE', (0, 255, 0))

        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, img)
        print(f'Saved annotated image to {output_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Inference Script")
    parser.add_argument('--input_dir', type=str, required=True, help=r'C:\Users\Zidan\OneDrive\Desktop\finale\Task 4 and 5\input')
    parser.add_argument('--output_dir', type=str, required=True, help=r'C:\Users\Zidan\OneDrive\Desktop\finale\Task 4 and 5\output')
    parser.add_argument('--person_det_model', type=str, required=True, help=r'weights\person.pt')
    parser.add_argument('--ppe_detection_model', type=str, required=True, help=r'weights\ppe.pt')
    args = parser.parse_args()

    process_images(args.input_dir, args.output_dir, args.person_det_model, args.ppe_detection_model)
