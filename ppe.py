from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    model.train(data="Task 2 and 3\ppe_data.yaml", epochs=10,name='yolov8_ppe')
 

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
