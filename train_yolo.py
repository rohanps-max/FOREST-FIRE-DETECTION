import os
from roboflow import Roboflow
from ultralytics import YOLO

def main():
    # 1. Download the dataset from Roboflow
    print("Downloading dataset from Roboflow...")
    rf = Roboflow(api_key="A1x6xFmBHlXOAVivasvM")
    project = rf.workspace("-jwzpw").project("continuous_fire")
    version = project.version(6)
    dataset = version.download("yolov8")
    
    # The dataset downloads to a folder locally. We get the location from the dataset object.
    data_yaml_path = f"{dataset.location}/data.yaml"
    
    print(f"Dataset downloaded to: {dataset.location}")
    print(f"Starting YOLO training using {data_yaml_path}...")
    
    # 2. Train the YOLO model
    # This is the Python equivalent of the !yolo command used in the notebook
    model = YOLO('yolov8s.pt')
    results = model.train(
        data=data_yaml_path,
        epochs=1,
        imgsz=640,
        plots=True
    )
    
    print("Training complete! Check the 'runs/detect/train/' folder for the results and plots.")

if __name__ == '__main__':
    main()





