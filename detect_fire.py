import cv2
from ultralytics import YOLO

def main():
    # Load the trained YOLO model
    # Note: 'runs/detect/train/weights/best.pt' is the default path where YOLO saves the best model
    # Make sure this file exists before running this script!
    try:
        model = YOLO('runs/detect/train8/weights/best.pt')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Have you finished training the model first? Run 'python train_yolo.py' first!")
        return

    # Enter the path to the image or video you want to test
    source = input("Enter the path to the image or video you want to test (e.g., 'test.jpg' or 'video.mp4'): ")

    # Run inference on the source
    # save=True will save the output with bounding boxes to runs/detect/predict/
    # show=True will open a window showing the result (works best for videos or if you have a display)
    results = model.predict(source=source, save=True, show=True)
    
    print("\nInference complete!")
    print("Check the 'runs/detect/predict/' folder for the saved results.")

if __name__ == '__main__':
    main()
