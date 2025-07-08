import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/HazeData/train132/weights/best.pt')
    results = model.val(data='datasets/Haze_data/data.yaml',
                       split='val',
                       imgsz=640,
                       batch=32,
                       # rect=False,
                       # save_json=True, 
                       project='runs/HazeData',
                       name='val',
                       ) 
    # print(results.box.map)  # map50-95
    # print(results.box.maps)  # a list contains map50-95 of each category

    # # Print the results
    # print("results:")
    # print(f"mAP@0.5: {results.maps[0]:.4f}")  # mAP@0.5
    # print(f"mAP@0.5:0.95: {results.maps.mean():.4f}")  # mAP@0.5:0.95

    # Print additional metrics
    print("additional metrics:")
    for key, value in results.results_dict.items():
        print(f"{key}: {value:.4f}")