from ultralytics import YOLO
from ray import tune
import torch
import time
import pickle
import ray


def main():
    st = time.time()
    # Load a model
    model = YOLO("yolov8m.yaml")  # build a new model from scratch
    

    model.train(data="C:/Users/CAD6/Documents/TFM/Code/src/config11.yaml", epochs=400, patience=15, mosaic = 0.0, lr0=0.0009,lrf=0.0250,momentum=0.8545,
                weight_decay=0.0005,warmup_epochs=2.5658,box=3.6613,cls=3.7552,cos_lr=True,optimizer='AdamW',imgsz=640,batch=12) 
    
    """ result = model.tune(
        data="C:/Users/CAD6/Documents/TFM/Code/src/config2.yaml",
        space={"lr0": tune.uniform(1e-5, 0.01),
               "lrf": tune.uniform(0.001, 0.1),
               "momentum": tune.uniform(0.6, 0.98),
               "weight_decay": tune.uniform(0.0, 0.001),
               "warmup_epochs": tune.uniform(0.0, 5.0),
               "warmup_momentum": tune.uniform(0.0, 0.95),
               "box": tune.uniform(0.02, 10),
               "cls": tune.uniform(0.2, 4.0),
               "cos_lr":  tune.choice([True, False]),
               "optimizer": tune.choice(['SGD', 'Adam', 'AdamW'])},
        max_samples=25,
        train_args={"epochs": 100, "patience":10, "mosaic":0, 'device':0})


    filename = 'C:/Users/CAD6/Documents/TFM/Code/src/mypickle0114.pkl'
    with open(filename, 'wb') as fi:
        # dump your data into the file
        pickle.dump(result, fi)

    print(result.get_best_result(metric = 'metrics/mAP50(B)', mode = 'max'))"""

if __name__ == '__main__':
    main()