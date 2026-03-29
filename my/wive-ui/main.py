# from onnxslim.core import freeze

from ultralytics import YOLO
from ultralytics import RTDETR

if __name__ == "__main__":
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("./yolov8_C2FTSSA2.yaml")
    # model = YOLO("./model.pt")
    # model.load('./model.pt')
    results = model.train(
        data="./data.yaml",
        epochs=100,
        # device=[1,0]
        # freeze=[0,1,2,3,4,5,6,7,8,9,12,17,22,27]
    )

    results = model.val()
    success = model.export(format="onnx")