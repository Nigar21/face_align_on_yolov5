# Face roll score fixing with Yolov5 Face Detection

## Description
The project is a wrap over [yolov5-face](https://github.com/deepcam-cn/yolov5-face) repo. Model detects faces on images and returns bounding boxes and coordinates of 5 facial keypoints, which can be used for face alignment. Taking the keypoints, we measure the roll alignment for a given face, and correct it using cosine rule. 
## Installation
```bash
pip install -r requirements.txt
```


## Other pretrained models
You can use any model from [yolov5-face](https://github.com/deepcam-cn/yolov5-face) repo. Default models are saved as entire torch module and are bound to the specific classes and the exact directory structure used when the model was saved by authors. To make model portable and run it via my interface you must save it as pytorch state_dict and put new weights in `weights/` folder. Example below:


## Result example
<img src="/anne_hathaway.jpg" width="360"/> <img src="/anne_hathaway_aligned.jpg" width="360"/>
<img src="/matthew_mcconaughey.jpg" width="360"/> <img src="/matthew_mcconaughey_aligned.jpg" width="360"/>


## Citiation
Thanks [deepcam-cn](https://github.com/deepcam-cn/yolov5-face) for pretrained models.

Thanks [elyha7](https://github.com/elyha7/yoloface) for detailed repo on yolov5-face  

