# TechTask - Building classifier for breast ultrasound lesion

Build “shape” and “orientation” feature classifiers for the breast ultrasound lesion image.

## Installation

Clone repo and install requirements.txt in a Python>=3.7.0 environment.

```bash
git clone https://github.com/yikai28/techtask.git  # clone
cd techtask
pip install -r requirements.txt  # install
```

## Code structure

    .
    ├── data                          # data
    │   ├── tech_task_data            # image files (inside all images (.png))
    │   ├── annotations.json          # annotation files
    ├── model  
    │   ├── cnn.py                    # CNN model class
    ├── utils  
    │   ├── read_data.py              # read data class
    │   ├── prep_data.py              # prepare data class
    ├── results  
    │   ├── eval_metric_results.json  # test set evaluation metrics results
    ├── predictions 
    │   ├── predictions.json          # extra dataset (if provided) predict result 
    ├── main.py                       # main function
    ├── README.md                     # readme
    ├── requirements.txt              # all the requirement packages
    └──weights_best.h5                # the best weight from train

## Usage
Put the dataset under the data directory, follow the data structure. Otherwise you have to feed your own --img-dir and --annotations to the function
```
python main.py [-h] --mode MODE --img-dir $IMG_DIR --annotations $ANNOTATIONS --eval-save-dir $EVAL_SAVE_DIR  --pred-save-dir $PRED_SAVE_DIR

optional arguments:
  -h, --help                      Show this help message and exit
  --mode MODE                     Choose from ['parameter_search', 'train', 'test', 'pred']
  --img-dir $IMG_DIR              Directory of images
  --annotations $ANNOTATIONS      Input json annotation file
  --eval-save-dir $EVAL_SAVE_DIR  Results save directory
  --pred-save-dir $PRED_SAVE_DIR  Extra predictions save directory
```

## Train
```bash
[CUDA_VISIBLE_DEVICES=0] python main.py --mode train
```
## Parameter Search 
You can specific the Hyperparameter you want to search on parameter_search function in cnn.py
```bash
[CUDA_VISIBLE_DEVICES=0] python main.py --mode parameter_search
```
## Test
```bash
[CUDA_VISIBLE_DEVICES=0] python main.py --mode test
```
It will automatically generate the file under results/eval_metric_results.json 

Including the metrics TP, FP, FN, F1, average shape feature f1, average orientation feature f1, average total f1
e.g. 
```json
{
        "metrics": ["oval", "round", "irregular", "parallel", "not_parallel"],
        "TP":      [9, 0, 6, 13, 7],
        "FP":      [5, 2, 3, 1, 2],
        "FN":      [1, 4, 3, 2, 1],
        "F1":      [0.75, 0.0, 0.67, 0.9, 0.82],
        "shape acc":  0.65,
        "ori acc":  0.87,
        "average shape f1": 0.47,
        "average ori f1": 0.86,
        "average total": 0.67
},
 ```

## Predict
```bash
[CUDA_VISIBLE_DEVICES=0] python main.py --mode pred --img-dir $IMG_DIR --annotations $ANNOTATIONS
```
It will automatically generate the file under results/predictions.json
e.g. 
```json
{
        "img_names": "0.png",
        "shape": "oval",
        "orientation": "not_parallel"
},
 ```


## Contributing

This repro is developed by Yikai Yang for the interview purpose.

If you have any questions, please email to yangyikai28@gmail.com

## License

[MIT](https://choosealicense.com/licenses/mit/)