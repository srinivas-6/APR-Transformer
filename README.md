## Learning Ego Pose Regression using Multi-Camera with Transformers

![Model architecture](assets/apr-tranformer.drawio.png)


### Repository Overview 

This code implements:

1. Training of a Transformer-based architecture for absolute ego pose regression
2. Testing the same model against various datasets 


---

### Prerequisites

In order to run this repository you will need:

1. Python3 (tested with Python 3.7.7, 3.8.5), PyTorch
2. Set up the conda environment with ```conda env create -f environment.yml```
3. Benchmarking on various datasets
3. Download the [DeepLoc dataset](http://deeploc.cs.uni-freiburg.de/)
4. Download the Oxford Robot Car dataset [Oxford Robot Car](https://robotcar-dataset.robots.ox.ac.uk/) or use the [RobustLoc](https://github.com/sijieaaa/RobustLoc) project that provide the Oxford RobotCar dataset that has been [pre-processed](https://github.com/sijieaaa/RobustLoc) 


### Pretrained Models 
TBD

### Usage

  For detailed explanation of the options run:)
  ```
  python main.py -h
  ```
  For example, in order to train the model on the DeepLoc dataset or Beintelli run: 
  ```
python main.py --model_name=ms-transposenet --config_file=LocationRetrival_config.json --mode=train --experiment {EXP_NAME} --entity {WANDB_USERNAME}
  ```
  Your checkpoints (.pth file saved based on the number you specify in the configuration file) and log file
  will be saved under an 'out' folder.

  **You will need a wandb account for logging the training metrics. Please pass your wandb username for the 'entity' flag**

  
  
  In order to test your model:
  ```
  python main.py --model_name=ms-transposenet --backbone_path=efficientnet --config_file=LocationRetrival_config.json --mode=test --checkpoint_path <path to your checkpoint .pth> --experiment {EXP_NAME} --entity {WANDB_USERNAME}
  ```

  Convert the trained pytorch model checkpoint to onnx for deployment
  ```
  python torch_to_onnx.py --help

  ```
  Run inference on the converted onnx model
  
  ```
  python inference.py --help

  ```

  
  
### Results

TBD
  
