# Multi-Task Network for joint lane and object detection

### Requirements

Python 3.7 \
PyTorch 1.4.0\
Cuda 10.0\
Torchvision 0.5.0

### Installation
1. Clone this repository
    ```
    git clone https://github.com/shecker/AegisMultitask.git
    ```
2. Create conda environment
    ```
    conda create -n MTL python=3.7
    ```
3. Install requirements
    ```
    pip install -r requirements.txt
    ```

The configuration file contains all parameters relevant for training, testing, and evaluation. 
Please refer to 'Configuration' below for further detail.

### Training
```
python train.py
```

### Inference
```
python test.py -p2w /path/to/weights/ -p2i /path/to/image/folder/ -s /path/to/folder/to/save/results
```

### Evaluation
Set *eval_only=true* and specify *path_to_weights* in *config.json*.
```
python train.py
```

### Speed test
```
python speed_test_real.py -bb "depth_of_backbone" -w_ "input_width" -h_ "input_height" 
```

### Configuration
Modifications can be directly performed in *config.json*
```
### Architecture
"type": "AegisMTModel",
"backbone_layers": Depth of backbone. Default: "34"
"pretrained_bb": Bool. Indicates whether backbone uses pretrained weights from ImageNet. Default: true
"use_aux": Bool. Indicates wheter auxiliary segmentation branch is used. Default: true
"griding_num": Number of grid columns. Default: 100
"cls_num_per_lane": Number of points per lane. Default: 56
"head_conv": Number of intermediate channels for backbone layers. Default: 64
"num_lanes": Maximal number of lanes per image. Default: 4
"num_obj_classes": Number of object classes. Default: 5
"max_obj_number": Maximum number of objects per image. Default: 128
"input_width": Width of input. Default: 800
"input_height": Height of input. Default: 288

### Optimizer
"lr": Initial learning rate. Default: 1e-5
"weight_decay": Weight decay. Default: 1e-4
"amsgrad": Bool. Indicates whether amsgrad is used for updating weights. Default: false

### Learning Rate Scheduler
"step_size": Step interval to update learning rate. Default: 60
"gamma": Learning rate multiplier. Default: 0.5

### Trainer
"stochastic": Bool. Indicates whether training is stochastic. Default: true
"obj_prob": Probability of using a batch of object-annotated images. Default: 0.5
"epochs": Number of epochs. Default: 140
"save_period": Interval of epochs to save model weights. Default: 25
"save_best": Bool. indicates whether best model wrt validation is saved. Default: true
"resume": Bool. Indicates whether pre-trained weights are used to resume training. Update path to weights. Default: false
"eval_only": Bool. Indicates whether model is evaluated only. Update path to weights. Default: false
"path_to_weights": Path to weights for resuming or evaluating
"log_dir": path to logging directory. Written dynamically by scripts
"checkpoint_dir": path do directory where model weights are saved. Written dynamically by scripts
"save_dir": Name of directory to save results. Default: "results"
"inference_dir": Name of directory to save inference results. Written dynamically by script
"tensorboard": Writes results to tensorboard. Default: true
"early_stop": Stop training if validation performance does not improve for a number of epochs. Default: 30

### Datasets
"obj": Name of object dataset if a separate one is used for lanes and objects. Default: "bdd100k"
"lanes": Name of lane dataset if a separate one is used for lanes and objects. Default: "tusimple"
"obj_and_lanes": Bool. Indicates whether a dataset with both objects and lanes annotations are used. Default: false

### Dataset Augmentations
"motionBlur": Probability of applying motion blur to images. Default: 0.0
"brightness": Probability of applying brightness to images. Default: 0.0
"shiftScaleRotate": Probability of applying shift, scale, and rotation. Default: 1.0
"flip": Probability of flipping object annotated images. Default: 0.5
"scale": Bool. Indicates whether object annotated images are scaled. Default: true
"shift": Bool. Indicates whether object annotated images are shifted. Default: true
"rotate": Bool. Indicates whether object annotated images are rotated. Default: false

### Loss
"sim_loss_w": Weight of similarity lane loss. Default: 1.0
"shp_loss_w": Weight of lane shape loss. Default: 0.0
"loss_multiplier_lanes": Total loss multiplier. Default: 1.0
"loss_multiplier": Lane loss multiplier. Default: 1.0

### Dataloaders
"type": Name of dataloader. Convention: Name of dataset + 'DataLoader'. Default: "TuSimpleDataloader" and "BDD100kDataLoader"
"dataset": Name of dataset for specific dataset. Default. "TuSimple" and "BDD100k"
"data_dir": Path to folder where dataset is stored
"batch_size": Batch size . Default: 32
"shuffle": Bool. Indicates whether images are shuffled. Default: true
"validation_split": Not used
"num_workers": Number of workers. Default: 1
"drop_last": Bool. Indicates whether last batch of images is dropped if toal number of images is not a multiple of the batch size. Default: true
"griding_num": Number of grid columns. Relevant for dataloaders of lane datastes. Default: 100
"cls_num_per_lane": Number of points per lane. Relevant for dataloader of lane datasets. Default: 56
"num_lanes": Maximal number of lanes per image. Relevant for dataloaders of lane datasets. Default: 4
```
