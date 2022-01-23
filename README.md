# IVU

### DATASET

The complete processed dataset can be downloaded from [complete dataset](https://drive.google.com/file/d/1qRHneW23Pgmem38N3yCUeoJhRn8UYTww/view?usp=sharing). 
Dataset which is split into train and test can be downloaded from [split data for training and testing](https://drive.google.com/drive/folders/1U9_Aw5AxK3iqqWGW__O0Ezx9CeaaE3rf?usp=sharing).

### Description of the data

| Value                      | Description                                                                       |
|----------------------------|-----------------------------------------------------------------------------------|
| key_points                 | 17 body keypoints                                                                 |
| normalized_key_points      | key points are normalized with respect to body position                           |
| distance_matrix            | pair wise distance between key points                                             |
| normalized_distance_matrix | pair wise distance between key points is normalized with respect to body position |
| class_label                | name of the class                                                                 |
| class_label_idx            | integer equivalent of class_label                                                 |
| file                       | name of the file                                                                  |
| frame_number               | respective frame of the file                                                      |
| frame_details              | details with respect to over all files                                            |

The size of features that are used for training for ```key_points``` & `normalized_key_points` is `51` , 
and for `distance_matrix` & `normalized_distance_matrix` is `136`.

### Training

- #### Config SetUp
  In order to start the training, all the required training parameters, must be 
      specified in a config file. 
    
    ```yaml
    data:
      validation_split: 0.2
      
      # Number of frames that are to considered as a single sequence during training 
      stride: 32
    
    optimizer:
      name: Adam
      parameters:
        lr: 0.001
    
    callbacks:
      ReduceLROnPlateau:
        parameters:
          patience: 3
          verbose: 1
          factor: 0.2
          min_lr: 0.000001
    
      EarlyStopping:
        parameters:
          min_delta: 0.001
          patience: 8
          verbose: 1
    
      ModelCheckpoint:
        parameters:
          verbose: 1
          save_best_only: True
          save_weights_only: False
    loss:
      name: CategoricalCrossentropy
      parameters:
        from_logits: True
    
    batch_size: 32
    epochs: 300
    shuffle: True
    
    model :
      # name of the model to use for training
      name : 
      # model input parameters
      parameters:
        # size of input features
        input_features: 136 
        # total number of classes
        n_classes: 7
    
    log_dir : logs/
    ```
    #### LSTM model configuration
 
    ```yaml
    model :
      name : lstm_kar_model
      parameters:
        hidden_units: 64
        input_features: # value based on type of data
        n_classes: 7
        penalty: 0.0001
    ```
    #### Convolution model configuration
    ```yaml
    model:
      name: temporal_model
      parameters:
        input_features: # value based on type of data
        n_classes: 7
    ```
  
    Based on the type of data used for training, the size of `input_features` will vary. 
If data used for training is either ```key_points``` or `normalized_key_points` set `input_features: 51` , 
else if data used is `distance_matrix` or `normalized_distance_matrix` set `input_features: 136`.

- #### Start Training
    Once th config is set, then based on the choice of the type of the data that is being
used on can perform the training.
    
    #### Train when data is Normalized Distance Matrix
    
    ```python
    from ivu.trainer import Trainer
    
    trainer = Trainer.train_with_normalized_distance_matrix(train_pth=r"path_to_train_pickle",
                                                            test_pth=r"path_to_test_pickle",
                                                            conf_pth=r"path_to_config")
    trainer.start_training()
    ```

    #### Train when data is Distance Matrix
    
    ```python
    from ivu.trainer import Trainer
    
    trainer = Trainer.train_with_distance_matrix(train_pth=r"path_to_train_pickle",
                                                            test_pth=r"path_to_test_pickle",
                                                            conf_pth=r"path_to_config")
    trainer.start_training()
    ```
  
    #### Train when data is Normalized KeyPoints
    
    ```python
    from ivu.trainer import Trainer
    
    trainer = Trainer.train_with_normalized_key_points(train_pth=r"path_to_train_pickle",
                                                            test_pth=r"path_to_test_pickle",
                                                            conf_pth=r"path_to_config")
    trainer.start_training()
    ```

### Run Experiments
Predefined config can be found in `config/`, which can be used to perform experiments.
Just run the ```train_experiments.py```, with path to data, which will start performing experiments on those configurations present in the folder.

If both `train_key_points.pickle` and `test_key_points.pickle` are available the run the following command to perform experiments using those files
```shell
python run_train_and_testing.py path/to/train.pickle path/to/test.pickle
```

If just `train_key_points.pickle` is available the run the following command to perform experiments that file without test set.
```shell
python run_train_and_testing.py path/to/train.pickle None
```

The results of the experiment can be viewed in `logs/` folder, which is generated in the present working directly. 
The folder contains saved model in the folder named `chk`, training graphs saved in the folder name `graphs`, a pdf containing the confusion matrix and its test metric.

To visualize the training graph run the following command in the terminal
```shell
tensorboard --logdir path/till/graph
```


### Inference

```shell
INFER_FOR = ["normalized_distance_matrix", "distance_matrix", "normalized_key_points"]
```

Based on what type of data the model was trained on choose the appropriate mode, 
below command is when mode is `normalized_distance_matrix` and stride of `32`
```shell
python run_example.py normalized_distance_matrix path/till/the/folder/chk path/where/my/testing/videos/are path/where/to/save/results 32
```

### Run Demo Using pretrained weights
- Download pretrained weights from [here]()
- Download the videos for demo [here]()
- Extract the downloaded model.zip, the extracted folder will contain a `chk` folder, the path till the `chk` folder is the model path
- Extract the downloaded demo_videos.zip
- Run the following command to run inference on the downloaded model.
```shell
python run_example.py normalized_key_points path/till/the/extracted/model/chk path/till/the/extracted/demo_videos path/where/to/save/results 64
```
