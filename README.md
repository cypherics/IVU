# IVU


#### Folder structure


        Dataset_dir
        |-- class_1
            |-img/video
            |-img/video
            |...
            |...
        |-- class_2
            |-img/video
            |-img/video
            |...
            |...
        |-- class_n
            |-img/video
            |-img/video
            |...
            |...

#### Generate Data From images

```python
from ivu.data.creation import training_data_set_over_images

where_is_my_data = "where_is_my_data_path"
where_i_want_to_store_my_data = "where_i_want_to_store_my_data_path"
training_data_set_over_images(where_is_my_data, where_i_want_to_store_my_data)
```

#### Generate Data from Videos with Parameters
```python
from ivu.data.creation import training_data_set_over_videos

where_is_my_data = "where_is_my_data_path"
where_i_want_to_store_my_data = "where_i_want_to_store_my_data_path"

# Set the width and height to a non negative integer if the frame has to be resized
training_data_set_over_videos(where_is_my_data, where_i_want_to_store_my_data, width=-1, height=-1)
```

#### Generate Data from Videos with config
```python
from ivu.data.creation import training_data_set_over_videos_using_conf
training_data_set_over_videos_using_conf("path_to_dat_config")
```

#### Load generated data


```python
from ivu.utils import load_pickle

path_to_pickle = "path_to_pickle"
data = load_pickle(path_to_pickle)
```

#### Structure for the data generated

Data Generated From Videos contain following columns

`['key_points', 'normalized_key_points', 'normalized_distance_matrix', 'distance_matrix', 'class_label', 'class_label_index', 'file', 'frame_number', 'frame_details']`

Data Generated From Images contain following columns

`['key_points', 'normalized_key_points', 'normalized_distance_matrix', 'distance_matrix', 'class_label', 'class_label_index', 'file']`

#### Train on data
```python
from ivu.trainer import Trainer

trainer = Trainer.train_with_normalized_distance_matrix(data_pth=r"path_to_pickle",
                                                        conf_pth=r"path_to_config")
trainer.start_training()
```

#### Inference on data

```python
from ivu.inference import Inference

vd = Inference.init_inference_from_config_pth("path_to_config_file")
vd.run()

```
