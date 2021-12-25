# IVU
### Workout Assistant 

#### Data Generation

- Input 


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


- From images
  
```python
from ivu.dataset import data_set_over_images

where_is_my_data = "where_is_my_data_path"
where_i_want_to_store_my_data = "where_i_want_to_store_my_data_path"
data_set_over_images(where_is_my_data, where_i_want_to_store_my_data)
```


- From Videos

```python
from ivu.dataset import data_set_over_videos

where_is_my_data = "where_is_my_data_path"
where_i_want_to_store_my_data = "where_i_want_to_store_my_data_path"

# Set the width and height to a non negative integer if the frame has to be resized
data_set_over_videos(where_is_my_data, where_i_want_to_store_my_data, width=-1, height=-1)
```

- Output

```python
from ivu.utils import load_pickle

path_to_pickle = "path_to_pickle"
data = load_pickle(path_to_pickle)
```

Data Generated From Videos contain following columns

`['key_points', 'normalized_key_points', 'normalized_distance_matrix', 'distance_matrix', 'class_label', 'class_label_index', 'file', 'frame_number', 'frame_details']`

Data Generated From Images contain following columns

`['key_points', 'normalized_key_points', 'normalized_distance_matrix', 'distance_matrix', 'class_label', 'class_label_index', 'file']`


