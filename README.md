# unseen_6d_pose_correpsondence_labelling

## Dependencies

graspnetAPI

open3d==0.12.0.0

tensorboard

tqdm

torch

Minkowski Engine

## Prepare codes

replace the 
```bash
GRASPNET_ROOT = '/place/your/directory/to/synthetic/graspnet/dataset/here'
GRASPNET_ROOT_REAL = '/place/your/directory/to/real/graspnet/dataset/here'
LABEL_DIR = '/place/your/directory/to/label/synthetic/here'
LABEL_DIR_REAL = '/place/your/to/label/synthetic/here'
```
by the data set root and the directory you want to place the labels

```GRASPNET_ROOT``` is the root directory to the synthetic dataset donated by https://graspnet.net/unseenpose.html
```GRASPNET_ROOT_REAL``` is the root directory to the real data donated by https://graspnet.net/datasets.html


### Run the code with 
```bash
bash ./generate_labels.sh $current_process_id $total_process_count $isreal
```

You can run multi processes simultaneously by having ```$total_process_count > 1```. The total dataset will be divided into 
```$total_process_count``` part.

```$current_process_id``` means this process is processing the ```$current_process_id```th part.

```isreal``` means is ```True``` when your dataset real elsewise ```False``` 
