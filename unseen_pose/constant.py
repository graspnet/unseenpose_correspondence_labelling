OBJ_NUM = 1031
KEY_POINT_NUM = 5057
# MODEL_KEY_POINT_DIR = 'voxel_down_sampling'


VOXEL_SIZE = 0.002


IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280
RANDOM_SEED = 27000

with open('/etc/hostname', 'r') as f:
    host_name = f.read()

# GRASPNET_ROOT  = '/Users/fulingyue/Desktop/graspnet'
if host_name.startswith('tmulab'):
    GRASPNET_ROOT = '/home/gmh/graspnet'
elif host_name.startswith('MVIG-24500'):
    # GRASPNET_ROOT = '/ssd1/graspnet'
    GRASPNET_ROOT = '/DATA3/Benchmark/BlenderProcGoogle1000/google1000'
    GRASPNET_ROOT_REAL = '/DATA3/Benchmark/BlenderProcGoogle1000/graspnet'
    BOP_ROOT = '/DATA3/Benchmark/BOP_datasets'
    MODEL_ROOT = '/DATA3/Benchmark/BlenderProcGoogle1000/net_weights'
    LABEL_DIR = '/DATA3/Benchmark/BlenderProcGoogle1000/google1000/graspnet_labels_v3/'
    LABEL_DIR_REAL = '/DATA3/Benchmark/BlenderProcGoogle1000/graspnet/graspnet_labels_v3/'
    EVAL_DIR = '/DATA3/Benchmark/BlenderProcGoogle1000/google1000/evaluation'
    MODEL_KEY_POINT_DIR = '/DATA3/Benchmark/BlenderProcGoogle1000/google1000/models_key_point '
    MODEL_DOWNSAMPLED_DIR = '/DATA3/Benchmark/BlenderProcGoogle1000/google1000/models_down'
    MODEL_DOWNSAMPLED_DIR_REAL = '/DATA3/Benchmark/BlenderProcGoogle1000/graspnet/models_down'
elif host_name.startswith('MVIG-27000'):
    GRASPNET_ROOT_REAL = '/DATA1/Benchmark/unseenpose/graspnet'
    GRASPNET_ROOT = '/DATA1/Benchmark/unseenpose/google1000'
    BOP_ROOT = '/DATA1/Benchmark/unseenpose'
    MODEL_ROOT = '/DATA1/Benchmark/unseenpose/net_weights'
    LABEL_DIR = '/DATA1/Benchmark/unseenpose/google1000/graspnet_labels_v3/'
    LABEL_DIR_REAL = '/DATA1/Benchmark/unseenpose/graspnet/graspnet_labels_v3/'
    EVAL_DIR = '/DATA1/Benchmark/unseenpose/google1000/evaluation'
    MODEL_KEY_POINT_DIR = '/DATA1/Benchmark/unseenpose/google1000/models_key_point '
    MODEL_DOWNSAMPLED_DIR = '/DATA1/Benchmark/unseenpose/google1000/models_down'
    MODEL_DOWNSAMPLED_DIR_REAL = '/DATA1/Benchmark/unseenpose/graspnet/models_down'
elif host_name.startswith('MVIG-28000'):
    GRASPNET_ROOT = '/ssd1/graspnet'
elif host_name.startswith('MVIG-29000'):
    GRASPNET_ROOT = '/home/minghao/graspnet'
elif host_name.startswith('MVIG-100'):
    GRASPNET_ROOT = '/disk1/graspnet'
elif host_name.startswith('vinjohn-TM1801'):
    GRASPNET_ROOT = '/disk1/minghao/graspnet'
else:
    GRASPNET_ROOT = '/place/your/directory/to/synthetic/graspnet/dataset/here'
    GRASPNET_ROOT_REAL = '/place/your/directory/to/real/graspnet/dataset/here'
    LABEL_DIR = '/place/your/directory/to/label/synthetic/here'
    LABEL_DIR_REAL = '/place/your/to/label/synthetic/here'
