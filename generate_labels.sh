source /home/haolin/anaconda3/bin/activate mink38
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export OMP_NUM_THREADS=12
python generate_labels.py $1 $2 $3
