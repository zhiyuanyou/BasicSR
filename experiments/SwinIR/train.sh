export PYTHONPATH=../../:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1
python -m torch.distributed.launch --nproc_per_node=$2 --master_port=$3 ../../basicsr/train.py -opt config.yml --launcher pytorch
