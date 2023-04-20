export PYTHONPATH=../../:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2,3:$CUDA_VISIBLE_DEVICES
python -m torch.distributed.launch --nproc_per_node=2 --master_port=5322 ../../basicsr/train.py -opt config.yml --launcher pytorch
