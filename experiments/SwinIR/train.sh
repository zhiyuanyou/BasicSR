export PYTHONPATH=../../:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0:$CUDA_VISIBLE_DEVICES
python ../../basicsr/train.py -opt config.yml
