export PYTHONPATH=../../:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1
python ../../basicsr/test.py -opt opt_val_div.yml --ckpt_iter $2
