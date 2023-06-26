export PYTHONPATH=/opt/data/private/142/BasicSR/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
/root/anaconda3/envs/basicsr/bin/python /opt/data/private/142/BasicSR/basicsr/test.py -opt val_MSRResNet.yml
