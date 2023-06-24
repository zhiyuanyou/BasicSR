export PYTHONPATH=/opt/data/private/142/BasicSR/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
/root/anaconda3/envs/basicsr/bin/python -m torch.distributed.launch --nproc_per_node=4 --master_port=8822 /opt/data/private/142/BasicSR/basicsr/train.py -opt train_MSRResNet.yml --launcher pytorch --auto_resume
