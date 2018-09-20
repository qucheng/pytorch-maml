exp='maml-cifar100-3way-5shot-TEST'
dataset='cifar100'
num_cls=3
num_inst=5
batch=1
m_batch=32
num_updates=15000
num_inner_updates=5
lr='1e-1'
meta_lr='1e-3'
gpu=0
python2 maml.py $exp --dataset $dataset --num_cls $num_cls --num_inst $num_inst --batch $batch --m_batch $m_batch --num_updates $num_updates --num_inner_updates $num_inner_updates --lr $lr --meta_lr $meta_lr --gpu $gpu 2>&1 | tee ../logs/$exp
