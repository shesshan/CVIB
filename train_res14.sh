GPU_ID=0
source_domain='rest'
target_domain='rest'
config_file='config/bert_config.json'
num_train_epochs=30
per_gpu_train_batch_sizes=(32)
per_gpu_eval_batch_sizes=(16)
learning_rate=5e-5
nd_lr=1e-3
weight_decay=0
max_grad_norm=1.0
# RGAT
num_heads=6
dropout=0.3
num_mlps=2
final_hidden_size=300
# clr
sdcl_facs=(0.5)
sd_temps=(0.05)
# vib
kl_facs=(2.5e-7)
ib_lrs=(1e-2)
ib_wd=0

seed=2022
'''
# VIB-SCL
for per_gpu_train_batch_size in "${per_gpu_train_batch_sizes[@]}"
do
    for per_gpu_eval_batch_size in "${per_gpu_eval_batch_sizes[@]}"
    do
        for sdcl_fac in "${sdcl_facs[@]}"
        do
            for sd_temp in "${sd_temps[@]}"
            do
                for kl_fac in "${kl_facs[@]}"
                do 
                    for ib_learning_rate in "${ib_lrs[@]}"
                    do
                        CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --seed $seed --cuda_id $GPU_ID --prune --spc --gat_bert --embedding_type bert --source_domain $source_domain --target_domain $target_domain --kl_fac $kl_fac --ib_lr $ib_learning_rate --ib_wd $ib_wd --sdcl_fac $sdcl_fac --sd_temp $sd_temp --per_gpu_train_batch_size $per_gpu_train_batch_size --per_gpu_eval_batch_size $per_gpu_eval_batch_size --output_dir data/output-gcn-rest --num_heads $num_heads --dropout $dropout --num_mlps $num_mlps --final_hidden_size $final_hidden_size --lr $learning_rate --nd_lr $nd_lr --weight_decay $weight_decay --max_grad_norm $max_grad_norm --num_train_epochs $num_train_epochs --config_file $config_file
                    done
                done
            done
        done
    done
done
'''
'''
# w/o CLR
per_gpu_train_batch_size=32
per_gpu_eval_batch_size=32
learning_rate=5e-5
dropout=0.3
final_hidden_size=200
kl_fac=1e-7
ib_learning_rate=1e-3
CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --seed $seed --cuda_id $GPU_ID --spc --gat_bert --embedding_type bert --source_domain $source_domain --target_domain $target_domain --per_gpu_train_batch_size $per_gpu_train_batch_size --per_gpu_eval_batch_size $per_gpu_eval_batch_size --kl_fac $kl_fac --ib_lr $ib_learning_rate --ib_wd $ib_wd --output_dir data/output-gcn-rest --num_mlps $num_mlps --dropout $dropout --num_heads $num_heads --final_hidden_size $final_hidden_size --lr $learning_rate --nd_lr $nd_lr --weight_decay $weight_decay --num_train_epochs $num_train_epochs --config_file $config_file
'''
'''
# w/o VIB
per_gpu_train_batch_size=32
per_gpu_eval_batch_size=32
learning_rate=5e-5
dropout=0.3
final_hidden_size=200
sdcl_fac=0.25
sd_temp=0.05
CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --cuda_id $GPU_ID --spc --prune --source_domain $source_domain --target_domain $target_domain --gat_bert --embedding_type bert --per_gpu_train_batch_size $per_gpu_train_batch_size --per_gpu_eval_batch_size $per_gpu_eval_batch_size --prune_percent 0.3 --sdcl_fac $sdcl_fac --sd_temp $sd_temp --output_dir data/output-gcn-rest --num_mlps $num_mlps --dropout $dropout --num_heads $num_heads --final_hidden_size $final_hidden_size --lr $learning_rate --nd_lr $nd_lr --weight_decay $weight_decay --num_train_epochs $num_train_epochs --config_file $config_file
'''
'''
# RGAT-BERT
seed=2019
per_gpu_train_batch_size=16
per_gpu_eval_batch_size=32
learning_rate=5e-5
nd_lr=5e-5
weight_decay=0.0
num_heads=6
dropout=0.3
num_mlps=2
final_hidden_size=300
CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --seed $seed --cuda_id $GPU_ID --spc --gat_bert --source_domain $source_domain --target_domain $target_domain --embedding_type bert --per_gpu_train_batch_size $per_gpu_train_batch_size --per_gpu_eval_batch_size $per_gpu_eval_batch_size --output_dir data/output-gcn-rest --num_heads $num_heads --dropout $dropout --num_mlps $num_mlps --final_hidden_size $final_hidden_size --lr $learning_rate --nd_lr $nd_lr --weight_decay $weight_decay --num_train_epochs $num_train_epochs --config_file $config_file # RGAT-BERT in rest14
'''
'''
# RGAT
seed=2019
per_gpu_train_batch_size=16
per_gpu_eval_batch_size=32
learning_rate=5e-5
nd_lr=5e-5
weight_decay=0.0
num_heads=7
dropout=0.8
CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --seed $seed --cuda_id $GPU_ID --spc --gat_our --highway --source_domain $source_domain --target_domain $target_domain --per_gpu_train_batch_size $per_gpu_train_batch_size --per_gpu_eval_batch_size $per_gpu_eval_batch_size --output_dir data/output-gcn-rest --num_heads $num_heads --dropout $dropout --lr $learning_rate --nd_lr $nd_lr --weight_decay $weight_decay --num_train_epochs $num_train_epochs --config_file $config_file # RGAT in restaurant
'''
'''
# ASGCN
seed=2019
per_gpu_train_batch_size=16
per_gpu_eval_batch_size=32
learning_rate=5e-5
nd_lr=5e-5
weight_decay=0.0
num_heads=7
dropout=0.8
CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --seed $seed --cuda_id $GPU_ID --spc --gat_our --highway --gat_attention_type gcn --source_domain $source_domain --target_domain $target_domain --per_gpu_train_batch_size $per_gpu_train_batch_size --per_gpu_eval_batch_size $per_gpu_eval_batch_size --output_dir data/output-gcn-rest --num_heads $num_heads --dropout $dropout --num_mlps $num_mlps --final_hidden_size $final_hidden_size --lr $learning_rate --nd_lr $nd_lr --weight_decay $weight_decay --num_train_epochs $num_train_epochs --config_file $config_file # ASGCN in restaurant
'''
'''
# CapsNet-BERT
num_train_epochs=20
per_gpu_train_batch_size=16
per_gpu_eval_batch_size=16
learning_rate=2e-5
nd_lr=2e-5
weight_decay=0.0
dropout=0.1
max_grad_norm=5.0
#clr
sdcl_fac=0.5
sd_temp=0.1
# vib
kl_fac=2.5e-7
ib_learning_rate=5e-3
ib_wd=0.0
CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --seed $seed --cuda_id $GPU_ID --capsnet_bert --embedding_type bert --source_domain $source_domain --target_domain $target_domain --per_gpu_train_batch_size $per_gpu_train_batch_size --per_gpu_eval_batch_size $per_gpu_eval_batch_size --output_dir data/output-gcn-rest --dropout $dropout --lr $learning_rate --nd_lr $nd_lr --weight_decay $weight_decay --max_grad_norm $max_grad_norm --num_train_epochs $num_train_epochs --config_file $config_file --kl_fac $kl_fac --ib_lr $ib_learning_rate --ib_wd $ib_wd --sdcl_fac $sdcl_fac --sd_temp $sd_temp
'''
'''
# BERT-SPC
num_train_epochs=30
per_gpu_train_batch_size=32
per_gpu_eval_batch_size=32
learning_rate=5e-5
nd_lr=1e-3
weight_decay=0
dropout=0.3
final_hidden_size=300
num_heads=6
num_mlps=2
#clr
sdcl_fac=0.25
sd_temp=0.05
# vib
kl_fac=1e-7
ib_learning_rate=3e-3
ib_wd=0.0
CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --seed $seed --cuda_id $GPU_ID --spc --pure_bert --embedding_type bert --source_domain $source_domain --target_domain $target_domain --per_gpu_train_batch_size $per_gpu_train_batch_size --per_gpu_eval_batch_size $per_gpu_eval_batch_size --output_dir data/output-gcn-rest --dropout $dropout --num_mlps $num_mlps --final_hidden_size $final_hidden_size --num_heads $num_heads --lr $learning_rate --nd_lr $nd_lr --weight_decay $weight_decay --num_train_epochs $num_train_epochs --config_file $config_file
'''
# ATAE-LSTM


# MemNet

# IAN
