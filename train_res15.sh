GPU_ID=0
config_file='config/bert_config.json'
source_domain='rest15'
target_domain='rest15'
num_train_epochs=40
per_gpu_train_batch_sizes=(32)
per_gpu_eval_batch_sizes=(32)
learning_rates=(5e-5)
nd_lr=1e-3
weight_decay=0
max_grad_norm=1.0
# RGAT
num_heads=8
dropout=0.3
num_mlps=2
final_hidden_sizes=(300)
# CLR
sdcl_facs=(0.25)
sd_temps=(0.05)
# VIB
kl_facs=(1e-7)
ib_lrs=(5e-3)
ib_wd=0

seed=2022

# CVIB
for per_gpu_train_batch_size in "${per_gpu_train_batch_sizes[@]}"
do
    for per_gpu_eval_batch_size in "${per_gpu_eval_batch_sizes[@]}"
    do
        for final_hidden_size in "${final_hidden_sizes[@]}"
        do
            for learning_rate in "${learning_rates[@]}"
            do
                for sdcl_fac in "${sdcl_facs[@]}"
                do
                    for sd_temp in "${sd_temps[@]}"
                    do
                        for kl_fac in "${kl_facs[@]}"
                        do      
                            for ib_learning_rate in "${ib_lrs[@]}"
                            do
                                CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --cuda_id $GPU_ID --seed $seed --prune --spc --gat_bert --embedding_type bert --source_domain $source_domain --target_domain $target_domain --output_dir data/output-gcn-rest15 --num_mlps $num_mlps --num_heads $num_heads --dropout $dropout --final_hidden_size $final_hidden_size --per_gpu_train_batch_size $per_gpu_train_batch_size --per_gpu_eval_batch_size $per_gpu_eval_batch_size --kl_fac $kl_fac --ib_lr $ib_learning_rate --ib_wd $ib_wd --sdcl_fac $sdcl_fac --sd_temp $sd_temp --lr $learning_rate --nd_lr $nd_lr --weight_decay $weight_decay --max_grad_norm $max_grad_norm --num_train_epochs $num_train_epochs --config_file $config_file
                            done
                        done
                    done
                done
            done
        done
    done
done

'''
# w/o CLR
per_gpu_train_batch_size=32
per_gpu_eval_batch_size=32
learning_rate=5e-5
final_hidden_size=300
kl_fac=1e-7
ib_learning_rate=5e-3
CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --seed $seed --cuda_id $GPU_ID --spc --gat_bert --embedding_type bert --source_domain $source_domain --target_domain $target_domain --per_gpu_train_batch_size $per_gpu_train_batch_size --per_gpu_eval_batch_size $per_gpu_eval_batch_size --output_dir data/output-gcn-rest15 --num_mlps $num_mlps --num_heads $num_heads --dropout $dropout --final_hidden_size $final_hidden_size --kl_fac $kl_fac --ib_lr $ib_learning_rate --ib_wd $ib_wd --lr $learning_rate --nd_lr $nd_lr --weight_decay $weight_decay --num_train_epochs $num_train_epochs --config_file $config_file
'''
'''
# w/o VIB
per_gpu_train_batch_size=32
per_gpu_eval_batch_size=32
learning_rate=5e-5
final_hidden_size=300
sdcl_fac=0.25
sd_temp=0.05
CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --seed $seed --cuda_id $GPU_ID --prune --gat_bert --embedding_type bert --source_domain $source_domain --target_domain $target_domain --per_gpu_train_batch_size $per_gpu_train_batch_size --per_gpu_eval_batch_size $per_gpu_eval_batch_size --prune_percent 0.3 --sdcl_fac $sdcl_fac --sd_temp $sd_temp --output_dir data/output-gcn-rest15 --num_mlps $num_mlps --num_heads $num_heads --dropout $dropout --final_hidden_size $final_hidden_size --lr $learning_rate --nd_lr $nd_lr --weight_decay $weight_decay --num_train_epochs $num_train_epochs --config_file $config_file
'''
'''
# RGAT-BERT
seed=2019
per_gpu_train_batch_size=16
per_gpu_eval_batch_size=32
learning_rate=5e-5
nd_lr=5e-5
weight_decay=0.0
num_heads=8
dropout=0.3
num_mlps=2
final_hidden_size=300
CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --seed $seed --cuda_id $GPU_ID --spc --gat_bert --embedding_type bert --per_gpu_train_batch_size $per_gpu_train_batch_size --per_gpu_eval_batch_size $per_gpu_eval_batch_size --source_domain $source_domain --target_domain $target_domain --output_dir data/output-gcn-rest15 --num_heads $num_heads --dropout $dropout --num_mlps $num_mlps --final_hidden_size $final_hidden_size --lr $learning_rate --nd_lr $nd_lr --weight_decay $weight_decay --num_train_epochs $num_train_epochs --config_file $config_file # RGAT-BERT in res15
'''
'''
# RGAT
seed=2019
num_train_epochs=30
per_gpu_train_batch_size=16
per_gpu_eval_batch_size=16
num_heads=8
dropout=0.7
num_layers=3
learning_rate=3e-4
CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --seed $seed --cuda_id $GPU_ID --spc --gat_our --highway --num_layers $num_layers --source_domain $source_domain --target_domain $target_domain --per_gpu_train_batch_size $per_gpu_train_batch_size --per_gpu_eval_batch_size $per_gpu_eval_batch_size --output_dir data/output-gcn-rest15 --num_heads $num_heads --dropout $dropout --num_mlps $num_mlps --lr $learning_rate --nd_lr $nd_lr --weight_decay $weight_decay --num_train_epochs $num_train_epochs --config_file $config_file # RGAT in rest15
'''
'''
# CapsNet-BERT
num_train_epochs=30
per_gpu_train_batch_size=32
per_gpu_eval_batch_size=32
learning_rate=5e-5
nd_lr=5e-5
weight_decay=0.0
dropout=0.1
max_grad_norm=5.0
CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --seed $seed --cuda_id $GPU_ID --capsnet_bert --embedding_type bert --source_domain $source_domain --target_domain $target_domain --per_gpu_train_batch_size $per_gpu_train_batch_size --per_gpu_eval_batch_size $per_gpu_eval_batch_size --output_dir data/output-gcn-rest15 --dropout $dropout --lr $learning_rate --nd_lr $nd_lr --weight_decay $weight_decay --max_grad_norm $max_grad_norm --num_train_epochs $num_train_epochs --config_file $config_file 
#--kl_fac $kl_fac --ib_lr $ib_learning_rate --ib_wd $ib_wd --sdcl_fac $sdcl_fac --sd_temp $sd_temp
'''