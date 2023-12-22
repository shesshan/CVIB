GPU_ID=0
seed=2023
data_root_dir='./ABSA_RGAT' # set data dir 
source_domain='rest14'
target_domain='rest14'
config_file='./config/bert_config.json'
num_train_epochs=30
per_gpu_train_batch_sizes=(32)
per_gpu_eval_batch_sizes=(32)
max_grad_norm=1.0

model_name='rgat_bert'
optimizer_type='adamw'
learning_rate=5e-5
nd_lr=1e-3
weight_decay=0

dropout=0.1
gat_dropout=0.5

# RGAT
num_heads=6
num_mlps=2
final_hidden_size=300
# SCL
sdcl_facs=(0.5)
sd_temps=(0.1)
# VIB
kl_facs=(2.5e-7)
ib_lrs=(5e-3)
ib_wd=0

# RGAT-BERT + CVIB
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
                        CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --seed $seed \
                        --cuda_id $GPU_ID \
                        --prune \
                        --use_ib \
                        --spc \
                        --model_name $model_name \
                        --data_root_dir $data_root_dir \
                        --source_domain $source_domain \
                        --target_domain $target_domain \
                        --config_file $config_file \
                        --per_gpu_train_batch_size $per_gpu_train_batch_size \
                        --per_gpu_eval_batch_size $per_gpu_eval_batch_size \
                        --num_train_epochs $num_train_epochs \
                        --max_grad_norm $max_grad_norm \
                        --optimizer_type $optimizer_type \
                        --lr $learning_rate \
                        --nd_lr $nd_lr \
                        --weight_decay $weight_decay \
                        --num_mlps $num_mlps \
                        --final_hidden_size $final_hidden_size \
                        --num_heads $num_heads \
                        --dropout $dropout \
                        --gat_dropout $gat_dropout \
                        --kl_fac $kl_fac \
                        --ib_lr $ib_learning_rate \
                        --ib_wd $ib_wd \
                        --sdcl_fac $sdcl_fac \
                        --sd_temp $sd_temp
                    done
                done
            done
        done
    done
done
