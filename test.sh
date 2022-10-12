GPU_ID=0
per_gpu_test_batch_size=8
source_domain='rest'
target_domain='rest'
output_dir='data/output-gcn-rest'

config_file='config/bert_config.json'
log_file='imb_study.log'

seed=2019

'''
# RGAT-BERT & CVIB
# rest-CVIB
dropout=0.3 # drop before MLPs
num_heads=6 # relational heads
num_mlps=2  # MLP layers
final_hidden_size=300
CUDA_VISIBLE_DEVICES=$GPU_ID python test.py --seed $seed --cuda_id $GPU_ID --imb_study --spc --gat_bert --log_file $log_file --source_domain $source_domain --target_domain $target_domain --output_dir $output_dir --embedding_type bert --per_gpu_test_batch_size $per_gpu_test_batch_size --num_mlps $num_mlps --final_hidden_size $final_hidden_size --num_heads $num_heads --dropout $dropout --config_file $config_file
'''
'''
# capsnet
num_train_epochs=20
dropout=0.1 # default 0.1
CUDA_VISIBLE_DEVICES=$GPU_ID python test.py --seed $seed --cuda_id $GPU_ID --imb_study --spc --capsnet_bert --embedding_type bert --log_file $log_file --source_domain $source_domain --target_domain $target_domain --output_dir $output_dir --per_gpu_test_batch_size $per_gpu_test_batch_size --dropout $dropout --config_file $config_file
'''
'''
# bert-spc
dropout=0.3
final_hidden_size=300
num_mlps=2
CUDA_VISIBLE_DEVICES=$GPU_ID python test.py --seed $seed --cuda_id $GPU_ID --imb_study --spc --pure_bert --embedding_type bert --log_file $log_file --source_domain $source_domain --target_domain $target_domain --output_dir $output_dir --per_gpu_test_batch_size $per_gpu_test_batch_size --dropout $dropout --num_mlps $num_mlps --final_hidden_size $final_hidden_size --config_file $config_file
'''
'''
# RGAT
num_heads=7
dropout=0.8
CUDA_VISIBLE_DEVICES=$GPU_ID python test.py --seed $seed --cuda_id $GPU_ID --imb_study --spc --gat_our --highway --source_domain $source_domain --target_domain $target_domain --per_gpu_test_batch_size $per_gpu_test_batch_size --output_dir $output_dir --num_heads $num_heads --dropout $dropout --config_file $config_file # RGAT in restaurant
'''
# ASGCN
num_heads=7
dropout=0.8
CUDA_VISIBLE_DEVICES=$GPU_ID python test.py --seed $seed --cuda_id $GPU_ID --imb_study --spc --gat_our --highway --gat_attention_type gcn --source_domain $source_domain --target_domain $target_domain --per_gpu_test_batch_size $per_gpu_test_batch_size --output_dir $output_dir --num_heads $num_heads --dropout $dropout --config_file $config_file # ASGCN in restaurant
