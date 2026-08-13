# Fine-tuning with LoRA

# 分布式训练配置
num_nodes=1
node_rank=0
nproc_per_node=4
master_addr=127.0.0.1
master_port=29500

# 路径配置
model_path="./models/BAGEL-7B-MoT"           # 预训练 BAGEL 模型权重
output_path="./results_3d_caption"               # 日志和指标输出目录
ckpt_path="./checkpoints_3d_caption"             # checkpoint 保存目录
data_config="./data/configs/example.yaml"        # 数据配置文件

torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=$nproc_per_node \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/sft_lora.py \
  --dataset_config_file $data_config \
  --model_path $model_path \
  --resume-from $model_path \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --visual_und True \
  --visual_gen False \
  --freeze_und False \
  --freeze_vae True \
  --use_lora True \
  --llm_lora_r 8 \
  --llm_lora_alpha 16 \
  --llm_lora_dropout 0.05 \
  --llm_lora_target_modules "q_proj,v_proj,k_proj,o_proj" \
  --vit_lora_r 16 \
  --vit_lora_alpha 32 \
  --vit_lora_dropout 0.1 \
  --vit_lora_target_modules "q_proj,v_proj" \
  --lr 2e-4 \
  --lr_scheduler cosine \
  --warmup_steps 500 \
  --total_steps 50000 \
  --save_every 1000 \
  --log_every 10 \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240 \
  --num_workers 0 \
  --prefetch_factor 0 \
  --sharding_strategy FULL_SHARD \
  --num_shard 4 \
  --cpu_offload True \
  --results_dir $output_path \
  --checkpoint_dir $ckpt_path \
  --wandb_project bagel_3d_caption \
  --wandb_name run1 \
  --wandb_offline False