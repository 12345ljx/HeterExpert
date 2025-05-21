export WANDB_DISABLED='true'
# for task_name in 'arc_easy' 'arc_challenge' 'piqa' 'openbookqa' 'winogrande' 'sciq' 'siqa'

device=0

top_k() {
    local split_mode=$1
    local function_name=$2
    local num_expert=$3
    local num_select=$4

    for task_name in 'arc_easy' 'arc_challenge' 'piqa' 'openbookqa' 'winogrande' 'sciq' 'siqa'
    do
        source ./LLaMA-Factory/scripts/task_setting.sh $task_name
        CUDA_VISIBLE_DEVICES=$device python ./LLaMA-Factory/src/train.py \
            --seed 42 \
            --stage sft \
            --do_train \
            --do_eval \
            --model_name llama3.2-1b \
            --model_name_or_path ./models/llama3.2-1b \
            --task_name $task_name \
            --dataset $dataset \
            --dataset_dir ./LLaMA-Factory/data/harness \
            --val_size 0.05 \
            --function_name $function_name \
            --moeficate \
            --static false \
            --split_mode $split_mode \
            --gate_mode top_k \
            --balance_loss_weight 0.0005 \
            --begin_layer 4 \
            --end_layer 15 \
            --num_expert $num_expert \
            --num_selected_expert $num_select \
            --template llama3 \
            --finetuning_type lora \
            --training_parts gate,lora \
            --lora_target q_proj,k_proj,v_proj,o_proj \
            --additional_target w_logits,weight_noise,gate_layer1,gate_layer2 \
            --lora_rank 8 \
            --lora_dropout 0.05 \
            --output_dir ./LLaMA-Factory/results \
            --overwrite_output_dir \
            --per_device_train_batch_size $batchsize \
            --gradient_accumulation_steps $accsteps \
            --logging_steps 10 \
            --save_strategy steps \
            --save_steps 200 \
            --eval_steps 10 \
            --learning_rate 3e-4 \
            --num_train_epochs $epochs \
            --warmup_ratio 0.05 \
            --lr_scheduler_type cosine_with_min_lr \
            --lr_scheduler_kwargs="{\"min_lr\": 3e-5}" \
            --plot_loss \
            --bf16
    done
}


top_k ilp 'domains(r4l2)' 8 4