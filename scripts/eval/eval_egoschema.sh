
MODEL_NAME="llava-ov-7B-bin_llava558k_pt_flat_zip_baseline"
MODEL_PATH='checkpoints/llava-onevision--siglip-so400m-patch14-384-Qwen2.5-7B-Instruct-llava558k-pt-oryx-ft-flat'

# MODEL_NAME="llava-ov-0.5B-bin_llava558k_pt_oryx_flat_baseline"
# MODEL_PATH='checkpoints/llava-onevision-siglip-so400m-patch14-384-Qwen2.5-0.5B-Instruct-llava558k-pt-oryx-ft-flat'

accelerate launch --num_processes 8 --main_process_port 23553 -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=$MODEL_PATH,conv_template=qwen_2,model_name=llava_qwen_zip \
    --tasks egoschema \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix eval \
    --output_path ./logs/$MODEL_NAME

echo $MODEL_NAME
echo $MODEL_PATH