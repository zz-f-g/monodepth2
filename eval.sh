modelname=bwd
for i in {0..19}  # 修改这里的范围，例如从 1 到 10
do
    weight_folder="/home/gzz/workspace/assignment/research/code/monodepth2/ckpts/${modelname}/models/weights_${i}"
    echo "Running with weights folder: ${weight_folder}"
    python evaluate_depth.py \
        --data_path /mnt/disk/kitti \
        --load_weights_folder "${weight_folder}" \
        --model_name ${modelname} \
        --eval_mono
    # 可以在这里添加其他命令或操作，例如保存结果等
done
