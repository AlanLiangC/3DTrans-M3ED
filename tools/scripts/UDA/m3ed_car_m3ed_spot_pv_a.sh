#!/bin/bash
#SBATCH --job-name=multi_gpu_train      # 任务名称
#SBATCH --output=training_output_%j.log # 标准输出和错误日志文件
#SBATCH --error=training_error_%j.log   # 错误日志文件
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=4                      # 总任务数（通常等于总GPU数）
#SBATCH --gres=gpu:nv:2                    # 每个节点使用的GPU数
#SBATCH -C cuda75
#SBATCH --mem=64G                       # 每个节点使用的内存
#SBATCH --time=36:00:00                 # 任务运行的最长时间
#SBATCH --mail-type=ALL            # 任务结束或失败时发送邮件
#SBATCH --mail-user=a_liang@u.nus.edu # 接收邮件的地

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

srun python -m torch.distributed.launch --nproc_per_node=2 train_uda.py --launcher pytorch --tcp_port ${PORT} \
    --batch_size 8 \
    --cfg_file cfgs/DA/m3ed_m3ed/car_spot/pvrcnn_st3d_w_ros_veh_feat_3.yaml \
    --pretrained_model ../output/m3edseq_models/pvrcnn_veh_feat_3/default/ckpt/checkpoint_epoch_40.pth \
    --eval_fov_only

srun python -m torch.distributed.launch --nproc_per_node=2 train_uda.py --launcher pytorch --tcp_port ${PORT} \
    --batch_size 8 \
    --cfg_file cfgs/DA/m3ed_m3ed/car_spot/pvrcnn_st3d_wo_ros_veh_feat_3.yaml \
    --pretrained_model ../output/m3edseq_models/pvrcnn_veh_feat_3/default/ckpt/checkpoint_epoch_40.pth \
    --eval_fov_only

srun python -m torch.distributed.launch --nproc_per_node=2 train_uda.py --launcher pytorch --tcp_port ${PORT} \
    --batch_size 8 \
    --cfg_file cfgs/DA/m3ed_m3ed/car_spot/pvrcnn_st3d_plus_w_ros_veh_feat_3.yaml \
    --pretrained_model ../output/m3edseq_models/pvrcnn_veh_feat_3/default/ckpt/checkpoint_epoch_40.pth \
    --eval_fov_only

srun python -m torch.distributed.launch --nproc_per_node=2 train_uda.py --launcher pytorch --tcp_port ${PORT} \
    --batch_size 8 \
    --cfg_file cfgs/DA/m3ed_m3ed/car_spot/pvrcnn_st3d_plus_wo_ros_veh_feat_3.yaml \
    --pretrained_model ../output/m3edseq_models/pvrcnn_veh_feat_3/default/ckpt/checkpoint_epoch_40.pth \
    --eval_fov_only

srun python -m torch.distributed.launch --nproc_per_node=2 train_uda.py --launcher pytorch --tcp_port ${PORT} \
    --batch_size 8 \
    --cfg_file cfgs/DA/m3ed_m3ed/car_spot/pvrcnn_ms3d_veh_feat_3.yaml \
    --pretrained_model ../output/m3edseq_models/pvrcnn_veh_feat_3/default/ckpt/checkpoint_epoch_40.pth \
    --eval_fov_only

srun python -m torch.distributed.launch --nproc_per_node=2 train_uda.py --launcher pytorch --tcp_port ${PORT} \
    --batch_size 8 \
    --cfg_file cfgs/DA/m3ed_m3ed/car_spot/pvrcnn_our_v0_veh_feat_3.yaml \
    --pretrained_model ../output/m3edseq_models/pvrcnn_veh_feat_3/default/ckpt/checkpoint_epoch_40.pth \
    --eval_fov_only