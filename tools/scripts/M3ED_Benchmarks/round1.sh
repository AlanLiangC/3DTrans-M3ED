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

# srun python -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --tcp_port ${PORT} \
#     --batch_size 8 \
#     --cfg_file cfgs/m3ed_benchmark_models/grid-based/centerpoint_vehi.yaml

srun python -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --tcp_port ${PORT} \
    --batch_size 8 \
    --cfg_file cfgs/m3ed_benchmark_models/grid-based/pillarnet.yaml

srun python -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --tcp_port ${PORT} \
    --batch_size 8 \
    --cfg_file cfgs/m3ed_benchmark_models/grid-based/pointpillar.yaml

srun python -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --tcp_port ${PORT} \
    --batch_size 8 \
    --cfg_file cfgs/m3ed_benchmark_models/point-based/IA-SSD_vehi.yaml

# srun python -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --tcp_port ${PORT} \
#     --batch_size 8 \
#     --cfg_file cfgs/m3ed_benchmark_models/grid-based/second_iou.yaml
