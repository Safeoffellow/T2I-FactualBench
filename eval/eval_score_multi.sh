#!/bin/bash
NAS_ADDR="9a40a493bf-eio4.cn-zhangjiakou.nas.aliyuncs.com" # /mnt/workspace/

datatime=$(date +"%Y_%m_%d_%H_%M_%S")


entry=eval_clip_dino_multi.py

# QUEUE=content_aigc_img # Replace with the actual queue name
# QUEUE=content_aigc_mm
# QUEUE=smart_algo_aigc
nebula_project=content_aigc_new
# QUEUE=smart_algo_aigc
QUEUE=content_aigc_mm_1
# QUEUE=smart_algo_content_aigc
algo_name=pytorch220 # Replace with your algorithm name if different
WORLD_SIZE=4
version="${datatime}_${WORLD_SIZE}_ssr_eval_one"
echo "version:$version"
# yansong_id=84912
user_id=433548
args="--result_path /mnt/workspace/ziwei/T2I_Knowledge_bench/sd3.5_new \
    --level level_all \
    --model sd3.5_new
"

# oss配置，用于fuse oss以后读取样本
OSS_ACCESS_ID="cZpR7MraH7vM2HXk"
OSS_ACCESS_KEY="QPHc6hi2ErK43XMOdtEWm4PUNZw4bI"
# OSS_BUCKET="lanke-data"
OSS_BUCKET="lanke-all"
# OSS_ENDPOINT="oss-cn-hangzhou.aliyuncs.com"
OSS_ENDPOINT="oss-cn-zhangjiakou.aliyuncs.com"

# Now run the online training using the 'nebulactl' command
nebulactl run mdl --queue=${QUEUE} \
                    --entry=${entry} \
                    --worker_count=${WORLD_SIZE} \
                    --user_params="$args" \
                    --file.cluster_file=./cluster.json \
                    --user_id=$user_id \
                    --oss_access_id=${OSS_ACCESS_ID} \
                    --oss_access_key=${OSS_ACCESS_KEY} \
                    --oss_bucket=${OSS_BUCKET} \
                    --oss_endpoint=${OSS_ENDPOINT} \
                    --nebula_project ${nebula_project} \
                    --algo_name=${algo_name} \
                    --nas_file_system_id=${NAS_ADDR} \
                    --job_name="${version}"
