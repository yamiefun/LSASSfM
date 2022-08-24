DATASET_PATH=$1
num_images_ub=$2
log_folder=$3
completeness_ratio=$4

/home/apple/F/henry/DAGSfM/build/src/exe/colmap exhaustive_matcher \
--database_path=$DATASET_PATH/database.db \
--SiftMatching.num_threads=8 \
--SiftMatching.use_gpu=1 \
--SiftMatching.gpu_index=-1
