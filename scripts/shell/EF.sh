DATASET_PATH=$1
num_images_ub=$2
log_folder=$3
completeness_ratio=$4
#VOC_TREE_PATH=$5
# image_overlap=$3
# max_num_cluster_pairs=$4


/home/apple/F/henry/DAGSfM/build/src/exe/colmap feature_extractor \
--database_path=$DATASET_PATH/database.db \
--image_path=$DATASET_PATH \
--SiftExtraction.num_threads=8 \
--SiftExtraction.use_gpu=1 \
--SiftExtraction.gpu_index=-1
