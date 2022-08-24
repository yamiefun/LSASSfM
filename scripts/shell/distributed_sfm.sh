DATASET_PATH=$1
num_images_ub=$2
log_folder=$3
completeness_ratio=$4
#VOC_TREE_PATH=$5
# image_overlap=$3
# max_num_cluster_pairs=$4


/home/apple/F/henry/LASSfM/build/src/exe/colmap feature_extractor \
--database_path=$DATASET_PATH/database.db \
--image_path=$DATASET_PATH \
--SiftExtraction.num_threads=8 \
--SiftExtraction.use_gpu=1 \
--SiftExtraction.gpu_index=-1

/home/apple/F/henry/LASSfM/build/src/exe/colmap exhaustive_matcher \
--database_path=$DATASET_PATH/database.db \
--SiftMatching.num_threads=8 \
--SiftMatching.use_gpu=1 \
--SiftMatching.gpu_index=-1
## Or use vocabulary tree matcher
# /home/amax/Projects/colmap/build/src/exe/colmap vocab_tree_matcher \
# --database_path=$DATASET_PATH/database.db \
# --SiftMatching.num_threads=8 \
# --SiftMatching.use_gpu=1 \
# --SiftMatching.gpu_index=0 \
# --VocabTreeMatching.num_images=100 \
# --VocabTreeMatching.num_nearest_neighbors=5 \
# --VocabTreeMatching.vocab_tree_path=$VOC_TREE_PATH

cp $DATASET_PATH/database.db $DATASET_PATH/database.db.bak

/home/apple/F/henry/LASSfM/build/src/exe/colmap distributed_mapper \
$DATASET_PATH/$log_folder \
--database_path=$DATASET_PATH/database.db \
--image_path=$DATASET_PATH \
--output_path=$DATASET_PATH/sparse \
--num_workers=1 \
--graph_dir=$DATASET_PATH/$log_folder \
--num_images_ub=$num_images_ub \
--completeness_ratio=$completeness_ratio \
--relax_ratio=1.3 \
--cluster_type=NCUT \
--imu=/home/apple/F/EC/dataset_0530/vio/vins \
--video=/home/apple/F/exp/0824/video \
--build_graph=1
