DATASET_PATH=$1
/home/apple/F/henry/DAGSfM/build/src/exe/colmap automatic_reconstructor \
--image_path=$DATASET_PATH \
--workspace_path=$DATASET_PATH \
--dense=0 #\
#--single_camera=1
