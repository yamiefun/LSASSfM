DATASET_PATH=$1

#/home/apple/F/henry/DAGSfM/build/src/exe/colmap mapper \
/home/apple/F/DAGSfM_master/build/src/exe/colmap mapper \
--database_path=$DATASET_PATH/database.db \
--image_path=$DATASET_PATH \
--output_path=$DATASET_PATH/sparse/ 
#--Mapper.init_image_id1=107 \
#--Mapper.init_image_id2=245
