#ifndef SRC_CONTROLLERS_DISTRIBUTED_MAPPER_CONTROLLER_H_
#define SRC_CONTROLLERS_DISTRIBUTED_MAPPER_CONTROLLER_H_

#include <string>
#include <memory>

#include "util/threading.h"
#include "base/reconstruction_manager.h"
#include "clustering/image_clustering.h"
#include "clustering/scene_clustering.h"
#include "controllers/incremental_mapper_controller.h"
#include "clustering/image_clustering.h"

using namespace colmap;

namespace GraphSfM {
struct Coordinate
{
    double x;
    double y;
    double z;
};
struct FindVIO
{
    bool found;
    GraphSfM::Coordinate coordinate;
};
struct VIOInfo
{
    bool has_VIO;
    GraphSfM::Coordinate coordinate;
    uint VIO_group_id;
};
struct ImageInfo
{
    std::string image_name;
    GraphSfM:: VIOInfo VIO_info;
};
struct VIOThresh
{
    double height_thresh;
    double drift_thresh;
};
struct VideoInfo
{
    uint video_id;
    int serial_id;
};

// Distributed mapping for very large scale structure from motion.
// This mapper first partition images into the given number of clusters, 
// then reconstruct each cluster with incremental/global/hybrid SfM approaches.
// At last, all clusters are aligned together using a robust graph-based aligner.
// A very large scale bundle adjustment should be performed after the alignment.
class DistributedMapperController : public Thread
{
public:
    struct Options
    {
        // the path to the image folder which are used as input
        std::string image_path;

        // The path to store reconstructions.
        std::string output_path;

        // The path to the database file which is used as input.
        std::string database_path;

        // The maximum number of trials to initialize a cluster.
        int init_num_trials = 10;

        // The number of workers used to reconstruct clusters in parallel.
        int num_workers = -1;

        // The path to the folder of imu information file which is used as input.
        std::string VIO_folder_path;

        // The path to the folder of video sequences.
        std::string video_folder_path;

        // If need to build view graph with VIO
        bool build_graph = true;

        bool Check() const;
    };
    
    DistributedMapperController(
        const Options& options,
        const ImageClustering::Options& clustering_options,
        const IncrementalMapperOptions& mapper_options,
        ReconstructionManager* reconstruction_manager);

    BundleAdjustmentOptions GlobalBundleAdjustment() const;

private:
    void Run() override;

    bool IsPartialReconsExist(std::vector<Reconstruction*>& recons) const;

    void LoadData(std::vector<std::pair<image_t, image_t>>& image_pairs,
                  std::vector<int>& num_inliers,
                  std::unordered_map<image_t, std::string>& image_id_to_name,
                  std::vector<image_t>& image_ids);

    void LoadVIO(std::vector<std::vector<std::pair<std::string, GraphSfM::Coordinate>>>& ts_to_VIO,
                 std::vector<std::pair<std::string, std::string>>& VIO_summary);
    
    void LoadVideo(std::vector<std::string>& video_paths);

    void ParseVideoFrames(
        std::vector<std::string>& video_paths,
        std::unordered_map<image_t, GraphSfM::VideoInfo>& video_frame_info,
        std::unordered_map<image_t, std::string>& image_id_to_name
    );

    void ParseVideoFramePath(
        std::string& image_full_path,
        std::string& image_path,
        std::string& image_name
    );

    void CalculateVIOThresh(
        std::vector<GraphSfM::VIOThresh>& vio_thresh,
        std::vector<std::vector<std::pair<std::string, GraphSfM::Coordinate>>>& ts_to_VIO);

    std::string ParseImageNameFromPath(std::string path);

    bool CHECK_TS_LE(std::string ts1, std::string ts2);

    GraphSfM::VIOInfo CheckImageWithVIO(
        std::vector<std::pair<std::string, std::string>>& VIO_summary,
        std::vector<std::vector<std::pair<std::string, GraphSfM::Coordinate>>>& ts_to_VIO,
        std::string& target);

    void MatchImageWithCoordinate(
        std::unordered_map<image_t, GraphSfM::ImageInfo>& image_information,
        std::unordered_map<image_t, std::string>& image_id_to_name,
        std::vector<std::vector<std::pair<std::string, GraphSfM::Coordinate>>>& ts_to_VIO,
        std::vector<std::pair<std::string, std::string>>& VIO_summary);

    bool CHECK_STR_LT(std::string& ts1, std::string& ts2);

    int ts_diff(std::string ts1, std::string ts2);

    double square_distance_3d(GraphSfM::Coordinate& coord1, GraphSfM::Coordinate& coord2);

    double CalculateScore(
        image_t& image_id,
        image_t& ref_id,
        std::unordered_map<ViewIdPair, int>& edges,
        std::unordered_map<image_t, std::set<image_t>>& graph,
        std::set<image_t>& images_with_vio,
        double& weight_vio, double& weight_no_vio
    );

    double CalculateScore(
        image_t& image_id,
        image_t& ref_id,
        std::unordered_map<ViewIdPair, int>& edges,
        std::unordered_map<image_t, std::set<image_t>>& graph,
        std::unordered_map<image_t, GraphSfM::VideoInfo>& video_frame_info,
        double& weight_vio, double& weight_no_vio
    );

    void CreateImageViewGraph(
        std::unordered_map<image_t, GraphSfM::ImageInfo>& image_information,
        std::vector<std::pair<image_t, image_t>>& image_pairs,
        std::vector<int>& num_inliers,
        std::vector<image_t>& image_ids,
        ImageCluster& image_cluster,
        std::vector<GraphSfM::VIOThresh>& vio_thresh);

    void CreateImageViewGraphWithVideos(
        std::vector<std::pair<image_t, image_t>>& image_pairs,
        std::vector<int>& num_inliers,
        std::vector<image_t>& image_ids,
        ImageCluster& image_cluster,
        std::unordered_map<image_t, GraphSfM::VideoInfo>& video_frame_info,
        std::unordered_map<image_t, std::string>& image_id_to_name,
        size_t& vide_num);

    void CreateImageViewGraphWithVideo(
        std::vector<std::pair<image_t, image_t>>& image_pairs,
        std::vector<int>& num_inliers,
        std::vector<image_t>& image_ids,
        ImageCluster& image_cluster,
        std::unordered_map<image_t, GraphSfM::VideoInfo>& video_frame_info,
        std::unordered_map<image_t, std::string>& image_id_to_name);

    std::vector<ImageCluster> ClusteringScenes(
        const std::vector<std::pair<image_t, image_t>>& image_pairs,
        const std::vector<int>& num_inliers,
        const std::vector<image_t>& image_ids);

    std::vector<ImageCluster> ClusteringScenesWithViewGraph(
        ImageCluster& image_cluster);


    void ReconstructPartitions(
        const std::unordered_map<image_t, std::string>& image_id_to_name,
        std::vector<ImageCluster>& inter_clusters,
        std::unordered_map<const ImageCluster*, ReconstructionManager>& reconstruction_managers,
        std::vector<Reconstruction*>& reconstructions);

    void MergeClusters(
        const std::vector<ImageCluster>& inter_clusters,
        std::vector<Reconstruction*>& reconstructions,
        std::unordered_map<const ImageCluster*, ReconstructionManager>& reconstruction_managers,
        const int num_eff_threads);

    const Options options_;

    ImageClustering::Options clustering_options_;

    std::unique_ptr<ImageClustering> image_clustering_;

    const IncrementalMapperOptions mapper_options_;

    ReconstructionManager* reconstruction_manager_;
};

} // namespace GraphSfM

#endif