#include <Eigen/Eigen>
#include <iostream>
#include <cmath>
#include <memory>
#include <utility>
#include <octomap_server/OctomapServer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include "airsim_ros_wrapper.h"

// Define this to work around the AirSim bug that saves binvox files with the
// inverse scale.
#define AIRSIM_SCALE


struct VoxelGrid {
	uint16_t dim_x = 0;
	uint16_t dim_y = 0;
	uint16_t dim_z = 0;
	float t_x = 0.0f;
	float t_y = 0.0f;
	float t_z = 0.0f;
	float scale = 1.0f;
	std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> voxels;

	Eigen::Vector3i indexToVoxelCoords(size_t idx) const
	{
		const size_t step_x = dim_z * dim_y;
		Eigen::Vector3i v;
		v.x() = idx / step_x;
		size_t rem = idx % step_x;
		v.z() = rem / dim_y;
		v.y() = rem % dim_y;
		return v;
	}

	Eigen::Vector3f voxelCoordsToCoords(const Eigen::Vector3i& v) const
	{
		const Eigen::Array3f nv =
			(v.array().cast<float>() + Eigen::Array3f::Constant(0.5f))
			/ Eigen::Array3f(dim_x, dim_y, dim_z);
		return (scale * nv + Eigen::Array3f(t_x, t_y, t_z)).matrix();
	}

	Eigen::Vector3f indexToCoords(size_t idx) const
	{
		return voxelCoordsToCoords(indexToVoxelCoords(idx));
	}

	VoxelGrid(const std::string& filename)
	{
		std::ifstream f(filename, std::ios::binary);
		// Read the header.
		std::string token;
		f >> token;
		if (token != "#binvox") {
			throw std::runtime_error("Expected the first line to be: #binvox 1");
		}
		int version;
		f >> version;
		if (version != 1) {
			throw std::runtime_error("Expected binvox version 1 but got "
				+ std::to_string(version));
		}
		for (f >> token; token != "data"; f >> token) {
			if (token == "dim") {
				f >> dim_x >> dim_y >> dim_z;
			} else if (token == "translate") {
				f >> t_x >> t_y >> t_z;
			} else if (token == "scale") {
				f >> scale;
#ifdef AIRSIM_SCALE
				scale = 1.0f / scale;
#endif
			}
		}
		// Discard the newline of the data line.
		uint8_t discard;
		f.read(reinterpret_cast<char*>(&discard), 1);
		// Read the run-length-encoded voxel data.
		for (size_t i = 0; i < dim_x * dim_y * dim_z; ) {
			uint8_t value = 0, count = 0;
			f.read(reinterpret_cast<char*>(&value), 1);
			f.read(reinterpret_cast<char*>(&count), 1);
			// Only store occupied voxel coordinates.
			if (value) {
				for (size_t v = i; v < i + count; v++) {
					voxels.push_back(indexToCoords(v));
				}
			}
			i += count;
		}
	}
};



// Warn if the AirSim number of voxels truncation bug is triggered.
void airsim_bug_1_warning(int grid_dim, float grid_res)
{
	const float num_voxels_actual = grid_dim / grid_res;
	const int num_voxels_airsim = grid_dim / grid_res;
	const float diff = num_voxels_actual - num_voxels_airsim;
	if (diff < 0.0f || diff > 0.00001f) {
		ROS_WARN("The resolution (%f) doesn't evenly divide the grid dimensions (%d), the voxel grid will be unreliable",
			grid_res, grid_dim);
	}
}



int main(int argc, char **argv) {
	ros::init(argc, argv, "map_node");
	ros::NodeHandle nh("~");
	ros::NodeHandle private_nh("~");

	std::string host_ip;
	double resolution;
	int grid_dim;
	double grid_origin_x;
	double grid_origin_y;
	double grid_origin_z;
	std::string world_frameid;
	bool use_octree;
	nh.param("host_ip", host_ip, std::string("localhost"));
	nh.param("resolution", resolution, 0.1);
	nh.param("grid_dim", grid_dim, 20);
	nh.param("grid_origin_x", grid_origin_x, 0.0);
	nh.param("grid_origin_y", grid_origin_y, 0.0);
	nh.param("grid_origin_z", grid_origin_z, 0.0);
	nh.param("world_frame_id", world_frameid, std::string("world_enu"));
	nh.param("use_octree", use_octree, false);
	// The origin of the grid expressed in the NED frame. The x/y swap
	// performed internally in AirSim when saving the voxel grid is
	// probably cancelled out because the pipelien uses the ENU frame. The
	// z coordinate doesn't get negated though which is weird.
	const msr::airlib::Vector3r grid_origin (grid_origin_x, grid_origin_y, grid_origin_z);

	airsim_bug_1_warning(grid_dim, resolution);

	std::unique_ptr<octomap_server::OctomapServer> server_drone;
	if (use_octree) {
		server_drone = std::unique_ptr<octomap_server::OctomapServer>(new octomap_server::OctomapServer(private_nh, nh, world_frameid));
		server_drone->m_octree->clear();
	}

	ros::Publisher airsim_map_pub =
		nh.advertise<sensor_msgs::PointCloud2>("/airsim_global_map", 1, true);
	msr::airlib::RpcLibClientBase airsim_client_map_(host_ip);
	airsim_client_map_.confirmConnection();

	// Get the MAV's AABB. Reduce its size in the z direction because it's
	// too large and increase it by the resolution in the others to avoid
	// off-by-one issues due to truncation.
	const std::string mav_name = "drone_1";
	const Eigen::Vector3f mav_box = airsim_client_map_.simGetObjectScale(mav_name)
		+ Eigen::Vector3f(resolution, resolution, -0.7f);
	Eigen::Vector3f mav_position = airsim_client_map_.simGetObjectPose(mav_name).position;
	if (world_frameid == "world_enu") {
		// Hacky conversion from NED to ENU.
		std::swap(mav_position.x(), mav_position.y());
		mav_position.z() *= -1.0f;
	}

	// Save and then load the voxel map.
	const std::string filename = "/tmp/airsim_map.binvox";
	if (!airsim_client_map_.simCreateVoxelGrid(grid_origin, grid_dim, grid_dim, grid_dim, resolution, filename)) {
		ROS_FATAL("Error writing binvox file %s", filename.c_str());
	}
	const VoxelGrid grid (filename);
	ROS_INFO("Loaded binvox map with %zu occupied voxels", grid.voxels.size());

	// Publish the map as a pointcloud message and update the OctoMap.
	{
		sensor_msgs::PointCloud2 globalMap_pcd;
		pcl::PointCloud<pcl::PointXYZ> cloudMap;
		for (const auto& point : grid.voxels) {
			// Only consider voxels outside the MAV's AABB.
			const bool in_mav = ((mav_position - point).array().abs() <= mav_box.array() / 2.0f).all();
			if (!in_mav) {
				cloudMap.points.emplace_back(point.x(), point.y(), point.z());
				if (use_octree) {
					server_drone->m_octree->updateNode(
						point.x() + 1e-5, point.y() + 1e-5, point.z() + 1e-5, true);
				}
			}
		}
		cloudMap.width = cloudMap.points.size();
		cloudMap.height = 1;
		cloudMap.is_dense = true;
		pcl::toROSMsg(cloudMap, globalMap_pcd);
		globalMap_pcd.header.frame_id = world_frameid;
		airsim_map_pub.publish(globalMap_pcd);
	}

	ros::Rate rate(1);
	while (ros::ok()) {
		ros::spinOnce();
		if (use_octree) {
			server_drone->publishAll();
		}
		rate.sleep();
	}
	return 0;
}

/*
{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "Vehicles": {
    "drone_1": {
      "VehicleType": "SimpleFlight",
      "DefaultVehicleState": "Armed",
      "Sensors": {
        "Barometer": {
          "SensorType": 1,
          "Enabled" : false
        },
        "Imu": {
          "SensorType": 2,
          "Enabled" : true
        },
        "Gps": {
          "SensorType": 3,
          "Enabled" : false
        },
        "Magnetometer": {
          "SensorType": 4,
          "Enabled" : false
        },
        "Distance": {
          "SensorType": 5,
          "Enabled" : false
        },
        "Lidar": {
          "SensorType": 6,
          "Enabled" : false,
          "NumberOfChannels": 16,
          "RotationsPerSecond": 10,
          "PointsPerSecond": 100000,
          "X": 0, "Y": 0, "Z": -1,
          "Roll": 0, "Pitch": 0, "Yaw" : 0,
          "VerticalFOVUpper": 0,
          "VerticalFOVLower": -0,
          "HorizontalFOVStart": -90,
          "HorizontalFOVEnd": 90,
          "DrawDebugPoints": true,
          "DataFrame": "SensorLocalFrame"
        }
      },
      "Cameras": {
        "front_center_custom": {
          "CaptureSettings": [
            {
              "PublishToRos": 1,
              "ImageType": 3,
              "Width": 320,
              "Height": 240,
              "FOV_Degrees": 90,
              "DepthOfFieldFstop": 2.8,
              "DepthOfFieldFocalDistance": 200.0,
              "DepthOfFieldFocalRegion": 200.0,
              "TargetGamma": 1.5
            }
          ],
          "Pitch": 0.0,
          "Roll": 0.0,
          "X": 0.25,
          "Y": 0.0,
          "Yaw": 0.0,
          "Z": 0.3
        }
      },
      "X": 0, "Y": 0, "Z": 0,
      "Pitch": 0, "Roll": 0, "Yaw": 0
    }
  },
  "SubWindows": [
    {"WindowID": 1, "ImageType": 3, "CameraName": "front_center_custom", "Visible": false}
  ]
}
*/
