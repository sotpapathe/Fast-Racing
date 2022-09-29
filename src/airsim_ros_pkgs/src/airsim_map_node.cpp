#include <Eigen/Eigen>
#include <geometry_msgs/Point32.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3.h>
#include <iostream>
#include <math.h>
#include <memory>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <octomap/OcTreeKey.h>
#include <octomap/octomap.h>
#include <octomap_msgs/BoundingBoxQuery.h>
#include <octomap_msgs/GetOctomap.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <octomap_ros/conversions.h>
#include <octomap_server/OctomapServer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/search/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>
#include <random>
#include <ros/console.h>
#include <ros/ros.h>
#include <ros/spinner.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "airsim_ros_wrapper.h"

#define BINVOX_FILE "/tmp/airsim_map.binvox"

// Define this to work around the AirSim bug that saves binvox files with the
// inverse scale.
#define AIRSIM_SCALE


using namespace octomap;
using namespace octomap_msgs;
using namespace octomap_server;

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



int main(int argc, char **argv) {
	ros::init(argc, argv, "map_node");
	ros::NodeHandle nh("~");
	ros::NodeHandle private_nh("~");

	std::string host_ip;
	double resolution;
	std::string world_frameid;
	bool use_octree;
	nh.param("host_ip", host_ip, std::string("localhost"));
	nh.param("resolution", resolution, 0.08);
	nh.param("world_frame_id", world_frameid, std::string("world_enu"));
	nh.param("use_octree", use_octree, false);

	std::unique_ptr<OctomapServer> server_drone;
	if (use_octree) {
		server_drone = std::unique_ptr<OctomapServer>(new OctomapServer(private_nh, nh, world_frameid));
	}

	ros::Publisher airsim_map_pub =
		nh.advertise<sensor_msgs::PointCloud2>("/airsim_global_map", 1);
	msr::airlib::RpcLibClientBase airsim_client_map_(host_ip);
	airsim_client_map_.confirmConnection();

	// Save and the load the voxel map.
	msr::airlib::Vector3r origin (0, 0, 0);
	constexpr double grid_dim = 20.0;
	airsim_client_map_.simCreateVoxelGrid(origin, grid_dim, grid_dim, grid_dim, resolution, BINVOX_FILE);
	const VoxelGrid grid (BINVOX_FILE);
	ROS_INFO("Loaded binvox map with %zu occupied voxels and pose", grid.voxels.size());

	ros::Rate rate(1);
	while (ros::ok()) {
		ros::spinOnce();
		if (use_octree) {
			server_drone->m_octree->clear();
		}

		pcl::PointCloud<pcl::PointXYZ> cloudMap;
		for (const auto& point : grid.voxels) {
			const Eigen::Vector3d p = point.cast<double>();
			cloudMap.points.emplace_back(p.x(), p.y(), p.z());
			if (use_octree) {
				server_drone->m_octree->updateNode(
					p.x() + 1e-5, p.y() + 1e-5, p.z() + 1e-5, true);
			}
		}
		if (use_octree) {
			server_drone->publishAll();
		}
		cloudMap.width = cloudMap.points.size();
		cloudMap.height = 1;
		cloudMap.is_dense = true;
		sensor_msgs::PointCloud2 globalMap_pcd;
		pcl::toROSMsg(cloudMap, globalMap_pcd);
		globalMap_pcd.header.frame_id = world_frameid;
		airsim_map_pub.publish(globalMap_pcd);
		ROS_INFO("Published global map");
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
