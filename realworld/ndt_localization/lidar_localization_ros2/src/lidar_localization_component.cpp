#include <lidar_localization/lidar_localization_component.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

PCLLocalization::PCLLocalization(const rclcpp::NodeOptions& options)
    : rclcpp_lifecycle::LifecycleNode("lidar_localization", options),
      clock_(RCL_ROS_TIME),
      tfbuffer_(std::make_shared<rclcpp::Clock>(clock_)),
      tflistener_(tfbuffer_),
      broadcaster_(this) {
  declare_parameter("global_frame_id", "map");
  declare_parameter("odom_frame_id", "livox_odom");
  declare_parameter("base_frame_id", "base_link");
  declare_parameter("imu_topic", "/livox/imu");
  declare_parameter("registration_method", "NDT");
  declare_parameter("score_threshold", 2.0);
  declare_parameter("reloc_yaw_search_deg", 60.0);
  declare_parameter("reloc_search_enabled", true);
  declare_parameter("reloc_prior_pos_weight", 0.0);
  declare_parameter("reloc_prior_yaw_weight", 0.0);
  declare_parameter("ndt_resolution", 1.0);
  declare_parameter("ndt_step_size", 0.1);
  declare_parameter("ndt_max_iterations", 35);
  declare_parameter("ndt_num_threads", 4);
  declare_parameter("transform_epsilon", 0.01);
  declare_parameter("voxel_leaf_size", 0.2);
  declare_parameter("map_viz_leaf_size", 0.4);
  declare_parameter("scan_max_range", 100.0);
  declare_parameter("scan_min_range", 1.0);
  declare_parameter("scan_period", 0.1);
  declare_parameter("use_pcd_map", false);
  declare_parameter("map_path", "/map/map.pcd");
  declare_parameter("set_initial_pose", false);
  declare_parameter("initial_pose_x", 0.0);
  declare_parameter("initial_pose_y", 0.0);
  declare_parameter("initial_pose_z", 0.0);
  declare_parameter("initial_pose_qx", 0.0);
  declare_parameter("initial_pose_qy", 0.0);
  declare_parameter("initial_pose_qz", 0.0);
  declare_parameter("initial_pose_qw", 1.0);
  declare_parameter("use_odom", false);
  declare_parameter("use_imu", false);
  declare_parameter("enable_debug", false);
}

using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

CallbackReturn PCLLocalization::on_configure(const rclcpp_lifecycle::State&) {
  RCLCPP_INFO(get_logger(), "Configuring");

  initializeParameters();
  initializePubSub();
  initializeRegistration();

  path_ptr_ = std::make_shared<nav_msgs::msg::Path>();
  path_ptr_->header.frame_id = global_frame_id_;

  RCLCPP_INFO(get_logger(), "Configuring end");
  return CallbackReturn::SUCCESS;
}

CallbackReturn PCLLocalization::on_activate(const rclcpp_lifecycle::State&) {
  RCLCPP_INFO(get_logger(), "Activating");

  pose_pub_->on_activate();
  path_pub_->on_activate();
  initial_map_pub_->on_activate();

  if (set_initial_pose_) {
    auto msg =
        std::make_shared<geometry_msgs::msg::PoseWithCovarianceStamped>();

    msg->header.stamp = now();
    msg->header.frame_id = global_frame_id_;
    msg->pose.pose.position.x = initial_pose_x_;
    msg->pose.pose.position.y = initial_pose_y_;
    msg->pose.pose.position.z = initial_pose_z_;
    msg->pose.pose.orientation.x = initial_pose_qx_;
    msg->pose.pose.orientation.y = initial_pose_qy_;
    msg->pose.pose.orientation.z = initial_pose_qz_;
    msg->pose.pose.orientation.w = initial_pose_qw_;

    geometry_msgs::msg::PoseStamped::SharedPtr pose_stamped(
        new geometry_msgs::msg::PoseStamped);
    pose_stamped->header.stamp = msg->header.stamp;
    pose_stamped->header.frame_id = global_frame_id_;
    pose_stamped->pose = msg->pose.pose;
    path_ptr_->poses.push_back(*pose_stamped);

    initialPoseReceived(msg);
  }

  if (use_pcd_map_) {
    const auto load_t0 = std::chrono::steady_clock::now();
    pcl::PointCloud<pcl::PointXYZI>::Ptr map_cloud_ptr(
        new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile(map_path_, *map_cloud_ptr) < 0 ||
        map_cloud_ptr->empty()) {
      RCLCPP_FATAL(get_logger(),
                   "Failed to load the map PCD (or it is empty): %s",
                   map_path_.c_str());
      return CallbackReturn::FAILURE;
    }
    const double load_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - load_t0)
            .count();
    RCLCPP_INFO(get_logger(), "Map Size %ld (loaded in %.2fs)",
                map_cloud_ptr->size(), load_sec);

    pcl::PointCloud<pcl::PointXYZI>::Ptr viz_cloud_ptr(
        new pcl::PointCloud<pcl::PointXYZI>());
    if (map_viz_leaf_size_ > 0.0) {
      pcl::VoxelGrid<pcl::PointXYZI> viz_filter;
      viz_filter.setInputCloud(map_cloud_ptr);
      viz_filter.setLeafSize(map_viz_leaf_size_, map_viz_leaf_size_,
                             map_viz_leaf_size_);
      viz_filter.filter(*viz_cloud_ptr);
    } else {
      viz_cloud_ptr = map_cloud_ptr;
    }

    sensor_msgs::msg::PointCloud2::SharedPtr map_msg_ptr(
        new sensor_msgs::msg::PointCloud2);
    pcl::toROSMsg(*viz_cloud_ptr, *map_msg_ptr);
    map_msg_ptr->header.frame_id = global_frame_id_;
    initial_map_pub_->publish(*map_msg_ptr);
    RCLCPP_INFO(get_logger(), "Initial Map Published (%ld pts for viz)",
                viz_cloud_ptr->size());

    if (registration_method_ == "GICP" || registration_method_ == "GICP_OMP") {
      pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud_ptr(
          new pcl::PointCloud<pcl::PointXYZI>());
      voxel_grid_filter_.setInputCloud(map_cloud_ptr);
      voxel_grid_filter_.filter(*filtered_cloud_ptr);
      registration_->setInputTarget(filtered_cloud_ptr);
    } else {
      const auto target_t0 = std::chrono::steady_clock::now();
      registration_->setInputTarget(map_cloud_ptr);
          RCLCPP_INFO(
          get_logger(), "NDT target built in %.2fs",
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        target_t0)
              .count());
    }

    map_recieved_ = true;
  }

  RCLCPP_INFO(get_logger(), "Activating end");
  return CallbackReturn::SUCCESS;
}

CallbackReturn PCLLocalization::on_deactivate(const rclcpp_lifecycle::State&) {
  RCLCPP_INFO(get_logger(), "Deactivating");

  pose_pub_->on_deactivate();
  path_pub_->on_deactivate();
  initial_map_pub_->on_deactivate();

  RCLCPP_INFO(get_logger(), "Deactivating end");
  return CallbackReturn::SUCCESS;
}

CallbackReturn PCLLocalization::on_cleanup(const rclcpp_lifecycle::State&) {
  RCLCPP_INFO(get_logger(), "Cleaning Up");
  initial_pose_sub_.reset();
  initial_map_pub_.reset();
  path_pub_.reset();
  pose_pub_.reset();
  odom_sub_.reset();
  cloud_sub_.reset();
  imu_sub_.reset();

  RCLCPP_INFO(get_logger(), "Cleaning Up end");
  return CallbackReturn::SUCCESS;
}

CallbackReturn PCLLocalization::on_shutdown(
    const rclcpp_lifecycle::State& state) {
  RCLCPP_INFO(get_logger(), "Shutting Down from %s", state.label().c_str());

  return CallbackReturn::SUCCESS;
}

CallbackReturn PCLLocalization::on_error(const rclcpp_lifecycle::State& state) {
  RCLCPP_FATAL(get_logger(), "Error Processing from %s", state.label().c_str());

  return CallbackReturn::SUCCESS;
}

void PCLLocalization::initializeParameters() {
  RCLCPP_INFO(get_logger(), "initializeParameters");
  get_parameter("global_frame_id", global_frame_id_);
  get_parameter("odom_frame_id", odom_frame_id_);
  get_parameter("imu_topic", imu_topic_);
  get_parameter("base_frame_id", base_frame_id_);
  get_parameter("registration_method", registration_method_);
  get_parameter("score_threshold", score_threshold_);
  get_parameter("reloc_yaw_search_deg", reloc_yaw_search_deg_);
  get_parameter("reloc_search_enabled", reloc_search_enabled_);
  get_parameter("reloc_prior_pos_weight", reloc_prior_pos_weight_);
  get_parameter("reloc_prior_yaw_weight", reloc_prior_yaw_weight_);
  get_parameter("ndt_resolution", ndt_resolution_);
  get_parameter("ndt_step_size", ndt_step_size_);
  get_parameter("ndt_num_threads", ndt_num_threads_);
  get_parameter("ndt_max_iterations", ndt_max_iterations_);
  get_parameter("transform_epsilon", transform_epsilon_);
  get_parameter("voxel_leaf_size", voxel_leaf_size_);
  get_parameter("map_viz_leaf_size", map_viz_leaf_size_);
  get_parameter("scan_max_range", scan_max_range_);
  get_parameter("scan_min_range", scan_min_range_);
  get_parameter("scan_period", scan_period_);
  get_parameter("use_pcd_map", use_pcd_map_);
  get_parameter("map_path", map_path_);
  get_parameter("set_initial_pose", set_initial_pose_);
  get_parameter("initial_pose_x", initial_pose_x_);
  get_parameter("initial_pose_y", initial_pose_y_);
  get_parameter("initial_pose_z", initial_pose_z_);
  get_parameter("initial_pose_qx", initial_pose_qx_);
  get_parameter("initial_pose_qy", initial_pose_qy_);
  get_parameter("initial_pose_qz", initial_pose_qz_);
  get_parameter("initial_pose_qw", initial_pose_qw_);
  get_parameter("use_odom", use_odom_);
  get_parameter("use_imu", use_imu_);
  get_parameter("enable_debug", enable_debug_);

  RCLCPP_INFO(get_logger(), "global_frame_id: %s", global_frame_id_.c_str());
  RCLCPP_INFO(get_logger(), "odom_frame_id: %s", odom_frame_id_.c_str());
  RCLCPP_INFO(get_logger(), "imu_topic: %s", imu_topic_.c_str());
  RCLCPP_INFO(get_logger(), "base_frame_id: %s", base_frame_id_.c_str());
  RCLCPP_INFO(get_logger(), "registration_method: %s",
              registration_method_.c_str());
  RCLCPP_INFO(get_logger(), "ndt_resolution: %lf", ndt_resolution_);
  RCLCPP_INFO(get_logger(), "ndt_step_size: %lf", ndt_step_size_);
  RCLCPP_INFO(get_logger(), "ndt_num_threads: %d", ndt_num_threads_);
  RCLCPP_INFO(get_logger(), "transform_epsilon: %lf", transform_epsilon_);
  RCLCPP_INFO(get_logger(), "voxel_leaf_size: %lf", voxel_leaf_size_);
  RCLCPP_INFO(get_logger(), "scan_max_range: %lf", scan_max_range_);
  RCLCPP_INFO(get_logger(), "scan_min_range: %lf", scan_min_range_);
  RCLCPP_INFO(get_logger(), "scan_period: %lf", scan_period_);
  RCLCPP_INFO(get_logger(), "use_pcd_map: %d", use_pcd_map_);
  RCLCPP_INFO(get_logger(), "map_path: %s", map_path_.c_str());
  RCLCPP_INFO(get_logger(), "set_initial_pose: %d", set_initial_pose_);
  RCLCPP_INFO(get_logger(), "use_odom: %d", use_odom_);
  RCLCPP_INFO(get_logger(), "use_imu: %d", use_imu_);
  RCLCPP_INFO(get_logger(), "enable_debug: %d", enable_debug_);
}

void PCLLocalization::initializePubSub() {
  RCLCPP_INFO(get_logger(), "initializePubSub");

  pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "pcl_pose",
      rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable());

  path_pub_ = create_publisher<nav_msgs::msg::Path>(
      "path", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable());

  initial_map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "initial_map",
      rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable());

  initial_pose_sub_ =
      create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
          "initialpose", rclcpp::SystemDefaultsQoS(),
          std::bind(&PCLLocalization::initialPoseReceived, this,
                    std::placeholders::_1));

  map_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "map", rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable(),
      std::bind(&PCLLocalization::mapReceived, this, std::placeholders::_1));

  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "odom", rclcpp::SensorDataQoS(),
      std::bind(&PCLLocalization::odomReceived, this, std::placeholders::_1));

  cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "cloud", rclcpp::SensorDataQoS(),
      std::bind(&PCLLocalization::cloudReceived, this, std::placeholders::_1));

  imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_, rclcpp::SensorDataQoS(),
      std::bind(&PCLLocalization::imuReceived, this, std::placeholders::_1));

  RCLCPP_INFO(get_logger(), "initializePubSub end");
}

void PCLLocalization::initializeRegistration() {
  RCLCPP_INFO(get_logger(), "initializeRegistration");

  if (registration_method_ == "GICP") {
    boost::shared_ptr<
        pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI>>
        gicp(new pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZI,
                                                       pcl::PointXYZI>());
    gicp->setTransformationEpsilon(transform_epsilon_);
    registration_ = gicp;
  } else if (registration_method_ == "NDT") {
    boost::shared_ptr<
        pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>>
        ndt(new pcl::NormalDistributionsTransform<pcl::PointXYZI,
                                                  pcl::PointXYZI>());
    ndt->setStepSize(ndt_step_size_);
    ndt->setResolution(ndt_resolution_);
    ndt->setTransformationEpsilon(transform_epsilon_);
    registration_ = ndt;
  } else if (registration_method_ == "NDT_OMP") {
    pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>::Ptr
        ndt_omp(new pclomp::NormalDistributionsTransform<pcl::PointXYZI,
                                                         pcl::PointXYZI>());
    ndt_omp->setStepSize(ndt_step_size_);
    ndt_omp->setResolution(ndt_resolution_);
    ndt_omp->setTransformationEpsilon(transform_epsilon_);
    if (ndt_num_threads_ > 0) {
      ndt_omp->setNumThreads(ndt_num_threads_);
    } else {
      ndt_omp->setNumThreads(omp_get_max_threads());
    }
    registration_ = ndt_omp;
  } else if (registration_method_ == "GICP_OMP") {
    pclomp::GeneralizedIterativeClosestPoint<pcl::PointXYZI,
                                             pcl::PointXYZI>::Ptr
        gicp_omp(
            new pclomp::GeneralizedIterativeClosestPoint<pcl::PointXYZI,
                                                         pcl::PointXYZI>());
    gicp_omp->setTransformationEpsilon(transform_epsilon_);
    registration_ = gicp_omp;
  } else {
    RCLCPP_ERROR(get_logger(), "Invalid registration method.");
    exit(EXIT_FAILURE);
  }
  registration_->setMaximumIterations(ndt_max_iterations_);

  voxel_grid_filter_.setLeafSize(voxel_leaf_size_, voxel_leaf_size_,
                                 voxel_leaf_size_);
  RCLCPP_INFO(get_logger(), "initializeRegistration end");
}

void PCLLocalization::initialPoseReceived(
    const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
  RCLCPP_INFO(get_logger(), "initialPoseReceived");
  if (msg->header.frame_id != global_frame_id_) {
    RCLCPP_WARN(this->get_logger(),
                "initialpose_frame_id does not match global_frame_id");
    return;
  }
  initialpose_recieved_ = true;
  relocalizing_ = reloc_search_enabled_;
  corrent_pose_with_cov_stamped_ptr_ = msg;
  pose_pub_->publish(*corrent_pose_with_cov_stamped_ptr_);

  if (use_odom_ && !odom_queue_.empty()) {
    Eigen::Affine3d map_to_base, odom_to_base;
    tf2::fromMsg(msg->pose.pose, map_to_base);
    tf2::fromMsg(odom_queue_.back().pose.pose, odom_to_base);
    map_to_odom_ = map_to_base * odom_to_base.inverse();
    map_to_odom_valid_ = true;
  }

  if (last_scan_ptr_) {
    cloudReceived(last_scan_ptr_);
  }
  RCLCPP_INFO(get_logger(), "initialPoseReceived end");
}

void PCLLocalization::mapReceived(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
  RCLCPP_INFO(get_logger(), "mapReceived");
  pcl::PointCloud<pcl::PointXYZI>::Ptr map_cloud_ptr(
      new pcl::PointCloud<pcl::PointXYZI>);

  if (msg->header.frame_id != global_frame_id_) {
    RCLCPP_WARN(this->get_logger(),
                "map_frame_id does not match　global_frame_id");
    return;
  }

  pcl::fromROSMsg(*msg, *map_cloud_ptr);

  if (registration_method_ == "GICP" || registration_method_ == "GICP_OMP") {
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud_ptr(
        new pcl::PointCloud<pcl::PointXYZI>());
    voxel_grid_filter_.setInputCloud(map_cloud_ptr);
    voxel_grid_filter_.filter(*filtered_cloud_ptr);
    registration_->setInputTarget(filtered_cloud_ptr);

  } else {
    registration_->setInputTarget(map_cloud_ptr);
  }

  map_recieved_ = true;
  RCLCPP_INFO(get_logger(), "mapReceived end");
}

bool PCLLocalization::odomToBaseAt(const rclcpp::Time& stamp,
                                   Eigen::Affine3d& odom_to_base) {
  if (odom_queue_.empty()) {
    return false;
  }

  const rclcpp::Time oldest(odom_queue_.front().header.stamp);
  const rclcpp::Time newest(odom_queue_.back().header.stamp);

  if (stamp < oldest) {
    RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Scan is older than the buffered odometry (%.3fs); "
        "map->odom not updated. Check clock sync or lower the scan rate.",
        (oldest - stamp).seconds());
    return false;
  }
  if (stamp >= newest) {
    tf2::fromMsg(odom_queue_.back().pose.pose, odom_to_base);
    return true;
  }

  for (std::size_t i = 1; i < odom_queue_.size(); ++i) {
    const rclcpp::Time t1(odom_queue_[i].header.stamp);
    if (t1 < stamp) {
      continue;
    }
    const rclcpp::Time t0(odom_queue_[i - 1].header.stamp);
    const double span = (t1 - t0).seconds();
    const double ratio = span > 1e-9 ? (stamp - t0).seconds() / span : 0.0;

    Eigen::Affine3d a0, a1;
    tf2::fromMsg(odom_queue_[i - 1].pose.pose, a0);
    tf2::fromMsg(odom_queue_[i].pose.pose, a1);

    const Eigen::Quaterniond q0(a0.rotation());
    const Eigen::Quaterniond q1(a1.rotation());
    odom_to_base = Eigen::Affine3d::Identity();
    odom_to_base.translate(Eigen::Vector3d(
        a0.translation() + ratio * (a1.translation() - a0.translation())));
    odom_to_base.rotate(q0.slerp(ratio, q1));
    return true;
  }

  tf2::fromMsg(odom_queue_.back().pose.pose, odom_to_base);
  return true;
}

void PCLLocalization::odomReceived(
    const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
  if (!use_odom_) {
    return;
  }
  RCLCPP_INFO_ONCE(get_logger(), "odomReceived started");

  odom_queue_.push_back(*msg);
  const rclcpp::Time newest(msg->header.stamp);
  while (odom_queue_.size() > 1 &&
         (newest - rclcpp::Time(odom_queue_.front().header.stamp)).seconds() >
             kOdomBufferSec) {
    odom_queue_.pop_front();
  }

  if (!initialpose_recieved_ || !map_to_odom_valid_) {
    return;
  }

  if (map_recieved_) {
    geometry_msgs::msg::TransformStamped map_to_odom_tf;
    map_to_odom_tf.header.stamp = msg->header.stamp;
    map_to_odom_tf.header.frame_id = global_frame_id_;
    map_to_odom_tf.child_frame_id = odom_frame_id_;
    map_to_odom_tf.transform = tf2::eigenToTransform(map_to_odom_).transform;
    broadcaster_.sendTransform(map_to_odom_tf);
  }
}

void PCLLocalization::imuReceived(
    const sensor_msgs::msg::Imu::ConstSharedPtr msg) {
  if (!use_imu_) {
    return;
  }

  sensor_msgs::msg::Imu tf_converted_imu;

  try {
    const geometry_msgs::msg::TransformStamped transform =
        tfbuffer_.lookupTransform(base_frame_id_, msg->header.frame_id,
                                  tf2::TimePointZero);

    geometry_msgs::msg::Vector3Stamped angular_velocity, linear_acceleration,
        transformed_angular_velocity, transformed_linear_acceleration;
    geometry_msgs::msg::Quaternion transformed_quaternion;

    angular_velocity.header = msg->header;
    angular_velocity.vector = msg->angular_velocity;
    linear_acceleration.header = msg->header;
    linear_acceleration.vector = msg->linear_acceleration;

    tf2::doTransform(angular_velocity, transformed_angular_velocity, transform);
    tf2::doTransform(linear_acceleration, transformed_linear_acceleration,
                     transform);

    tf_converted_imu.angular_velocity = transformed_angular_velocity.vector;
    tf_converted_imu.linear_acceleration =
        transformed_linear_acceleration.vector;
    tf_converted_imu.orientation = transformed_quaternion;

  } catch (tf2::TransformException& ex) {
    std::cout << "Failed to lookup transform" << std::endl;
    RCLCPP_WARN(this->get_logger(), "Failed to lookup transform.");
    return;
  }

  Eigen::Vector3f angular_velo{
      static_cast<float>(tf_converted_imu.angular_velocity.x),
      static_cast<float>(tf_converted_imu.angular_velocity.y),
      static_cast<float>(tf_converted_imu.angular_velocity.z)};
  Eigen::Vector3f acc{
      static_cast<float>(tf_converted_imu.linear_acceleration.x),
      static_cast<float>(tf_converted_imu.linear_acceleration.y),
      static_cast<float>(tf_converted_imu.linear_acceleration.z)};
  Eigen::Quaternionf quat{static_cast<float>(msg->orientation.w),
                          static_cast<float>(msg->orientation.x),
                          static_cast<float>(msg->orientation.y),
                          static_cast<float>(msg->orientation.z)};
  double imu_time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

  lidar_undistortion_.getImu(angular_velo, acc, quat, imu_time);
}

void PCLLocalization::cloudReceived(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
  if (!map_recieved_ || !initialpose_recieved_) {
    return;
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*msg, *cloud_ptr);

  if (use_imu_) {
    double received_time =
        msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    lidar_undistortion_.adjustDistortion(cloud_ptr, received_time);
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr cropped_ptr(
      new pcl::PointCloud<pcl::PointXYZI>());
  for (const auto& p : cloud_ptr->points) {
    double r = std::sqrt(p.x * p.x + p.y * p.y);
    if (!std::isfinite(p.x) || r <= scan_min_range_ || r >= scan_max_range_) {
      continue;
    }
    cropped_ptr->push_back(p);
  }
  pcl::PointCloud<pcl::PointXYZI>::Ptr tmp_ptr(
      new pcl::PointCloud<pcl::PointXYZI>());
  voxel_grid_filter_.setInputCloud(cropped_ptr);
  voxel_grid_filter_.filter(*tmp_ptr);
  registration_->setInputSource(tmp_ptr);

  Eigen::Affine3d affine;
  Eigen::Affine3d odom_to_base_at_scan;
  bool have_odom_at_scan =
      use_odom_ && map_to_odom_valid_ &&
      odomToBaseAt(rclcpp::Time(msg->header.stamp), odom_to_base_at_scan);
  if (have_odom_at_scan) {
    affine = map_to_odom_ * odom_to_base_at_scan;
  } else {
    tf2::fromMsg(corrent_pose_with_cov_stamped_ptr_->pose.pose, affine);
  }
  Eigen::Matrix4f init_guess = affine.matrix().cast<float>();

  pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);

  const bool did_relocalize_this_frame = relocalizing_;
  if (relocalizing_) {
    relocalizing_ = false;
    const auto t_reloc_start = std::chrono::steady_clock::now();

    constexpr size_t kCoarsePoints = 1000;
    pcl::PointCloud<pcl::PointXYZI>::Ptr coarse_src(
        new pcl::PointCloud<pcl::PointXYZI>(*tmp_ptr));
    if (coarse_src->size() > kCoarsePoints) {
      std::mt19937 thin_rng(2026);
      std::shuffle(coarse_src->points.begin(), coarse_src->points.end(), thin_rng);
      coarse_src->points.resize(kCoarsePoints);
      coarse_src->width = static_cast<uint32_t>(kCoarsePoints);
      coarse_src->height = 1;
    }

    const int max_iter_backup = ndt_max_iterations_;
    registration_->setInputSource(coarse_src);
    registration_->setMaximumIterations(4);

    auto screen = [&](const Eigen::Matrix4f& guess) {
      pcl::PointCloud<pcl::PointXYZI> out;
      registration_->align(out, guess);
      return registration_->hasConverged()
                 ? registration_->getFitnessScore()
                 : std::numeric_limits<double>::max();
    };
    auto with_yaw_xy = [&](double yaw_rad, double dx, double dy) {
      Eigen::Matrix4f g = init_guess;
      Eigen::Matrix3f r_yaw =
          Eigen::AngleAxisf(static_cast<float>(yaw_rad), Eigen::Vector3f::UnitZ())
              .toRotationMatrix();
      g.block<3, 3>(0, 0) = r_yaw * init_guess.block<3, 3>(0, 0);
      g(0, 3) += static_cast<float>(dx);
      g(1, 3) += static_cast<float>(dy);
      return g;
    };

    const double half = std::max(6.0, std::min(180.0, reloc_yaw_search_deg_));
    constexpr double kYawStepDeg = 12.0;
    const int kYawBins = static_cast<int>(std::floor(2.0 * half / kYawStepDeg)) + 1;
    std::vector<std::pair<double, double>> yaw_scored;
    for (int i = 0; i < kYawBins; ++i) {
      const double deg = -half + i * kYawStepDeg;
      const double yaw = deg * M_PI / 180.0;
      yaw_scored.emplace_back(screen(with_yaw_xy(yaw, 0.0, 0.0)), yaw);
    }
    std::sort(yaw_scored.begin(), yaw_scored.end());

    constexpr double kCoarseStep = 5.0;
    constexpr double kMaxR = 10.0;
    std::vector<std::pair<double, Eigen::Matrix4f>> cand;
    for (const auto& ys : yaw_scored) {
      for (int ix = -2; ix <= 2; ++ix) {
        for (int iy = -2; iy <= 2; ++iy) {
          const double dx = ix * kCoarseStep, dy = iy * kCoarseStep;
          if (std::hypot(dx, dy) > kMaxR + 1e-6) continue;
          const Eigen::Matrix4f g = with_yaw_xy(ys.second, dx, dy);
          cand.emplace_back(screen(g), g);
        }
      }
    }
    std::sort(cand.begin(), cand.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    constexpr size_t kSeeds = 6;
    constexpr double kFineStep = 2.0;
    std::vector<std::pair<double, Eigen::Matrix4f>> fine;
    for (size_t i = 0; i < std::min(kSeeds, cand.size()); ++i) {
      for (int ix = -1; ix <= 1; ++ix) {
        for (int iy = -1; iy <= 1; ++iy) {
          Eigen::Matrix4f g = cand[i].second;
          g(0, 3) += static_cast<float>(ix * kFineStep);
          g(1, 3) += static_cast<float>(iy * kFineStep);
          fine.emplace_back(screen(g), g);
        }
      }
    }
    cand.insert(cand.end(), fine.begin(), fine.end());
    std::sort(cand.begin(), cand.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    constexpr size_t kRefine = 16;
    const double kPriorPosWeight = reloc_prior_pos_weight_;
    const double kPriorYawWeight = reloc_prior_yaw_weight_;
    registration_->setInputSource(tmp_ptr);
    registration_->setMaximumIterations(max_iter_backup);
    const Eigen::Matrix4f prior_guess = init_guess;
    auto prior_penalty = [&](const Eigen::Matrix4f& m) {
      const double dpos = std::hypot(m(0, 3) - prior_guess(0, 3),
                                     m(1, 3) - prior_guess(1, 3));
      const Eigen::Matrix3f dr =
          prior_guess.block<3, 3>(0, 0).transpose() * m.block<3, 3>(0, 0);
      const double dyaw = std::abs(std::atan2(dr(1, 0), dr(0, 0)));
      return kPriorPosWeight * dpos / 10.0 + kPriorYawWeight * dyaw / M_PI;
    };
    double best_sel = std::numeric_limits<double>::max();
    double best_fitness = std::numeric_limits<double>::max();
    Eigen::Matrix4f best_guess = init_guess;
    int converged_count = 0;
    for (size_t i = 0; i < std::min(kRefine, cand.size()); ++i) {
      pcl::PointCloud<pcl::PointXYZI> out;
      registration_->align(out, cand[i].second);
      if (!registration_->hasConverged()) continue;
      ++converged_count;
      const double fit = registration_->getFitnessScore();
      const Eigen::Matrix4f final_m = registration_->getFinalTransformation();
      const double sel = fit + prior_penalty(final_m);
      if (sel < best_sel) {
        best_sel = sel;
        best_fitness = fit;
        best_guess = final_m;
      }
    }
    const double reloc_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_reloc_start)
                                .count();
    {
      const double dpos = std::hypot(best_guess(0, 3) - prior_guess(0, 3),
                                     best_guess(1, 3) - prior_guess(1, 3));
      const Eigen::Matrix3f dr =
          prior_guess.block<3, 3>(0, 0).transpose() * best_guess.block<3, 3>(0, 0);
      const double dyaw_deg =
          std::abs(std::atan2(dr(1, 0), dr(0, 0))) * 180.0 / M_PI;
      RCLCPP_INFO(get_logger(),
                  "Relocalization: %d yaw + %zu pose hypotheses over +-%.0f m, "
                  "%d of top-%zu converged, best fitness %.3f (threshold %.3f), "
                  "chosen pose is %.1f m / %.0f deg from the given prior, %.0f ms",
                  kYawBins, cand.size(), kMaxR, converged_count, kRefine,
                  best_fitness, score_threshold_, dpos, dyaw_deg, reloc_ms);
      if (dpos > 3.0 || dyaw_deg > 45.0) {
        RCLCPP_WARN(get_logger(),
                    "Relocalization landed far from the given prior (%.1f m / "
                    "%.0f deg). If the robot is actually at the clicked pose, "
                    "re-send the initial pose.",
                    dpos, dyaw_deg);
      }
    }
    init_guess = best_guess;
  }

  registration_->align(*output_cloud, init_guess);

  bool has_converged = registration_->hasConverged();
  double fitness_score = registration_->getFitnessScore();
  if (!has_converged) {
    RCLCPP_WARN(get_logger(), "The registration didn't converge.");
    return;
  }

  if (fitness_score > score_threshold_) {
    RCLCPP_WARN(get_logger(),
                "Fitness %.3f exceeds score_threshold %.3f — the pose may be "
                "a wrong local minimum.",
                fitness_score, score_threshold_);
  }

  if (!did_relocalize_this_frame) {
    const Eigen::Matrix4f final_m = registration_->getFinalTransformation();
    const double jump_pos = std::hypot(final_m(0, 3) - init_guess(0, 3),
                                       final_m(1, 3) - init_guess(1, 3));
    const Eigen::Matrix3f dr = init_guess.block<3, 3>(0, 0).transpose() *
                               final_m.block<3, 3>(0, 0);
    const double jump_rad = std::abs(std::atan2(dr(1, 0), dr(0, 0)));
    const rclcpp::Time scan_time(msg->header.stamp);
    double dt = 0.1;
    if (have_last_scan_time_) {
      const double d = (scan_time - last_scan_time_).seconds();
      if (d > 1e-3 && d < 2.0) { dt = d; }
    }
    constexpr double kMaxSpeed = 10.0;
    constexpr double kMaxOmega = 6.0;
    constexpr int kMaxConsecutiveRejects = 30;
    if (jump_pos / dt > kMaxSpeed || jump_rad / dt > kMaxOmega) {
      ++divergence_reject_count_;
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Rejecting diverged registration: %.1f m / %.0f deg in %.2fs "
          "(= %.0f m/s) away from the odometry prediction; the robot cannot "
          "move like that. Pose held (%d consecutive).",
          jump_pos, jump_rad * 180.0 / M_PI, dt, jump_pos / dt,
          divergence_reject_count_);
      if (divergence_reject_count_ >= kMaxConsecutiveRejects) {
        RCLCPP_ERROR(get_logger(),
                     "Registration has been diverging for %d frames — trying a "
                     "multi-hypothesis relocalization around the last good "
                     "prediction. Re-send /initialpose if this does not recover.",
                     divergence_reject_count_);
        divergence_reject_count_ = 0;
        relocalizing_ = reloc_search_enabled_;
      }
      last_scan_time_ = scan_time;
      have_last_scan_time_ = true;
      return;
    }
    divergence_reject_count_ = 0;
    last_scan_time_ = scan_time;
    have_last_scan_time_ = true;
  }

  Eigen::Matrix4f final_transformation =
      registration_->getFinalTransformation();
  Eigen::Matrix3d rot_mat =
      final_transformation.block<3, 3>(0, 0).cast<double>();
  Eigen::Quaterniond quat_eig(rot_mat);

  corrent_pose_with_cov_stamped_ptr_->header.stamp = msg->header.stamp;
  corrent_pose_with_cov_stamped_ptr_->header.frame_id = global_frame_id_;
  corrent_pose_with_cov_stamped_ptr_->pose.pose.position.x =
      static_cast<double>(final_transformation(0, 3));
  corrent_pose_with_cov_stamped_ptr_->pose.pose.position.y =
      static_cast<double>(final_transformation(1, 3));
  corrent_pose_with_cov_stamped_ptr_->pose.pose.position.z =
      static_cast<double>(final_transformation(2, 3));
  corrent_pose_with_cov_stamped_ptr_->pose.pose.orientation =
      tf2::toMsg(quat_eig);

  double pose_covariance = std::max(0.1, fitness_score);
  for (int i = 0; i < 36; i++)
    corrent_pose_with_cov_stamped_ptr_->pose.covariance[i] = 0;
  corrent_pose_with_cov_stamped_ptr_->pose.covariance[0] =
      corrent_pose_with_cov_stamped_ptr_->pose.covariance[7] =
          corrent_pose_with_cov_stamped_ptr_->pose.covariance[14] =
              corrent_pose_with_cov_stamped_ptr_->pose.covariance[21] =
                  corrent_pose_with_cov_stamped_ptr_->pose.covariance[28] =
                      corrent_pose_with_cov_stamped_ptr_->pose.covariance[35] =
                          pose_covariance;

  pose_pub_->publish(*corrent_pose_with_cov_stamped_ptr_);

  if (use_odom_ &&
      (have_odom_at_scan ||
       odomToBaseAt(rclcpp::Time(msg->header.stamp), odom_to_base_at_scan))) {
    Eigen::Affine3d map_to_base_at_scan;
    tf2::fromMsg(corrent_pose_with_cov_stamped_ptr_->pose.pose,
                 map_to_base_at_scan);
    map_to_odom_ = map_to_base_at_scan * odom_to_base_at_scan.inverse();
    map_to_odom_valid_ = true;
  }

  if (!use_odom_) {
    geometry_msgs::msg::TransformStamped transform_stamped;
    transform_stamped.header.stamp = msg->header.stamp;
    transform_stamped.header.frame_id = global_frame_id_;
    transform_stamped.child_frame_id = base_frame_id_;
    transform_stamped.transform.translation.x =
        corrent_pose_with_cov_stamped_ptr_->pose.pose.position.x;
    transform_stamped.transform.translation.y =
        corrent_pose_with_cov_stamped_ptr_->pose.pose.position.y;
    transform_stamped.transform.translation.z =
        corrent_pose_with_cov_stamped_ptr_->pose.pose.position.z;
    transform_stamped.transform.rotation =
        corrent_pose_with_cov_stamped_ptr_->pose.pose.orientation;
    broadcaster_.sendTransform(transform_stamped);
  }

  geometry_msgs::msg::PoseStamped::SharedPtr pose_stamped_ptr(
      new geometry_msgs::msg::PoseStamped);
  pose_stamped_ptr->header = corrent_pose_with_cov_stamped_ptr_->header;
  pose_stamped_ptr->pose = corrent_pose_with_cov_stamped_ptr_->pose.pose;
  path_ptr_->poses.push_back(*pose_stamped_ptr);
  path_pub_->publish(*path_ptr_);

  last_scan_ptr_ = msg;
}
