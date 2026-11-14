/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#ifndef PCL_GICP_OMP_H_
#define PCL_GICP_OMP_H_

#include <pcl/registration/icp.h>
#include <pcl/registration/bfgs.h>
#include <boost/shared_ptr.hpp>

namespace pclomp
{
  template <typename PointSource, typename PointTarget>
  class GeneralizedIterativeClosestPoint : public pcl::IterativeClosestPoint<PointSource, PointTarget>
  {
    public:
      using pcl::IterativeClosestPoint<PointSource, PointTarget>::reg_name_;
      using pcl::IterativeClosestPoint<PointSource, PointTarget>::getClassName;
      using pcl::IterativeClosestPoint<PointSource, PointTarget>::indices_;
      using pcl::IterativeClosestPoint<PointSource, PointTarget>::target_;
      using pcl::IterativeClosestPoint<PointSource, PointTarget>::input_;
      using pcl::IterativeClosestPoint<PointSource, PointTarget>::tree_;
      using pcl::IterativeClosestPoint<PointSource, PointTarget>::tree_reciprocal_;
      using pcl::IterativeClosestPoint<PointSource, PointTarget>::nr_iterations_;
      using pcl::IterativeClosestPoint<PointSource, PointTarget>::max_iterations_;
      using pcl::IterativeClosestPoint<PointSource, PointTarget>::previous_transformation_;
      using pcl::IterativeClosestPoint<PointSource, PointTarget>::final_transformation_;
      using pcl::IterativeClosestPoint<PointSource, PointTarget>::transformation_;
      using pcl::IterativeClosestPoint<PointSource, PointTarget>::transformation_epsilon_;
      using pcl::IterativeClosestPoint<PointSource, PointTarget>::converged_;
      using pcl::IterativeClosestPoint<PointSource, PointTarget>::corr_dist_threshold_;
      using pcl::IterativeClosestPoint<PointSource, PointTarget>::inlier_threshold_;
      using pcl::IterativeClosestPoint<PointSource, PointTarget>::min_number_correspondences_;
      using pcl::IterativeClosestPoint<PointSource, PointTarget>::update_visualizer_;

      using PointCloudSource = pcl::PointCloud<PointSource>;
      using PointCloudSourcePtr = typename PointCloudSource::Ptr;
      using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

      using PointCloudTarget = pcl::PointCloud<PointTarget>;
      using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
      using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

      using PointIndicesPtr = pcl::PointIndices::Ptr;
      using PointIndicesConstPtr = pcl::PointIndices::ConstPtr;

      using MatricesVector = std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> >;
      using MatricesVectorPtr = boost::shared_ptr<MatricesVector>;
      using MatricesVectorConstPtr = boost::shared_ptr<const MatricesVector>;

      using InputKdTree = typename pcl::Registration<PointSource, PointTarget>::KdTree;
      using InputKdTreePtr = typename pcl::Registration<PointSource, PointTarget>::KdTreePtr;

      using Ptr = boost::shared_ptr<GeneralizedIterativeClosestPoint<PointSource, PointTarget> >;
      using ConstPtr = boost::shared_ptr<const GeneralizedIterativeClosestPoint<PointSource, PointTarget> >;


      using Vector6d = Eigen::Matrix<double, 6, 1>;

      GeneralizedIterativeClosestPoint ()
        : k_correspondences_(20)
        , gicp_epsilon_(0.001)
        , rotation_epsilon_(2e-3)
        , mahalanobis_(0)
        , max_inner_iterations_(20)
      {
        min_number_correspondences_ = 4;
        reg_name_ = "GeneralizedIterativeClosestPoint";
        max_iterations_ = 200;
        transformation_epsilon_ = 5e-4;
        corr_dist_threshold_ = 5.;
        rigid_transformation_estimation_ = [this] (const PointCloudSource& cloud_src,
                                                   const std::vector<int>& indices_src,
                                                   const PointCloudTarget& cloud_tgt,
                                                   const std::vector<int>& indices_tgt,
                                                   Eigen::Matrix4f& transformation_matrix)
        {
          estimateRigidTransformationBFGS (cloud_src, indices_src, cloud_tgt, indices_tgt, transformation_matrix);
        };
      }

      inline void
      setInputSource (const PointCloudSourceConstPtr &cloud) override
      {

        if (cloud->points.empty ())
        {
          PCL_ERROR ("[pcl::%s::setInputSource] Invalid or empty point cloud dataset given!\n", getClassName ().c_str ());
          return;
        }
        PointCloudSource input = *cloud;
        for (size_t i = 0; i < input.size (); ++i)
          input[i].data[3] = 1.0;

        pcl::IterativeClosestPoint<PointSource, PointTarget>::setInputSource (cloud);
        input_covariances_.reset ();
      }

      inline void
      setSourceCovariances (const MatricesVectorPtr& covariances)
      {
        input_covariances_ = covariances;
      }

      inline void
      setInputTarget (const PointCloudTargetConstPtr &target) override
      {
        pcl::IterativeClosestPoint<PointSource, PointTarget>::setInputTarget(target);
        target_covariances_.reset ();
      }

	    inline void
      setTargetCovariances (const MatricesVectorPtr& covariances)
      {
        target_covariances_ = covariances;
      }

      void
      estimateRigidTransformationBFGS (const PointCloudSource &cloud_src,
                                       const std::vector<int> &indices_src,
                                       const PointCloudTarget &cloud_tgt,
                                       const std::vector<int> &indices_tgt,
                                       Eigen::Matrix4f &transformation_matrix);

      inline const Eigen::Matrix4f& mahalanobis(size_t index) const
      {
        assert(index < mahalanobis_.size());
        return mahalanobis_[index];
      }

      void
      computeRDerivative(const Vector6d &x, const Eigen::Matrix3d &R, Vector6d &g) const;

      inline void
      setRotationEpsilon (double epsilon) { rotation_epsilon_ = epsilon; }

      inline double
      getRotationEpsilon () { return (rotation_epsilon_); }

      void
      setCorrespondenceRandomness (int k) { k_correspondences_ = k; }

      int
      getCorrespondenceRandomness () { return (k_correspondences_); }

      void
      setMaximumOptimizerIterations (int max) { max_inner_iterations_ = max; }

      int
      getMaximumOptimizerIterations () { return (max_inner_iterations_); }

    protected:

      int k_correspondences_;

      double gicp_epsilon_;

      double rotation_epsilon_;

      Eigen::Matrix4f base_transformation_;

      const PointCloudSource *tmp_src_;

      const PointCloudTarget  *tmp_tgt_;

      const std::vector<int> *tmp_idx_src_;

      const std::vector<int> *tmp_idx_tgt_;


      MatricesVectorPtr input_covariances_;

      MatricesVectorPtr target_covariances_;

      std::vector<Eigen::Matrix4f> mahalanobis_;

      int max_inner_iterations_;

      template<typename PointT>
      void computeCovariances(typename pcl::PointCloud<PointT>::ConstPtr cloud,
                              const typename pcl::search::KdTree<PointT>::ConstPtr tree,
                              MatricesVector& cloud_covariances);

      inline double
      matricesInnerProd(const Eigen::MatrixXd& mat1, const Eigen::MatrixXd& mat2) const
      {
        double r = 0.;
        size_t n = mat1.rows();
        for(size_t i = 0; i < n; i++)
          for(size_t j = 0; j < n; j++)
            r += mat1 (j, i) * mat2 (i,j);
        return r;
      }

      void
      computeTransformation (PointCloudSource &output, const Eigen::Matrix4f &guess) override;

      inline bool
      searchForNeighbors (const PointSource &query, std::vector<int>& index, std::vector<float>& distance) const
      {
        int k = tree_->nearestKSearch (query, 1, index, distance);
        if (k == 0)
          return (false);
        return (true);
      }

      void applyState(Eigen::Matrix4f &t, const Vector6d& x) const;

      struct OptimizationFunctorWithIndices : public BFGSDummyFunctor<double,6>
      {
        OptimizationFunctorWithIndices (const GeneralizedIterativeClosestPoint* gicp)
          : BFGSDummyFunctor<double,6> (), gicp_(gicp) {}
        double operator() (const Vector6d& x) override;
        void  df(const Vector6d &x, Vector6d &df) override;
        void fdf(const Vector6d &x, double &f, Vector6d &df) override;

        const GeneralizedIterativeClosestPoint *gicp_;
      };

      std::function<void(const pcl::PointCloud<PointSource> &cloud_src,
                           const std::vector<int> &src_indices,
                           const pcl::PointCloud<PointTarget> &cloud_tgt,
                           const std::vector<int> &tgt_indices,
                           Eigen::Matrix4f &transformation_matrix)> rigid_transformation_estimation_;
  };
}

#endif
