/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2012, Willow Garage, Inc.
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

#ifndef PCL_REGISTRATION_NDT_OMP_H_
#define PCL_REGISTRATION_NDT_OMP_H_

#include <pcl/registration/registration.h>
#include <pcl/search/impl/search.hpp>
#include "voxel_grid_covariance_omp.h"

#include <unsupported/Eigen/NonLinearOptimization>

namespace pclomp
{
	enum NeighborSearchMethod {
		KDTREE,
		DIRECT26,
		DIRECT7,
		DIRECT1
	};

	template<typename PointSource, typename PointTarget>
	class NormalDistributionsTransform : public pcl::Registration<PointSource, PointTarget>
	{
	protected:

		typedef typename pcl::Registration<PointSource, PointTarget>::PointCloudSource PointCloudSource;
		typedef typename PointCloudSource::Ptr PointCloudSourcePtr;
		typedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;

		typedef typename pcl::Registration<PointSource, PointTarget>::PointCloudTarget PointCloudTarget;
		typedef typename PointCloudTarget::Ptr PointCloudTargetPtr;
		typedef typename PointCloudTarget::ConstPtr PointCloudTargetConstPtr;

		typedef pcl::PointIndices::Ptr PointIndicesPtr;
		typedef pcl::PointIndices::ConstPtr PointIndicesConstPtr;

		typedef pclomp::VoxelGridCovariance<PointTarget> TargetGrid;
		typedef TargetGrid* TargetGridPtr;
		typedef const TargetGrid* TargetGridConstPtr;
		typedef typename TargetGrid::LeafConstPtr TargetGridLeafConstPtr;


	public:

		typedef boost::shared_ptr< NormalDistributionsTransform<PointSource, PointTarget> > Ptr;
		typedef boost::shared_ptr< const NormalDistributionsTransform<PointSource, PointTarget> > ConstPtr;


		NormalDistributionsTransform();

		virtual ~NormalDistributionsTransform() {}

    void setNumThreads(int n) {
      num_threads_ = n;
    }

		inline void
			setInputTarget(const PointCloudTargetConstPtr &cloud)
		{
			pcl::Registration<PointSource, PointTarget>::setInputTarget(cloud);
			init();
		}

		inline void
			setResolution(float resolution)
		{
			if (resolution_ != resolution)
			{
				resolution_ = resolution;
				if (input_)
					init();
			}
		}

		inline float
			getResolution() const
		{
			return (resolution_);
		}

		inline double
			getStepSize() const
		{
			return (step_size_);
		}

		inline void
			setStepSize(double step_size)
		{
			step_size_ = step_size;
		}

		inline double
			getOulierRatio() const
		{
			return (outlier_ratio_);
		}

		inline void
			setOulierRatio(double outlier_ratio)
		{
			outlier_ratio_ = outlier_ratio;
		}

		inline void setNeighborhoodSearchMethod(NeighborSearchMethod method) {
			search_method = method;
		}

		inline double
			getTransformationProbability() const
		{
			return (trans_probability_);
		}

		inline int
			getFinalNumIteration() const
		{
			return (nr_iterations_);
		}

		static void
			convertTransform(const Eigen::Matrix<double, 6, 1> &x, Eigen::Affine3f &trans)
		{
			trans = Eigen::Translation<float, 3>(float(x(0)), float(x(1)), float(x(2))) *
				Eigen::AngleAxis<float>(float(x(3)), Eigen::Vector3f::UnitX()) *
				Eigen::AngleAxis<float>(float(x(4)), Eigen::Vector3f::UnitY()) *
				Eigen::AngleAxis<float>(float(x(5)), Eigen::Vector3f::UnitZ());
		}

		static void
			convertTransform(const Eigen::Matrix<double, 6, 1> &x, Eigen::Matrix4f &trans)
		{
			Eigen::Affine3f _affine;
			convertTransform(x, _affine);
			trans = _affine.matrix();
		}

		double calculateScore(const PointCloudSource& cloud) const;

	protected:

		using pcl::Registration<PointSource, PointTarget>::reg_name_;
		using pcl::Registration<PointSource, PointTarget>::getClassName;
		using pcl::Registration<PointSource, PointTarget>::input_;
		using pcl::Registration<PointSource, PointTarget>::indices_;
		using pcl::Registration<PointSource, PointTarget>::target_;
		using pcl::Registration<PointSource, PointTarget>::nr_iterations_;
		using pcl::Registration<PointSource, PointTarget>::max_iterations_;
		using pcl::Registration<PointSource, PointTarget>::previous_transformation_;
		using pcl::Registration<PointSource, PointTarget>::final_transformation_;
		using pcl::Registration<PointSource, PointTarget>::transformation_;
		using pcl::Registration<PointSource, PointTarget>::transformation_epsilon_;
		using pcl::Registration<PointSource, PointTarget>::converged_;
		using pcl::Registration<PointSource, PointTarget>::corr_dist_threshold_;
		using pcl::Registration<PointSource, PointTarget>::inlier_threshold_;

		using pcl::Registration<PointSource, PointTarget>::update_visualizer_;

		virtual void
			computeTransformation(PointCloudSource &output)
		{
			computeTransformation(output, Eigen::Matrix4f::Identity());
		}

		virtual void
			computeTransformation(PointCloudSource &output, const Eigen::Matrix4f &guess);

		void inline
			init()
		{
			target_cells_.setLeafSize(resolution_, resolution_, resolution_);
			target_cells_.setInputCloud(target_);
			target_cells_.filter(true);
		}

		double
			computeDerivatives(Eigen::Matrix<double, 6, 1> &score_gradient,
				Eigen::Matrix<double, 6, 6> &hessian,
				PointCloudSource &trans_cloud,
				Eigen::Matrix<double, 6, 1> &p,
				bool compute_hessian = true);

		double
			updateDerivatives(Eigen::Matrix<double, 6, 1> &score_gradient,
				Eigen::Matrix<double, 6, 6> &hessian,
				const Eigen::Matrix<float, 4, 6> &point_gradient_,
				const Eigen::Matrix<float, 24, 6> &point_hessian_,
				const Eigen::Vector3d &x_trans, const Eigen::Matrix3d &c_inv,
				bool compute_hessian = true) const;

		void
			computeAngleDerivatives(Eigen::Matrix<double, 6, 1> &p, bool compute_hessian = true);

		void
			computePointDerivatives(Eigen::Vector3d &x, Eigen::Matrix<double, 3, 6>& point_gradient_, Eigen::Matrix<double, 18, 6>& point_hessian_, bool compute_hessian = true) const;

		void
			computePointDerivatives(Eigen::Vector3d &x, Eigen::Matrix<float, 4, 6>& point_gradient_, Eigen::Matrix<float, 24, 6>& point_hessian_, bool compute_hessian = true) const;

		void
			computeHessian(Eigen::Matrix<double, 6, 6> &hessian,
				PointCloudSource &trans_cloud,
				Eigen::Matrix<double, 6, 1> &p);

		void
			updateHessian(Eigen::Matrix<double, 6, 6> &hessian,
				const Eigen::Matrix<double, 3, 6> &point_gradient_,
				const Eigen::Matrix<double, 18, 6> &point_hessian_,
				const Eigen::Vector3d &x_trans, const Eigen::Matrix3d &c_inv) const;

		double
			computeStepLengthMT(const Eigen::Matrix<double, 6, 1> &x,
				Eigen::Matrix<double, 6, 1> &step_dir,
				double step_init,
				double step_max, double step_min,
				double &score,
				Eigen::Matrix<double, 6, 1> &score_gradient,
				Eigen::Matrix<double, 6, 6> &hessian,
				PointCloudSource &trans_cloud);

		bool
			updateIntervalMT(double &a_l, double &f_l, double &g_l,
				double &a_u, double &f_u, double &g_u,
				double a_t, double f_t, double g_t);

		double
			trialValueSelectionMT(double a_l, double f_l, double g_l,
				double a_u, double f_u, double g_u,
				double a_t, double f_t, double g_t);

		inline double
			auxilaryFunction_PsiMT(double a, double f_a, double f_0, double g_0, double mu = 1.e-4)
		{
			return (f_a - f_0 - mu * g_0 * a);
		}

		inline double
			auxilaryFunction_dPsiMT(double g_a, double g_0, double mu = 1.e-4)
		{
			return (g_a - mu * g_0);
		}

		TargetGrid target_cells_;


		float resolution_;

		double step_size_;

		double outlier_ratio_;

		double gauss_d1_, gauss_d2_, gauss_d3_;

		double trans_probability_;

		Eigen::Vector3d j_ang_a_, j_ang_b_, j_ang_c_, j_ang_d_, j_ang_e_, j_ang_f_, j_ang_g_, j_ang_h_;

		Eigen::Matrix<float, 8, 4> j_ang;

		Eigen::Vector3d h_ang_a2_, h_ang_a3_,
			h_ang_b2_, h_ang_b3_,
			h_ang_c2_, h_ang_c3_,
			h_ang_d1_, h_ang_d2_, h_ang_d3_,
			h_ang_e1_, h_ang_e2_, h_ang_e3_,
			h_ang_f1_, h_ang_f2_, h_ang_f3_;

		Eigen::Matrix<float, 16, 4> h_ang;



    int num_threads_;

	public:
		NeighborSearchMethod search_method;

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};

}

#endif
