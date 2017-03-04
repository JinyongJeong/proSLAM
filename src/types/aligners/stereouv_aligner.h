#pragma once
#include "../contexts/frame.h"
#include "base_aligner.h"

namespace gslam {

  class StereoUVAligner: public BaseAligner6_4 {
    public: EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    //ds object handling
    protected:

      //ds instantiation controlled by aligner factory
      StereoUVAligner(): BaseAligner6_4(1e-3, 9, 1) {}
      ~StereoUVAligner() {}

    //ds required interface
    public:

      //ds initialize aligner with minimal entity TODO purify this
      virtual void init(BaseContext* context_, const TransformMatrix3D& robot_to_world_ = TransformMatrix3D::Identity());

      //ds linearize the system: to be called inside oneRound
      virtual void linearize(const bool& ignore_outliers_);

      //ds solve alignment problem for one round: to be called inside converge
      virtual void oneRound(const bool& ignore_outliers_);

      //ds solve alignment problem until convergence is reached
      virtual void converge();

      //ds additional accessors
    public:

      //ds getters/setters
      const TransformMatrix3D robotToWorld() const {return _robot_to_world;}
      const TransformMatrix3D worldToRobot() const {return _world_to_robot;}
      void setWeightFramepoint(const gt_real& weight_framepoint_) {_weight_framepoint = weight_framepoint_;}

    //ds aligner specific
    protected:

      //ds context
      Frame* _context = 0;

      //ds objective
      TransformMatrix3D _world_to_camera_left = TransformMatrix3D::Identity();
      TransformMatrix3D _camera_left_to_world = TransformMatrix3D::Identity();

      //ds optimization wrapping
      TransformMatrix3D _world_to_robot = TransformMatrix3D::Identity();
      TransformMatrix3D _robot_to_world = TransformMatrix3D::Identity();
      ProjectionMatrix _projection_matrix_left  = ProjectionMatrix::Zero();
      ProjectionMatrix _projection_matrix_right = ProjectionMatrix::Zero();
      Count _image_rows                         = 0;
      Count _image_cols                         = 0;

      //ds others
      TransformMatrix3D _robot_to_world_previous = TransformMatrix3D::Identity();
      gt_real _weight_framepoint                 = 1;

    //ds grant access to factory: ctor/dtor
    friend AlignerFactory;

  }; //class StereoUVAligner
} //namespace gslam
