
#include "base_tracker.h"

#include "aligners/stereouv_aligner.h"

namespace proslam
{
using namespace srrg_core;

BaseTracker::BaseTracker(BaseTrackerParameters* parameters_) : _parameters(parameters_), _has_odometry(false)
{
  LOG_INFO(std::cerr << "BaseTracker::BaseTracker|constructed" << std::endl)
}

void BaseTracker::configure()
{
  LOG_INFO(std::cerr << "BaseTracker::configure|configuring" << std::endl)
  assert(_camera_left);
  assert(_pose_optimizer);
  _previous_to_current_camera.setIdentity();
  _lost_points.clear();
  _projection_tracking_distance_pixels =
      _framepoint_generator->parameters()->maximum_projection_tracking_distance_pixels;

  // ds print tracker configuration (with dynamic type of parameters)
  LOG_INFO(std::cerr << "BaseTracker::configure|configured" << std::endl)
}

// ds dynamic cleanup
BaseTracker::~BaseTracker()
{
  LOG_INFO(std::cerr << "BaseTracker::~BaseTracker|destroying" << std::endl)
  _lost_points.clear();

  // ds free dynamics
  delete _framepoint_generator;
  delete _pose_optimizer;
  LOG_INFO(std::cerr << "BaseTracker::~BaseTracker|destroyed" << std::endl)
}

// Camera image를 넣어주고 compute() 실행
void BaseTracker::compute()
{
  assert(_camera_left);
  assert(_context);

  // ds reset point configurations
  _number_of_tracked_points = 0;
  _number_of_lost_points = 0;
  _number_of_recovered_points = 0;
  _context->currentlyTrackedLandmarks().clear();

  // ds relative camera motion guess
  TransformMatrix3D previous_to_current = TransformMatrix3D::Identity();

  // ds check if initial guess can be refined with a motion model or other input
  switch (_parameters->motion_model)
  {
    // ds use previous motion as initial guess for current motion
    case Parameters::MotionModel::CONSTANT_VELOCITY:
    {
      previous_to_current = _previous_to_current_camera;
      break;
    }

    // ds camera odometry as motion guess
    case Parameters::MotionModel::CAMERA_ODOMETRY:
    {
      if (!_context->currentFrame())
      {
        _previous_odometry = _odometry;
      }
      TransformMatrix3D odom_delta = _odometry.inverse() * _previous_odometry;
      _previous_odometry = _odometry;
      previous_to_current = odom_delta;
      break;
    }

    // ds no guess (identity)
    case Parameters::MotionModel::NONE:
    {
      previous_to_current = TransformMatrix3D::Identity();
      break;
    }

    // ds invalid setting
    default:
    {
      throw std::runtime_error("motion model undefined");
    }
  }

  // ds create new frame
  // camera, intensity image로 frame을 만든다. (worldmap에서 최신 frame을 이전 frame에 넣고 새로운 frame 생성)
  Frame* current_frame = _createFrame();
  // status: localizing or tracking
  current_frame->setStatus(_status);
  // previous frame은 worldmap이 넣어준 frame (_createFrame 함수에서)
  Frame* previous_frame = current_frame->previous();

  // ds initialize framepoint generator
  // 현재 frame에서 keypoint추출 -> descriptor 추출 -> matcher에 데이터 넣어줌
  _framepoint_generator->initialize(current_frame);

  // ds if possible - attempt to track the points from the previous frame
  if (previous_frame)
  {
    // ds see if we're tracking by appearance (expecting large photometric displacements between images)
    // ds always do it for the first 2 frames (repeats after track is lost)
    bool track_by_appearance = (_status == Frame::Localizing);

    // ds search point tracks
    // 이전 frame이 유효하면 tracking 실행
    // 이전 frame의 feature와 maching되는 keypoint를 current left/right에서 찾는다
    // left/right에서 모두 찾으면 tracking 성공/ 아니면 실패
    // 여기까지 pose 보정은 없음
    CHRONOMETER_START(tracking);
    _track(previous_frame, current_frame, previous_to_current, track_by_appearance);
    CHRONOMETER_STOP(tracking);
  }

  // ds check previous tracker status
  _number_of_potential_points = _framepoint_generator->numberOfDetectedKeypoints();
  switch (_status)
  {
    // 상황에 따라서 localizatino mode와 tracking모드로 나뉘어서 계산이 수행된다.

    // ds localization mode - always running on tracking by appearance with maximum window size

    // Localization 모드
    case Frame::Localizing:
    {
      // ds if we got not enough tracks to evaluate for position tracking
      // tracking된 point가 너무 적으면 종료
      if (_number_of_tracked_points < _parameters->minimum_number_of_landmarks_to_track)
      {
        // ds reset state and stick to previous solution
        LOG_WARNING(
            std::printf("BaseTracker::compute|skipping position tracking due to insufficient number of tracks: %lu\n",
                        _number_of_tracked_points))
        _previous_to_current_camera = TransformMatrix3D::Identity();
        break;
      }

      // ds if we have a previous frame
      if (previous_frame)
      {
        LOG_INFO(std::cerr << "BaseTracker::compute|STATE: LOCALIZING|current tracks: " << _number_of_tracked_points
                           << "/" << previous_frame->points().size() << " at image: " << current_frame->identifier()
                           << std::endl)

        // ds solve pose on frame points only
        CHRONOMETER_START(pose_optimization);
        // pose optimizer 실행
        _pose_optimizer->setEnableWeightsTranslation(false);
        // Optimizer 초기화 / 데이터 정리
        _pose_optimizer->initialize(previous_frame, current_frame, previous_to_current);
        // Optimize 실행 (point reporjection을 이용해서 relative pose 최적화)
        _pose_optimizer->converge();
        CHRONOMETER_STOP(pose_optimization);

        // ds if the pose computation is acceptable
        if (_pose_optimizer->numberOfInliers() > _parameters->minimum_number_of_landmarks_to_track)
        {
          // ds compute resulting motion
          _previous_to_current_camera = _pose_optimizer->previousToCurrent();
          const real delta_angular = WorldMap::toOrientationRodrigues(_previous_to_current_camera.linear()).norm();
          const real delta_translational = _previous_to_current_camera.translation().norm();

          // ds if the posit result is significant enough
          if (delta_angular > _parameters->minimum_delta_angular_for_movement ||
              delta_translational > _parameters->minimum_delta_translational_for_movement)
          {
            // ds update tracker - TODO purge this transform chaos
            const TransformMatrix3D world_to_camera_current =
                _previous_to_current_camera * previous_frame->worldToCameraLeft();
            const TransformMatrix3D world_to_robot_current = _camera_left->cameraToRobot() * world_to_camera_current;
            current_frame->setRobotToWorld(world_to_robot_current.inverse());
            LOG_WARNING(std::cerr << "BaseTracker::compute|using posit on frame points (experimental) inliers: "
                                  << _pose_optimizer->numberOfInliers()
                                  << " outliers: " << _pose_optimizer->numberOfOutliers() << " average error: "
                                  << _pose_optimizer->totalError() / _pose_optimizer->numberOfInliers() << std::endl)
          }
          else
          {
            // ds keep previous solution
            current_frame->setRobotToWorld(previous_frame->robotToWorld());
            _previous_to_current_camera = TransformMatrix3D::Identity();
          }

          // ds update context position
          _context->setRobotToWorld(current_frame->robotToWorld());
        }
      }
      break;
    }

    // ds on the track
    case Frame::Tracking:
    {
      _registerRecursive(previous_frame, current_frame, previous_to_current);
      break;
    }
    default:
    {
      throw std::runtime_error("invalid tracker state");
      break;
    }
  }

  // ds trigger landmark creation and framepoint update
  _updatePoints(_context, current_frame);

  // ds check if we switch to tracking state
  _status_previous = _status;
  if (_number_of_active_landmarks > _parameters->minimum_number_of_landmarks_to_track)
  {
    _status = Frame::Tracking;
  }

  // ds compute remaining points in frame
  CHRONOMETER_START(track_creation)
  _framepoint_generator->compute(current_frame);
  CHRONOMETER_STOP(track_creation)
  current_frame->setStatus(_status);

  // ds update bookkeeping
  _number_of_tracked_landmarks_previous = _context->currentlyTrackedLandmarks().size();
  _total_number_of_tracked_points += _number_of_tracked_points;

  // ds update stats
  _mean_number_of_keypoints =
      (_mean_number_of_keypoints * (_context->frames().size() - 1) + current_frame->_number_of_detected_keypoints) /
      _context->frames().size();
  _mean_number_of_framepoints =
      (_mean_number_of_framepoints * (_context->frames().size() - 1) + current_frame->points().size()) /
      _context->frames().size();
}

// ds retrieves framepoint correspondences between previous and current frame
void BaseTracker::_track(Frame* previous_frame_, Frame* current_frame_, const TransformMatrix3D& previous_to_current_,
                         const bool& track_by_appearance_)
{
  // ds check state for current framepoint tracking configuration
  // apearance 로 tracking할 때 search 윈도우 크기를 최대로 (이전 frame과 현재 frame의 correspondence 찾을 때)
  if (track_by_appearance_)
  {
    // ds maximimize tracking window
    _projection_tracking_distance_pixels =
        _framepoint_generator->parameters()->maximum_projection_tracking_distance_pixels;
  }

  // ds configure and track points in current frame
  _framepoint_generator->setProjectionTrackingDistancePixels(_projection_tracking_distance_pixels);
  // 이전 frame의 feature와 maching되는 keypoint를 current left/right에서 찾는다
  // left/right에서 모두 찾으면 tracking 성공/ 아니면 실패
  // _lost_points: tracking에 실패한 point들이 들어감
  // tracking파트는 motion 모델(여기선 previous_to_curernt)의 모션으로 prediction한 걸로 tracking수행
  // previous keypoint의 correpondace를 left에서 찾을때는 사각형 영역에서 찾음.
  // left keypoint의 correpondace를 right에서 찾을 때는 좌우 일정 거리 + 위아래 약간의 범위 에서 search
  _framepoint_generator->track(current_frame_, previous_frame_, previous_to_current_, _lost_points,
                               track_by_appearance_);

  // ds adjust bookkeeping
  // tracking 실패한 point의 갯수
  _number_of_lost_points = _lost_points.size();
  // tracking 성공하면 이전 point에서 landmark 정보를 가져옴.
  _number_of_tracked_landmarks = _framepoint_generator->numberOfTrackedLandmarks();
  _number_of_tracked_points = current_frame_->points().size();

  // 얼마만큼의 point가 tracking되었는지
  _tracking_ratio = static_cast<real>(_number_of_tracked_points) / previous_frame_->points().size();

  // ds if we're below the target - raise tracking window for next image
  if (_tracking_ratio < 0.1)
  {
    LOG_WARNING(std::cerr << "BaseTracker::_trackFramepoints|low point tracking ratio: " << _tracking_ratio << " ("
                          << _number_of_tracked_points << "/" << previous_frame_->points().size() << ")" << std::endl)

    // ds maximimze tracking range
    _projection_tracking_distance_pixels =
        _framepoint_generator->parameters()->maximum_projection_tracking_distance_pixels;

    // ds narrow tracking window
  }
  else
  {
    // ds if we still can reduce the tracking window size
    if (_projection_tracking_distance_pixels >
        _framepoint_generator->parameters()->minimum_projection_tracking_distance_pixels)
    {
      _projection_tracking_distance_pixels = std::max(
          _projection_tracking_distance_pixels * _parameters->tunnel_vision_ratio,  // tracking search 범위를 점점 줄임
          static_cast<real>(_framepoint_generator->parameters()->minimum_projection_tracking_distance_pixels));
    }
  }

  // ds stats
  // tracking ratio 들의 평균을 업데이트
  _mean_tracking_ratio =
      (_number_of_tracked_frames * _mean_tracking_ratio + _tracking_ratio) / (1 + _number_of_tracked_frames);
  ++_number_of_tracked_frames;

  // ds update frame with current points

  _total_number_of_landmarks += _number_of_tracked_landmarks;

  // ds VISUALIZATION ONLY
  current_frame_->setProjectionTrackingDistancePixels(_projection_tracking_distance_pixels);
}

void BaseTracker::_registerRecursive(Frame* frame_previous_, Frame* frame_current_,
                                     TransformMatrix3D previous_to_current_, const Count& recursion_)
{
  assert(_number_of_tracked_landmarks_previous != 0);

  // ds current number of tracked landmarks
  real relative_number_of_tracked_landmarks_to_previous =
      static_cast<real>(_number_of_tracked_landmarks) / _number_of_tracked_landmarks_previous;

  // ds if we got not enough tracks to evaluate for position tracking: robustness
  if (_number_of_tracked_landmarks == 0 || relative_number_of_tracked_landmarks_to_previous < 0.1)
  {
    // ds if we have recursions left (currently only two)
    if (recursion_ < 2)
    {
      // ds fallback to no motion model
      previous_to_current_ = TransformMatrix3D::Identity();

      // ds attempt tracking by appearance (maximum window size)
      _framepoint_generator->initialize(frame_current_, false);
      _track(frame_previous_, frame_current_, previous_to_current_, true);
      _registerRecursive(frame_previous_, frame_current_, previous_to_current_, recursion_ + 1);
      ++_number_of_recursive_registrations;
      return;
    }
    else
    {
      // ds failed
      breakTrack(frame_current_);
      return;
    }
  }

  // ds compute ratio between landmarks and tracked points: informative only
  const real percentage_landmarks = static_cast<real>(_number_of_tracked_landmarks) / _number_of_tracked_points;
  if (percentage_landmarks < 0.1)
  {
    LOG_WARNING(std::cerr << "BaseTracker::compute|low percentage of tracked landmarks over framepoints: "
                          << percentage_landmarks << " (landmarks/framepoints: " << _number_of_tracked_landmarks << "/"
                          << _number_of_tracked_points << ")" << std::endl)
  }

  // ds call pose solver
  CHRONOMETER_START(pose_optimization)
  _pose_optimizer->setEnableWeightsTranslation(true);
  _pose_optimizer->initialize(frame_previous_, frame_current_, previous_to_current_);
  _pose_optimizer->converge();
  CHRONOMETER_STOP(pose_optimization)

  // ds solver deltas
  const Count number_of_inliers = _pose_optimizer->numberOfInliers();
  const real average_error = _pose_optimizer->totalError() / number_of_inliers;

  // ds if we have enough inliers in the pose optimization
  if (number_of_inliers > _parameters->minimum_number_of_landmarks_to_track)
  {
    // ds info
    if (recursion_ > 0)
    {
      LOG_INFO(std::cerr << frame_current_->identifier() << "|BaseTracker::_registerRecursive|recursion: " << recursion_
                         << "|inliers: " << number_of_inliers << " average error: " << average_error << std::endl)
    }

    // ds setup
    _previous_to_current_camera = _pose_optimizer->previousToCurrent();
    const real delta_angular = WorldMap::toOrientationRodrigues(_previous_to_current_camera.linear()).norm();
    const real delta_translational = _previous_to_current_camera.translation().norm();

    // ds if the posit result is significant enough
    if (delta_angular > _parameters->minimum_delta_angular_for_movement ||
        delta_translational > _parameters->minimum_delta_translational_for_movement)
    {
      // ds update tracker - TODO purge this transform chaos
      const TransformMatrix3D world_to_camera_current =
          _previous_to_current_camera * frame_previous_->worldToCameraLeft();
      const TransformMatrix3D world_to_robot_current = _camera_left->cameraToRobot() * world_to_camera_current;
      frame_current_->setRobotToWorld(world_to_robot_current.inverse());
    }
    else
    {
      // ds keep previous solution
      frame_current_->setRobotToWorld(frame_previous_->robotToWorld());
      _previous_to_current_camera = TransformMatrix3D::Identity();
    }

    // ds visualization only (we need to clear and push_back in order to not crash the gui since its decoupled -
    // otherwise we could use resize)
    _context->currentlyTrackedLandmarks().reserve(_number_of_tracked_landmarks);

    // ds prune current frame points
    _prunePoints(frame_current_);
    assert(_context->currentlyTrackedLandmarks().size() <= _number_of_tracked_landmarks);
    assert(_number_of_tracked_points >= number_of_inliers);

    // ds recover lost points based on refined pose
    if (_parameters->enable_landmark_recovery)
    {
      CHRONOMETER_START(point_recovery)
      _recoverPoints(frame_current_);
      CHRONOMETER_STOP(point_recovery)
    }

    // ds update tracks
    _context->setRobotToWorld(frame_current_->robotToWorld());
  }
  else
  {
    LOG_WARNING(std::cerr << frame_current_->identifier()
                          << "|BaseTracker::_registerRecursive|recursion: " << recursion_
                          << "|inliers: " << number_of_inliers << " average error: " << average_error << std::endl)

    // ds if we have recursions left (currently only two)
    if (recursion_ < 2)
    {
      // ds if we still can increase the tracking window size
      if (_projection_tracking_distance_pixels <
          _framepoint_generator->parameters()->maximum_projection_tracking_distance_pixels)
      {
        ++_projection_tracking_distance_pixels;
      }

      // ds attempt new tracking with the increased window size
      _framepoint_generator->initialize(frame_current_, false);
      _track(frame_previous_, frame_current_, previous_to_current_);
      _registerRecursive(frame_previous_, frame_current_, previous_to_current_, recursion_ + 1);
      ++_number_of_recursive_registrations;
    }
    else
    {
      // ds failed
      breakTrack(frame_current_);
    }
  }
}

//! @breaks the track at the current frame
void BaseTracker::breakTrack(Frame* frame_)
{
  // ds reset state
  LOG_WARNING(std::printf("BaseTracker::breakTrack|LOST TRACK at frame [%06lu] with tracks: %lu\n",
                          frame_->identifier(), _number_of_tracked_points))
  _status_previous = Frame::Localizing;
  _status = Frame::Localizing;

  // ds stick to previous solution - simulating a fresh start
  frame_->setRobotToWorld(frame_->previous()->robotToWorld());
  _previous_to_current_camera = TransformMatrix3D::Identity();
  _number_of_recovered_points = 0;
  _number_of_tracked_points = 0;

  // ds reset frame in world context, triggering a restart of the pipeline
  _context->breakTrack(frame_);
}

// ds prunes invalid points after pose optimization
void BaseTracker::_prunePoints(Frame* frame_)
{
  // ds filter invalid points of pose optimization
  _number_of_tracked_points = 0;
  for (Index index_point = 0; index_point < frame_->points().size(); index_point++)
  {
    assert(frame_->points()[index_point]->previous());

    // ds keep points which were not suppressed in the optimization
    if (_pose_optimizer->errors()[index_point] != -1)
    {
      // ds if we have a new point keep it
      // if (!frame_->points()[index_point]->landmark()) {
      frame_->points()[_number_of_tracked_points] = frame_->points()[index_point];
      ++_number_of_tracked_points;

      // ds keep landmarks only if they were inliers
      //} else if (_pose_optimizer->inliers()[index_point]) {
      //  frame_->points()[_number_of_tracked_points] = frame_->points()[index_point];
      //  ++_number_of_tracked_points;
      //}
    }
  }
  //  std::cerr << "dropped points REGISTRATION: " << frame_->points().size()-_number_of_tracked_points << "/" <<
  //  frame_->points().size() << std::endl;
  frame_->points().resize(_number_of_tracked_points);
}

// ds updates existing or creates new landmarks for framepoints of the provided frame
void BaseTracker::_updatePoints(WorldMap* context_, Frame* frame_)
{
  CHRONOMETER_START(landmark_optimization)

  // ds buffer current pose
  const TransformMatrix3D& robot_to_world = frame_->robotToWorld();

  // ds start landmark generation/update
  _number_of_active_landmarks = 0;
  for (FramePoint* point : frame_->points())
  {
    point->setWorldCoordinates(robot_to_world * point->robotCoordinates());

    // ds skip point if tracking and not mature enough to be a landmark - for localizing state this is skipped
    if (point->trackLength() < _parameters->minimum_track_length_for_landmark_creation)
    {
      continue;
    }

    // ds check if the point is linked to a landmark
    Landmark* landmark = point->landmark();

    // ds if there's no landmark yet
    if (!landmark)
    {
      // ds create a landmark and associate it with the current framepoint
      landmark = context_->createLandmark(point);
    }

    // ds update landmark position based on current point (triggered as we linked the landmark to the point)
    landmark->update(point);
    point->setCameraCoordinatesLeftLandmark(frame_->worldToCameraLeft() * landmark->coordinates());
    ++_number_of_active_landmarks;

    // ds VISUALIZATION ONLY: add landmarks to currently visible ones
    landmark->setIsCurrentlyTracked(true);
    context_->currentlyTrackedLandmarks().push_back(landmark);
  }
  CHRONOMETER_STOP(landmark_optimization)
}
}
