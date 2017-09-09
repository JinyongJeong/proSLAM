#include "world_map.h"

#include <fstream>
#include <iomanip>

namespace proslam {
  using namespace srrg_core;

  WorldMap::WorldMap(const WorldMapParameters* parameters_): _parameters(parameters_) {
    LOG_DEBUG(std::cerr << "WorldMap::WorldMap|constructing" << std::endl)
    clear();
    LOG_DEBUG(std::cerr << "WorldMap::WorldMap|constructed" << std::endl)
  };

  WorldMap::~WorldMap() {
    LOG_DEBUG(std::cerr << "WorldMap::~WorldMap|destroying" << std::endl)
    clear();
    LOG_DEBUG(std::cerr << "WorldMap::~WorldMap|destroyed" << std::endl)
  }
  
  //ds clears all internal structures
  void WorldMap::clear() {

    //ds free landmarks
    for(LandmarkPointerMap::iterator it = _landmarks.begin(); it != _landmarks.end(); ++it) {
      delete it->second;
    }

    //ds free all frames
    for(FramePointerMap::iterator it = _frames.begin(); it != _frames.end(); ++it) {
      delete it->second;
    }

    //ds free all local maps
    for(const LocalMap* local_map: _local_maps) {
      delete local_map;
    }

    //ds clear containers
    _frame_queue_for_local_map.clear();
    _landmarks.clear();
    _frames.clear();
    _local_maps.clear();
    _currently_tracked_landmarks.clear();
  }

  Frame* WorldMap::createFrame(const TransformMatrix3D& robot_to_world_,
                               const real& maximum_depth_near_,
                               const double& timestamp_image_left_seconds_){

    //ds update current frame
    _previous_frame = _current_frame;
    _current_frame  = new Frame(this, _previous_frame, 0, robot_to_world_, maximum_depth_near_, timestamp_image_left_seconds_);

    //ds check if the frame has a predecessor
    if (_previous_frame) {
      _previous_frame->setNext(_current_frame);
    } else {

      //ds we have a new root frame
      _root_frame = _current_frame;
      _current_frame ->setRoot(_root_frame);
    }

    //ds bookkeeping
    _frames.insert(std::make_pair(_current_frame->identifier(), _current_frame));
    _frame_queue_for_local_map.push_back(_current_frame);

    //ds done
    return _current_frame;
  }

  Landmark* WorldMap::createLandmark(FramePoint* origin_) {
    Landmark* landmark = new Landmark(origin_, _parameters->landmark);
    _landmarks.insert(std::make_pair(landmark->identifier(), landmark));
    return landmark;
  }

  const bool WorldMap::createLocalMap(const bool& drop_framepoints_) {
    if (_previous_frame == 0) {
      return false;
    }

    //ds reset closure status
    _relocalized = false;

    //ds update distance traveled and last pose
    const TransformMatrix3D robot_pose_last_to_current = _previous_frame->worldToRobot()*_current_frame->robotToWorld();
    _distance_traveled_window += robot_pose_last_to_current.translation().norm();
    _degrees_rotated_window   += toOrientationRodrigues(robot_pose_last_to_current.linear()).norm();

    //ds check if we can generate a keyframe - if generated by translation only a minimum number of frames in the buffer is required - or a new tracking context
    if (_degrees_rotated_window   > _parameters->minimum_degrees_rotated_for_local_map ||
        (_distance_traveled_window > _parameters->minimum_distance_traveled_for_local_map && _frame_queue_for_local_map.size() > _parameters->minimum_number_of_frames_for_local_map)||
        (_frame_queue_for_local_map.size() > _parameters->minimum_number_of_frames_for_local_map && _local_maps.size() < 5)) {

      //ds create the new keyframe and add it to the keyframe database
      _current_local_map = new LocalMap(_frame_queue_for_local_map,
                                        _parameters->local_map,
                                        _root_local_map,
                                        _current_local_map);
      _local_maps.push_back(_current_local_map);
      assert(_current_frame->isKeyframe());
      assert(_current_frame->localMap() == _current_local_map);

      //ds set local map root
      if (!_root_local_map) {
        _root_local_map = _current_local_map;
        _root_local_map->setRoot(_current_local_map);
      }

      //ds reset generation properties
      resetWindowForLocalMapCreation(drop_framepoints_);

      //ds local map generated
      return true;
    } else {

      //ds no local map generated
      return false;
    }
  }

  //ds resets the window for the local map generation
  void WorldMap::resetWindowForLocalMapCreation(const bool& drop_framepoints_) {
    _distance_traveled_window = 0;
    _degrees_rotated_window   = 0;

    //ds free memory if desired (saves a lot of memory costs a little computation)
    if (drop_framepoints_) {

      //ds the last frame we'll need for the next tracking step
      _frame_queue_for_local_map.pop_back();

      //ds the pre-last frame is needed for visualization only (optical flow)
      _frame_queue_for_local_map.pop_back();

      //ds purge the rest
      for (Frame* frame: _frame_queue_for_local_map) {
        frame->clear();
      }
    }
    _frame_queue_for_local_map.clear();
  }

  void WorldMap::addCorrespondence(LocalMap* query_,
                                   const LocalMap* reference_,
                                   const TransformMatrix3D& query_to_reference_,
                                   const CorrespondencePointerVector& landmark_correspondences_,
                                   const real& information_) {

    //ds check if we relocalized after a lost track
    if (_frames.at(0)->root() != _current_frame->root()) {
      assert(_current_frame->localMap() == query_);

      //ds rudely link the current frame into the list (proper map merging will be coming soon!)
      setTrack(_current_frame);
    }

    //ds add loop closure information to the world map
    query_->addCorrespondence(reference_, query_to_reference_, landmark_correspondences_, information_);
    _relocalized = true;

    //ds if merging of corresponding landmarks is desired (TODO beta)
    if (_parameters->merge_landmarks) {

      //ds for all correspondences
      for (const LandmarkCorrespondence* landmark_correspondence: landmark_correspondences_) {

        //ds merge components
        Landmark* landmark_query     = landmark_correspondence->query->landmark;
        Landmark* landmark_reference = landmark_correspondence->reference->landmark;

        //ds merge query with reference (original) landmark
        landmark_reference->merge(landmark_query);

        //ds check if the landmark is in the currently tracked ones and update it accordingly
        for (uint64_t index = 0; index < _currently_tracked_landmarks.size(); ++index) {
          if (_currently_tracked_landmarks[index] == landmark_query) {
            _currently_tracked_landmarks[index] = landmark_reference;
          }
        }

        //ds free input landmark from bookkeeping
        _landmarks.erase(landmark_query->identifier());
        delete landmark_query;
      }
    }

    //ds informative only
    ++_number_of_closures;
  }

  void WorldMap::writeTrajectoryKITTI(const std::string& filename_) const {

    //ds construct filename
    std::string filename_kitti(filename_);

    //ds if not set
    if (filename_ == "") {

      //ds generate generic filename with timestamp
      filename_kitti = "trajectory_kitti-"+std::to_string(static_cast<uint64_t>(std::round(srrg_core::getTime())))+".txt";
    }

    //ds open file stream for kitti (overwriting)
    std::ofstream outfile_trajectory(filename_kitti, std::ifstream::out);
    assert(outfile_trajectory.good());
    outfile_trajectory << std::fixed;
    outfile_trajectory << std::setprecision(9);

    //ds for each frame (assuming continuous, sequential indexing)
    for (const FramePointerMapElement frame: _frames) {

      //ds buffer transform
      const TransformMatrix3D& robot_to_world = frame.second->robotToWorld();

      //ds dump transform according to KITTI format
      for (uint8_t u = 0; u < 3; ++u) {
        for (uint8_t v = 0; v < 4; ++v) {
          outfile_trajectory << robot_to_world(u,v) << " ";
        }
      }
      outfile_trajectory << "\n";
    }
    outfile_trajectory.close();
    LOG_INFO(std::cerr << "WorldMap::WorldMap|saved trajectory (KITTI format) to: " << filename_kitti << std::endl)
  }

  void WorldMap::writeTrajectoryTUM(const std::string& filename_) const {

    //ds construct filename
    std::string filename_tum(filename_);

    //ds if not set
    if (filename_ == "") {

      //ds generate generic filename with timestamp
      filename_tum = "trajectory_tum-"+std::to_string(static_cast<uint64_t>(std::round(srrg_core::getTime())))+".txt";
    }

    //ds open file stream for tum (overwriting)
    std::ofstream outfile_trajectory(filename_tum, std::ifstream::out);
    assert(outfile_trajectory.good());
    outfile_trajectory << std::fixed;
    outfile_trajectory << std::setprecision(9);

    //ds for each frame (assuming continuous, sequential indexing)
    for (const FramePointerMapElement frame: _frames) {

      //ds buffer transform
      const TransformMatrix3D& robot_to_world = frame.second->robotToWorld();
      const Quaternion orientation = Quaternion(robot_to_world.linear());

      //ds dump transform according to TUM format
      outfile_trajectory << frame.second->timestampImageLeftSeconds() << " ";
      outfile_trajectory << robot_to_world.translation().x() << " ";
      outfile_trajectory << robot_to_world.translation().y() << " ";
      outfile_trajectory << robot_to_world.translation().z() << " ";
      outfile_trajectory << orientation.x() << " ";
      outfile_trajectory << orientation.y() << " ";
      outfile_trajectory << orientation.z() << " ";
      outfile_trajectory << orientation.w() << " ";
      outfile_trajectory << "\n";
    }
    outfile_trajectory.close();
    LOG_INFO(std::cerr << "WorldMap::WorldMap|saved trajectory (TUM format) to: " << filename_tum << std::endl)
  }

  void WorldMap::breakTrack(const Frame* frame_) {

    //ds if the track is not already broken
    if (_last_frame_before_track_break == 0)
    {
      _last_frame_before_track_break     = _previous_frame;
      _last_local_map_before_track_break = _current_local_map;
    }

    //ds purge previous and set new root - this will trigger a new start
    _previous_frame = 0;
    _root_frame     = frame_;
    _root_local_map = 0;

    //ds reset current head
    _currently_tracked_landmarks.clear();
    resetWindowForLocalMapCreation();
    setRobotToWorld(frame_->robotToWorld());
  }

  void WorldMap::setTrack(Frame* frame_) {
    assert(frame_->localMap());
    assert(_last_local_map_before_track_break);
    LOG_INFO(std::printf("WorldMap::setTrack|RELOCALIZED - connecting [Frame] < [LocalMap]: [%06lu] < [%06lu] with [%06lu] < [%06lu]\n",
                _last_frame_before_track_break->identifier(), _last_local_map_before_track_break->identifier(), frame_->identifier(), frame_->localMap()->identifier()))

    //ds return to original roots
    _root_frame = _last_frame_before_track_break->root();
    frame_->setRoot(_root_frame);
    _root_local_map = _last_local_map_before_track_break->root();
    frame_->localMap()->setRoot(_root_local_map);

    //ds connect the given frame to the last one before the track broke and vice versa
    _last_frame_before_track_break->setNext(frame_);
    frame_->setPrevious(frame_);

    //ds connect local maps
    _last_local_map_before_track_break->setNext(frame_->localMap());
    frame_->localMap()->setPrevious(_last_local_map_before_track_break);

    _last_frame_before_track_break     = 0;
    _last_local_map_before_track_break = 0;
  }
}

