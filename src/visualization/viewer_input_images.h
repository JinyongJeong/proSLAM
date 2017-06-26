#pragma once
#include <mutex>
#include "types/world_map.h"

namespace proslam {

class ViewerInputImages {

public:
  static struct KeyStroke {

#if CV_MAJOR_VERSION == 2

    constexpr static int Escape      = 1048603;
    constexpr static int NumpadPlus  = 1114027;
    constexpr static int NumpadMinus = 1114029;
    constexpr static int Space       = 1048608;
    constexpr static int Backspace   = 1113864;
    constexpr static int Num1 = 1048625;
    constexpr static int Num2 = 1048626;
    constexpr static int Num3 = 1048627;
#elif CV_MAJOR_VERSION == 3

    constexpr static int Escape    = 27;
    constexpr static int Space     = 32;
    constexpr static int Backspace = 8;
    constexpr static int Num1 = 1048625;
    constexpr static int Num2 = 1048626;
    constexpr static int Num3 = 1048627;
#else
#error OpenCV version not supported
#endif

  } KeyStroke;

public:

  ViewerInputImages(const std::string& window_name_ = "input: images");
  ~ViewerInputImages();

public:

  void update(const Frame* frame_);
  void drawFeatures();
  void drawFeatureTracking();
  const bool updateGUI();
  void switchMode();

protected:

  //ds DEPRECATED
  Count _cv_wait_key_timeout_milliseconds;

  //! @brief viewer window title
  const std::string _window_name;

  //! @brief mutex for data exchange, owned by the viewer
  std::mutex _mutex_data_exchange;

  //! @brief active framepoint vector copy from tracker (updated with update method)
  std::vector<const FramePoint*> _active_framepoints;

  //! @brief currently displayed image
  cv::Mat _current_image;
};
}
