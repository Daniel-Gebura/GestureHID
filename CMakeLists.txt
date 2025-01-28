cmake_minimum_required(VERSION 3.5.0)
project(GestureHID VERSION 0.1.0 LANGUAGES C CXX)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(GestureHID src/main.cpp)

target_link_libraries( GestureHID ${OpenCV_LIBS} )

include(CTest)
enable_testing()

