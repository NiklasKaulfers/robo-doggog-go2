cmake_minimum_required(VERSION 3.16)

project(follow)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# recognition code that only needs OpenCV, shared by the app and the offline tools
add_library(recognition STATIC
        src/ml/PersonDetection.h
        src/ml/FaceRecognizer.cpp
        src/ml/FaceRecognizer.h
        src/ml/Appearance.cpp
        src/ml/Appearance.h
        src/tracking/CameraGeometry.h
        src/tracking/TargetTracker.cpp
        src/tracking/TargetTracker.h
)
target_link_libraries(recognition ${OpenCV_LIBS})

# terrain memory and route planning, plain C++ (the DDS subscription is in the app), so
# it is shared by the app and the offline tools as well
add_library(navigation STATIC
        src/core/ObstacleMap.cpp
        src/core/ObstacleMap.h
)

# checks the reference photos of the person to follow, runs without the robot
add_executable(facecheck tools/facecheck.cpp)
target_link_libraries(facecheck recognition ${OpenCV_LIBS})

# self-test of the tracking logic with drawn people, runs without the robot
add_executable(trackertest tools/trackertest.cpp)
target_link_libraries(trackertest recognition ${OpenCV_LIBS})

# self-test of the obstacle map and route planning with synthetic height maps, needs
# neither the robot nor the models
add_executable(maptest tools/maptest.cpp)
target_link_libraries(maptest navigation)

# the robot app needs the Unitree SDK and LiteRT, which only exist on the Go2's Jetson.
# Set FOLLOW_TOOLS_ONLY=ON to build just the offline tools elsewhere
option(FOLLOW_TOOLS_ONLY "build only the offline tools, not the robot app" OFF)
if (NOT FOLLOW_TOOLS_ONLY)
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_LIST_DIR}/../sdk/install/)
    find_package(unitree_sdk2 REQUIRED CONFIG)

    set(LITERT_INSTALL_LOCATION ${CMAKE_SOURCE_DIR}/../sdk/litert/)

    add_subdirectory(${LITERT_INSTALL_LOCATION}/litert_cc_sdk ${LITERT_INSTALL_LOCATION}/litert_cc_sdk_build)
    include_directories(${LITERT_INSTALL_LOCATION}/litert_cc_sdk)

    # needed or else litert_cc_api will fail
    add_compile_options(-fpermissive)

    add_executable( app
            src/main.cpp
            src/core/Control.cpp
            src/core/Control.h
            src/core/ObstacleMapSubscriber.cpp
            src/core/ObstacleMapSubscriber.h
            src/ml/movenet/Movenet.cpp
            src/ml/movenet/Movenet.h
            src/ml/movenet/Multipose.cpp
            src/ml/movenet/Multipose.h
            src/ml/movenet/Singlepose.cpp
            src/ml/movenet/Singlepose.h
            src/tricks/TurnToPerson.cpp
            src/tricks/TurnToPerson.h
            src/tricks/SitStandApproach.cpp
            src/tricks/SitStandApproach.h
            src/tricks/FollowPerson.cpp
            src/tricks/FollowPerson.h
            src/tricks/Explore.cpp
            src/tricks/Explore.h
    )
    target_link_libraries(app recognition navigation unitree_sdk2 litert_cc_api ${OpenCV_LIBS})
endif ()
