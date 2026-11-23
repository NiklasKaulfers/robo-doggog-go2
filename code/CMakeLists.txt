cmake_minimum_required(VERSION 3.16)

project(follow)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# test builds only: build everything with these sanitizers, e.g. "address,undefined"
# (see scripts/pre-install-check.sh). Any finding aborts the test that triggered it
set(FOLLOW_SANITIZE "" CACHE STRING "sanitizers for test builds, e.g. address,undefined")
if (FOLLOW_SANITIZE)
    add_compile_options(-fsanitize=${FOLLOW_SANITIZE} -fno-omit-frame-pointer -fno-sanitize-recover=all)
    add_link_options(-fsanitize=${FOLLOW_SANITIZE})
endif ()

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

# times the route planning with synthetic height maps, needs neither the robot nor the models
add_executable(mapbench tools/mapbench.cpp)
target_link_libraries(mapbench navigation)

# unit, simulation and command line tests that must pass before the code is installed on
# the robot, run them with ctest or scripts/pre-install-check.sh. They need neither the
# robot, nor the Unitree SDK, nor LiteRT: the robot code is tested against fakes of those.
# Added before the app so the tests don't inherit its -fpermissive and LiteRT includes
option(FOLLOW_TESTS "build the tests (run them with ctest)" OFF)
if (FOLLOW_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

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
