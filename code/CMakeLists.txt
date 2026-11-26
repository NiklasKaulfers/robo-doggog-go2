cmake_minimum_required(VERSION 3.16)

project(follow)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Threads REQUIRED)

# SQLite for the log store: the system's if it has one (macOS does), else the official
# amalgamation, downloaded at configure time and built here, so nothing has to be
# installed on the Jetson host
find_package(SQLite3 QUIET)
if (SQLite3_FOUND)
    add_library(robo_sqlite3 INTERFACE)
    target_include_directories(robo_sqlite3 INTERFACE ${SQLite3_INCLUDE_DIRS})
    target_link_libraries(robo_sqlite3 INTERFACE ${SQLite3_LIBRARIES})
else ()
    include(FetchContent)
    FetchContent_Declare(sqlite_amalgamation
            URL https://www.sqlite.org/2024/sqlite-amalgamation-3460100.zip
            URL_HASH SHA256=77823cb110929c2bcb0f5d48e4833b5c59a8a6e40cdea3936b99e199dbbe5784
    )
    # the zip has no CMakeLists.txt, so this only downloads and unpacks it
    FetchContent_MakeAvailable(sqlite_amalgamation)
    add_library(robo_sqlite3 STATIC ${sqlite_amalgamation_SOURCE_DIR}/sqlite3.c)
    target_include_directories(robo_sqlite3 SYSTEM PUBLIC ${sqlite_amalgamation_SOURCE_DIR})
    target_compile_definitions(robo_sqlite3 PRIVATE SQLITE_THREADSAFE=1 SQLITE_OMIT_LOAD_EXTENSION SQLITE_DQS=0)
    target_link_libraries(robo_sqlite3 PUBLIC Threads::Threads)
endif ()

# structured log: records saved in a local SQLite outbox until uploaded, plain C++ (no
# robot), shared by the app, the libraries below and the offline tools
add_library(logging STATIC
        src/logging/LogRecord.cpp
        src/logging/LogRecord.h
        src/logging/LogStore.cpp
        src/logging/LogStore.h
        src/logging/Logger.cpp
        src/logging/Logger.h
        src/logging/Uploader.cpp
        src/logging/Uploader.h
)
target_link_libraries(logging PRIVATE robo_sqlite3 PUBLIC Threads::Threads)

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
target_link_libraries(recognition logging ${OpenCV_LIBS})

# terrain memory and route planning, plain C++ (the DDS subscription is in the app), so
# it is shared by the app and the offline tools as well
add_library(navigation STATIC
        src/core/ObstacleMap.cpp
        src/core/ObstacleMap.h
        src/core/ExplorePlanner.cpp
        src/core/ExplorePlanner.h
)
target_link_libraries(navigation logging)

# following one person and searching for them when lost, on top of recognition and
# navigation; needs no robot either, so it is shared by the app and the offline tools
add_library(following STATIC
        src/follow/FollowPlanner.cpp
        src/follow/FollowPlanner.h
)
target_link_libraries(following recognition navigation ${OpenCV_LIBS})

# self-test of the logging library (log store, logger, uploading), needs nothing
add_executable(logtest tools/logtest.cpp)
target_link_libraries(logtest logging)

# checks the reference photos of the person to follow, runs without the robot
add_executable(facecheck tools/facecheck.cpp)
target_link_libraries(facecheck recognition ${OpenCV_LIBS})

# self-test of the tracking logic with drawn people, runs without the robot
add_executable(trackertest tools/trackertest.cpp)
target_link_libraries(trackertest recognition ${OpenCV_LIBS})

# self-test of the clothes model (body regions, color signatures, learned views) with
# drawn people, needs neither the robot nor the models
add_executable(appearancetest tools/appearancetest.cpp)
target_link_libraries(appearancetest recognition ${OpenCV_LIBS})

# self-test of the obstacle map and route planning with synthetic height maps, needs
# neither the robot nor the models
add_executable(maptest tools/maptest.cpp)
target_link_libraries(maptest navigation)

# self-test of the exploring logic with a simulated robot on synthetic terrain, needs
# neither the robot nor the models
add_executable(exploretest tools/exploretest.cpp)
target_link_libraries(exploretest navigation)

# self-test of following and searching with a simulated robot, simulated people and
# synthetic terrain, needs only the face models in ./res
add_executable(followtest tools/followtest.cpp)
target_link_libraries(followtest following ${OpenCV_LIBS})

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
    target_link_libraries(app following recognition navigation logging unitree_sdk2 litert_cc_api ${OpenCV_LIBS})
endif ()
