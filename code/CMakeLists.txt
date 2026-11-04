cmake_minimum_required(VERSION 3.16)

project(follow)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_LIST_DIR}/../sdk/install/)
find_package(unitree_sdk2 REQUIRED CONFIG)

set(LITERT_INSTALL_LOCATION ${CMAKE_SOURCE_DIR}/../sdk/litert/)

add_subdirectory(${LITERT_INSTALL_LOCATION}/litert_cc_sdk ${LITERT_INSTALL_LOCATION}/litert_cc_sdk_build)
include_directories(${LITERT_INSTALL_LOCATION}/litert_cc_sdk)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# needed or else litert_cc_api will fail
add_compile_options(-fpermissive)

add_executable( app
        src/main.cpp
        src/core/Control.cpp
        src/core/Control.h
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
)
target_link_libraries(app unitree_sdk2 litert_cc_api ${OpenCV_LIBS})
