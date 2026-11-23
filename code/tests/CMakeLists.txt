# Tests that must pass before the code is installed on the robot, run by
# scripts/pre-install-check.sh. Built with -DFOLLOW_TESTS=ON, run with ctest:
#
#   unit  the recognition, tracking and map logic on its own (deterministic, no models)
#   sim   the robot code (Control, FollowPerson, Explore, main) compiled unchanged
#         against fakes of the Unitree SDK and LiteRT, in a simulated world
#   cli   the programs run as a whole: arguments, exit codes, output
#
# Every doctest test case becomes its own ctest test (and process). They run with
# code/ as working directory, like the app, so ./res is found.

# the app is built as C++20 on the robot (LiteRT requires it), the tests match that
set(CMAKE_CXX_STANDARD 20)
# test names are sentences (spaces, colons, +)
if (POLICY CMP0110)
    cmake_policy(SET CMP0110 NEW)
endif ()

include(${CMAKE_CURRENT_SOURCE_DIR}/third_party/doctest/doctest.cmake)
find_package(Threads REQUIRED)

add_library(doctest INTERFACE)
target_include_directories(doctest INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# main() for every test executable plus the shared helpers in support/
add_library(test_main STATIC support/doctest_main.cpp)
target_link_libraries(test_main PUBLIC doctest Threads::Threads)
target_include_directories(test_main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/support ${PROJECT_SOURCE_DIR}/src)

# a doctest executable whose test cases are registered with ctest under a label
function(follow_test name label source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE test_main ${ARGN})
    doctest_discover_tests(${name}
            TEST_PREFIX "${name}: "
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            PROPERTIES LABELS ${label} TIMEOUT 120)
endfunction()

# --- unit tests, one executable per unit ---
follow_test(test_camera_geometry unit unit/test_camera_geometry.cpp recognition)
follow_test(test_appearance unit unit/test_appearance.cpp recognition)
follow_test(test_face_recognizer unit unit/test_face_recognizer.cpp recognition)
follow_test(test_obstacle_map unit unit/test_obstacle_map.cpp navigation)
follow_test(test_target_tracker unit unit/test_target_tracker.cpp recognition)

# --- the robot code in a simulated world, and the programs as a whole ---
add_subdirectory(sim)
