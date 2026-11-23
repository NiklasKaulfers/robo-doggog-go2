# The robot code against a simulated robot (label sim) and the app as a whole (label cli).
# Control, ObstacleMapSubscriber, FollowPerson, Explore, the MoveNet wrappers and main()
# are compiled unchanged; only the Unitree SDK and LiteRT underneath them are fakes
# (fake_sdk/, fake_litert/), and SimWorld plays the robot and its surroundings.

# the fakes of the vendor SDKs and the simulated world
add_library(sim_fakes STATIC
        fake_sdk/FakeSdk.cpp
        fake_litert/FakeLitert.cpp
        SimWorld.cpp)
target_include_directories(sim_fakes PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/fake_sdk
        ${CMAKE_CURRENT_SOURCE_DIR}/fake_litert
        ${CMAKE_CURRENT_SOURCE_DIR}/../support
        ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(sim_fakes PUBLIC recognition ${OpenCV_LIBS} Threads::Threads)

# the robot code itself
add_library(sim_app STATIC
        ${PROJECT_SOURCE_DIR}/src/core/Control.cpp
        ${PROJECT_SOURCE_DIR}/src/core/ObstacleMapSubscriber.cpp
        ${PROJECT_SOURCE_DIR}/src/tricks/FollowPerson.cpp
        ${PROJECT_SOURCE_DIR}/src/tricks/Explore.cpp
        ${PROJECT_SOURCE_DIR}/src/ml/movenet/Movenet.cpp
        ${PROJECT_SOURCE_DIR}/src/ml/movenet/Multipose.cpp)
target_link_libraries(sim_app PUBLIC sim_fakes recognition navigation)

# scenarios in one process each, in real time
follow_test(test_control sim test_control.cpp sim_app)
follow_test(test_safety sim test_safety.cpp sim_app)
follow_test(test_follow sim test_follow.cpp sim_app)
# scenarios the code doesn't handle yet: not part of the gate, `ctest -L known` runs them
follow_test(test_known_issues known test_known_issues.cpp sim_app)

# main() with the fakes; FOLLOW_SIM_SCENARIO picks what the simulated robot sees
add_executable(app_sim ${PROJECT_SOURCE_DIR}/src/main.cpp AppSimScenarios.cpp)
target_link_libraries(app_sim PRIVATE sim_app)

# --- command line tests: whole programs, checked by exit code and output ---
set(EXPECT ${CMAKE_CURRENT_SOURCE_DIR}/../cli/expect.cmake)

# runs program with args (separated by |) and checks its exit code and output
function(cli_test name program args exitCode)
    cmake_parse_arguments(CLI "" "SCENARIO" "OUTPUT;FORBID" ${ARGN})
    add_test(NAME "cli: ${name}"
            COMMAND ${CMAKE_COMMAND}
            "-DPROGRAM=$<TARGET_FILE:${program}>" "-DARGS=${args}" "-DEXIT_CODE=${exitCode}"
            "-DOUTPUT=${CLI_OUTPUT}" "-DFORBID=${CLI_FORBID}"
            -P ${EXPECT}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    set(environment "")
    if (CLI_SCENARIO)
        set(environment "FOLLOW_SIM_SCENARIO=${CLI_SCENARIO}")
    endif ()
    set_tests_properties("cli: ${name}" PROPERTIES LABELS cli TIMEOUT 120 ENVIRONMENT "${environment}")
endfunction()

cli_test("app prints its usage with --help" app_sim "--help" 0 OUTPUT "usage:")
cli_test("app rejects an unknown argument" app_sim "--bogus" 1 OUTPUT "unknown argument --bogus")
cli_test("app rejects --target without a folder" app_sim "--target" 1 OUTPUT "unknown argument --target")
cli_test("app rejects a photos folder that does not exist" app_sim "/nonexistent/photos" 1
        OUTPUT "photos folder /nonexistent/photos does not exist")
cli_test("app stands up and follows and hands control back on Ctrl+C" app_sim "--no-target" 0
        SCENARIO follow
        OUTPUT "no photos given, following the first person seen" "\\[Tracker\\] found the person"
               "\\[sim\\] moved toward the person: yes" "\\[sim\\] control handed back to the remote: yes"
               "\\[sim\\] SportClient calls besides standing up: none" "\\[sim\\] collision: none"
        FORBID "error in standup")
cli_test("app exits when the robot does not stand up" app_sim "--no-target" 1
        SCENARIO standup_fails
        OUTPUT "error in standup" "\\[sim\\] movement commands: 0")

cli_test("facecheck prints its usage without arguments" facecheck "" 1 OUTPUT "usage: facecheck")
cli_test("facecheck fails without the face models" facecheck "/nonexistent/photos|--res|/nonexistent/res" 1
        OUTPUT "unable to load the face models")
cli_test("facecheck fails on a photos folder that does not exist" facecheck "/nonexistent/photos" 1
        OUTPUT "no usable face in /nonexistent/photos")
