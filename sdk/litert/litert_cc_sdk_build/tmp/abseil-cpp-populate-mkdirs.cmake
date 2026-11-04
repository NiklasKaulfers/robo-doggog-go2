# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/ubuntu/proj/playground/sdk/litert/litert_cc_sdk_build/abseil-cpp")
  file(MAKE_DIRECTORY "/home/ubuntu/proj/playground/sdk/litert/litert_cc_sdk_build/abseil-cpp")
endif()
file(MAKE_DIRECTORY
  "/home/ubuntu/proj/playground/sdk/litert/litert_cc_sdk_build/_deps/abseil-cpp-build"
  "/home/ubuntu/proj/playground/sdk/litert/litert_cc_sdk_build"
  "/home/ubuntu/proj/playground/sdk/litert/litert_cc_sdk_build/tmp"
  "/home/ubuntu/proj/playground/sdk/litert/litert_cc_sdk_build/src/abseil-cpp-populate-stamp"
  "/home/ubuntu/proj/playground/sdk/litert/litert_cc_sdk_build/src"
  "/home/ubuntu/proj/playground/sdk/litert/litert_cc_sdk_build/src/abseil-cpp-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ubuntu/proj/playground/sdk/litert/litert_cc_sdk_build/src/abseil-cpp-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ubuntu/proj/playground/sdk/litert/litert_cc_sdk_build/src/abseil-cpp-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
