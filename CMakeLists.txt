cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(PMRI)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(SimpleITK REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Boost COMPONENTS python310 REQUIRED)

SET(PYTHON_INCLUDE_DIRS "/usr/include/python3.10/" )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")




include_directories(${PROJECT_SOURCE_DIR}/headers ${PYTHON_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
# Get all the C++ files in the src/ directory
file(GLOB_RECURSE SOURCE_FILES source/*.cpp)

add_executable(PMRI ${SOURCE_FILES})
target_link_libraries(PMRI "${PROJECT_SOURCE_DIR}/headers" "${TORCH_LIBRARIES}" "${OpenCV_LIBRARIES}" "${SimpleITK_LIBRARIES}" "${ITK_LIBRARIES}" "${PYTHON_LIBRARIES}" "${Boost_LIBRARIES}")
target_link_libraries(PMRI Boost::python310)
target_link_libraries(PMRI ${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so)
set_property(TARGET PMRI PROPERTY CXX_STANDARD 17)
