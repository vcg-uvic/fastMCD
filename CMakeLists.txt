CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND CMAKE_POLICY)


SET (PROJECT_NAME "fastMCD")
project(${PROJECT_NAME})

find_package( PkgConfig )
find_package(OpenCV REQUIRED) # core imgproc highgui flann video calib3d nonfree objdetect)

set(HDRS )
file(GLOB "FILES" "src/*.cpp")
set( SRCS src/main.cpp ${FILES})

add_executable(${PROJECT_NAME} ${SRCS} )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBRARIES} )
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

include_directories(${PROJECT_SOURCE_DIR}/. ${PROJECT_SOURCE_DIR}/src)
# add OpenCV include directories
INCLUDE_DIRECTORIES(${OPENCV_INCLUDE_DIR})
