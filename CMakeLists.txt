cmake_minimum_required(VERSION 3.9)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(k3dom LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

set(CUDA_NVCC_FLAGS -g;-G)

option(BUILD_TESTS "Build unit tests" ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/")

find_package(OpenCV)
if (NOT OpenCV_FOUND)
	set(OpenCV_INCLUDE_DIRS "" CACHE PATH "" )
	set(OpenCV_LIBS "" CACHE FILEPATH "" )
endif()

add_definitions(-D_USE_MATH_DEFINES)  # Required to make M_PI from cmath available in MSVC

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  pcl_ros
  roscpp
  visualization_msgs
)

find_package(OpenGL REQUIRED)

find_package(GLM)
if (NOT GLM_FOUND)
	set(GLM_INCLUDE_DIR "" CACHE PATH "")
endif()

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   geometry_msgs#   nav_msgs#   visualization_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
 INCLUDE_DIRS include
#  LIBRARIES
#  CATKIN_DEPENDS geometry_msgs nav_msgs pcl_ros roscpp roscpp visualization_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

LIST (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# INCLUDE(GoogleTestDownloadAndBuild)

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(
# # include
#   ${catkin_INCLUDE_DIRS}
# )

include_directories(
	SYSTEM
	include
	${catkin_INCLUDE_DIRS}
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	${GLM_INCLUDE_DIR}
)

set(K3DOM_SRC_FILES
	src/k3dom/kernel/mass_update.cu
	src/k3dom/dom.cu
)

set(COMMON_SRC_FILES
	src/common/kernel/ego_motion_compensation.cu
	src/common/kernel/init.cu
	src/common/kernel/init_new_particles.cu
	src/common/kernel/particle_to_grid.cu
	src/common/kernel/predict.cu
	src/common/kernel/resampling.cu
	src/common/kernel/statistical_moments.cu
	src/common/kernel/update_persistent_particles.cu
	src/common/dom.cu)

set(DSPHDMIB_SRC_FILES
	src/dsphdmib/kernel/mass_update.cu
	src/dsphdmib/kernel/measurement_grid.cu
	src/dsphdmib/dom.cu
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/rfs3ddm.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/rfs3ddm_node.cpp)

add_executable(k3dom_node
    ${K3DOM_SRC_FILES}
	${COMMON_SRC_FILES}
    src/k3dom/k3dom_node.cpp
)

add_executable(dsphdmib_node
    ${DSPHDMIB_SRC_FILES}
	${COMMON_SRC_FILES}
    src/dsphdmib/dsphdmib_node.cpp
)

add_executable(tf2pose_node
	src/tf2pose_node.cpp
)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

# target_include_directories(k3dom PUBLIC include)
# target_include_directories(k3dom_node PUBLIC ${OpenCV_INCLUDE_DIRS} ${GLM_INCLUDE_DIR})
# target_include_directories(dsphdmib_node PUBLIC ${OpenCV_INCLUDE_DIRS} ${GLM_INCLUDE_DIR})


target_link_libraries(k3dom_node
	PRIVATE ${OPENGL_LIBRARY}
    ${catkin_LIBRARIES}
	${OpenCV_LIBS}
)

target_link_libraries(dsphdmib_node
	PRIVATE ${OPENGL_LIBRARY}
    ${catkin_LIBRARIES}
	${OpenCV_LIBS}
)

target_link_libraries(tf2pose_node
	${catkin_LIBRARIES}
)

# support for gpu lambdas
string(APPEND CMAKE_CUDA_FLAGS " --expt-extended-lambda")
# supress glm warnings
#string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
set(CUDA_PROPAGATE_HOST_FLAGS ON)

# set_target_properties(k3dom PROPERTIES PUBLIC_HEADER "include/dom.h;include/dom_types.h")

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#INSTALL(TARGETS dom
#		ARCHIVE DESTINATION lib/dom
#        LIBRARY DESTINATION lib/dom
#        PUBLIC_HEADER DESTINATION include/dom
#)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_rfs3ddm.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)

# if(BUILD_TESTS)
#     enable_testing()
#     add_subdirectory(test)
# endif()