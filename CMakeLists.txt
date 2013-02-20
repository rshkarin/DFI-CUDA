cmake_minimum_required(VERSION 2.6)

project(DFI)
set(PROJECT dfi)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/common/cmake")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/")

# --- Options -----------------------------------------------------------------
option(WITH_PROFILING "Enable profiling" OFF)
if (WITH_PROFILING)
    add_definitions("-pg")
    set(CMAKE_C_FLAGS "-pg")
endif ()

# --- Find packages and libraries ---------------------------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB2 glib-2.0>=2.24 REQUIRED)
pkg_check_modules(GOBJECT2 gobject-2.0>=2.22 REQUIRED)

include_directories(
    ${GLIB2_INCLUDE_DIRS}
)

add_definitions(-std=c99 -Wall -fPIC -g3)
add_definitions(-DG_LOG_DOMAIN="DFI")
add_definitions(-DMULTIPLIER=2)

add_subdirectory(src)

