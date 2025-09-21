cmake_minimum_required(VERSION 3.21...3.29)



set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
project(Grafter LANGUAGES CXX CUDA)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native -flto")
endif()


set(CMAKE_CUDA_STANDARD 20)

if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    enable_language("RC")
    set (WIN32_RESOURCES "${CMAKE_CURRENT_SOURCE_DIR}/assets/grafter.rc")
    if (NOT EXISTS ${WIN32_RESOURCES})
        message(FATAL_ERROR "Windows resource file not found: ${WIN32_RESOURCES}")
    else()
        message(STATUS "Adding Windows resources: ${WIN32_RESOURCES}")
    endif()
endif()


include(FetchContent)

FetchContent_Declare(
        webview
        GIT_REPOSITORY https://github.com/webview/webview
        GIT_TAG 0.12.0)
FetchContent_MakeAvailable(webview)



include_directories(calculate)
include_directories(calculate/commands)
include_directories(calculate/commands/real)
include_directories(calculate/commands/complex)
include_directories(calculate/commands/draw)
include_directories(images)
include_directories(utils)
include_directories(utils/parsers)
include_directories(server)
include_directories(utils/device)
add_executable(Grafter ${WIN32_RESOURCES} kernel.cu
        calculate/commands/command.h
        calculate/scaler.h
        calculate/commands/real/add.h
        calculate/commands/real/sub.h
        calculate/commands/real/mul.h
        calculate/commands/__init.h
        images/draw.h
        calculate/commands/const.h
        utils/utils.h
        calculate/commands/comp.h
        calculate/commands/real/pow.h
        calculate/commands/real/intf.h
        utils/formats.h
        calculate/commands/real/trigonometric.h
        calculate/commands/real/div.h
        calculate/commands/real/diff.h
        calculate/commands/real/log.h
        calculate/commands/real/abs.h
        calculate/commands/complex/utils.h
        calculate/commands/complex/visual.h
        calculate/commands/draw/explicit.h
        calculate/commands/draw/rgb2D.h
        calculate/graph.h
        utils/parsers/cgc.h
        utils/parsers/formulaParser.h
        calculate/commands/draw/implicit_polar.h
        utils/device/cpu.h
        env.h
        server/server.h
        server/httplib.h
        utils/device/device.h
        utils/assets.h
        calculate/commands/draw/implicit.h
        utils/parsers/jsonParser.h
        gui/wingui.h
        main.cpp
)

set_target_properties(Grafter PROPERTIES OUTPUT_NAME "grafter")


find_package(cxxopts CONFIG REQUIRED)
target_link_libraries(Grafter PRIVATE cxxopts::cxxopts)

target_link_libraries(Grafter PRIVATE webview::core)

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})
target_link_libraries(Grafter PRIVATE ${OpenCV_LIBS})
#target_include_directories(Grafter PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(Grafter PRIVATE webview::core_static)


find_package(SymEngine CONFIG REQUIRED)
target_link_libraries(Grafter PRIVATE symengine)

set_target_properties(Grafter PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
add_custom_target(copy_assets ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/assets
        ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Copying assets folder..."
)
add_dependencies(Grafter copy_assets)