# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.18.1)

# Declares and names the project.

project("opencvminist4android")

#定义变量opencvlibs使后面的命令可以使用定位具体的库文件
set(opencvlibs ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs)

#调用头文件的具体路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#增加OpenCV的动态库
add_library(libopencv_java4 SHARED IMPORTED)

#建立链接
set_target_properties(libopencv_java4 PROPERTIES IMPORTED_LOCATION
        "${opencvlibs}/${ANDROID_ABI}/libopencv_java4.so")


# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

file(GLOB native_srcs "*.cpp")

add_library( # Sets the name of the library.
        opencvminist4android

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        ${native_srcs})

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        opencvminist4android
        jnigraphics
        libopencv_java4

        # Links the target library to the log library
        # included in the NDK.
        ${log-lib})