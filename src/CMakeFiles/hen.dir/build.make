# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/cmake/bin/cmake

# The command to remove a file.
RM = /usr/local/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/allensll/github/hen

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/allensll/github/hen

# Include any dependencies generated for this target.
include src/CMakeFiles/hen.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/hen.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/hen.dir/flags.make

src/CMakeFiles/hen.dir/cloud.cpp.o: src/CMakeFiles/hen.dir/flags.make
src/CMakeFiles/hen.dir/cloud.cpp.o: src/cloud.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/allensll/github/hen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/hen.dir/cloud.cpp.o"
	cd /home/allensll/github/hen/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hen.dir/cloud.cpp.o -c /home/allensll/github/hen/src/cloud.cpp

src/CMakeFiles/hen.dir/cloud.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hen.dir/cloud.cpp.i"
	cd /home/allensll/github/hen/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/allensll/github/hen/src/cloud.cpp > CMakeFiles/hen.dir/cloud.cpp.i

src/CMakeFiles/hen.dir/cloud.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hen.dir/cloud.cpp.s"
	cd /home/allensll/github/hen/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/allensll/github/hen/src/cloud.cpp -o CMakeFiles/hen.dir/cloud.cpp.s

src/CMakeFiles/hen.dir/dataset.cpp.o: src/CMakeFiles/hen.dir/flags.make
src/CMakeFiles/hen.dir/dataset.cpp.o: src/dataset.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/allensll/github/hen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/hen.dir/dataset.cpp.o"
	cd /home/allensll/github/hen/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hen.dir/dataset.cpp.o -c /home/allensll/github/hen/src/dataset.cpp

src/CMakeFiles/hen.dir/dataset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hen.dir/dataset.cpp.i"
	cd /home/allensll/github/hen/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/allensll/github/hen/src/dataset.cpp > CMakeFiles/hen.dir/dataset.cpp.i

src/CMakeFiles/hen.dir/dataset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hen.dir/dataset.cpp.s"
	cd /home/allensll/github/hen/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/allensll/github/hen/src/dataset.cpp -o CMakeFiles/hen.dir/dataset.cpp.s

src/CMakeFiles/hen.dir/hen.cpp.o: src/CMakeFiles/hen.dir/flags.make
src/CMakeFiles/hen.dir/hen.cpp.o: src/hen.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/allensll/github/hen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/hen.dir/hen.cpp.o"
	cd /home/allensll/github/hen/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hen.dir/hen.cpp.o -c /home/allensll/github/hen/src/hen.cpp

src/CMakeFiles/hen.dir/hen.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hen.dir/hen.cpp.i"
	cd /home/allensll/github/hen/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/allensll/github/hen/src/hen.cpp > CMakeFiles/hen.dir/hen.cpp.i

src/CMakeFiles/hen.dir/hen.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hen.dir/hen.cpp.s"
	cd /home/allensll/github/hen/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/allensll/github/hen/src/hen.cpp -o CMakeFiles/hen.dir/hen.cpp.s

src/CMakeFiles/hen.dir/models.cpp.o: src/CMakeFiles/hen.dir/flags.make
src/CMakeFiles/hen.dir/models.cpp.o: src/models.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/allensll/github/hen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/hen.dir/models.cpp.o"
	cd /home/allensll/github/hen/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hen.dir/models.cpp.o -c /home/allensll/github/hen/src/models.cpp

src/CMakeFiles/hen.dir/models.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hen.dir/models.cpp.i"
	cd /home/allensll/github/hen/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/allensll/github/hen/src/models.cpp > CMakeFiles/hen.dir/models.cpp.i

src/CMakeFiles/hen.dir/models.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hen.dir/models.cpp.s"
	cd /home/allensll/github/hen/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/allensll/github/hen/src/models.cpp -o CMakeFiles/hen.dir/models.cpp.s

src/CMakeFiles/hen.dir/nn.cpp.o: src/CMakeFiles/hen.dir/flags.make
src/CMakeFiles/hen.dir/nn.cpp.o: src/nn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/allensll/github/hen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/CMakeFiles/hen.dir/nn.cpp.o"
	cd /home/allensll/github/hen/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hen.dir/nn.cpp.o -c /home/allensll/github/hen/src/nn.cpp

src/CMakeFiles/hen.dir/nn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hen.dir/nn.cpp.i"
	cd /home/allensll/github/hen/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/allensll/github/hen/src/nn.cpp > CMakeFiles/hen.dir/nn.cpp.i

src/CMakeFiles/hen.dir/nn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hen.dir/nn.cpp.s"
	cd /home/allensll/github/hen/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/allensll/github/hen/src/nn.cpp -o CMakeFiles/hen.dir/nn.cpp.s

src/CMakeFiles/hen.dir/optim.cpp.o: src/CMakeFiles/hen.dir/flags.make
src/CMakeFiles/hen.dir/optim.cpp.o: src/optim.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/allensll/github/hen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/CMakeFiles/hen.dir/optim.cpp.o"
	cd /home/allensll/github/hen/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hen.dir/optim.cpp.o -c /home/allensll/github/hen/src/optim.cpp

src/CMakeFiles/hen.dir/optim.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hen.dir/optim.cpp.i"
	cd /home/allensll/github/hen/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/allensll/github/hen/src/optim.cpp > CMakeFiles/hen.dir/optim.cpp.i

src/CMakeFiles/hen.dir/optim.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hen.dir/optim.cpp.s"
	cd /home/allensll/github/hen/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/allensll/github/hen/src/optim.cpp -o CMakeFiles/hen.dir/optim.cpp.s

src/CMakeFiles/hen.dir/pack.cpp.o: src/CMakeFiles/hen.dir/flags.make
src/CMakeFiles/hen.dir/pack.cpp.o: src/pack.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/allensll/github/hen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object src/CMakeFiles/hen.dir/pack.cpp.o"
	cd /home/allensll/github/hen/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hen.dir/pack.cpp.o -c /home/allensll/github/hen/src/pack.cpp

src/CMakeFiles/hen.dir/pack.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hen.dir/pack.cpp.i"
	cd /home/allensll/github/hen/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/allensll/github/hen/src/pack.cpp > CMakeFiles/hen.dir/pack.cpp.i

src/CMakeFiles/hen.dir/pack.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hen.dir/pack.cpp.s"
	cd /home/allensll/github/hen/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/allensll/github/hen/src/pack.cpp -o CMakeFiles/hen.dir/pack.cpp.s

src/CMakeFiles/hen.dir/tensor.cpp.o: src/CMakeFiles/hen.dir/flags.make
src/CMakeFiles/hen.dir/tensor.cpp.o: src/tensor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/allensll/github/hen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object src/CMakeFiles/hen.dir/tensor.cpp.o"
	cd /home/allensll/github/hen/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hen.dir/tensor.cpp.o -c /home/allensll/github/hen/src/tensor.cpp

src/CMakeFiles/hen.dir/tensor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hen.dir/tensor.cpp.i"
	cd /home/allensll/github/hen/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/allensll/github/hen/src/tensor.cpp > CMakeFiles/hen.dir/tensor.cpp.i

src/CMakeFiles/hen.dir/tensor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hen.dir/tensor.cpp.s"
	cd /home/allensll/github/hen/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/allensll/github/hen/src/tensor.cpp -o CMakeFiles/hen.dir/tensor.cpp.s

src/CMakeFiles/hen.dir/user.cpp.o: src/CMakeFiles/hen.dir/flags.make
src/CMakeFiles/hen.dir/user.cpp.o: src/user.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/allensll/github/hen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object src/CMakeFiles/hen.dir/user.cpp.o"
	cd /home/allensll/github/hen/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hen.dir/user.cpp.o -c /home/allensll/github/hen/src/user.cpp

src/CMakeFiles/hen.dir/user.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hen.dir/user.cpp.i"
	cd /home/allensll/github/hen/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/allensll/github/hen/src/user.cpp > CMakeFiles/hen.dir/user.cpp.i

src/CMakeFiles/hen.dir/user.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hen.dir/user.cpp.s"
	cd /home/allensll/github/hen/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/allensll/github/hen/src/user.cpp -o CMakeFiles/hen.dir/user.cpp.s

src/CMakeFiles/hen.dir/utils.cpp.o: src/CMakeFiles/hen.dir/flags.make
src/CMakeFiles/hen.dir/utils.cpp.o: src/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/allensll/github/hen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object src/CMakeFiles/hen.dir/utils.cpp.o"
	cd /home/allensll/github/hen/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hen.dir/utils.cpp.o -c /home/allensll/github/hen/src/utils.cpp

src/CMakeFiles/hen.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hen.dir/utils.cpp.i"
	cd /home/allensll/github/hen/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/allensll/github/hen/src/utils.cpp > CMakeFiles/hen.dir/utils.cpp.i

src/CMakeFiles/hen.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hen.dir/utils.cpp.s"
	cd /home/allensll/github/hen/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/allensll/github/hen/src/utils.cpp -o CMakeFiles/hen.dir/utils.cpp.s

# Object files for target hen
hen_OBJECTS = \
"CMakeFiles/hen.dir/cloud.cpp.o" \
"CMakeFiles/hen.dir/dataset.cpp.o" \
"CMakeFiles/hen.dir/hen.cpp.o" \
"CMakeFiles/hen.dir/models.cpp.o" \
"CMakeFiles/hen.dir/nn.cpp.o" \
"CMakeFiles/hen.dir/optim.cpp.o" \
"CMakeFiles/hen.dir/pack.cpp.o" \
"CMakeFiles/hen.dir/tensor.cpp.o" \
"CMakeFiles/hen.dir/user.cpp.o" \
"CMakeFiles/hen.dir/utils.cpp.o"

# External object files for target hen
hen_EXTERNAL_OBJECTS =

lib/libhen.a: src/CMakeFiles/hen.dir/cloud.cpp.o
lib/libhen.a: src/CMakeFiles/hen.dir/dataset.cpp.o
lib/libhen.a: src/CMakeFiles/hen.dir/hen.cpp.o
lib/libhen.a: src/CMakeFiles/hen.dir/models.cpp.o
lib/libhen.a: src/CMakeFiles/hen.dir/nn.cpp.o
lib/libhen.a: src/CMakeFiles/hen.dir/optim.cpp.o
lib/libhen.a: src/CMakeFiles/hen.dir/pack.cpp.o
lib/libhen.a: src/CMakeFiles/hen.dir/tensor.cpp.o
lib/libhen.a: src/CMakeFiles/hen.dir/user.cpp.o
lib/libhen.a: src/CMakeFiles/hen.dir/utils.cpp.o
lib/libhen.a: src/CMakeFiles/hen.dir/build.make
lib/libhen.a: src/CMakeFiles/hen.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/allensll/github/hen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Linking CXX static library ../lib/libhen.a"
	cd /home/allensll/github/hen/src && $(CMAKE_COMMAND) -P CMakeFiles/hen.dir/cmake_clean_target.cmake
	cd /home/allensll/github/hen/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hen.dir/link.txt --verbose=$(VERBOSE)
	cd /home/allensll/github/hen/src && /usr/local/cmake/bin/cmake -E make_directory /home/allensll/github/hen/include
	cd /home/allensll/github/hen/src && /usr/local/cmake/bin/cmake -E copy cloud.h dataset.h hen.h models.h nn.h optim.h pack.h tensor.h user.h utils.h /home/allensll/github/hen/include

# Rule to build all files generated by this target.
src/CMakeFiles/hen.dir/build: lib/libhen.a

.PHONY : src/CMakeFiles/hen.dir/build

src/CMakeFiles/hen.dir/clean:
	cd /home/allensll/github/hen/src && $(CMAKE_COMMAND) -P CMakeFiles/hen.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/hen.dir/clean

src/CMakeFiles/hen.dir/depend:
	cd /home/allensll/github/hen && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/allensll/github/hen /home/allensll/github/hen/src /home/allensll/github/hen /home/allensll/github/hen/src /home/allensll/github/hen/src/CMakeFiles/hen.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/hen.dir/depend

