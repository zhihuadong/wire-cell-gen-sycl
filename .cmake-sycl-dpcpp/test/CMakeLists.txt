set(all_execs
  test_syclarray.cpp
)

set(source_dir ${PROJECT_SOURCE_DIR}/../test)

foreach(source_file IN LISTS all_execs)
  get_filename_component(exec_name ${source_file} NAME_WE)
  add_executable(${exec_name} ${source_dir}/${source_file})
  target_link_directories(${exec_name} PRIVATE $ENV{WIRECELL_LIB})
  target_link_libraries(${exec_name} PRIVATE PkgConfig::FFTW WireCellGenSycl)
  target_include_directories(${exec_name}
    PRIVATE
      ${PROJECT_SOURCE_DIR}/../inc
      $ENV{WIRECELL_INC}
  )
#set_target_properties(${exec_name}
#        PROPERTIES COMPILE_OPTIONS "-DSYCL_TARGET_HIP")

endforeach()
