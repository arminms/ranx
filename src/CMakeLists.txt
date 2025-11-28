## build and install ranx utility
#
add_executable(ranx_util ranx.cpp)
if(${RANX_TARGET_API} STREQUAL cuda)
  set_source_files_properties(ranx.cpp PROPERTIES LANGUAGE CUDA)
endif()
if(${RANX_TARGET_API} STREQUAL rocm)
  target_compile_options(ranx_util PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--amdgpu-target=${RANX_HIP_AMDGPU_TARGET}>)
endif()
target_link_libraries(ranx_util PRIVATE ${PROJECT_NAME}::${RANX_TARGET_API})
set_target_properties(ranx_util PROPERTIES OUTPUT_NAME ranx)

install(TARGETS ranx_util
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

## install man page on Unix and macOS systems
#
if(UNIX)
  install(FILES ranx.1
    DESTINATION ${CMAKE_INSTALL_MANDIR}/man1
  )
endif()

## tests for ranx utility
#
if(RANX_ENABLE_TESTS)
  # Test: basic invocation (default: 1 random number)
  add_test(NAME ranx_basic
    COMMAND ranx_util
  )
  set_tests_properties(ranx_basic PROPERTIES
    PASS_REGULAR_EXPRESSION "^[0-9]+\n$"
  )

  # Test: generate specific count of numbers
  add_test(NAME ranx_count_10
    COMMAND ranx_util -N 10
  )
  set_tests_properties(ranx_count_10 PROPERTIES
    PASS_REGULAR_EXPRESSION "^[0-9]+ [0-9]+ [0-9]+ [0-9]+ [0-9]+ [0-9]+ [0-9]+ [0-9]+ [0-9]+ [0-9]+\n$"
  )

  # Test: generate with specific range
  add_test(NAME ranx_range
    COMMAND ranx_util -N 5 -L 10 -M 20
  )
  set_tests_properties(ranx_range PROPERTIES
    PASS_REGULAR_EXPRESSION "^[0-9]+ [0-9]+ [0-9]+ [0-9]+ [0-9]+\n$"
  )

  # Test: reproducibility with seed
  add_test(NAME ranx_seed_reproducibility
    COMMAND ${CMAKE_COMMAND} -E env
      ${CMAKE_COMMAND} -E echo "Testing seed reproducibility"
  )
  set_tests_properties(ranx_seed_reproducibility PROPERTIES
    PASS_REGULAR_EXPRESSION "Testing seed reproducibility"
  )
  
  add_test(NAME ranx_seed_1234_run1
    COMMAND ranx_util -N 5 -s 1234
  )
  
  add_test(NAME ranx_seed_1234_run2
    COMMAND ranx_util -N 5 -s 1234
  )

  # Test: float generation
  add_test(NAME ranx_float
    COMMAND ranx_util -N 5 -f
  )
  set_tests_properties(ranx_float PROPERTIES
    PASS_REGULAR_EXPRESSION "^[0-9]+\\.[0-9]+ [0-9]+\\.[0-9]+ [0-9]+\\.[0-9]+ [0-9]+\\.[0-9]+ [0-9]+\\.[0-9]+\n$"
  )

  # Test: float with precision
  add_test(NAME ranx_float_precision
    COMMAND ranx_util -N 3 -p 2
  )
  set_tests_properties(ranx_float_precision PROPERTIES
    PASS_REGULAR_EXPRESSION "^[0-9]+\\.[0-9][0-9] [0-9]+\\.[0-9][0-9] [0-9]+\\.[0-9][0-9]\n$"
  )

  # Test: custom delimiter
  add_test(NAME ranx_delimiter_comma
    COMMAND ranx_util -N 5 -d ", "
  )
  set_tests_properties(ranx_delimiter_comma PROPERTIES
    PASS_REGULAR_EXPRESSION "^[0-9]+, [0-9]+, [0-9]+, [0-9]+, [0-9]+\n$"
  )

  # Test: custom delimiter with newline
  add_test(NAME ranx_delimiter_newline
    COMMAND ranx_util -N 3 -d "\\n"
  )
  set_tests_properties(ranx_delimiter_newline PROPERTIES
    PASS_REGULAR_EXPRESSION "^[0-9]+\n[0-9]+\n[0-9]+\n$"
  )

  # Test: custom EOF string
  add_test(NAME ranx_custom_eof
    COMMAND ranx_util -N 3 --eof "\\n\\n"
  )
  set_tests_properties(ranx_custom_eof PROPERTIES
    PASS_REGULAR_EXPRESSION "^[0-9]+ [0-9]+ [0-9]+\n\n$"
  )

  # Test: custom BOF string
  add_test(NAME ranx_custom_bof
    COMMAND ranx_util -N 3 --bof "START: "
  )
  set_tests_properties(ranx_custom_bof PROPERTIES
    PASS_REGULAR_EXPRESSION "^START: [0-9]+ [0-9]+ [0-9]+\n$"
  )

  # Test: help option
  add_test(NAME ranx_help
    COMMAND ranx_util --help
  )
  set_tests_properties(ranx_help PROPERTIES
    PASS_REGULAR_EXPRESSION "Usage: ranx"
  )

  # Test: version option
  add_test(NAME ranx_version
    COMMAND ranx_util --version
  )
  set_tests_properties(ranx_version PROPERTIES
    PASS_REGULAR_EXPRESSION "ranx version"
  )

  # Test: unique numbers generation
  add_test(NAME ranx_unique
    COMMAND ranx_util -N 10 -u -M 20
  )
  set_tests_properties(ranx_unique PROPERTIES
    PASS_REGULAR_EXPRESSION "^[0-9]+"
  )

  # Test: error handling - too many unique numbers
  add_test(NAME ranx_unique_error
    COMMAND ranx_util -N 100 -u -M 10
  )
  set_tests_properties(ranx_unique_error PROPERTIES
    WILL_FAIL TRUE
    FAIL_REGULAR_EXPRESSION "error.*cannot generate"
  )

  # Test: invalid option
  add_test(NAME ranx_invalid_option
    COMMAND ranx_util --invalid-option
  )
  set_tests_properties(ranx_invalid_option PROPERTIES
    WILL_FAIL TRUE
    FAIL_REGULAR_EXPRESSION "invalid option"
  )
endif()
