## defining target for unit tests
#
add_executable(unit_tests
  unit_tests_${RANX_TARGET_API}.$<IF:$<STREQUAL:${RANX_TARGET_API},cuda>,cu,cpp>
)

## defining link libraries for unit tests
#
target_link_libraries(unit_tests PRIVATE
  ${PROJECT_NAME}::${RANX_TARGET_API}
  Catch2::Catch2WithMain
)

## finally adding unit tests
#
catch_discover_tests(unit_tests)
