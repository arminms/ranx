## defining target for benchmarks
#
add_executable(benchmarks
  benchmarks_${RANX_TARGET_API}.$<IF:$<STREQUAL:${RANX_TARGET_API},cuda>,cu,cpp>
)

## defining link libraries for benchmarks
#
target_link_libraries(benchmarks PRIVATE
  ${PROJECT_NAME}::${RANX_TARGET_API}
  benchmark::benchmark
)
