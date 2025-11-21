## build and install ranx utility
#
add_executable(ranx_util ranx.cpp)
if(${RANX_TARGET_API} STREQUAL cuda)
  set_source_files_properties(ranx.cpp PROPERTIES LANGUAGE CUDA)
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
