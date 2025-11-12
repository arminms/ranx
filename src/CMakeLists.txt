add_executable(ranx_util ranx.cpp)
target_link_libraries(ranx_util PRIVATE ${PROJECT_NAME}::${RANX_TARGET_API})
set_target_properties(ranx_util PROPERTIES OUTPUT_NAME ranx)

install(TARGETS ranx_util
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
