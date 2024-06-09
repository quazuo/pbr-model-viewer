if(MSVC AND NOT "${MSVC_VERSION}" LESS 1400)
    add_definitions( "/MP" )
endif()

### GLFW ###

add_subdirectory(glfw)

include_directories(
    glfw/include/GLFW/
)
