# include libraries
include_directories (${aia_SOURCE_DIR}/utils)
include_directories (${aia_SOURCE_DIR}/3rdparty)

# find sources
file(GLOB project1_sources *.h *.hpp *.cpp)

# create executable from sources
add_executable(project1 ${project1_sources})

# link the executable to other modules / libraries 
target_link_libraries(project1 aiaUtils ucasUtils ${OpenCV_LIBS})
