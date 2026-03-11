cmake_minimum_required(VERSION 3.16.3)

project(communicate)
add_executable(client client.c) 
add_executable(server server.c) 

target_link_libraries(client pthread)
target_link_libraries(server pthread)