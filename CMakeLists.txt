cmake_minimum_required(VERSION 3.26)

project(torchsim LANGUAGES CXX)

# Both kernels are plain CPython extensions against the stable ABI: they call
# no PyTorch API and link no PyTorch library, which is why one wheel per
# platform serves every supported interpreter and why that wheel is a couple of
# megabytes rather than the size of libtorch.
set(TORCHSIM_ABI3_VERSION 3.10)

find_package(Python 3.10 REQUIRED
    COMPONENTS Interpreter Development.Module Development.SABIModule
)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

function(torchsim_add_kernel name source)
    python_add_library(${name} MODULE
        USE_SABI ${TORCHSIM_ABI3_VERSION}
        WITH_SOABI
        "${source}"
    )
    # The shared headers sit beside the sources they are shared by.
    target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/torchsim")
    target_link_libraries(${name} PRIVATE Threads::Threads)

    # Nothing the C++ runtime owns crosses the module boundary: the kernel
    # takes raw buffers from the interpreter and hands back objects the
    # interpreter allocated. So the runtime goes inside the extension, and a
    # wheel carries no library beside it and needs no redistributable
    # installed to import.
    if(MSVC)
        set_property(TARGET ${name} PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
        )
    endif()

    # The mask a host kernel reads decides which terms it evaluates, and the
    # tests hold a gated launch to a whole one bit for bit. Contraction is free
    # to fuse a multiply and an add on one side of that and not the other, which
    # moves the last place of a result that has to be identical. MSVC does not
    # contract across statements under its default /fp:precise.
    if(MSVC)
        target_compile_options(${name} PRIVATE /O2)
    else()
        target_compile_options(${name} PRIVATE -O3 -ffp-contract=off)
    endif()

    install(TARGETS ${name} DESTINATION torchsim)
endfunction()

torchsim_add_kernel(_epg_cpu  src/torchsim/_epg_cpu.cpp)
torchsim_add_kernel(_perk_cpu src/torchsim/_perk_cpu.cpp)
