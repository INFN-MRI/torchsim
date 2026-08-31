#ifndef TORCHSIM_IFUNC_HPP
#define TORCHSIM_IFUNC_HPP

// ``__GLIBC__`` comes from a libc header rather than from the compiler, so one
// has to have been included before the test below can read it.
#include <cstdlib>

// Whether the loader on this target resolves an indirect function.
//
// That is how ``target_clones`` dispatches: the compiler emits one copy of a
// function per instruction set and the resolver picks the copy the processor
// can run. musl implements no ifunc and rejects the call outright, and it
// announces itself with no macro of its own -- so the test is for the glibc
// whose absence on Linux leaves only musl.
#if defined(__linux__) && !defined(__GLIBC__)
#define TORCHSIM_HAS_IFUNC 0
#else
#define TORCHSIM_HAS_IFUNC 1
#endif

// Whether the EPG kernels are compiled once per instruction set. It takes the
// GNU spelling of ``target_clones``, which clang does not implement and MSVC
// rejects, a loader that resolves an indirect function, and an x86 to have
// instruction sets to choose between.
//
// A build that multiversions runs a clone the code around it does not, and
// leaves a vector state behind that everything after it pays for until
// something clears it; see ``release_vector_state`` in ``_threads.hpp``. A
// build that does not has one code path and nothing to clear, so the module
// reports this and a caller measuring the difference knows whether there is
// one to measure.
#if defined(__GNUC__) && !defined(__clang__) && TORCHSIM_HAS_IFUNC \
    && (defined(__x86_64__) || defined(__i386__))
#define TORCHSIM_MULTIVERSIONED 1
#else
#define TORCHSIM_MULTIVERSIONED 0
#endif

#endif  // TORCHSIM_IFUNC_HPP
