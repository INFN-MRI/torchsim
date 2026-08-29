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

#endif  // TORCHSIM_IFUNC_HPP
