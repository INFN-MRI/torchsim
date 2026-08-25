#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "_threads.hpp"

// The PERK feature map and its regression, fused.
//
//   y[n] = parameter_mean + (scale * cos(W x[n] + b) - feature_mean) @ weight^T
//
// Written as separate operations that builds the whole (voxels x features)
// matrix and moves it through memory twice. Here a block of features is formed
// and consumed into the output accumulator while it is still in cache, so the
// matrix never exists.
//
// The cosine is the reason this is not simply a call to a BLAS: measured on
// this expression it costs more than the matrix product feeding it, because
// libm's scalar cosf is what a compiler emits and it does not vectorize. The
// polynomial below is written so that -O3 can, which is the whole difference.

// A wheel has to run on whatever the machine is, so the build cannot assume
// AVX2 -- and a baseline build is half the speed of the vendor BLAS this
// replaces, where an AVX2 one is better than half again as fast as it.
// ``target_clones`` resolves that at load time: the compiler emits one copy of
// the function per instruction set and the dynamic linker picks the copy this
// processor can run.
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define TORCHSIM_VECTORIZED __attribute__((target_clones("avx2", "default")))
#else
#define TORCHSIM_VECTORIZED
#endif

namespace {

using torchsim::WorkerPool;
using torchsim::worker_count;

// Voxels per block, and features per block. The feature block is what has to
// stay in cache across the contrast loop.
constexpr std::int64_t VOXEL_BLOCK = 64;
constexpr std::int64_t FEATURE_BLOCK = 128;

constexpr float TWO_PI = 6.28318530717958647692f;
constexpr float INV_TWO_PI = 0.15915494309189533577f;
constexpr float PI = 3.14159265358979323846f;
constexpr float HALF_PI = 1.57079632679489661923f;

// cos on [-pi, pi] as a degree-10 even polynomial in x^2, from the minimax
// coefficients of the Taylor series. Straight-line code with no branches and
// no calls, so a vectorizing compiler can put eight of them in a register.
inline float fast_cos(float x) {
    // Fold to [-pi, pi], then reflect about +-pi/2 so the polynomial only has
    // to cover a quadrant. Both steps are arithmetic rather than branches,
    // because a branch here would stop the loop around it vectorizing and the
    // vectorizing is the entire point.
    const float turns = std::nearbyint(x * INV_TWO_PI);
    const float folded = x - turns * TWO_PI;
    const float over = folded > HALF_PI ? 1.0f : 0.0f;
    const float under = folded < -HALF_PI ? 1.0f : 0.0f;
    const float reduced = folded + over * (PI - 2.0f * folded)
        + under * (-PI - 2.0f * folded);
    const float sign = 1.0f - 2.0f * (over + under);
    const float square = reduced * reduced;
    float value = -2.605e-07f;
    value = value * square + 2.4760e-05f;
    value = value * square - 1.3888387e-03f;
    value = value * square + 4.16666418e-02f;
    value = value * square - 4.999999963e-01f;
    value = value * square + 1.0f;
    return sign * value;
}

inline float fast_sin(float x) { return fast_cos(x - HALF_PI); }

struct Problem {
    const float* signals;
    const float* frequency;
    const float* frequency_t;
    const float* phase;
    const float* feature_mean;
    const float* weight;
    const float* parameter_mean;
    const float* cotangent;
    float* output;
    std::int64_t voxels;
    std::int64_t contrasts;
    std::int64_t features;
    std::int64_t parameters;
    float scale;
};

// The angle of every feature for a block of voxels, into ``tile``.
//
// The loop order is what makes this worth writing: with the contrast outermost
// and the feature innermost, the inner statement is an AXPY along contiguous
// memory that a compiler vectorizes, where the natural order -- a dot product
// per feature -- is a reduction it cannot.
TORCHSIM_VECTORIZED inline void angles(
    const Problem& problem,
    const std::int64_t begin,
    const std::int64_t width,
    const std::int64_t first,
    const std::int64_t span,
    float* const tile
) {
    for (std::int64_t row = 0; row < width; ++row) {
        float* const target = tile + row * span;
        for (std::int64_t feature = 0; feature < span; ++feature) {
            target[feature] = problem.phase[first + feature];
        }
    }
    for (std::int64_t contrast = 0; contrast < problem.contrasts; ++contrast) {
        const float* const row_of_weights =
            problem.frequency_t + contrast * problem.features + first;
        for (std::int64_t row = 0; row < width; ++row) {
            const float value =
                problem.signals[(begin + row) * problem.contrasts + contrast];
            float* const target = tile + row * span;
            for (std::int64_t feature = 0; feature < span; ++feature) {
                target[feature] += value * row_of_weights[feature];
            }
        }
    }
}

// One block of voxels, from signal to parameters.
TORCHSIM_VECTORIZED void regress_block(
    const Problem& problem,
    const std::int64_t begin,
    const std::int64_t end,
    std::vector<float>& mapped
) {
    const std::int64_t width = end - begin;
    const std::int64_t parameters = problem.parameters;
    std::vector<float> total(
        static_cast<std::size_t>(width * parameters), 0.0f
    );

    for (std::int64_t first = 0; first < problem.features;
         first += FEATURE_BLOCK) {
        const std::int64_t last =
            std::min(first + FEATURE_BLOCK, problem.features);
        const std::int64_t span = last - first;
        float* const tile = mapped.data();
        angles(problem, begin, width, first, span, tile);

        for (std::int64_t row = 0; row < width; ++row) {
            float* const target = tile + row * span;
            for (std::int64_t feature = 0; feature < span; ++feature) {
                target[feature] = problem.scale * fast_cos(target[feature])
                    - problem.feature_mean[first + feature];
            }
        }

        for (std::int64_t row = 0; row < width; ++row) {
            const float* const features = tile + row * span;
            float* const accumulator = total.data() + row * parameters;
            for (std::int64_t parameter = 0; parameter < parameters;
                 ++parameter) {
                const float* const column =
                    problem.weight + parameter * problem.features + first;
                float sum = 0.0f;
                for (std::int64_t feature = 0; feature < span; ++feature) {
                    sum += features[feature] * column[feature];
                }
                accumulator[parameter] += sum;
            }
        }
    }

    for (std::int64_t row = 0; row < width; ++row) {
        float* const destination = problem.output + (begin + row) * parameters;
        const float* const accumulator = total.data() + row * parameters;
        for (std::int64_t parameter = 0; parameter < parameters; ++parameter) {
            destination[parameter] =
                accumulator[parameter] + problem.parameter_mean[parameter];
        }
    }
}

// The derivative of one block with respect to its signals. The angle is
// rebuilt rather than kept, which is the same trade the forward pass makes.
TORCHSIM_VECTORIZED void regress_vjp_block(
    const Problem& problem,
    const std::int64_t begin,
    const std::int64_t end,
    std::vector<float>& mapped
) {
    const std::int64_t width = end - begin;
    const std::int64_t contrasts = problem.contrasts;

    for (std::int64_t row = 0; row < width; ++row) {
        float* const destination = problem.output + (begin + row) * contrasts;
        for (std::int64_t contrast = 0; contrast < contrasts; ++contrast) {
            destination[contrast] = 0.0f;
        }
    }

    for (std::int64_t first = 0; first < problem.features;
         first += FEATURE_BLOCK) {
        const std::int64_t last =
            std::min(first + FEATURE_BLOCK, problem.features);
        const std::int64_t span = last - first;
        float* const tile = mapped.data();
        angles(problem, begin, width, first, span, tile);

        for (std::int64_t row = 0; row < width; ++row) {
            const float* const seed =
                problem.cotangent + (begin + row) * problem.parameters;
            float* const target = tile + row * span;
            for (std::int64_t feature = 0; feature < span; ++feature) {
                float through = 0.0f;
                for (std::int64_t parameter = 0;
                     parameter < problem.parameters; ++parameter) {
                    through += seed[parameter]
                        * problem.weight[parameter * problem.features + first
                                         + feature];
                }
                target[feature] =
                    -problem.scale * fast_sin(target[feature]) * through;
            }
        }

        for (std::int64_t row = 0; row < width; ++row) {
            float* const destination =
                problem.output + (begin + row) * contrasts;
            const float* const through = tile + row * span;
            for (std::int64_t feature = 0; feature < span; ++feature) {
                const float* const weights =
                    problem.frequency + (first + feature) * contrasts;
                const float factor = through[feature];
                for (std::int64_t contrast = 0; contrast < contrasts;
                     ++contrast) {
                    destination[contrast] += factor * weights[contrast];
                }
            }
        }
    }
}

void run_blocks(
    const Problem& problem,
    const int requested_threads,
    const bool adjoint
) {
    const std::int64_t blocks =
        (problem.voxels + VOXEL_BLOCK - 1) / VOXEL_BLOCK;
    const unsigned int slots =
        worker_count(requested_threads, blocks);
    WorkerPool::instance().run(slots, [&](const unsigned int slot) {
        std::vector<float> mapped(
            static_cast<std::size_t>(VOXEL_BLOCK * FEATURE_BLOCK)
        );
        for (std::int64_t block = static_cast<std::int64_t>(slot);
             block < blocks; block += static_cast<std::int64_t>(slots)) {
            const std::int64_t begin = block * VOXEL_BLOCK;
            const std::int64_t end =
                std::min(begin + VOXEL_BLOCK, problem.voxels);
            if (adjoint) {
                regress_vjp_block(problem, begin, end, mapped);
            } else {
                regress_block(problem, begin, end, mapped);
            }
        }
    });
}

bool parse_pointer(PyObject* sequence, const Py_ssize_t index, void** pointer) {
    PyObject* value = PySequence_GetItem(sequence, index);
    if (value == nullptr) {
        return false;
    }
    *pointer = PyLong_AsVoidPtr(value);
    Py_DECREF(value);
    return *pointer != nullptr || !PyErr_Occurred();
}

// The buffers a call hands over, as data pointers in a fixed order.
bool unpack(
    PyObject* pointers, const Py_ssize_t expected, std::vector<void*>& raw
) {
    if (PySequence_Size(pointers) != expected) {
        PyErr_SetString(PyExc_ValueError, "wrong number of buffers");
        return false;
    }
    raw.resize(static_cast<std::size_t>(expected));
    for (Py_ssize_t index = 0; index < expected; ++index) {
        if (!parse_pointer(
                pointers, index, &raw[static_cast<std::size_t>(index)]
            )) {
            return false;
        }
    }
    return true;
}

void describe(
    Problem& problem,
    const long long voxels,
    const long long contrasts,
    const long long features,
    const long long parameters
) {
    problem.voxels = voxels;
    problem.contrasts = contrasts;
    problem.features = features;
    problem.parameters = parameters;
    problem.scale = static_cast<float>(
        std::sqrt(2.0 / static_cast<double>(features))
    );
}

PyObject* regress(PyObject*, PyObject* arguments) {
    PyObject* pointers = nullptr;
    long long voxels = 0;
    long long contrasts = 0;
    long long features = 0;
    long long parameters = 0;
    int threads = 0;
    if (!PyArg_ParseTuple(
            arguments, "OLLLLi", &pointers, &voxels, &contrasts, &features,
            &parameters, &threads
        )) {
        return nullptr;
    }
    std::vector<void*> raw;
    if (!unpack(pointers, 8, raw)) {
        return nullptr;
    }
    Problem problem{};
    describe(problem, voxels, contrasts, features, parameters);
    problem.signals = static_cast<const float*>(raw[0]);
    problem.frequency = static_cast<const float*>(raw[1]);
    problem.frequency_t = static_cast<const float*>(raw[2]);
    problem.phase = static_cast<const float*>(raw[3]);
    problem.feature_mean = static_cast<const float*>(raw[4]);
    problem.weight = static_cast<const float*>(raw[5]);
    problem.parameter_mean = static_cast<const float*>(raw[6]);
    problem.output = static_cast<float*>(raw[7]);
    problem.cotangent = nullptr;
    Py_BEGIN_ALLOW_THREADS
    run_blocks(problem, threads, false);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

PyObject* regress_vjp(PyObject*, PyObject* arguments) {
    PyObject* pointers = nullptr;
    long long voxels = 0;
    long long contrasts = 0;
    long long features = 0;
    long long parameters = 0;
    int threads = 0;
    if (!PyArg_ParseTuple(
            arguments, "OLLLLi", &pointers, &voxels, &contrasts, &features,
            &parameters, &threads
        )) {
        return nullptr;
    }
    std::vector<void*> raw;
    if (!unpack(pointers, 7, raw)) {
        return nullptr;
    }
    Problem problem{};
    describe(problem, voxels, contrasts, features, parameters);
    problem.signals = static_cast<const float*>(raw[0]);
    problem.frequency = static_cast<const float*>(raw[1]);
    problem.frequency_t = static_cast<const float*>(raw[2]);
    problem.phase = static_cast<const float*>(raw[3]);
    problem.weight = static_cast<const float*>(raw[4]);
    problem.cotangent = static_cast<const float*>(raw[5]);
    problem.output = static_cast<float*>(raw[6]);
    problem.feature_mean = nullptr;
    problem.parameter_mean = nullptr;
    Py_BEGIN_ALLOW_THREADS
    run_blocks(problem, threads, true);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {"regress", regress, METH_VARARGS, "Fused PERK feature map and regression."},
    {"regress_vjp", regress_vjp, METH_VARARGS,
     "Derivative of the fused regression with respect to its signals."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "torchsim._perk_cpu", nullptr, -1, methods,
    nullptr, nullptr, nullptr, nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__perk_cpu() { return PyModule_Create(&module); }
