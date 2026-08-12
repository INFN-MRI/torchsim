#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

constexpr std::uint8_t PRE_SHIFT = 1;
constexpr std::uint8_t POST_SHIFT = 2;
constexpr std::uint8_t INVERSION = 4;
constexpr std::uint8_t SPOIL_AFTER = 8;
constexpr std::uint8_t SHIFT_AFTER = 16;
constexpr std::uint8_t RECORD = 32;
constexpr float PI = 3.14159265358979323846F;

struct Buffers {
    const float* t1;
    const float* t2;
    const float* m0;
    const float* b1;
    const float* b1_phase;
    const float* b0;
    const float* inversion_efficiency;
    const float* duration;
    const std::int32_t* kind;
    const float* flip;
    const float* phase;
    const std::uint8_t* action;
    const std::int32_t* output_index;
    float* output_real;
    float* output_imag;
};

struct JvpBuffers {
    Buffers primal;
    const float* t1;
    const float* t2;
    const float* m0;
    const float* b1;
    const float* b1_phase;
    const float* b0;
    const float* inversion_efficiency;
    const float* duration;
    const float* flip;
    const float* phase;
};

using Complex = std::complex<float>;

struct DualComplex {
    Complex value;
    Complex tangent;
};

struct DualCoefficient {
    Complex value;
    Complex tangent;
};

inline Complex multiply(const Complex left, const Complex right) {
    return left * right;
}

inline void shift(std::vector<Complex>& fplus, std::vector<Complex>& fminus) {
    const std::size_t count = fplus.size();
    for (std::size_t state = 0; state + 1 < count; ++state) {
        fminus[state] = fminus[state + 1];
    }
    fminus[count - 1] = Complex{};
    for (std::size_t state = count - 1; state > 0; --state) {
        fplus[state] = fplus[state - 1];
    }
    fplus[0] = std::conj(fminus[0]);
}

inline void rotate(
    std::vector<Complex>& fplus,
    std::vector<Complex>& fminus,
    std::vector<Complex>& longitudinal,
    const float alpha,
    const float phi
) {
    const float cosine = std::cos(alpha);
    const float sine = std::sin(alpha);
    const float cosine_half_sq = 0.5F * (1.0F + cosine);
    const float sine_half_sq = 0.5F * (1.0F - cosine);
    const Complex phase_one = std::polar(1.0F, phi);
    const Complex phase_two = phase_one * phase_one;
    const Complex t00(cosine_half_sq, 0.0F);
    const Complex t01 = sine_half_sq * phase_two;
    const Complex t02 = Complex(0.0F, -sine) * phase_one;
    const Complex t10 = std::conj(t01);
    const Complex t11 = t00;
    const Complex t12 = Complex(0.0F, sine) * std::conj(phase_one);
    const Complex t20 = Complex(0.0F, -0.5F * sine) * std::conj(phase_one);
    const Complex t21 = Complex(0.0F, 0.5F * sine) * phase_one;
    const Complex t22(cosine, 0.0F);

    for (std::size_t state = 0; state < fplus.size(); ++state) {
        const Complex fp = fplus[state];
        const Complex fm = fminus[state];
        const Complex z = longitudinal[state];
        fplus[state] = multiply(t00, fp) + multiply(t01, fm) + multiply(t02, z);
        fminus[state] = multiply(t10, fp) + multiply(t11, fm) + multiply(t12, z);
        longitudinal[state] =
            multiply(t20, fp) + multiply(t21, fm) + multiply(t22, z);
    }
}

inline DualComplex apply(
    const DualCoefficient coefficient,
    const DualComplex state
) {
    return {
        coefficient.value * state.value,
        coefficient.tangent * state.value + coefficient.value * state.tangent,
    };
}

inline DualComplex add(
    const DualComplex first,
    const DualComplex second,
    const DualComplex third
) {
    return {
        first.value + second.value + third.value,
        first.tangent + second.tangent + third.tangent,
    };
}

inline void shift(
    std::vector<DualComplex>& fplus,
    std::vector<DualComplex>& fminus
) {
    const std::size_t count = fplus.size();
    for (std::size_t state = 0; state + 1 < count; ++state) {
        fminus[state] = fminus[state + 1];
    }
    fminus[count - 1] = DualComplex{};
    for (std::size_t state = count - 1; state > 0; --state) {
        fplus[state] = fplus[state - 1];
    }
    fplus[0] = {
        std::conj(fminus[0].value),
        std::conj(fminus[0].tangent),
    };
}

inline void rotate(
    std::vector<DualComplex>& fplus,
    std::vector<DualComplex>& fminus,
    std::vector<DualComplex>& longitudinal,
    const float alpha,
    const float alpha_tangent,
    const float phi,
    const float phi_tangent
) {
    const float cosine = std::cos(alpha);
    const float sine = std::sin(alpha);
    const float cosine_tangent = -sine * alpha_tangent;
    const float sine_tangent = cosine * alpha_tangent;
    const float cosine_half_sq = 0.5F * (1.0F + cosine);
    const float sine_half_sq = 0.5F * (1.0F - cosine);
    const float cosine_half_tangent = 0.5F * cosine_tangent;
    const float sine_half_tangent = -0.5F * cosine_tangent;
    const Complex phase_one = std::polar(1.0F, phi);
    const Complex phase_one_tangent = Complex(0.0F, phi_tangent) * phase_one;
    const Complex phase_two = phase_one * phase_one;
    const Complex phase_two_tangent =
        Complex(0.0F, 2.0F * phi_tangent) * phase_two;

    const DualCoefficient t00{
        Complex(cosine_half_sq, 0.0F),
        Complex(cosine_half_tangent, 0.0F),
    };
    const DualCoefficient t01{
        sine_half_sq * phase_two,
        sine_half_tangent * phase_two + sine_half_sq * phase_two_tangent,
    };
    const DualCoefficient t02{
        Complex(0.0F, -sine) * phase_one,
        Complex(0.0F, -sine_tangent) * phase_one
            + Complex(0.0F, -sine) * phase_one_tangent,
    };
    const DualCoefficient t10{std::conj(t01.value), std::conj(t01.tangent)};
    const DualCoefficient t11 = t00;
    const DualCoefficient t12{
        Complex(0.0F, sine) * std::conj(phase_one),
        Complex(0.0F, sine_tangent) * std::conj(phase_one)
            + Complex(0.0F, sine) * std::conj(phase_one_tangent),
    };
    const DualCoefficient t20{
        Complex(0.0F, -0.5F * sine) * std::conj(phase_one),
        Complex(0.0F, -0.5F * sine_tangent) * std::conj(phase_one)
            + Complex(0.0F, -0.5F * sine) * std::conj(phase_one_tangent),
    };
    const DualCoefficient t21{
        Complex(0.0F, 0.5F * sine) * phase_one,
        Complex(0.0F, 0.5F * sine_tangent) * phase_one
            + Complex(0.0F, 0.5F * sine) * phase_one_tangent,
    };
    const DualCoefficient t22{
        Complex(cosine, 0.0F),
        Complex(cosine_tangent, 0.0F),
    };

    for (std::size_t state = 0; state < fplus.size(); ++state) {
        const DualComplex fp = fplus[state];
        const DualComplex fm = fminus[state];
        const DualComplex z = longitudinal[state];
        fplus[state] = add(apply(t00, fp), apply(t01, fm), apply(t02, z));
        fminus[state] = add(apply(t10, fp), apply(t11, fm), apply(t12, z));
        longitudinal[state] =
            add(apply(t20, fp), apply(t21, fm), apply(t22, z));
    }
}

#if defined(__GNUC__) && !defined(__clang__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target_clones("default", "sse4.2", "avx2", "avx512f")))
#endif
void simulate_jvp_range(
    const JvpBuffers& buffers,
    const std::int64_t atom_begin,
    const std::int64_t atom_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count
) {
    const Buffers& primal = buffers.primal;
    for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
        std::vector<DualComplex> fplus(static_cast<std::size_t>(state_count));
        std::vector<DualComplex> fminus(static_cast<std::size_t>(state_count));
        std::vector<DualComplex> longitudinal(static_cast<std::size_t>(state_count));
        longitudinal[0].value = Complex(1.0F, 0.0F);

        const float t1 = primal.t1[atom];
        const float t2 = primal.t2[atom];
        const float r1 = 1000.0F / t1;
        const float r2 = 1000.0F / t2;
        for (std::int64_t event = 0; event < event_count; ++event) {
            const float dt = primal.duration[event];
            const float dt_tangent = buffers.duration[event];
            const float e1 = std::exp(-r1 * dt);
            const float e2 = std::exp(-r2 * dt);
            const float e1_tangent = e1 * (
                1000.0F * dt * buffers.t1[atom] / (t1 * t1)
                - r1 * dt_tangent
            );
            const float e2_tangent = e2 * (
                1000.0F * dt * buffers.t2[atom] / (t2 * t2)
                - r2 * dt_tangent
            );
            const float angle = -2.0F * PI * primal.b0[atom] * dt;
            const float angle_tangent = -2.0F * PI * (
                buffers.b0[atom] * dt + primal.b0[atom] * dt_tangent
            );
            const Complex phase = std::polar(1.0F, angle);
            const Complex phase_tangent = Complex(0.0F, angle_tangent) * phase;
            const Complex off_resonance = e2 * phase;
            const Complex off_resonance_tangent =
                e2_tangent * phase + e2 * phase_tangent;
            const Complex conjugate_off_resonance = std::conj(off_resonance);
            const Complex conjugate_off_tangent =
                std::conj(off_resonance_tangent);
            for (std::int64_t state = 0; state < state_count; ++state) {
                DualComplex& fp = fplus[static_cast<std::size_t>(state)];
                DualComplex& fm = fminus[static_cast<std::size_t>(state)];
                DualComplex& z = longitudinal[static_cast<std::size_t>(state)];
                fp.tangent = fp.tangent * off_resonance
                    + fp.value * off_resonance_tangent;
                fp.value *= off_resonance;
                fm.tangent = fm.tangent * conjugate_off_resonance
                    + fm.value * conjugate_off_tangent;
                fm.value *= conjugate_off_resonance;
                z.tangent = z.tangent * e1 + z.value * e1_tangent;
                z.value *= e1;
            }
            longitudinal[0].value += Complex(1.0F - e1, 0.0F);
            longitudinal[0].tangent -= Complex(e1_tangent, 0.0F);

            const std::uint8_t action = primal.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift(fplus, fminus);
            }
            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const float efficiency = -primal.inversion_efficiency[atom];
                    const float efficiency_tangent =
                        -buffers.inversion_efficiency[atom];
                    for (DualComplex& value : longitudinal) {
                        value.tangent = efficiency_tangent * value.value
                            + efficiency * value.tangent;
                        value.value *= efficiency;
                    }
                } else {
                    const float alpha = primal.flip[event] * primal.b1[atom];
                    const float alpha_tangent =
                        buffers.flip[event] * primal.b1[atom]
                        + primal.flip[event] * buffers.b1[atom];
                    const float phi = primal.phase[event] + primal.b1_phase[atom];
                    const float phi_tangent =
                        buffers.phase[event] + buffers.b1_phase[atom];
                    rotate(
                        fplus,
                        fminus,
                        longitudinal,
                        alpha,
                        alpha_tangent,
                        phi,
                        phi_tangent
                    );
                }
            } else if (primal.kind[event] == 2 && (action & RECORD) != 0) {
                const std::int64_t output = primal.output_index[event];
                const Complex demodulation = std::polar(1.0F, -primal.phase[event]);
                const Complex demodulation_tangent =
                    Complex(0.0F, -buffers.phase[event]) * demodulation;
                const DualComplex fp = fplus[0];
                const Complex signal_tangent =
                    buffers.m0[atom] * fp.value * demodulation
                    + primal.m0[atom] * fp.tangent * demodulation
                    + primal.m0[atom] * fp.value * demodulation_tangent;
                const std::int64_t index = atom * output_count + output;
                primal.output_real[index] = signal_tangent.real();
                primal.output_imag[index] = signal_tangent.imag();
            }
            if ((action & POST_SHIFT) != 0) {
                shift(fplus, fminus);
            }
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(fplus.begin(), fplus.end(), DualComplex{});
                std::fill(fminus.begin(), fminus.end(), DualComplex{});
            } else if ((action & SHIFT_AFTER) != 0) {
                shift(fplus, fminus);
            }
        }
    }
}

#if defined(__GNUC__) && !defined(__clang__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target_clones("default", "sse4.2", "avx2", "avx512f")))
#endif
void simulate_range(
    const Buffers& buffers,
    const std::int64_t atom_begin,
    const std::int64_t atom_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count
) {
    for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
        std::vector<Complex> fplus(static_cast<std::size_t>(state_count));
        std::vector<Complex> fminus(static_cast<std::size_t>(state_count));
        std::vector<Complex> longitudinal(static_cast<std::size_t>(state_count));
        longitudinal[0] = Complex(1.0F, 0.0F);

        const float r1 = 1000.0F / buffers.t1[atom];
        const float r2 = 1000.0F / buffers.t2[atom];
        for (std::int64_t event = 0; event < event_count; ++event) {
            const float dt = buffers.duration[event];
            const float e1 = std::exp(-r1 * dt);
            const float e2 = std::exp(-r2 * dt);
            const Complex off_resonance =
                std::polar(e2, -2.0F * PI * buffers.b0[atom] * dt);
            const Complex conjugate_off_resonance = std::conj(off_resonance);
            for (std::int64_t state = 0; state < state_count; ++state) {
                fplus[static_cast<std::size_t>(state)] *= off_resonance;
                fminus[static_cast<std::size_t>(state)] *= conjugate_off_resonance;
                longitudinal[static_cast<std::size_t>(state)] *= e1;
            }
            longitudinal[0] += Complex(1.0F - e1, 0.0F);

            const std::uint8_t action = buffers.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift(fplus, fminus);
            }
            if (buffers.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const float efficiency = -buffers.inversion_efficiency[atom];
                    for (Complex& value : longitudinal) {
                        value *= efficiency;
                    }
                } else {
                    rotate(
                        fplus,
                        fminus,
                        longitudinal,
                        buffers.flip[event] * buffers.b1[atom],
                        buffers.phase[event] + buffers.b1_phase[atom]
                    );
                }
            } else if (buffers.kind[event] == 2 && (action & RECORD) != 0) {
                const std::int64_t output = buffers.output_index[event];
                const Complex demodulation = std::polar(1.0F, -buffers.phase[event]);
                const Complex signal = buffers.m0[atom] * fplus[0] * demodulation;
                const std::int64_t index = atom * output_count + output;
                buffers.output_real[index] = signal.real();
                buffers.output_imag[index] = signal.imag();
            }
            if ((action & POST_SHIFT) != 0) {
                shift(fplus, fminus);
            }
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(fplus.begin(), fplus.end(), Complex{});
                std::fill(fminus.begin(), fminus.end(), Complex{});
            } else if ((action & SHIFT_AFTER) != 0) {
                shift(fplus, fminus);
            }
        }
    }
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

PyObject* simulate(PyObject*, PyObject* arguments) {
    PyObject* pointers = nullptr;
    long long atom_count = 0;
    long long event_count = 0;
    long long state_count = 0;
    long long output_count = 0;
    int requested_threads = 0;
    if (!PyArg_ParseTuple(
            arguments,
            "OLLLLi",
            &pointers,
            &atom_count,
            &event_count,
            &state_count,
            &output_count,
            &requested_threads
        )) {
        return nullptr;
    }
    if (!PySequence_Check(pointers) || PySequence_Size(pointers) != 15) {
        PyErr_SetString(PyExc_ValueError, "expected fifteen buffer pointers");
        return nullptr;
    }
    if (atom_count < 0 || event_count < 0 || state_count < 1 || output_count < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid EPG buffer dimensions");
        return nullptr;
    }

    void* raw[15]{};
    for (Py_ssize_t index = 0; index < 15; ++index) {
        if (!parse_pointer(pointers, index, &raw[index])) {
            return nullptr;
        }
    }
    const Buffers buffers{
        static_cast<const float*>(raw[0]),
        static_cast<const float*>(raw[1]),
        static_cast<const float*>(raw[2]),
        static_cast<const float*>(raw[3]),
        static_cast<const float*>(raw[4]),
        static_cast<const float*>(raw[5]),
        static_cast<const float*>(raw[6]),
        static_cast<const float*>(raw[7]),
        static_cast<const std::int32_t*>(raw[8]),
        static_cast<const float*>(raw[9]),
        static_cast<const float*>(raw[10]),
        static_cast<const std::uint8_t*>(raw[11]),
        static_cast<const std::int32_t*>(raw[12]),
        static_cast<float*>(raw[13]),
        static_cast<float*>(raw[14]),
    };

    unsigned int thread_count = requested_threads > 0
        ? static_cast<unsigned int>(requested_threads)
        : std::thread::hardware_concurrency();
    thread_count = std::max(1U, thread_count);
    thread_count = std::min(thread_count, static_cast<unsigned int>(std::max(1LL, atom_count)));

    Py_BEGIN_ALLOW_THREADS
    if (thread_count == 1) {
        simulate_range(buffers, 0, atom_count, event_count, state_count, output_count);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(thread_count);
        const std::int64_t block = (atom_count + thread_count - 1) / thread_count;
        for (unsigned int thread = 0; thread < thread_count; ++thread) {
            const std::int64_t begin = static_cast<std::int64_t>(thread) * block;
            const std::int64_t end = std::min<std::int64_t>(
                static_cast<std::int64_t>(atom_count), begin + block
            );
            if (begin < end) {
                workers.emplace_back(
                    simulate_range,
                    std::cref(buffers),
                    begin,
                    end,
                    event_count,
                    state_count,
                    output_count
                );
            }
        }
        for (std::thread& worker : workers) {
            worker.join();
        }
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

PyObject* simulate_jvp(PyObject*, PyObject* arguments) {
    PyObject* pointers = nullptr;
    long long atom_count = 0;
    long long event_count = 0;
    long long state_count = 0;
    long long output_count = 0;
    int requested_threads = 0;
    if (!PyArg_ParseTuple(
            arguments,
            "OLLLLi",
            &pointers,
            &atom_count,
            &event_count,
            &state_count,
            &output_count,
            &requested_threads
        )) {
        return nullptr;
    }
    if (!PySequence_Check(pointers) || PySequence_Size(pointers) != 25) {
        PyErr_SetString(PyExc_ValueError, "expected twenty-five buffer pointers");
        return nullptr;
    }
    if (atom_count < 0 || event_count < 0 || state_count < 1 || output_count < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid EPG buffer dimensions");
        return nullptr;
    }

    void* raw[25]{};
    for (Py_ssize_t index = 0; index < 25; ++index) {
        if (!parse_pointer(pointers, index, &raw[index])) {
            return nullptr;
        }
    }
    const Buffers primal{
        static_cast<const float*>(raw[0]),
        static_cast<const float*>(raw[1]),
        static_cast<const float*>(raw[2]),
        static_cast<const float*>(raw[3]),
        static_cast<const float*>(raw[4]),
        static_cast<const float*>(raw[5]),
        static_cast<const float*>(raw[6]),
        static_cast<const float*>(raw[7]),
        static_cast<const std::int32_t*>(raw[8]),
        static_cast<const float*>(raw[9]),
        static_cast<const float*>(raw[10]),
        static_cast<const std::uint8_t*>(raw[11]),
        static_cast<const std::int32_t*>(raw[12]),
        static_cast<float*>(raw[23]),
        static_cast<float*>(raw[24]),
    };
    const JvpBuffers buffers{
        primal,
        static_cast<const float*>(raw[13]),
        static_cast<const float*>(raw[14]),
        static_cast<const float*>(raw[15]),
        static_cast<const float*>(raw[16]),
        static_cast<const float*>(raw[17]),
        static_cast<const float*>(raw[18]),
        static_cast<const float*>(raw[19]),
        static_cast<const float*>(raw[20]),
        static_cast<const float*>(raw[21]),
        static_cast<const float*>(raw[22]),
    };

    unsigned int thread_count = requested_threads > 0
        ? static_cast<unsigned int>(requested_threads)
        : std::thread::hardware_concurrency();
    thread_count = std::max(1U, thread_count);
    thread_count = std::min(
        thread_count,
        static_cast<unsigned int>(std::max(1LL, atom_count))
    );

    Py_BEGIN_ALLOW_THREADS
    if (thread_count == 1) {
        simulate_jvp_range(
            buffers,
            0,
            atom_count,
            event_count,
            state_count,
            output_count
        );
    } else {
        std::vector<std::thread> workers;
        workers.reserve(thread_count);
        const std::int64_t block = (atom_count + thread_count - 1) / thread_count;
        for (unsigned int thread = 0; thread < thread_count; ++thread) {
            const std::int64_t begin = static_cast<std::int64_t>(thread) * block;
            const std::int64_t end = std::min<std::int64_t>(
                static_cast<std::int64_t>(atom_count), begin + block
            );
            if (begin < end) {
                workers.emplace_back(
                    simulate_jvp_range,
                    std::cref(buffers),
                    begin,
                    end,
                    event_count,
                    state_count,
                    output_count
                );
            }
        }
        for (std::thread& worker : workers) {
            worker.join();
        }
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {"simulate", simulate, METH_VARARGS, "Run a fused CPU EPG state machine."},
    {"simulate_jvp", simulate_jvp, METH_VARARGS, "Run a fused CPU EPG JVP."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_epg_cpu",
    "Torch-free CPU EPG kernels.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__epg_cpu() {
    return PyModule_Create(&module);
}
