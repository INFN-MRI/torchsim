#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace {

constexpr std::uint8_t PRE_SHIFT = 1;
constexpr std::uint8_t POST_SHIFT = 2;
constexpr std::uint8_t INVERSION = 4;
constexpr std::uint8_t SPOIL_AFTER = 8;
constexpr std::uint8_t SHIFT_AFTER = 16;
constexpr std::uint8_t RECORD = 32;
constexpr float PI = 3.14159265358979323846F;

// The kernels' input list, mirrored by sequence/_parameters.py: the tissue
// properties, then the packed per-event buffers. The pointer arrays below are
// sized from these, so a parameter appended there is appended here too.
constexpr std::size_t TISSUE_COUNT = 9;
constexpr std::size_t EVENT_COUNT = 7;
constexpr std::size_t PACKED_COUNT = TISSUE_COUNT + EVENT_COUNT;
// Every tissue property carries a gradient; of the event buffers only
// duration, flip and phase do.
constexpr std::size_t FLOAT_COUNT = TISSUE_COUNT + 3;

// Where the transmit pair sits among the tissue properties.
constexpr std::size_t B1_INDEX = 3;
constexpr std::size_t B1_PHASE_INDEX = 4;

// Rows of one voxel each that a tissue parameter's gradient takes. A pulse
// reaches only the shim it drives, so the transmit pair takes a row per shim;
// every other property belongs to the voxel alone and takes one.
inline std::size_t tissue_rows(const std::size_t parameter, const std::int64_t shims) {
    return (parameter == B1_INDEX || parameter == B1_PHASE_INDEX)
        ? static_cast<std::size_t>(shims)
        : 1U;
}

// Where each parameter's gradient starts within the plane, in those rows. At a
// single shim every base is its own parameter index, which is the flat layout
// a sequence without a transmit array uses.
struct TissueLayout {
    std::size_t base[TISSUE_COUNT];
    std::size_t rows;

    explicit TissueLayout(const std::int64_t shims) : base{}, rows(0U) {
        for (std::size_t parameter = 0U; parameter < TISSUE_COUNT; ++parameter) {
            base[parameter] = rows;
            rows += tissue_rows(parameter, shims);
        }
    }
};

// Workers outlive the call that first needs them, so a kernel pays for a
// handoff rather than for thread creation.
//
// A job hands slot 0 to the calling thread and the rest to the pool. Slots are
// numbered, not owned by any particular worker, so which worker runs a slot
// cannot affect a result: everything the kernels index by "thread" is really
// indexed by slot.
class WorkerPool {
public:
    static WorkerPool& instance() {
        // Deliberately never destroyed. An extension can be torn down with
        // threads still parked, and joining them there risks a deadlock at
        // interpreter exit for no benefit.
        static WorkerPool* const pool = new WorkerPool();
        return *pool;
    }

    void run(const unsigned int slots, const std::function<void(unsigned int)>& task) {
        if (slots <= 1) {
            task(0);
            return;
        }
        // One job at a time. A kernel already spreads across every core, so a
        // second caller gains nothing by interleaving with the first.
        const std::lock_guard<std::mutex> submission(submission_mutex_);
        {
            const std::lock_guard<std::mutex> guard(mutex_);
            while (workers_.size() + 1 < slots) {
                workers_.emplace_back([this] { serve(); });
            }
            task_ = &task;
            next_slot_ = 1;
            slot_limit_ = slots;
            outstanding_ = slots - 1;
        }
        ready_.notify_all();
        task(0);
        std::unique_lock<std::mutex> guard(mutex_);
        finished_.wait(guard, [this] { return outstanding_ == 0; });
        task_ = nullptr;
        slot_limit_ = 0;
    }

private:
    WorkerPool() = default;

    void serve() {
        while (true) {
            unsigned int slot = 0;
            const std::function<void(unsigned int)>* task = nullptr;
            {
                std::unique_lock<std::mutex> guard(mutex_);
                ready_.wait(guard, [this] { return next_slot_ < slot_limit_; });
                slot = next_slot_++;
                task = task_;
            }
            (*task)(slot);
            {
                const std::lock_guard<std::mutex> guard(mutex_);
                --outstanding_;
                if (outstanding_ == 0) {
                    finished_.notify_one();
                }
            }
        }
    }

    std::mutex submission_mutex_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable finished_;
    std::vector<std::thread> workers_;
    const std::function<void(unsigned int)>* task_ = nullptr;
    unsigned int next_slot_ = 0;
    unsigned int slot_limit_ = 0;
    unsigned int outstanding_ = 0;
};

// The processors this process may actually run on, which is not the same as the
// ones the machine has: a container or a taskset narrows the CPU set without
// changing what hardware_concurrency reports, and workers placed outside that
// set only contend for the ones inside it.
inline unsigned int usable_processors() {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        const int usable = CPU_COUNT(&set);
        if (usable > 0) {
            return static_cast<unsigned int>(usable);
        }
    }
#endif
    return std::max(1U, std::thread::hardware_concurrency());
}

// Even from a pool a worker has to be worth waking, so a problem that cannot
// give every slot a few work items runs faster with fewer slots. An explicit
// request is honoured as given, capped only by the work available.
constexpr std::int64_t MIN_WORK_PER_THREAD = 4;

inline unsigned int worker_count(
    const int requested, const std::int64_t work_count
) {
    const std::int64_t available = requested > 0
        ? static_cast<std::int64_t>(requested)
        : static_cast<std::int64_t>(usable_processors());
    const std::int64_t affordable = requested > 0
        ? work_count
        : work_count / MIN_WORK_PER_THREAD;
    const std::int64_t count =
        std::min(available, std::max<std::int64_t>(1, affordable));
    return static_cast<unsigned int>(std::max<std::int64_t>(1, count));
}

struct Buffers {
    const float* t1;
    const float* t2;
    const float* m0;
    const float* b1;
    const float* b1_phase;
    const float* b0;
    const float* inversion_efficiency;
    const float* diffusion;
    const float* velocity;
    const float* duration;
    const std::int32_t* kind;
    const float* flip;
    const float* phase;
    const std::uint8_t* action;
    const std::int32_t* output_index;
    // Which row of the transmit buffers each pulse drives. ``b1`` and
    // ``b1_phase`` hold ``shim_count`` rows of ``atom_count``, so a pulse reads
    // its own shim's field rather than one shared by the whole sequence.
    const std::int32_t* shim_index;
    float* output_real;
    float* output_imag;
    // ``duration``, ``flip`` and ``phase`` are (train_count, event_count)
    // row-major; every other event buffer describes structure shared by all
    // trains. Work items enumerate the (train, atom) product train-major, so a
    // range of consecutive items aligned to ``atom_count`` owns whole trains --
    // which is what lets threads write event gradients without sharing a slot.
    std::int64_t atom_count;
    std::int64_t train_count;
    // The sequence's gradient geometry, which turns a spin velocity into the
    // two rates it drives: ``flow_scale`` in rad/m is the winding an unbalanced
    // gradient puts across a metre, and ``washout_scale`` in 1/m is the
    // reciprocal voxel size. They are separate because washout depends on the
    // speed a spin leaves the voxel at whether or not a gradient is playing,
    // while flow dephasing depends on the winding it crosses.
    float flow_scale;
    float washout_scale;
    std::int64_t shim_count;
    // The rotation a shaped pulse performs, tabulated over slice position and
    // effective flip angle: ``locations`` rows of ``profile_bins`` knots, eight
    // floats each -- the Cayley-Klein pair and its slope in the flip angle,
    // interleaved so the two knots a read needs are contiguous. Null when the
    // sequence has no table, which is what selects the flip-and-phase operator
    // instead. Voxels run voxel-major over the slice, so a voxel's row is its
    // index modulo ``locations``.
    const float* profile;
    std::int64_t profile_bins;
    std::int64_t locations;
    float profile_step;
};

// Floats one knot of the transition table holds.
constexpr std::int64_t PROFILE_STRIDE = 8;

// Per-work-item view of the buffers that vary along the train axis.
struct TrainView {
    const float* duration;
    const float* flip;
    const float* phase;
    std::int64_t atom;
    std::int64_t train;
    std::int64_t output_base;
    std::int64_t event_base;
};

inline TrainView train_view(
    const Buffers& buffers,
    const std::int64_t work,
    const std::int64_t event_count,
    const std::int64_t output_count
) {
    const std::int64_t train = work / buffers.atom_count;
    const std::int64_t atom = work % buffers.atom_count;
    const std::int64_t event_base = train * event_count;
    return TrainView{
        buffers.duration + event_base,
        buffers.flip + event_base,
        buffers.phase + event_base,
        atom,
        train,
        (train * buffers.atom_count + atom) * output_count,
        event_base,
    };
}

struct JvpBuffers {
    Buffers primal;
    const float* t1;
    const float* t2;
    const float* m0;
    const float* b1;
    const float* b1_phase;
    const float* b0;
    const float* inversion_efficiency;
    const float* diffusion;
    const float* velocity;
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

struct DualFloat {
    float value;
    float tangent;
};

inline DualFloat operator+(const DualFloat a, const DualFloat b) {
    return {a.value + b.value, a.tangent + b.tangent};
}

inline DualFloat operator-(const DualFloat a, const DualFloat b) {
    return {a.value - b.value, a.tangent - b.tangent};
}

inline DualFloat operator*(const DualFloat a, const DualFloat b) {
    return {a.value * b.value, a.tangent * b.value + a.value * b.tangent};
}

inline DualFloat operator*(const float a, const DualFloat b) {
    return {a * b.value, a * b.tangent};
}

inline DualComplex operator+(const DualComplex a, const DualComplex b) {
    return {a.value + b.value, a.tangent + b.tangent};
}

inline DualComplex operator*(const DualComplex a, const DualComplex b) {
    return {a.value * b.value, a.tangent * b.value + a.value * b.tangent};
}

inline DualComplex operator*(const DualFloat a, const DualComplex b) {
    return {a.value * b.value, a.tangent * b.value + a.value * b.tangent};
}

inline DualComplex operator*(const DualComplex a, const DualFloat b) {
    return {a.value * b.value, a.tangent * b.value + a.value * b.tangent};
}

inline DualComplex operator*(const Complex a, const DualComplex b) {
    return {a * b.value, a * b.tangent};
}

inline DualComplex operator-(const DualComplex a, const DualComplex b) {
    return {a.value - b.value, a.tangent - b.tangent};
}

inline DualComplex conjugate(const DualComplex a) {
    return {std::conj(a.value), std::conj(a.tangent)};
}

inline DualFloat real_part(const DualComplex a) {
    return {a.value.real(), a.tangent.real()};
}

inline DualComplex to_complex(const DualFloat a) {
    return {Complex(a.value, 0.0F), Complex(a.tangent, 0.0F)};
}

inline DualFloat dual_exp(const DualFloat a) {
    const float value = std::exp(a.value);
    return {value, value * a.tangent};
}

inline DualFloat dual_cos(const DualFloat a) {
    return {std::cos(a.value), -std::sin(a.value) * a.tangent};
}

inline DualFloat dual_sin(const DualFloat a) {
    return {std::sin(a.value), std::cos(a.value) * a.tangent};
}

// exp(i * angle) for a dual angle
inline DualComplex dual_polar(const DualFloat angle) {
    const Complex value = std::polar(1.0F, angle.value);
    return {value, Complex(0.0F, angle.tangent) * value};
}

inline DualFloat dual_reciprocal(const DualFloat a) {
    const float inverse = 1.0F / a.value;
    return {inverse, -a.tangent * inverse * inverse};
}

// ---------------------------------------------------------------------------
// Diffusion damping.
//
// Over a free-precession interval a state at dephasing order l accumulates a
// b-factor weighted by l^2 while longitudinal and by l^2 + l + 1/3 while
// transverse. The rate arrives already carrying the sequence's gradient
// geometry, so ``rate * dt`` is the b-factor an order-1 state accumulates.
//
// Order zero has longitudinal weight zero, so its damping is exactly one: the
// affine recovery term needs no special case.
// ---------------------------------------------------------------------------

inline float longitudinal_weight(const std::size_t state) {
    const float order = static_cast<float>(state);
    return order * order;
}

inline float transverse_weight(const std::size_t state) {
    const float order = static_cast<float>(state);
    return order * order + order + (1.0F / 3.0F);
}

inline float exponential(const float a) {
    return std::exp(a);
}

inline DualFloat exponential(const DualFloat a) {
    return dual_exp(a);
}

inline bool is_zero(const float a) {
    return a == 0.0F;
}

inline bool is_zero(const DualFloat a) {
    return a.value == 0.0F && a.tangent == 0.0F;
}

inline void set_unit(float& target) {
    target = 1.0F;
}

inline void set_unit(DualFloat& target) {
    target = DualFloat{1.0F, 0.0F};
}

// Per-order damping factors for one interval, held across events so a train
// with no diffusion never evaluates a transcendental.
template <typename Scalar>
struct Damping {
    std::vector<Scalar> longitudinal;
    std::vector<Scalar> transverse;
    bool damped;

    explicit Damping(const std::size_t states)
        : longitudinal(states), transverse(states), damped(true) {
        reset();
    }

    void reset() {
        if (!damped) {
            return;
        }
        for (std::size_t state = 0; state < longitudinal.size(); ++state) {
            set_unit(longitudinal[state]);
            set_unit(transverse[state]);
        }
        damped = false;
    }

    // Set the factors for one interval, leaving them at one where there is no
    // diffusion to apply.
    void set(const Scalar rate, const Scalar dt) {
        if (is_zero(rate)) {
            reset();
            return;
        }
        const Scalar b_factor = rate * dt;
        for (std::size_t state = 0; state < longitudinal.size(); ++state) {
            longitudinal[state] =
                exponential(-longitudinal_weight(state) * b_factor);
            transverse[state] =
                exponential(-transverse_weight(state) * b_factor);
        }
        damped = true;
    }
};

// Phase each dephasing order turns through over one interval. The rate is the
// winding per unit order per second, so a longitudinal state at order l turns
// through l * rate * dt; the transverse states sit half an order further along
// the gradient. Order zero is left alone while longitudinal, so the recovery
// term is unaffected.
inline void flow_turn(
    const float rate,
    const float dt,
    const std::size_t state,
    float& longitudinal,
    float& transverse
) {
    const float turn = rate * dt;
    const float order = static_cast<float>(state);
    longitudinal = -order * turn;
    transverse = -(order + 0.5F) * turn;
}

// The same turn carried on dual numbers, for the kernels that differentiate.
template <typename Scalar>
inline void flow_turn_dual(
    const Scalar rate,
    const Scalar dt,
    const std::size_t state,
    Scalar& longitudinal,
    Scalar& transverse
) {
    const Scalar turn = rate * dt;
    const float order = static_cast<float>(state);
    longitudinal = (-order) * turn;
    transverse = (-(order + 0.5F)) * turn;
}

// The fraction of a voxel's spins replaced by inflowing ones over an interval,
// at a rate of ``|v| / voxel_size`` per second. Clamped at one: a spin cannot
// leave the voxel more than once, and past that the whole voxel has turned over.
//
// The inflowing spins are taken to be fully relaxed and unexcited -- no
// transverse magnetization and unit longitudinal -- which is the single
// compartment a state machine over one voxel can see. That makes washout an
// affine map of exactly the shape longitudinal recovery already has, and
//
//     wout * (Z * e1 + (1 - e1)) + win  ==  Z * (e1 * wout) + (1 - e1 * wout)
//
// so it needs no term of its own: scaling e1 and e2 by ``wout`` carries it,
// derivatives included.
inline float washout_out(const float rate, const float dt) {
    return 1.0F - std::min(rate * dt, 1.0F);
}

// The same fraction carried on dual numbers. Past the clamp the interval has
// replaced the voxel outright and nothing further depends on the rate, so the
// tangent is zero there.
inline DualFloat washout_out(const DualFloat rate, const DualFloat dt) {
    const DualFloat fraction = rate * dt;
    if (fraction.value >= 1.0F) {
        return DualFloat{0.0F, 0.0F};
    }
    return DualFloat{1.0F - fraction.value, -fraction.tangent};
}

// Which way a velocity points, and nothing at all where it is zero: washout
// depends on the speed, and |v| has no derivative at the origin.
inline float speed_direction(const float velocity) {
    return static_cast<float>(velocity > 0.0F) - static_cast<float>(velocity < 0.0F);
}

// Where an event's pulse reads the transmit field. Templated rather than
// branched so a sequence with one shim compiles to the bare atom index it
// would have had, with neither the row load nor the multiply.
template <bool SHIMMED>
inline std::int64_t transmit_row(
    const Buffers& buffers, const std::int64_t event, const std::int64_t atom
) {
    if constexpr (SHIMMED) {
        return static_cast<std::int64_t>(buffers.shim_index[event])
            * buffers.atom_count + atom;
    } else {
        (void)buffers;
        (void)event;
        return atom;
    }
}

// Whether any atom carries diffusion. The lane kernels keep transcendentals
// out of their state loops, which per-order damping would undo, so they are
// selected only when this is false.
inline bool any_diffusion(const float* rate, const std::int64_t atom_count) {
    for (std::int64_t atom = 0; atom < atom_count; ++atom) {
        if (rate[atom] != 0.0F) {
            return true;
        }
    }
    return false;
}

inline Complex multiply(const Complex left, const Complex right) {
    return left * right;
}

// Where a voxel sits along the slice. Voxels are spread over the profile
// voxel-major, so consecutive atoms walk the slice and wrap.
template <bool PROFILED>
inline std::int64_t slice_row(const Buffers& buffers, const std::int64_t atom) {
    if constexpr (PROFILED) {
        return atom % buffers.locations;
    } else {
        (void)buffers;
        (void)atom;
        return 0;
    }
}

// The pair the table holds at this flip angle, by cubic Hermite between the
// knots bracketing it. Cubic rather than linear because a linear read has no
// second derivative to give the second-order pass, and because storing the
// slope makes the cubic cost the same two loads.
//
// Clamped at both ends rather than extrapolated: a cubic run off its grid
// leaves the unit circle, and a pulse driven past the tabulated flip should
// saturate rather than diverge.
inline void profile_pair(
    const Buffers& buffers,
    const std::int64_t location,
    const float theta,
    Complex& a,
    Complex& b
) {
    const float last = static_cast<float>(buffers.profile_bins - 1);
    const float scaled = std::min(std::max(theta / buffers.profile_step, 0.0F), last);
    const float lower = std::min(std::floor(scaled), last - 1.0F);
    const float u = scaled - lower;
    const float* const near = buffers.profile
        + (location * buffers.profile_bins + static_cast<std::int64_t>(lower))
            * PROFILE_STRIDE;
    const float* const far = near + PROFILE_STRIDE;

    const float u2 = u * u;
    const float u3 = u2 * u;
    const float h00 = 2.0F * u3 - 3.0F * u2 + 1.0F;
    const float h10 = (u3 - 2.0F * u2 + u) * buffers.profile_step;
    const float h01 = -2.0F * u3 + 3.0F * u2;
    const float h11 = (u3 - u2) * buffers.profile_step;

    a = Complex(
        h00 * near[0] + h10 * near[4] + h01 * far[0] + h11 * far[4],
        h00 * near[1] + h10 * near[5] + h01 * far[1] + h11 * far[5]
    );
    b = Complex(
        h00 * near[2] + h10 * near[6] + h01 * far[2] + h11 * far[6],
        h00 * near[3] + h10 * near[7] + h01 * far[3] + h11 * far[7]
    );
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

// The pair and its derivative in the flip angle, from the same cubic. The
// derivative of a Hermite segment is another polynomial in the same four knot
// values, so reading both costs one extra combination rather than a second
// table.
//
// Past the ends the value is clamped and the derivative is that of the clamped
// curve, which is the last segment's -- callers refuse a pulse that reaches
// there, so this only has to stay finite.
inline void profile_pair_slope(
    const Buffers& buffers,
    const std::int64_t location,
    const float theta,
    Complex& a,
    Complex& b,
    Complex& slope_a,
    Complex& slope_b
) {
    const float last = static_cast<float>(buffers.profile_bins - 1);
    const float scaled = std::min(std::max(theta / buffers.profile_step, 0.0F), last);
    const float lower = std::min(std::floor(scaled), last - 1.0F);
    const float u = scaled - lower;
    const float* const near = buffers.profile
        + (location * buffers.profile_bins + static_cast<std::int64_t>(lower))
            * PROFILE_STRIDE;
    const float* const far = near + PROFILE_STRIDE;

    const float u2 = u * u;
    const float u3 = u2 * u;
    const float step = buffers.profile_step;
    const float h00 = 2.0F * u3 - 3.0F * u2 + 1.0F;
    const float h10 = (u3 - 2.0F * u2 + u) * step;
    const float h01 = -2.0F * u3 + 3.0F * u2;
    const float h11 = (u3 - u2) * step;
    // d/dtheta is d/du over the knot spacing.
    const float g00 = (6.0F * u2 - 6.0F * u) / step;
    const float g10 = 3.0F * u2 - 4.0F * u + 1.0F;
    const float g01 = (-6.0F * u2 + 6.0F * u) / step;
    const float g11 = 3.0F * u2 - 2.0F * u;

    float value[4];
    float slope[4];
    for (std::size_t part = 0; part < 4; ++part) {
        value[part] = h00 * near[part] + h10 * near[part + 4]
            + h01 * far[part] + h11 * far[part + 4];
        slope[part] = g00 * near[part] + g10 * near[part + 4]
            + g01 * far[part] + g11 * far[part + 4];
    }
    a = Complex(value[0], value[1]);
    b = Complex(value[2], value[3]);
    slope_a = Complex(slope[0], slope[1]);
    slope_b = Complex(slope[2], slope[3]);
}

// The rotation a pulse of any shape performs, named by its Cayley-Klein pair:
//
//   T = [ conj(a)^2   -conj(b)^2   -2 conj(a b) ]
//       [ -b^2         a^2         -2 a b       ]
//       [ conj(a) b    a conj(b)   |a|^2-|b|^2  ]
//
// `rotate` above is the case of an instantaneous pulse, kept separate rather
// than folded into this: it reaches the same matrix through different
// arithmetic, and a sequence with no table must not move in the last place for
// the sake of one that has a table.
inline void rotate_spinor(
    std::vector<Complex>& fplus,
    std::vector<Complex>& fminus,
    std::vector<Complex>& longitudinal,
    const Complex a,
    const Complex b
) {
    const Complex conj_a = std::conj(a);
    const Complex conj_b = std::conj(b);
    const Complex t00 = conj_a * conj_a;
    const Complex t01 = -conj_b * conj_b;
    const Complex t02 = -2.0F * std::conj(a * b);
    const Complex t10 = -b * b;
    const Complex t11 = a * a;
    const Complex t12 = -2.0F * a * b;
    const Complex t20 = conj_a * b;
    const Complex t21 = a * conj_b;
    const Complex t22(std::norm(a) - std::norm(b), 0.0F);

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

// The spinor rotation carrying a forward-mode tangent. Every entry of the
// matrix is a product of two factors drawn from the pair and its conjugate, so
// its tangent is the same product differentiated once.
inline void rotate_spinor(
    std::vector<DualComplex>& fplus,
    std::vector<DualComplex>& fminus,
    std::vector<DualComplex>& longitudinal,
    const DualComplex a,
    const DualComplex b
) {
    const DualComplex conj_a{std::conj(a.value), std::conj(a.tangent)};
    const DualComplex conj_b{std::conj(b.value), std::conj(b.tangent)};

    auto product = [](const DualComplex left, const DualComplex right) {
        return DualCoefficient{
            left.value * right.value,
            left.tangent * right.value + left.value * right.tangent,
        };
    };
    auto scaled = [](const DualCoefficient term, const float factor) {
        return DualCoefficient{factor * term.value, factor * term.tangent};
    };

    const DualCoefficient aa = product(conj_a, conj_a);
    const DualCoefficient bb = product(conj_b, conj_b);
    const DualCoefficient ab = product(conj_a, conj_b);
    const DualCoefficient plain_bb = product(b, b);
    const DualCoefficient plain_aa = product(a, a);
    const DualCoefficient plain_ab = product(a, b);

    const DualCoefficient t00 = aa;
    const DualCoefficient t01 = scaled(bb, -1.0F);
    const DualCoefficient t02 = scaled(ab, -2.0F);
    const DualCoefficient t10 = scaled(plain_bb, -1.0F);
    const DualCoefficient t11 = plain_aa;
    const DualCoefficient t12 = scaled(plain_ab, -2.0F);
    const DualCoefficient t20 = product(conj_a, b);
    const DualCoefficient t21 = product(a, conj_b);
    const DualCoefficient norm_a = product(a, conj_a);
    const DualCoefficient norm_b = product(b, conj_b);
    const DualCoefficient t22{
        norm_a.value - norm_b.value, norm_a.tangent - norm_b.tangent
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

template <bool SHIMMED, bool PROFILED>
#if defined(__GNUC__) && !defined(__clang__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target_clones("default", "sse4.2", "avx2", "avx512f")))
#endif
void simulate_jvp_range(
    const JvpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count
) {
    const Buffers& primal = buffers.primal;
    const std::size_t states = static_cast<std::size_t>(state_count);
    std::vector<DualComplex> fplus(states);
    std::vector<DualComplex> fminus(states);
    std::vector<DualComplex> longitudinal(states);
    Damping<DualFloat> damping(states);

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const TrainView view = train_view(primal, work, event_count, output_count);
        const std::int64_t atom = view.atom;
        const std::int64_t location = slice_row<PROFILED>(primal, atom);
        const float* const dot_duration = buffers.duration + view.event_base;
        const float* const dot_flip = buffers.flip + view.event_base;
        const float* const dot_phase = buffers.phase + view.event_base;
        std::fill(fplus.begin(), fplus.end(), DualComplex{});
        std::fill(fminus.begin(), fminus.end(), DualComplex{});
        std::fill(longitudinal.begin(), longitudinal.end(), DualComplex{});
        longitudinal[0].value = Complex(1.0F, 0.0F);

        const float t1 = primal.t1[atom];
        const float t2 = primal.t2[atom];
        const float r1 = 1000.0F / t1;
        const float r2 = 1000.0F / t2;
        const DualFloat damping_rate{
            primal.diffusion[atom], buffers.diffusion[atom]
        };
        const float velocity = primal.velocity[atom];
        const float dot_velocity = buffers.velocity[atom];
        const DualFloat flow_rate{
            velocity * primal.flow_scale, dot_velocity * primal.flow_scale
        };
        const DualFloat washout_rate{
            std::fabs(velocity) * primal.washout_scale,
            speed_direction(velocity) * dot_velocity * primal.washout_scale,
        };
        for (std::int64_t event = 0; event < event_count; ++event) {
            const float dt = view.duration[event];
            const float dt_tangent = dot_duration[event];
            const DualFloat interval{dt, dt_tangent};
            damping.set(damping_rate, interval);
            const DualFloat wout = washout_out(washout_rate, interval);
            const float dry1 = std::exp(-r1 * dt);
            const float dry2 = std::exp(-r2 * dt);
            const float e1 = dry1 * wout.value;
            const float e2 = dry2 * wout.value;
            const float e1_tangent = e1 * (
                1000.0F * dt * buffers.t1[atom] / (t1 * t1)
                - r1 * dt_tangent
            ) + dry1 * wout.tangent;
            const float e2_tangent = e2 * (
                1000.0F * dt * buffers.t2[atom] / (t2 * t2)
                - r2 * dt_tangent
            ) + dry2 * wout.tangent;
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
                const std::size_t index = static_cast<std::size_t>(state);
                const DualFloat damp_transverse = damping.transverse[index];
                const DualFloat damp_longitudinal = damping.longitudinal[index];
                DualComplex& fp = fplus[index];
                DualComplex& fm = fminus[index];
                DualComplex& z = longitudinal[index];
                DualFloat turn_longitudinal{};
                DualFloat turn_transverse{};
                flow_turn_dual(
                    flow_rate, interval, index, turn_longitudinal, turn_transverse
                );
                const Complex spin_transverse =
                    std::polar(1.0F, turn_transverse.value);
                const Complex spin_transverse_tangent =
                    Complex(0.0F, turn_transverse.tangent) * spin_transverse;
                const Complex spin_longitudinal =
                    std::polar(1.0F, turn_longitudinal.value);
                const Complex spin_longitudinal_tangent =
                    Complex(0.0F, turn_longitudinal.tangent) * spin_longitudinal;
                const Complex damped = off_resonance * damp_transverse.value;
                const Complex damped_tangent =
                    off_resonance_tangent * damp_transverse.value
                    + off_resonance * damp_transverse.tangent;
                const Complex off = damped * spin_transverse;
                const Complex off_tangent =
                    damped_tangent * spin_transverse + damped * spin_transverse_tangent;
                const Complex conjugate_off = std::conj(off);
                const Complex conjugate_tangent = std::conj(off_tangent);
                const float recovered = e1 * damp_longitudinal.value;
                const float recovered_tangent =
                    e1_tangent * damp_longitudinal.value
                    + e1 * damp_longitudinal.tangent;
                const Complex relaxation = recovered * spin_longitudinal;
                const Complex relaxation_tangent =
                    recovered_tangent * spin_longitudinal
                    + recovered * spin_longitudinal_tangent;
                fp.tangent = fp.tangent * off + fp.value * off_tangent;
                fp.value *= off;
                fm.tangent =
                    fm.tangent * conjugate_off + fm.value * conjugate_tangent;
                fm.value *= conjugate_off;
                z.tangent = z.tangent * relaxation + z.value * relaxation_tangent;
                z.value *= relaxation;
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
                    const std::int64_t transmit =
                        transmit_row<SHIMMED>(primal, event, atom);
                    const float alpha = view.flip[event] * primal.b1[transmit];
                    const float alpha_tangent =
                        dot_flip[event] * primal.b1[transmit]
                        + view.flip[event] * buffers.b1[transmit];
                    const float phi = view.phase[event] + primal.b1_phase[transmit];
                    const float phi_tangent =
                        dot_phase[event] + buffers.b1_phase[transmit];
                    if constexpr (PROFILED) {
                        Complex pair_a{};
                        Complex pair_b{};
                        Complex slope_a{};
                        Complex slope_b{};
                        profile_pair_slope(
                            primal, location, alpha, pair_a, pair_b, slope_a,
                            slope_b
                        );
                        // The flip angle carries the tangent into the table;
                        // the RF phase turns the axis after it comes out.
                        const Complex turn = std::polar(1.0F, -phi);
                        const Complex spun = pair_b * turn;
                        rotate_spinor(
                            fplus,
                            fminus,
                            longitudinal,
                            DualComplex{pair_a, slope_a * alpha_tangent},
                            DualComplex{
                                spun,
                                slope_b * turn * alpha_tangent
                                    - Complex(0.0F, phi_tangent) * spun,
                            }
                        );
                    } else {
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
                }
            } else if (primal.kind[event] == 2 && (action & RECORD) != 0) {
                const std::int64_t output = primal.output_index[event];
                const Complex demodulation = std::polar(1.0F, -view.phase[event]);
                const Complex demodulation_tangent =
                    Complex(0.0F, -dot_phase[event]) * demodulation;
                const DualComplex fp = fplus[0];
                const Complex signal_tangent =
                    buffers.m0[atom] * fp.value * demodulation
                    + primal.m0[atom] * fp.tangent * demodulation
                    + primal.m0[atom] * fp.value * demodulation_tangent;
                const std::int64_t index = view.output_base + output;
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
inline void shift_real(
    std::vector<float>& plus, std::vector<float>& minus, const std::size_t states
) {
    for (std::size_t state = 0; state + 1 < states; ++state) {
        minus[state] = minus[state + 1];
    }
    minus[states - 1] = 0.0F;
    for (std::size_t state = states - 1; state > 0; --state) {
        plus[state] = plus[state - 1];
    }
    plus[0] = -minus[0];
}

// ---------------------------------------------------------------------------
// Real-subspace forward.
//
// When every refocusing pulse shares a phase and there is no off-resonance or
// transmit phase, writing F+ = e^{i phi} i a, F- = e^{-i phi} i b, Z = c leaves
// a, b and c real for the whole train: relaxation scales them, the shift's
// conjugate coupling becomes a0 = -b0, and the RF rotation reduces to a real
// 3x3 in the flip angle alone. The recorded sample is then i * m0 * a0.
//
// Callers must have established those conditions; see real_subspace_axis.
// ---------------------------------------------------------------------------

void simulate_real_range(
    const Buffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count
) {
    const std::size_t states = static_cast<std::size_t>(state_count);
    std::vector<float> plus(states);
    std::vector<float> minus(states);
    std::vector<float> longitudinal(states);
    Damping<float> damping(states);

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const TrainView view = train_view(buffers, work, event_count, output_count);
        const std::int64_t atom = view.atom;
        std::fill(plus.begin(), plus.end(), 0.0F);
        std::fill(minus.begin(), minus.end(), 0.0F);
        std::fill(longitudinal.begin(), longitudinal.end(), 0.0F);
        longitudinal[0] = 1.0F;

        const float r1 = 1000.0F / buffers.t1[atom];
        const float r2 = 1000.0F / buffers.t2[atom];
        const float b1 = buffers.b1[atom];
        const float m0 = buffers.m0[atom];
        const float damping_rate = buffers.diffusion[atom];

        for (std::int64_t event = 0; event < event_count; ++event) {
            const float dt = view.duration[event];
            damping.set(damping_rate, dt);
            const float e1 = std::exp(-r1 * dt);
            const float e2 = std::exp(-r2 * dt);
            for (std::size_t state = 0; state < states; ++state) {
                const float damp_transverse = e2 * damping.transverse[state];
                plus[state] *= damp_transverse;
                minus[state] *= damp_transverse;
                longitudinal[state] *= e1 * damping.longitudinal[state];
            }
            longitudinal[0] += 1.0F - e1;

            const std::uint8_t action = buffers.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift_real(plus, minus, states);
            }
            if (buffers.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const float efficiency = -buffers.inversion_efficiency[atom];
                    for (std::size_t state = 0; state < states; ++state) {
                        longitudinal[state] *= efficiency;
                    }
                } else {
                    const float alpha = view.flip[event] * b1;
                    const float cosine = std::cos(alpha);
                    const float sine = std::sin(alpha);
                    const float cosine_half_sq = 0.5F * (1.0F + cosine);
                    const float sine_half_sq = 0.5F * (1.0F - cosine);
                    const float half_sine = 0.5F * sine;
                    for (std::size_t state = 0; state < states; ++state) {
                        const float p = plus[state];
                        const float m = minus[state];
                        const float z = longitudinal[state];
                        plus[state] = cosine_half_sq * p + sine_half_sq * m - sine * z;
                        minus[state] = sine_half_sq * p + cosine_half_sq * m + sine * z;
                        longitudinal[state] = half_sine * p - half_sine * m + cosine * z;
                    }
                }
            } else if (buffers.kind[event] == 2 && (action & RECORD) != 0) {
                const std::int64_t index =
                    view.output_base + buffers.output_index[event];
                buffers.output_real[index] = 0.0F;
                buffers.output_imag[index] = m0 * plus[0];
            }
            if ((action & POST_SHIFT) != 0) {
                shift_real(plus, minus, states);
            }
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(plus.begin(), plus.end(), 0.0F);
                std::fill(minus.begin(), minus.end(), 0.0F);
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_real(plus, minus, states);
            }
        }
    }
}

// Forward-mode through the real subspace. The tangent obeys the same closure as
// the value, so a, b and c each carry a derivative and the rotation is the same
// real 3x3 differentiated by the product rule.
//
// A tangent along b0, b1_phase or an RF phase would leave the subspace; callers
// must seed only directions that stay inside it.
void simulate_real_jvp_range(
    const JvpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count
) {
    const Buffers& primal = buffers.primal;
    const std::size_t states = static_cast<std::size_t>(state_count);
    std::vector<float> plus(states), minus(states), longitudinal(states);
    std::vector<float> dot_plus(states), dot_minus(states), dot_longitudinal(states);
    Damping<DualFloat> damping(states);

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const TrainView view = train_view(primal, work, event_count, output_count);
        const std::int64_t atom = view.atom;
        const float* const dot_duration = buffers.duration + view.event_base;
        const float* const dot_flip = buffers.flip + view.event_base;

        std::fill(plus.begin(), plus.end(), 0.0F);
        std::fill(minus.begin(), minus.end(), 0.0F);
        std::fill(longitudinal.begin(), longitudinal.end(), 0.0F);
        std::fill(dot_plus.begin(), dot_plus.end(), 0.0F);
        std::fill(dot_minus.begin(), dot_minus.end(), 0.0F);
        std::fill(dot_longitudinal.begin(), dot_longitudinal.end(), 0.0F);
        longitudinal[0] = 1.0F;

        const float t1 = primal.t1[atom];
        const float t2 = primal.t2[atom];
        const float r1 = 1000.0F / t1;
        const float r2 = 1000.0F / t2;
        const float b1 = primal.b1[atom];
        const float m0 = primal.m0[atom];
        const float dot_t1 = buffers.t1[atom];
        const float dot_t2 = buffers.t2[atom];
        const float dot_b1 = buffers.b1[atom];
        const float dot_m0 = buffers.m0[atom];
        const DualFloat damping_rate{
            primal.diffusion[atom], buffers.diffusion[atom]
        };

        for (std::int64_t event = 0; event < event_count; ++event) {
            const float dt = view.duration[event];
            const float dt_tangent = dot_duration[event];
            damping.set(damping_rate, DualFloat{dt, dt_tangent});
            const float e1 = std::exp(-r1 * dt);
            const float e2 = std::exp(-r2 * dt);
            const float e1_tangent =
                e1 * (1000.0F * dt * dot_t1 / (t1 * t1) - r1 * dt_tangent);
            const float e2_tangent =
                e2 * (1000.0F * dt * dot_t2 / (t2 * t2) - r2 * dt_tangent);
            for (std::size_t state = 0; state < states; ++state) {
                const DualFloat damp_transverse = damping.transverse[state];
                const DualFloat damp_longitudinal = damping.longitudinal[state];
                const float transverse = e2 * damp_transverse.value;
                const float transverse_tangent = e2_tangent * damp_transverse.value
                    + e2 * damp_transverse.tangent;
                const float relaxation = e1 * damp_longitudinal.value;
                const float relaxation_tangent = e1_tangent * damp_longitudinal.value
                    + e1 * damp_longitudinal.tangent;
                dot_plus[state] =
                    dot_plus[state] * transverse + plus[state] * transverse_tangent;
                plus[state] *= transverse;
                dot_minus[state] =
                    dot_minus[state] * transverse + minus[state] * transverse_tangent;
                minus[state] *= transverse;
                dot_longitudinal[state] = dot_longitudinal[state] * relaxation
                    + longitudinal[state] * relaxation_tangent;
                longitudinal[state] *= relaxation;
            }
            longitudinal[0] += 1.0F - e1;
            dot_longitudinal[0] -= e1_tangent;

            const std::uint8_t action = primal.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift_real(plus, minus, states);
                shift_real(dot_plus, dot_minus, states);
            }
            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const float efficiency = -primal.inversion_efficiency[atom];
                    const float efficiency_tangent =
                        -buffers.inversion_efficiency[atom];
                    for (std::size_t state = 0; state < states; ++state) {
                        dot_longitudinal[state] =
                            dot_longitudinal[state] * efficiency
                            + longitudinal[state] * efficiency_tangent;
                        longitudinal[state] *= efficiency;
                    }
                } else {
                    const float alpha = view.flip[event] * b1;
                    const float alpha_tangent =
                        dot_flip[event] * b1 + view.flip[event] * dot_b1;
                    const float cosine = std::cos(alpha);
                    const float sine = std::sin(alpha);
                    const float cosine_half_sq = 0.5F * (1.0F + cosine);
                    const float sine_half_sq = 0.5F * (1.0F - cosine);
                    const float half_sine = 0.5F * sine;
                    const float cosine_tangent = -sine * alpha_tangent;
                    const float sine_tangent = cosine * alpha_tangent;
                    const float cosine_half_sq_tangent = -0.5F * sine * alpha_tangent;
                    const float sine_half_sq_tangent = 0.5F * sine * alpha_tangent;
                    const float half_sine_tangent = 0.5F * cosine * alpha_tangent;
                    for (std::size_t state = 0; state < states; ++state) {
                        const float p = plus[state];
                        const float m = minus[state];
                        const float z = longitudinal[state];
                        const float dp = dot_plus[state];
                        const float dm = dot_minus[state];
                        const float dz = dot_longitudinal[state];
                        dot_plus[state] = cosine_half_sq * dp + cosine_half_sq_tangent * p
                            + sine_half_sq * dm + sine_half_sq_tangent * m
                            - sine * dz - sine_tangent * z;
                        dot_minus[state] = sine_half_sq * dp + sine_half_sq_tangent * p
                            + cosine_half_sq * dm + cosine_half_sq_tangent * m
                            + sine * dz + sine_tangent * z;
                        dot_longitudinal[state] = half_sine * dp + half_sine_tangent * p
                            - half_sine * dm - half_sine_tangent * m
                            + cosine * dz + cosine_tangent * z;
                        plus[state] = cosine_half_sq * p + sine_half_sq * m - sine * z;
                        minus[state] = sine_half_sq * p + cosine_half_sq * m + sine * z;
                        longitudinal[state] = half_sine * p - half_sine * m + cosine * z;
                    }
                }
            } else if (primal.kind[event] == 2 && (action & RECORD) != 0) {
                const std::int64_t index =
                    view.output_base + primal.output_index[event];
                primal.output_real[index] = 0.0F;
                primal.output_imag[index] = dot_m0 * plus[0] + m0 * dot_plus[0];
            }
            if ((action & POST_SHIFT) != 0) {
                shift_real(plus, minus, states);
                shift_real(dot_plus, dot_minus, states);
            }
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(plus.begin(), plus.end(), 0.0F);
                std::fill(minus.begin(), minus.end(), 0.0F);
                std::fill(dot_plus.begin(), dot_plus.end(), 0.0F);
                std::fill(dot_minus.begin(), dot_minus.end(), 0.0F);
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_real(plus, minus, states);
                shift_real(dot_plus, dot_minus, states);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Train-lane vectorization of the real-subspace kernels.
//
// A block of REAL_LANES trains shares one atom, so every per-atom quantity is
// uniform across lanes while duration and flip vary with the train. States live
// as [state][lane] with the lane axis innermost and contiguous, which turns each
// state loop into packed arithmetic over independent trains. The lane axis
// carries no reduction: the gradient sums that run across states in the scalar
// kernels become one running total per lane, so nothing forces a horizontal add
// inside the loop.
//
// A partial block repeats its first train into the unused lanes so the
// arithmetic stays uniform; those lanes are not written back.
// ---------------------------------------------------------------------------

constexpr std::size_t REAL_LANES = 8;

// Setting TORCHSIM_REAL_SCALAR=1 routes the real-subspace path through the
// one-train-at-a-time kernels instead. They compute the same thing, so this
// selects between two implementations of one contract: useful for timing one
// against the other, and as a way out if a host miscompiles the lane arithmetic.
inline bool lane_kernels_enabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("TORCHSIM_REAL_SCALAR");
        return value == nullptr || value[0] != '1';
    }();
    return enabled;
}

// A lane loop has a constant trip count, so the compiler unrolls it and then
// declines to re-form the pieces into packed arithmetic. Spelling the width out
// as a vector type settles it, and costs nothing where the extension is absent.
#if defined(__GNUC__) || defined(__clang__)
using LaneVector = float __attribute__((vector_size(sizeof(float) * REAL_LANES)));
#else
struct LaneVector {
    float lane[REAL_LANES];
};

inline LaneVector operator+(const LaneVector left, const LaneVector right) {
    LaneVector out{};
    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
        out.lane[lane] = left.lane[lane] + right.lane[lane];
    }
    return out;
}

inline LaneVector operator-(const LaneVector left, const LaneVector right) {
    LaneVector out{};
    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
        out.lane[lane] = left.lane[lane] - right.lane[lane];
    }
    return out;
}

inline LaneVector operator*(const LaneVector left, const LaneVector right) {
    LaneVector out{};
    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
        out.lane[lane] = left.lane[lane] * right.lane[lane];
    }
    return out;
}

inline LaneVector operator-(const LaneVector value) {
    LaneVector out{};
    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
        out.lane[lane] = -value.lane[lane];
    }
    return out;
}
#endif

// memcpy rather than a cast: the state planes carry only float alignment, and
// this is the spelling every compiler turns into a single unaligned move.
inline LaneVector lane_load(const float* const source) {
    LaneVector out;
    std::memcpy(&out, source, sizeof(out));
    return out;
}

inline void lane_store(float* const target, const LaneVector value) {
    std::memcpy(target, &value, sizeof(value));
}

inline LaneVector lane_splat(const float value) {
    float buffer[REAL_LANES];
    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
        buffer[lane] = value;
    }
    return lane_load(buffer);
}

struct LaneView {
    std::int64_t atom;
    std::int64_t train_begin;
    std::int64_t active;
};

inline std::int64_t lane_blocks(const std::int64_t train_count) {
    const std::int64_t lanes = static_cast<std::int64_t>(REAL_LANES);
    return (train_count + lanes - 1) / lanes;
}

inline LaneView lane_view(const Buffers& buffers, const std::int64_t work) {
    const std::int64_t atom = work % buffers.atom_count;
    const std::int64_t begin =
        (work / buffers.atom_count) * static_cast<std::int64_t>(REAL_LANES);
    return LaneView{
        atom,
        begin,
        std::min<std::int64_t>(
            static_cast<std::int64_t>(REAL_LANES), buffers.train_count - begin
        ),
    };
}

// One event's value for every lane, repeating the block's first train into the
// inactive lanes of a partial block.
inline void gather_lanes(
    const float* buffer,
    const LaneView& view,
    const std::int64_t event,
    const std::int64_t event_count,
    float* out
) {
    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
        const std::int64_t offset =
            static_cast<std::int64_t>(lane) < view.active
                ? static_cast<std::int64_t>(lane)
                : 0;
        out[lane] = buffer[(view.train_begin + offset) * event_count + event];
    }
}

inline void shift_real_lanes(float* plus, float* minus, const std::size_t states) {
    for (std::size_t state = 0; state + 1 < states; ++state) {
        lane_store(
            minus + state * REAL_LANES, lane_load(minus + (state + 1) * REAL_LANES)
        );
    }
    lane_store(minus + (states - 1) * REAL_LANES, lane_splat(0.0F));
    for (std::size_t state = states - 1; state > 0; --state) {
        lane_store(
            plus + state * REAL_LANES, lane_load(plus + (state - 1) * REAL_LANES)
        );
    }
    lane_store(plus, -lane_load(minus));
}

void simulate_real_jvp_lane_range(
    const JvpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count
) {
    const Buffers& primal = buffers.primal;
    const std::size_t states = static_cast<std::size_t>(state_count);
    const std::size_t width = states * REAL_LANES;
    std::vector<float> storage(6U * width);
    float* const __restrict__ plus = storage.data();
    float* const __restrict__ minus = plus + width;
    float* const __restrict__ longitudinal = minus + width;
    float* const __restrict__ dot_plus = longitudinal + width;
    float* const __restrict__ dot_minus = dot_plus + width;
    float* const __restrict__ dot_longitudinal = dot_minus + width;

    float dt[REAL_LANES], dt_dot[REAL_LANES];
    float flip[REAL_LANES], flip_dot[REAL_LANES];
    float e1[REAL_LANES], e2[REAL_LANES];
    float e1_dot[REAL_LANES], e2_dot[REAL_LANES];
    float cosine[REAL_LANES], sine[REAL_LANES], alpha_dot[REAL_LANES];
    const LaneVector half = lane_splat(0.5F);
    const LaneVector one = lane_splat(1.0F);

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const LaneView view = lane_view(primal, work);
        const std::int64_t atom = view.atom;
        std::fill(storage.begin(), storage.end(), 0.0F);
        for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
            longitudinal[lane] = 1.0F;
        }

        const float t1 = primal.t1[atom];
        const float t2 = primal.t2[atom];
        const float r1 = 1000.0F / t1;
        const float r2 = 1000.0F / t2;
        const float b1 = primal.b1[atom];
        const float m0 = primal.m0[atom];
        const float dot_b1 = buffers.b1[atom];
        const float dot_m0 = buffers.m0[atom];
        const float t1_scale = 1000.0F * buffers.t1[atom] / (t1 * t1);
        const float t2_scale = 1000.0F * buffers.t2[atom] / (t2 * t2);

        for (std::int64_t event = 0; event < event_count; ++event) {
            gather_lanes(primal.duration, view, event, event_count, dt);
            gather_lanes(buffers.duration, view, event, event_count, dt_dot);
            for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
                e1[lane] = std::exp(-r1 * dt[lane]);
                e2[lane] = std::exp(-r2 * dt[lane]);
                e1_dot[lane] = e1[lane] * (dt[lane] * t1_scale - r1 * dt_dot[lane]);
                e2_dot[lane] = e2[lane] * (dt[lane] * t2_scale - r2 * dt_dot[lane]);
            }
            const LaneVector v_e1 = lane_load(e1);
            const LaneVector v_e2 = lane_load(e2);
            const LaneVector v_e1_dot = lane_load(e1_dot);
            const LaneVector v_e2_dot = lane_load(e2_dot);
            for (std::size_t state = 0; state < states; ++state) {
                const std::size_t base = state * REAL_LANES;
                const LaneVector p = lane_load(plus + base);
                const LaneVector m = lane_load(minus + base);
                const LaneVector z = lane_load(longitudinal + base);
                lane_store(
                    dot_plus + base,
                    lane_load(dot_plus + base) * v_e2 + p * v_e2_dot
                );
                lane_store(plus + base, p * v_e2);
                lane_store(
                    dot_minus + base,
                    lane_load(dot_minus + base) * v_e2 + m * v_e2_dot
                );
                lane_store(minus + base, m * v_e2);
                lane_store(
                    dot_longitudinal + base,
                    lane_load(dot_longitudinal + base) * v_e1 + z * v_e1_dot
                );
                lane_store(longitudinal + base, z * v_e1);
            }
            lane_store(longitudinal, lane_load(longitudinal) + (one - v_e1));
            lane_store(dot_longitudinal, lane_load(dot_longitudinal) - v_e1_dot);

            const std::uint8_t action = primal.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift_real_lanes(plus, minus, states);
                shift_real_lanes(dot_plus, dot_minus, states);
            }
            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const float efficiency = -primal.inversion_efficiency[atom];
                    const float efficiency_dot = -buffers.inversion_efficiency[atom];
                    for (std::size_t slot = 0; slot < width; ++slot) {
                        dot_longitudinal[slot] = dot_longitudinal[slot] * efficiency
                            + longitudinal[slot] * efficiency_dot;
                        longitudinal[slot] *= efficiency;
                    }
                } else {
                    gather_lanes(primal.flip, view, event, event_count, flip);
                    gather_lanes(buffers.flip, view, event, event_count, flip_dot);
                    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
                        const float alpha = flip[lane] * b1;
                        cosine[lane] = std::cos(alpha);
                        sine[lane] = std::sin(alpha);
                        alpha_dot[lane] = flip_dot[lane] * b1 + flip[lane] * dot_b1;
                    }
                    const LaneVector c = lane_load(cosine);
                    const LaneVector s = lane_load(sine);
                    const LaneVector rate = lane_load(alpha_dot);
                    const LaneVector chs = half * (one + c);
                    const LaneVector shs = half * (one - c);
                    const LaneVector hs = half * s;
                    const LaneVector c_dot = -(s * rate);
                    const LaneVector s_dot = c * rate;
                    const LaneVector chs_dot = half * c_dot;
                    const LaneVector shs_dot = -(half * c_dot);
                    const LaneVector hs_dot = half * s_dot;
                    for (std::size_t state = 0; state < states; ++state) {
                        const std::size_t base = state * REAL_LANES;
                        const LaneVector p = lane_load(plus + base);
                        const LaneVector m = lane_load(minus + base);
                        const LaneVector z = lane_load(longitudinal + base);
                        const LaneVector dp = lane_load(dot_plus + base);
                        const LaneVector dm = lane_load(dot_minus + base);
                        const LaneVector dz = lane_load(dot_longitudinal + base);
                        lane_store(
                            dot_plus + base,
                            chs * dp + chs_dot * p + shs * dm + shs_dot * m
                                - s * dz - s_dot * z
                        );
                        lane_store(
                            dot_minus + base,
                            shs * dp + shs_dot * p + chs * dm + chs_dot * m
                                + s * dz + s_dot * z
                        );
                        lane_store(
                            dot_longitudinal + base,
                            hs * dp + hs_dot * p - hs * dm - hs_dot * m
                                + c * dz + c_dot * z
                        );
                        lane_store(plus + base, chs * p + shs * m - s * z);
                        lane_store(minus + base, shs * p + chs * m + s * z);
                        lane_store(longitudinal + base, hs * p - hs * m + c * z);
                    }
                }
            } else if (primal.kind[event] == 2 && (action & RECORD) != 0) {
                const std::int64_t offset = primal.output_index[event];
                for (std::int64_t lane = 0; lane < view.active; ++lane) {
                    const std::int64_t index =
                        ((view.train_begin + lane) * primal.atom_count + atom)
                            * output_count
                        + offset;
                    primal.output_real[index] = 0.0F;
                    primal.output_imag[index] =
                        dot_m0 * plus[lane] + m0 * dot_plus[lane];
                }
            }
            if ((action & POST_SHIFT) != 0) {
                shift_real_lanes(plus, minus, states);
                shift_real_lanes(dot_plus, dot_minus, states);
            }
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(plus, plus + width, 0.0F);
                std::fill(minus, minus + width, 0.0F);
                std::fill(dot_plus, dot_plus + width, 0.0F);
                std::fill(dot_minus, dot_minus + width, 0.0F);
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_real_lanes(plus, minus, states);
                shift_real_lanes(dot_plus, dot_minus, states);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Train-axis vectorization.
//
// A block of trains shares one atom, so every per-atom quantity is uniform
// across lanes while flip, phase and duration vary with the train. States are
// stored [state][lane] with the lane axis contiguous and innermost, which is
// what lets the state loops below compile to packed arithmetic. Transcendentals
// stay scalar: they are evaluated once per event, against a state loop that
// runs `state_count` times.
//
// Inactive lanes in a partial block carry a copy of the first train, so the
// arithmetic is uniform; their outputs are simply not written back.
// ---------------------------------------------------------------------------

constexpr std::size_t LANES = 8;

struct LaneStates {
    std::vector<float> storage;
    std::size_t states;
    float* plane[6];

    explicit LaneStates(const std::size_t state_count)
        : storage(6U * state_count * LANES, 0.0F), states(state_count) {
        for (std::size_t index = 0; index < 6U; ++index) {
            plane[index] = storage.data() + index * state_count * LANES;
        }
    }

    void reset() { std::fill(storage.begin(), storage.end(), 0.0F); }

    float* fplus_real() { return plane[0]; }
    float* fplus_imag() { return plane[1]; }
    float* fminus_real() { return plane[2]; }
    float* fminus_imag() { return plane[3]; }
    float* longitudinal_real() { return plane[4]; }
    float* longitudinal_imag() { return plane[5]; }
};

inline void shift_lanes(
    float* fplus_real,
    float* fplus_imag,
    float* fminus_real,
    float* fminus_imag,
    const std::size_t states
) {
    for (std::size_t state = 0; state + 1 < states; ++state) {
        const std::size_t destination = state * LANES;
        const std::size_t source = (state + 1) * LANES;
        for (std::size_t lane = 0; lane < LANES; ++lane) {
            fminus_real[destination + lane] = fminus_real[source + lane];
            fminus_imag[destination + lane] = fminus_imag[source + lane];
        }
    }
    for (std::size_t lane = 0; lane < LANES; ++lane) {
        fminus_real[(states - 1) * LANES + lane] = 0.0F;
        fminus_imag[(states - 1) * LANES + lane] = 0.0F;
    }
    for (std::size_t state = states - 1; state > 0; --state) {
        const std::size_t destination = state * LANES;
        const std::size_t source = (state - 1) * LANES;
        for (std::size_t lane = 0; lane < LANES; ++lane) {
            fplus_real[destination + lane] = fplus_real[source + lane];
            fplus_imag[destination + lane] = fplus_imag[source + lane];
        }
    }
    // fplus[0] = conj(fminus[0]), reading fminus after its own shift.
    for (std::size_t lane = 0; lane < LANES; ++lane) {
        fplus_real[lane] = fminus_real[lane];
        fplus_imag[lane] = -fminus_imag[lane];
    }
}

// Nine complex coefficients of the RF rotation, one set per lane.
struct RotationLanes {
    float real[9][LANES];
    float imag[9][LANES];
};

inline void build_rotation(
    RotationLanes& rotation,
    const float* alpha,
    const float* phi
) {
    for (std::size_t lane = 0; lane < LANES; ++lane) {
        const float cosine = std::cos(alpha[lane]);
        const float sine = std::sin(alpha[lane]);
        const float cosine_half_sq = 0.5F * (1.0F + cosine);
        const float sine_half_sq = 0.5F * (1.0F - cosine);
        const float phase_real = std::cos(phi[lane]);
        const float phase_imag = std::sin(phi[lane]);
        const float two_real = phase_real * phase_real - phase_imag * phase_imag;
        const float two_imag = 2.0F * phase_real * phase_imag;

        rotation.real[0][lane] = cosine_half_sq;
        rotation.imag[0][lane] = 0.0F;
        rotation.real[1][lane] = sine_half_sq * two_real;
        rotation.imag[1][lane] = sine_half_sq * two_imag;
        // (-i sine) * phase_one
        rotation.real[2][lane] = sine * phase_imag;
        rotation.imag[2][lane] = -sine * phase_real;
        rotation.real[3][lane] = rotation.real[1][lane];
        rotation.imag[3][lane] = -rotation.imag[1][lane];
        rotation.real[4][lane] = cosine_half_sq;
        rotation.imag[4][lane] = 0.0F;
        // (i sine) * conj(phase_one)
        rotation.real[5][lane] = sine * phase_imag;
        rotation.imag[5][lane] = sine * phase_real;
        // (-i sine / 2) * conj(phase_one)
        rotation.real[6][lane] = -0.5F * sine * phase_imag;
        rotation.imag[6][lane] = -0.5F * sine * phase_real;
        // (i sine / 2) * phase_one
        rotation.real[7][lane] = -0.5F * sine * phase_imag;
        rotation.imag[7][lane] = 0.5F * sine * phase_real;
        rotation.real[8][lane] = cosine;
        rotation.imag[8][lane] = 0.0F;
    }
}

inline void rotate_lanes(
    LaneStates& lane_states,
    const RotationLanes& rotation,
    const std::size_t states
) {
    float* fplus_real = lane_states.fplus_real();
    float* fplus_imag = lane_states.fplus_imag();
    float* fminus_real = lane_states.fminus_real();
    float* fminus_imag = lane_states.fminus_imag();
    float* longitudinal_real = lane_states.longitudinal_real();
    float* longitudinal_imag = lane_states.longitudinal_imag();

    for (std::size_t state = 0; state < states; ++state) {
        const std::size_t base = state * LANES;
        for (std::size_t lane = 0; lane < LANES; ++lane) {
            const std::size_t slot = base + lane;
            const float fp_re = fplus_real[slot];
            const float fp_im = fplus_imag[slot];
            const float fm_re = fminus_real[slot];
            const float fm_im = fminus_imag[slot];
            const float z_re = longitudinal_real[slot];
            const float z_im = longitudinal_imag[slot];

            fplus_real[slot] = rotation.real[0][lane] * fp_re
                - rotation.imag[0][lane] * fp_im
                + rotation.real[1][lane] * fm_re - rotation.imag[1][lane] * fm_im
                + rotation.real[2][lane] * z_re - rotation.imag[2][lane] * z_im;
            fplus_imag[slot] = rotation.real[0][lane] * fp_im
                + rotation.imag[0][lane] * fp_re
                + rotation.real[1][lane] * fm_im + rotation.imag[1][lane] * fm_re
                + rotation.real[2][lane] * z_im + rotation.imag[2][lane] * z_re;

            fminus_real[slot] = rotation.real[3][lane] * fp_re
                - rotation.imag[3][lane] * fp_im
                + rotation.real[4][lane] * fm_re - rotation.imag[4][lane] * fm_im
                + rotation.real[5][lane] * z_re - rotation.imag[5][lane] * z_im;
            fminus_imag[slot] = rotation.real[3][lane] * fp_im
                + rotation.imag[3][lane] * fp_re
                + rotation.real[4][lane] * fm_im + rotation.imag[4][lane] * fm_re
                + rotation.real[5][lane] * z_im + rotation.imag[5][lane] * z_re;

            longitudinal_real[slot] = rotation.real[6][lane] * fp_re
                - rotation.imag[6][lane] * fp_im
                + rotation.real[7][lane] * fm_re - rotation.imag[7][lane] * fm_im
                + rotation.real[8][lane] * z_re - rotation.imag[8][lane] * z_im;
            longitudinal_imag[slot] = rotation.real[6][lane] * fp_im
                + rotation.imag[6][lane] * fp_re
                + rotation.real[7][lane] * fm_im + rotation.imag[7][lane] * fm_re
                + rotation.real[8][lane] * z_im + rotation.imag[8][lane] * z_re;
        }
    }
}

// Blocks of trains for one atom; ``work`` indexes the (atom, block) product.
void simulate_lane_range(
    const Buffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count
) {
    const std::size_t states = static_cast<std::size_t>(state_count);
    const std::int64_t blocks_per_atom =
        (buffers.train_count + static_cast<std::int64_t>(LANES) - 1)
        / static_cast<std::int64_t>(LANES);

    LaneStates lane_states(states);
    RotationLanes rotation{};
    alignas(64) float duration[LANES];
    alignas(64) float recovery[LANES];
    alignas(64) float off_real[LANES];
    alignas(64) float off_imag[LANES];
    alignas(64) float alpha[LANES];
    alignas(64) float phi[LANES];
    alignas(64) float demodulation_real[LANES];
    alignas(64) float demodulation_imag[LANES];
    std::int64_t train_of[LANES];

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const std::int64_t atom = work / blocks_per_atom;
        const std::int64_t block = work % blocks_per_atom;
        const std::int64_t train_begin = block * static_cast<std::int64_t>(LANES);
        const std::int64_t active =
            std::min<std::int64_t>(LANES, buffers.train_count - train_begin);
        for (std::size_t lane = 0; lane < LANES; ++lane) {
            const std::int64_t offset = std::min<std::int64_t>(
                static_cast<std::int64_t>(lane), active - 1
            );
            train_of[lane] = train_begin + offset;
        }

        lane_states.reset();
        float* fplus_real = lane_states.fplus_real();
        float* fplus_imag = lane_states.fplus_imag();
        float* fminus_real = lane_states.fminus_real();
        float* fminus_imag = lane_states.fminus_imag();
        float* longitudinal_real = lane_states.longitudinal_real();
        float* longitudinal_imag = lane_states.longitudinal_imag();
        for (std::size_t lane = 0; lane < LANES; ++lane) {
            longitudinal_real[lane] = 1.0F;
        }

        const float r1 = 1000.0F / buffers.t1[atom];
        const float r2 = 1000.0F / buffers.t2[atom];
        const float b0 = buffers.b0[atom];
        const float b1 = buffers.b1[atom];
        const float b1_phase = buffers.b1_phase[atom];
        const float m0 = buffers.m0[atom];
        const float efficiency = -buffers.inversion_efficiency[atom];

        for (std::int64_t event = 0; event < event_count; ++event) {
            for (std::size_t lane = 0; lane < LANES; ++lane) {
                duration[lane] =
                    buffers.duration[train_of[lane] * event_count + event];
            }
            // Transcendentals first, once per lane; the state loops that follow
            // keep the lane axis innermost and contiguous so they vectorize.
            for (std::size_t lane = 0; lane < LANES; ++lane) {
                const float dt = duration[lane];
                const float angle = -2.0F * PI * b0 * dt;
                const float e2 = std::exp(-r2 * dt);
                recovery[lane] = std::exp(-r1 * dt);
                off_real[lane] = e2 * std::cos(angle);
                off_imag[lane] = e2 * std::sin(angle);
            }
            for (std::size_t state = 0; state < states; ++state) {
                const std::size_t base = state * LANES;
                for (std::size_t lane = 0; lane < LANES; ++lane) {
                    const std::size_t slot = base + lane;
                    const float fp_re = fplus_real[slot];
                    const float fp_im = fplus_imag[slot];
                    fplus_real[slot] = fp_re * off_real[lane] - fp_im * off_imag[lane];
                    fplus_imag[slot] = fp_re * off_imag[lane] + fp_im * off_real[lane];
                    const float fm_re = fminus_real[slot];
                    const float fm_im = fminus_imag[slot];
                    fminus_real[slot] = fm_re * off_real[lane] + fm_im * off_imag[lane];
                    fminus_imag[slot] = -fm_re * off_imag[lane] + fm_im * off_real[lane];
                    longitudinal_real[slot] *= recovery[lane];
                    longitudinal_imag[slot] *= recovery[lane];
                }
            }
            for (std::size_t lane = 0; lane < LANES; ++lane) {
                longitudinal_real[lane] += 1.0F - recovery[lane];
            }

            const std::uint8_t action = buffers.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift_lanes(
                    fplus_real, fplus_imag, fminus_real, fminus_imag, states
                );
            }
            if (buffers.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    for (std::size_t slot = 0; slot < states * LANES; ++slot) {
                        longitudinal_real[slot] *= efficiency;
                        longitudinal_imag[slot] *= efficiency;
                    }
                } else {
                    for (std::size_t lane = 0; lane < LANES; ++lane) {
                        const std::int64_t index =
                            train_of[lane] * event_count + event;
                        alpha[lane] = buffers.flip[index] * b1;
                        phi[lane] = buffers.phase[index] + b1_phase;
                    }
                    build_rotation(rotation, alpha, phi);
                    rotate_lanes(lane_states, rotation, states);
                }
            } else if (buffers.kind[event] == 2 && (action & RECORD) != 0) {
                const std::int64_t output = buffers.output_index[event];
                for (std::size_t lane = 0; lane < LANES; ++lane) {
                    const float angle =
                        -buffers.phase[train_of[lane] * event_count + event];
                    demodulation_real[lane] = std::cos(angle);
                    demodulation_imag[lane] = std::sin(angle);
                }
                for (std::int64_t lane = 0; lane < active; ++lane) {
                    const float fp_re = fplus_real[lane];
                    const float fp_im = fplus_imag[lane];
                    const std::int64_t index =
                        ((train_begin + lane) * buffers.atom_count + atom)
                            * output_count
                        + output;
                    buffers.output_real[index] = m0
                        * (fp_re * demodulation_real[lane]
                           - fp_im * demodulation_imag[lane]);
                    buffers.output_imag[index] = m0
                        * (fp_re * demodulation_imag[lane]
                           + fp_im * demodulation_real[lane]);
                }
            }
            if ((action & POST_SHIFT) != 0) {
                shift_lanes(
                    fplus_real, fplus_imag, fminus_real, fminus_imag, states
                );
            }
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(fplus_real, fplus_real + states * LANES, 0.0F);
                std::fill(fplus_imag, fplus_imag + states * LANES, 0.0F);
                std::fill(fminus_real, fminus_real + states * LANES, 0.0F);
                std::fill(fminus_imag, fminus_imag + states * LANES, 0.0F);
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_lanes(
                    fplus_real, fplus_imag, fminus_real, fminus_imag, states
                );
            }
        }
    }
}

template <bool SHIMMED, bool PROFILED>
void simulate_range(
    const Buffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count
) {
    const std::size_t states = static_cast<std::size_t>(state_count);
    std::vector<Complex> fplus(states);
    std::vector<Complex> fminus(states);
    std::vector<Complex> longitudinal(states);
    Damping<float> damping(states);

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const TrainView view = train_view(buffers, work, event_count, output_count);
        const std::int64_t atom = view.atom;
        const std::int64_t location = slice_row<PROFILED>(buffers, atom);
        std::fill(fplus.begin(), fplus.end(), Complex{});
        std::fill(fminus.begin(), fminus.end(), Complex{});
        std::fill(longitudinal.begin(), longitudinal.end(), Complex{});
        longitudinal[0] = Complex(1.0F, 0.0F);

        const float r1 = 1000.0F / buffers.t1[atom];
        const float r2 = 1000.0F / buffers.t2[atom];
        const float damping_rate = buffers.diffusion[atom];
        const float flow_rate = buffers.velocity[atom] * buffers.flow_scale;
        const float washout_rate =
            std::fabs(buffers.velocity[atom]) * buffers.washout_scale;
        for (std::int64_t event = 0; event < event_count; ++event) {
            const float dt = view.duration[event];
            damping.set(damping_rate, dt);
            const float wout = washout_out(washout_rate, dt);
            const float e1 = std::exp(-r1 * dt) * wout;
            const float e2 = std::exp(-r2 * dt) * wout;
            const float off_angle = -2.0F * PI * buffers.b0[atom] * dt;
            for (std::int64_t state = 0; state < state_count; ++state) {
                const std::size_t index = static_cast<std::size_t>(state);
                const float damp_transverse = damping.transverse[index];
                float turn_longitudinal = 0.0F;
                float turn_transverse = 0.0F;
                flow_turn(flow_rate, dt, index, turn_longitudinal, turn_transverse);
                // Flow winds the transverse states through the same rotation
                // off-resonance does, so the two phases add before either is
                // taken; the longitudinal states carry a phase of their own.
                const Complex transverse =
                    std::polar(e2 * damp_transverse, off_angle + turn_transverse);
                fplus[index] *= transverse;
                fminus[index] *= std::conj(transverse);
                longitudinal[index] *= std::polar(
                    e1 * damping.longitudinal[index], turn_longitudinal
                );
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
                    const std::int64_t transmit =
                        transmit_row<SHIMMED>(buffers, event, atom);
                    const float theta = view.flip[event] * buffers.b1[transmit];
                    const float phi =
                        view.phase[event] + buffers.b1_phase[transmit];
                    if constexpr (PROFILED) {
                        Complex a{};
                        Complex b{};
                        profile_pair(buffers, location, theta, a, b);
                        // The table is built at zero RF phase, which turns the
                        // rotation axis and so reaches ``b`` alone.
                        rotate_spinor(
                            fplus, fminus, longitudinal, a,
                            b * std::polar(1.0F, -phi)
                        );
                    } else {
                        rotate(fplus, fminus, longitudinal, theta, phi);
                    }
                }
            } else if (buffers.kind[event] == 2 && (action & RECORD) != 0) {
                const std::int64_t output = buffers.output_index[event];
                const Complex demodulation = std::polar(1.0F, -view.phase[event]);
                const Complex signal = buffers.m0[atom] * fplus[0] * demodulation;
                const std::int64_t index = view.output_base + output;
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

// ---------------------------------------------------------------------------
// Reverse mode.
//
// The forward pass is a chain of linear maps on the state s = (F+, F-, Z),
// with an affine term from longitudinal recovery. Writing s_k = A_k s_{k-1} +
// b_k, the adjoint recursion is lambda_{k-1} = A_k^H lambda_k, and a real
// parameter theta collects
//
//     dL/dtheta += Re( conj(lambda_k) . (dA_k/dtheta) s_{k-1} + ... ).
//
// For a real loss, Torch hands back grad_output = dL/dRe(y) + i dL/dIm(y),
// which is exactly the adjoint convention used here: the adjoint of v = c u is
// u_bar = conj(c) v_bar, and the adjoint of conjugation is conjugation.
// ---------------------------------------------------------------------------

struct VjpBuffers {
    Buffers primal;
    const float* grad_output_real;
    const float* grad_output_imag;
    float* grad_t1;
    float* grad_t2;
    float* grad_m0;
    float* grad_b1;
    float* grad_b1_phase;
    float* grad_b0;
    float* grad_inversion_efficiency;
    float* grad_diffusion;
    float* grad_velocity;
    // Per-event gradients are shared by every atom. Workers accumulate into
    // private buffers which are reduced in a fixed order, so the result does
    // not depend on thread scheduling.
    float* grad_flip;
    float* grad_phase;
    float* grad_duration;
};

using State = std::vector<Complex>;

inline void shift_adjoint(State& fplus_bar, State& fminus_bar) {
    const std::size_t count = fplus_bar.size();
    // fplus[0] = conj(fminus[1]) couples the two branches; capture it before
    // the in-place shifts overwrite the source.
    const Complex carry = std::conj(fplus_bar[0]);
    for (std::size_t state = 0; state + 1 < count; ++state) {
        fplus_bar[state] = fplus_bar[state + 1];
    }
    fplus_bar[count - 1] = Complex{};
    for (std::size_t state = count - 1; state > 0; --state) {
        fminus_bar[state] = fminus_bar[state - 1];
    }
    fminus_bar[0] = Complex{};
    if (count > 1) {
        fminus_bar[1] += carry;
    }
}

inline void rotate_adjoint(
    const State& fplus_in,
    const State& fminus_in,
    const State& longitudinal_in,
    State& fplus_bar,
    State& fminus_bar,
    State& longitudinal_bar,
    const float alpha,
    const float phi,
    float& grad_alpha,
    float& grad_phi
) {
    const float cosine = std::cos(alpha);
    const float sine = std::sin(alpha);
    const Complex phase_one = std::polar(1.0F, phi);
    const Complex phase_two = phase_one * phase_one;
    const Complex phase_conj = std::conj(phase_one);

    const Complex t00(0.5F * (1.0F + cosine), 0.0F);
    const Complex t01 = 0.5F * (1.0F - cosine) * phase_two;
    const Complex t02 = Complex(0.0F, -sine) * phase_one;
    const Complex t10 = std::conj(t01);
    const Complex t11 = t00;
    const Complex t12 = Complex(0.0F, sine) * phase_conj;
    const Complex t20 = Complex(0.0F, -0.5F * sine) * phase_conj;
    const Complex t21 = Complex(0.0F, 0.5F * sine) * phase_one;
    const Complex t22(cosine, 0.0F);

    // d/dalpha of each coefficient.
    const Complex d00(-0.5F * sine, 0.0F);
    const Complex d01 = 0.5F * sine * phase_two;
    const Complex d02 = Complex(0.0F, -cosine) * phase_one;
    const Complex d10 = std::conj(d01);
    const Complex d11 = d00;
    const Complex d12 = Complex(0.0F, cosine) * phase_conj;
    const Complex d20 = Complex(0.0F, -0.5F * cosine) * phase_conj;
    const Complex d21 = Complex(0.0F, 0.5F * cosine) * phase_one;
    const Complex d22(-sine, 0.0F);

    // Every coefficient is of the form c * exp(i n phi), so d/dphi = i n t.
    const Complex imaginary(0.0F, 1.0F);
    float alpha_sum = 0.0F;
    float phi_sum = 0.0F;
    for (std::size_t state = 0; state < fplus_bar.size(); ++state) {
        const Complex a0 = fplus_bar[state];
        const Complex a1 = fminus_bar[state];
        const Complex a2 = longitudinal_bar[state];
        const Complex x0 = fplus_in[state];
        const Complex x1 = fminus_in[state];
        const Complex x2 = longitudinal_in[state];

        alpha_sum += std::real(
            std::conj(a0) * (d00 * x0 + d01 * x1 + d02 * x2)
            + std::conj(a1) * (d10 * x0 + d11 * x1 + d12 * x2)
            + std::conj(a2) * (d20 * x0 + d21 * x1 + d22 * x2)
        );
        phi_sum += std::real(
            std::conj(a0) * imaginary * (2.0F * t01 * x1 + t02 * x2)
            + std::conj(a1) * imaginary * (-2.0F * t10 * x0 - t12 * x2)
            + std::conj(a2) * imaginary * (-t20 * x0 + t21 * x1)
        );

        fplus_bar[state] =
            std::conj(t00) * a0 + std::conj(t10) * a1 + std::conj(t20) * a2;
        fminus_bar[state] =
            std::conj(t01) * a0 + std::conj(t11) * a1 + std::conj(t21) * a2;
        longitudinal_bar[state] =
            std::conj(t02) * a0 + std::conj(t12) * a1 + std::conj(t22) * a2;
    }
    grad_alpha += alpha_sum;
    grad_phi += phi_sum;
}

// The adjoint of the spinor rotation: the states back through the conjugate
// transpose, and the pair reached in closed form.
//
// Every entry of the matrix is a product of two factors drawn from the pair and
// its conjugate, so both Wirtinger halves of the pair's cotangent are linear in
// the outer product of the seed with the state the rotation acted on. That
// outer product is accumulated over the states first, so the pair is worked out
// once per pulse rather than once per state.
inline void rotate_adjoint_spinor(
    const State& fplus_in,
    const State& fminus_in,
    const State& longitudinal_in,
    State& fplus_bar,
    State& fminus_bar,
    State& longitudinal_bar,
    const Complex a,
    const Complex b,
    Complex& grad_a,
    Complex& grad_b
) {
    const Complex conj_a = std::conj(a);
    const Complex conj_b = std::conj(b);
    const Complex t00 = conj_a * conj_a;
    const Complex t01 = -conj_b * conj_b;
    const Complex t02 = -2.0F * std::conj(a * b);
    const Complex t10 = -b * b;
    const Complex t11 = a * a;
    const Complex t12 = -2.0F * a * b;
    const Complex t20 = conj_a * b;
    const Complex t21 = a * conj_b;
    const Complex t22(std::norm(a) - std::norm(b), 0.0F);

    Complex m[3][3]{};
    for (std::size_t state = 0; state < fplus_bar.size(); ++state) {
        const Complex a0 = fplus_bar[state];
        const Complex a1 = fminus_bar[state];
        const Complex a2 = longitudinal_bar[state];
        const Complex x0 = fplus_in[state];
        const Complex x1 = fminus_in[state];
        const Complex x2 = longitudinal_in[state];

        const Complex s0 = std::conj(a0);
        const Complex s1 = std::conj(a1);
        const Complex s2 = std::conj(a2);
        m[0][0] += s0 * x0;
        m[0][1] += s0 * x1;
        m[0][2] += s0 * x2;
        m[1][0] += s1 * x0;
        m[1][1] += s1 * x1;
        m[1][2] += s1 * x2;
        m[2][0] += s2 * x0;
        m[2][1] += s2 * x1;
        m[2][2] += s2 * x2;

        fplus_bar[state] =
            std::conj(t00) * a0 + std::conj(t10) * a1 + std::conj(t20) * a2;
        fminus_bar[state] =
            std::conj(t01) * a0 + std::conj(t11) * a1 + std::conj(t21) * a2;
        longitudinal_bar[state] =
            std::conj(t02) * a0 + std::conj(t12) * a1 + std::conj(t22) * a2;
    }

    const Complex holding_conj_a = 2.0F * a * m[1][1] - 2.0F * b * m[1][2]
        + conj_b * m[2][1] + conj_a * m[2][2];
    const Complex holding_a = 2.0F * conj_a * m[0][0] - 2.0F * conj_b * m[0][2]
        + b * m[2][0] + a * m[2][2];
    const Complex holding_conj_b = -2.0F * b * m[1][0] - 2.0F * a * m[1][2]
        + conj_a * m[2][0] - conj_b * m[2][2];
    const Complex holding_b = -2.0F * conj_b * m[0][1] - 2.0F * conj_a * m[0][2]
        + a * m[2][1] - b * m[2][2];
    grad_a += std::conj(holding_conj_a) + holding_a;
    grad_b += std::conj(holding_conj_b) + holding_b;
}

// ---------------------------------------------------------------------------
// Forward-over-reverse.
//
// The JVP computes ydot = J(theta) thetadot. Seeding its adjoint with sigma,
//
//     dL/dthetadot_i = Re<sigma, J_i(theta)>              (the first-order VJP)
//     dL/dtheta_i    = sum_j thetadot_j dV_i/dtheta_j     (its directional
//                                                          derivative)
//
// where V_i is that same first-order VJP. So running the adjoint above on dual
// numbers seeded with thetadot yields both halves at once: the value part is
// the tangent gradient, the tangent part is the primal gradient. No second
// derivative has to be written out by hand; dual arithmetic produces it.
// ---------------------------------------------------------------------------

using DualState = std::vector<DualComplex>;

inline void shift_adjoint(DualState& fplus_bar, DualState& fminus_bar) {
    const std::size_t count = fplus_bar.size();
    const DualComplex carry = conjugate(fplus_bar[0]);
    for (std::size_t state = 0; state + 1 < count; ++state) {
        fplus_bar[state] = fplus_bar[state + 1];
    }
    fplus_bar[count - 1] = DualComplex{};
    for (std::size_t state = count - 1; state > 0; --state) {
        fminus_bar[state] = fminus_bar[state - 1];
    }
    fminus_bar[0] = DualComplex{};
    if (count > 1) {
        fminus_bar[1] = fminus_bar[1] + carry;
    }
}

template <bool SHIMMED, bool PROFILED>
#if defined(__GNUC__) && !defined(__clang__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target_clones("default", "sse4.2", "avx2", "avx512f")))
#endif
void simulate_vjp_range(
    const VjpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count,
    float* grad_flip_local,
    float* grad_phase_local,
    float* grad_duration_local,
    // Per-atom accumulators laid out [row][atom], the rows given by
    // TissueLayout. Work items are split across the (atom, train) product, so
    // several threads reach the same atom and every train contributes to it.
    float* grad_tissue_local
) {
    const Buffers& primal = buffers.primal;
    const TissueLayout layout(primal.shim_count);
    const std::int64_t atoms = primal.atom_count;
    const std::size_t states = static_cast<std::size_t>(state_count);
    const std::size_t trajectory_stride = 3U * states;

    // State at the start of every event; intra-event intermediates are cheap
    // enough to replay from it during the reverse sweep.
    std::vector<Complex> trajectory(
        static_cast<std::size_t>(event_count) * trajectory_stride
    );
    State fplus(states);
    State fminus(states);
    State longitudinal(states);
    State fplus_bar(states);
    State fminus_bar(states);
    State longitudinal_bar(states);
    State fplus_relaxed(states);
    State fminus_relaxed(states);
    State longitudinal_relaxed(states);
    State fplus_shifted(states);
    State fminus_shifted(states);
    State longitudinal_pre(states);
    Damping<float> damping(states);

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const TrainView view = train_view(primal, work, event_count, output_count);
        const std::int64_t atom = view.atom;
        const std::int64_t location = slice_row<PROFILED>(primal, atom);
        float* const grad_flip_train = grad_flip_local + view.event_base;
        float* const grad_phase_train = grad_phase_local + view.event_base;
        float* const grad_duration_train = grad_duration_local + view.event_base;
        std::fill(fplus.begin(), fplus.end(), Complex{});
        std::fill(fminus.begin(), fminus.end(), Complex{});
        std::fill(longitudinal.begin(), longitudinal.end(), Complex{});
        longitudinal[0] = Complex(1.0F, 0.0F);

        const float t1 = primal.t1[atom];
        const float t2 = primal.t2[atom];
        const float r1 = 1000.0F / t1;
        const float r2 = 1000.0F / t2;
        const float b0 = primal.b0[atom];
        const float m0 = primal.m0[atom];
        // With one shim the transmit field is a property of the voxel and
        // lifts out of the event loop; with several it belongs to the shim a
        // pulse drives, and is read where the pulse is.
        const float b1 = primal.b1[atom];
        const float b1_phase = primal.b1_phase[atom];
        const float efficiency = primal.inversion_efficiency[atom];
        const float damping_rate = primal.diffusion[atom];
        const float velocity = primal.velocity[atom];
        const float flow_rate = velocity * primal.flow_scale;
        const float washout_rate = std::fabs(velocity) * primal.washout_scale;

        // ---- forward, recording the state entering each event ----
        for (std::int64_t event = 0; event < event_count; ++event) {
            Complex* slot = trajectory.data()
                + static_cast<std::size_t>(event) * trajectory_stride;
            std::copy(fplus.begin(), fplus.end(), slot);
            std::copy(fminus.begin(), fminus.end(), slot + states);
            std::copy(longitudinal.begin(), longitudinal.end(), slot + 2U * states);

            const float dt = view.duration[event];
            damping.set(damping_rate, dt);
            const float wout = washout_out(washout_rate, dt);
            const float e1 = std::exp(-r1 * dt) * wout;
            const float e2 = std::exp(-r2 * dt) * wout;
            const float off_angle = -2.0F * PI * b0 * dt;
            for (std::size_t state = 0; state < states; ++state) {
                float turn_longitudinal = 0.0F;
                float turn_transverse = 0.0F;
                flow_turn(flow_rate, dt, state, turn_longitudinal, turn_transverse);
                const Complex transverse = std::polar(
                    e2 * damping.transverse[state], off_angle + turn_transverse
                );
                fplus[state] *= transverse;
                fminus[state] *= std::conj(transverse);
                longitudinal[state] *= std::polar(
                    e1 * damping.longitudinal[state], turn_longitudinal
                );
            }
            longitudinal[0] += Complex(1.0F - e1, 0.0F);

            const std::uint8_t action = primal.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift(fplus, fminus);
            }
            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    for (Complex& value : longitudinal) {
                        value *= -efficiency;
                    }
                } else {
                    const std::int64_t transmit =
                        transmit_row<SHIMMED>(primal, event, atom);
                    const float alpha =
                        view.flip[event] * (SHIMMED ? primal.b1[transmit] : b1);
                    const float phi = view.phase[event]
                        + (SHIMMED ? primal.b1_phase[transmit] : b1_phase);
                    if constexpr (PROFILED) {
                        Complex pair_a{};
                        Complex pair_b{};
                        profile_pair(primal, location, alpha, pair_a, pair_b);
                        rotate_spinor(
                            fplus, fminus, longitudinal, pair_a,
                            pair_b * std::polar(1.0F, -phi)
                        );
                    } else {
                        rotate(fplus, fminus, longitudinal, alpha, phi);
                    }
                }
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

        // ---- reverse ----
        std::fill(fplus_bar.begin(), fplus_bar.end(), Complex{});
        std::fill(fminus_bar.begin(), fminus_bar.end(), Complex{});
        std::fill(longitudinal_bar.begin(), longitudinal_bar.end(), Complex{});
        float grad_t1 = 0.0F;
        float grad_t2 = 0.0F;
        float grad_m0 = 0.0F;
        float grad_b1 = 0.0F;
        float grad_b1_phase = 0.0F;
        // Transmit gradients are summed per shim: the running pair is flushed
        // to its row whenever the walk back reaches a pulse on a different
        // one. A single-shim sequence never changes row and flushes once.
        std::int64_t held = 0;
        float grad_b0 = 0.0F;
        float grad_efficiency = 0.0F;
        float grad_damping = 0.0F;
        float grad_flow = 0.0F;
        float grad_washout = 0.0F;

        for (std::int64_t event = event_count - 1; event >= 0; --event) {
            const Complex* slot = trajectory.data()
                + static_cast<std::size_t>(event) * trajectory_stride;
            std::copy(slot, slot + states, fplus.begin());
            std::copy(slot + states, slot + 2U * states, fminus.begin());
            std::copy(slot + 2U * states, slot + 3U * states, longitudinal.begin());

            const float dt = view.duration[event];
            damping.set(damping_rate, dt);
            const float dry1 = std::exp(-r1 * dt);
            const float dry2 = std::exp(-r2 * dt);
            const float wout = washout_out(washout_rate, dt);
            const float e1 = dry1 * wout;
            const float e2 = dry2 * wout;
            const float angle = -2.0F * PI * b0 * dt;
            const Complex phase = std::polar(1.0F, angle);
            const Complex off_resonance = e2 * phase;
            const Complex conjugate_off = std::conj(off_resonance);
            const std::uint8_t action = primal.action[event];

            // replay this event to recover the intra-event states
            for (std::size_t state = 0; state < states; ++state) {
                float turn_longitudinal = 0.0F;
                float turn_transverse = 0.0F;
                flow_turn(flow_rate, dt, state, turn_longitudinal, turn_transverse);
                const Complex transverse = std::polar(
                    e2 * damping.transverse[state], angle + turn_transverse
                );
                fplus_relaxed[state] = fplus[state] * transverse;
                fminus_relaxed[state] = fminus[state] * std::conj(transverse);
                longitudinal_relaxed[state] = longitudinal[state] * std::polar(
                    e1 * damping.longitudinal[state], turn_longitudinal
                );
            }
            longitudinal_relaxed[0] += Complex(1.0F - e1, 0.0F);

            fplus_shifted = fplus_relaxed;
            fminus_shifted = fminus_relaxed;
            longitudinal_pre = longitudinal_relaxed;
            if ((action & PRE_SHIFT) != 0) {
                shift(fplus_shifted, fminus_shifted);
            }

            // --- adjoint of the trailing shift/spoil ---
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(fplus_bar.begin(), fplus_bar.end(), Complex{});
                std::fill(fminus_bar.begin(), fminus_bar.end(), Complex{});
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_adjoint(fplus_bar, fminus_bar);
            }
            if ((action & POST_SHIFT) != 0) {
                shift_adjoint(fplus_bar, fminus_bar);
            }

            // --- adjoint of the ADC ---
            if (primal.kind[event] == 2 && (action & RECORD) != 0) {
                const std::int64_t output = primal.output_index[event];
                const std::int64_t index = view.output_base + output;
                const Complex seed(
                    buffers.grad_output_real[index],
                    buffers.grad_output_imag[index]
                );
                const Complex demodulation = std::polar(1.0F, -view.phase[event]);
                // an ADC event carries no RF, so the recorded state is the one
                // left by the pre-shift
                const Complex recorded = fplus_shifted[0];
                grad_m0 += std::real(std::conj(seed) * recorded * demodulation);
                grad_phase_train[event] += std::real(
                    std::conj(seed) * m0 * recorded * Complex(0.0F, -1.0F)
                        * demodulation
                );
                fplus_bar[0] += std::conj(m0 * demodulation) * seed;
            }

            // --- adjoint of the RF ---
            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    for (std::size_t state = 0; state < states; ++state) {
                        grad_efficiency += std::real(
                            std::conj(longitudinal_bar[state])
                            * (-longitudinal_pre[state])
                        );
                        longitudinal_bar[state] *= -efficiency;
                    }
                } else {
                    const std::int64_t transmit =
                        transmit_row<SHIMMED>(primal, event, atom);
                    const float pulse_b1 = SHIMMED ? primal.b1[transmit] : b1;
                    const float alpha = view.flip[event] * pulse_b1;
                    const float phi = view.phase[event]
                        + (SHIMMED ? primal.b1_phase[transmit] : b1_phase);
                    if constexpr (SHIMMED) {
                        const std::int64_t row = primal.shim_index[event];
                        if (row != held) {
                            grad_tissue_local[
                                (layout.base[B1_INDEX] + held) * atoms + atom
                            ] += grad_b1;
                            grad_tissue_local[
                                (layout.base[B1_PHASE_INDEX] + held) * atoms + atom
                            ] += grad_b1_phase;
                            grad_b1 = 0.0F;
                            grad_b1_phase = 0.0F;
                            held = row;
                        }
                    }
                    float grad_alpha = 0.0F;
                    float grad_phi = 0.0F;
                    if constexpr (PROFILED) {
                        Complex pair_a{};
                        Complex pair_b{};
                        Complex slope_a{};
                        Complex slope_b{};
                        profile_pair_slope(
                            primal, location, alpha, pair_a, pair_b, slope_a,
                            slope_b
                        );
                        const Complex turn = std::polar(1.0F, -phi);
                        const Complex spun = pair_b * turn;
                        Complex grad_a{};
                        Complex grad_b{};
                        rotate_adjoint_spinor(
                            fplus_shifted,
                            fminus_shifted,
                            longitudinal_pre,
                            fplus_bar,
                            fminus_bar,
                            longitudinal_bar,
                            pair_a,
                            spun,
                            grad_a,
                            grad_b
                        );
                        // The flip angle reaches the pair through the slope the
                        // table stores beside it; the RF phase turns the axis
                        // once the pair is out, so it reaches ``b`` alone.
                        grad_alpha = std::real(std::conj(grad_a) * slope_a)
                            + std::real(std::conj(grad_b) * slope_b * turn);
                        grad_phi = std::real(
                            std::conj(grad_b) * Complex(0.0F, -1.0F) * spun
                        );
                    } else {
                        rotate_adjoint(
                            fplus_shifted,
                            fminus_shifted,
                            longitudinal_pre,
                            fplus_bar,
                            fminus_bar,
                            longitudinal_bar,
                            alpha,
                            phi,
                            grad_alpha,
                            grad_phi
                        );
                    }
                    grad_flip_train[event] += grad_alpha * pulse_b1;
                    grad_b1 += grad_alpha * view.flip[event];
                    grad_phase_train[event] += grad_phi;
                    grad_b1_phase += grad_phi;
                }
            }

            if ((action & PRE_SHIFT) != 0) {
                shift_adjoint(fplus_bar, fminus_bar);
            }

            // --- adjoint of relaxation and precession ---
            float grad_e1 = 0.0F;
            float grad_e2 = 0.0F;
            float grad_angle = 0.0F;
            // The b-factor reaches every order through a different weight, so
            // its gradient is a separate weighted sum rather than a rescaling
            // of the relaxation gradients.
            float grad_b_factor = 0.0F;
            // Flow reaches each order through a turn of its own, so like the
            // b-factor it collects a weighted sum rather than a scalar one.
            float grad_turn = 0.0F;
            const Complex imaginary(0.0F, 1.0F);
            // Z[0] also carries the affine recovery term (1 - e1), whose
            // derivative contributes -Re(adjoint) exactly once. Order zero is
            // undamped, so that term takes the bare factor.
            grad_e1 -= std::real(longitudinal_bar[0]);
            for (std::size_t state = 0; state < states; ++state) {
                const Complex ap = fplus_bar[state];
                const Complex am = fminus_bar[state];
                const Complex az = longitudinal_bar[state];
                const float damp_transverse = damping.transverse[state];
                const float damp_longitudinal = damping.longitudinal[state];
                float turn_longitudinal = 0.0F;
                float turn_transverse = 0.0F;
                flow_turn(flow_rate, dt, state, turn_longitudinal, turn_transverse);
                const Complex spin_transverse = std::polar(1.0F, turn_transverse);
                const Complex spin_longitudinal =
                    std::polar(1.0F, turn_longitudinal);
                const Complex carried = phase * damp_transverse * spin_transverse;
                const Complex full_transverse = e2 * carried;
                const Complex full_longitudinal =
                    e1 * damp_longitudinal * spin_longitudinal;
                const float transverse_term = std::real(
                    std::conj(ap) * carried * fplus[state]
                    + std::conj(am) * std::conj(carried) * fminus[state]
                );
                grad_e2 += transverse_term;
                // A turn of the transverse states and the off-resonance angle
                // are the same derivative; only the weight they carry differs.
                const float angle_term = std::real(
                    std::conj(ap) * imaginary * full_transverse * fplus[state]
                    - std::conj(am) * imaginary * std::conj(full_transverse)
                        * fminus[state]
                );
                grad_angle += angle_term;
                const float longitudinal_term = std::real(
                    std::conj(az) * damp_longitudinal * spin_longitudinal
                        * longitudinal[state]
                );
                grad_e1 += longitudinal_term;
                const float longitudinal_angle_term = std::real(
                    std::conj(az) * imaginary * full_longitudinal * longitudinal[state]
                );
                grad_b_factor -= e2 * transverse_term * transverse_weight(state)
                    + e1 * longitudinal_term * longitudinal_weight(state);
                grad_turn -= angle_term * (static_cast<float>(state) + 0.5F)
                    + longitudinal_angle_term * static_cast<float>(state);
                fplus_bar[state] = std::conj(full_transverse) * ap;
                fminus_bar[state] = full_transverse * am;
                longitudinal_bar[state] = std::conj(full_longitudinal) * az;
            }

            // Washout scales both relaxation factors, so its gradient is the
            // one they already carry, taken against those factors as they
            // stand before that scaling. Past the clamp nothing depends on the
            // rate any more.
            const float grad_wout = washout_rate * dt < 1.0F
                ? -(dry1 * grad_e1 + dry2 * grad_e2)
                : 0.0F;

            grad_t1 += grad_e1 * e1 * 1000.0F * dt / (t1 * t1);
            grad_t2 += grad_e2 * e2 * 1000.0F * dt / (t2 * t2);
            grad_b0 += grad_angle * (-2.0F * PI * dt);
            grad_damping += grad_b_factor * dt;
            grad_flow += grad_turn * dt;
            grad_washout += grad_wout * dt;
            grad_duration_train[event] +=
                grad_e1 * (-r1 * e1) + grad_e2 * (-r2 * e2)
                + grad_angle * (-2.0F * PI * b0)
                + grad_b_factor * damping_rate
                + grad_turn * flow_rate
                + grad_wout * washout_rate;
        }

        grad_tissue_local[layout.base[0] * atoms + atom] += grad_t1;
        grad_tissue_local[layout.base[1] * atoms + atom] += grad_t2;
        grad_tissue_local[layout.base[2] * atoms + atom] += grad_m0;
        grad_tissue_local[(layout.base[B1_INDEX] + held) * atoms + atom] += grad_b1;
        grad_tissue_local[(layout.base[B1_PHASE_INDEX] + held) * atoms + atom] +=
            grad_b1_phase;
        grad_tissue_local[layout.base[5] * atoms + atom] += grad_b0;
        grad_tissue_local[layout.base[6] * atoms + atom] += grad_efficiency;
        grad_tissue_local[layout.base[7] * atoms + atom] += grad_damping;
        // One buffer drives two rates, so the velocity gradient is the sum of
        // what each geometry carries back.
        grad_tissue_local[layout.base[8] * atoms + atom] +=
            grad_flow * primal.flow_scale
            + grad_washout * speed_direction(velocity) * primal.washout_scale;
    }
}

struct VjpJvpBuffers {
    Buffers primal;
    // tangent directions for the ten differentiable inputs
    const float* dot_t1;
    const float* dot_t2;
    const float* dot_m0;
    const float* dot_b1;
    const float* dot_b1_phase;
    const float* dot_b0;
    const float* dot_inversion_efficiency;
    const float* dot_diffusion;
    const float* dot_velocity;
    const float* dot_duration;
    const float* dot_flip;
    const float* dot_phase;
    const float* grad_output_real;
    const float* grad_output_imag;
    // value part -> gradient w.r.t. the tangent inputs
    float* grad_dot_t1;
    float* grad_dot_t2;
    float* grad_dot_m0;
    float* grad_dot_b1;
    float* grad_dot_b1_phase;
    float* grad_dot_b0;
    float* grad_dot_inversion_efficiency;
    float* grad_dot_diffusion;
    float* grad_dot_velocity;
    float* grad_dot_duration;
    float* grad_dot_flip;
    float* grad_dot_phase;
    // tangent part -> gradient w.r.t. the primal inputs
    float* grad_t1;
    float* grad_t2;
    float* grad_m0;
    float* grad_b1;
    float* grad_b1_phase;
    float* grad_b0;
    float* grad_inversion_efficiency;
    float* grad_diffusion;
    float* grad_velocity;
    float* grad_duration;
    float* grad_flip;
    float* grad_phase;
};

inline void rotate_dual(
    DualState& fplus,
    DualState& fminus,
    DualState& longitudinal,
    const DualFloat alpha,
    const DualFloat phi
) {
    const DualFloat cosine = dual_cos(alpha);
    const DualFloat sine = dual_sin(alpha);
    const DualComplex phase_one = dual_polar(phi);
    const DualComplex phase_two = phase_one * phase_one;
    const DualComplex phase_conj = conjugate(phase_one);
    const DualFloat half(DualFloat{0.5F, 0.0F});
    const DualFloat one(DualFloat{1.0F, 0.0F});
    const DualComplex t00 = to_complex(half * (one + cosine));
    const DualComplex t01 = to_complex(half * (one - cosine)) * phase_two;
    const DualComplex t02 =
        DualComplex{Complex(0.0F, -sine.value), Complex(0.0F, -sine.tangent)}
        * phase_one;
    const DualComplex t10 = conjugate(t01);
    const DualComplex t11 = t00;
    const DualComplex t12 =
        DualComplex{Complex(0.0F, sine.value), Complex(0.0F, sine.tangent)}
        * phase_conj;
    const DualFloat half_sine = half * sine;
    const DualComplex t20 =
        DualComplex{
            Complex(0.0F, -half_sine.value), Complex(0.0F, -half_sine.tangent)
        }
        * phase_conj;
    const DualComplex t21 =
        DualComplex{
            Complex(0.0F, half_sine.value), Complex(0.0F, half_sine.tangent)
        }
        * phase_one;
    const DualComplex t22 = to_complex(cosine);

    for (std::size_t state = 0; state < fplus.size(); ++state) {
        const DualComplex x0 = fplus[state];
        const DualComplex x1 = fminus[state];
        const DualComplex x2 = longitudinal[state];
        fplus[state] = t00 * x0 + t01 * x1 + t02 * x2;
        fminus[state] = t10 * x0 + t11 * x1 + t12 * x2;
        longitudinal[state] = t20 * x0 + t21 * x1 + t22 * x2;
    }
}

inline void rotate_adjoint_dual(
    const DualState& fplus_in,
    const DualState& fminus_in,
    const DualState& longitudinal_in,
    DualState& fplus_bar,
    DualState& fminus_bar,
    DualState& longitudinal_bar,
    const DualFloat alpha,
    const DualFloat phi,
    DualFloat& grad_alpha,
    DualFloat& grad_phi
) {
    const DualFloat cosine = dual_cos(alpha);
    const DualFloat sine = dual_sin(alpha);
    const DualComplex phase_one = dual_polar(phi);
    const DualComplex phase_two = phase_one * phase_one;
    const DualComplex phase_conj = conjugate(phase_one);
    const DualFloat half{0.5F, 0.0F};
    const DualFloat one{1.0F, 0.0F};

    auto imaginary_of = [](const DualFloat value) {
        return DualComplex{
            Complex(0.0F, value.value), Complex(0.0F, value.tangent)
        };
    };

    const DualComplex t00 = to_complex(half * (one + cosine));
    const DualComplex t01 = to_complex(half * (one - cosine)) * phase_two;
    const DualComplex t02 = imaginary_of(DualFloat{0.0F, 0.0F} - sine) * phase_one;
    const DualComplex t10 = conjugate(t01);
    const DualComplex t11 = t00;
    const DualComplex t12 = imaginary_of(sine) * phase_conj;
    const DualComplex t20 =
        imaginary_of(DualFloat{0.0F, 0.0F} - (half * sine)) * phase_conj;
    const DualComplex t21 = imaginary_of(half * sine) * phase_one;
    const DualComplex t22 = to_complex(cosine);

    const DualComplex d00 = to_complex(DualFloat{0.0F, 0.0F} - (half * sine));
    const DualComplex d01 = to_complex(half * sine) * phase_two;
    const DualComplex d02 =
        imaginary_of(DualFloat{0.0F, 0.0F} - cosine) * phase_one;
    const DualComplex d10 = conjugate(d01);
    const DualComplex d11 = d00;
    const DualComplex d12 = imaginary_of(cosine) * phase_conj;
    const DualComplex d20 =
        imaginary_of(DualFloat{0.0F, 0.0F} - (half * cosine)) * phase_conj;
    const DualComplex d21 = imaginary_of(half * cosine) * phase_one;
    const DualComplex d22 = to_complex(DualFloat{0.0F, 0.0F} - sine);

    const Complex imaginary(0.0F, 1.0F);
    DualFloat alpha_sum{0.0F, 0.0F};
    DualFloat phi_sum{0.0F, 0.0F};
    for (std::size_t state = 0; state < fplus_bar.size(); ++state) {
        const DualComplex a0 = fplus_bar[state];
        const DualComplex a1 = fminus_bar[state];
        const DualComplex a2 = longitudinal_bar[state];
        const DualComplex x0 = fplus_in[state];
        const DualComplex x1 = fminus_in[state];
        const DualComplex x2 = longitudinal_in[state];

        alpha_sum = alpha_sum
            + real_part(
                conjugate(a0) * (d00 * x0 + d01 * x1 + d02 * x2)
                + conjugate(a1) * (d10 * x0 + d11 * x1 + d12 * x2)
                + conjugate(a2) * (d20 * x0 + d21 * x1 + d22 * x2)
            );
        const DualComplex two{Complex(2.0F, 0.0F), Complex{}};
        phi_sum = phi_sum
            + real_part(
                conjugate(a0) * (imaginary * (two * (t01 * x1) + t02 * x2))
                + conjugate(a1)
                    * (imaginary
                       * (DualComplex{Complex(-2.0F, 0.0F), Complex{}} * (t10 * x0)
                          + DualComplex{Complex(-1.0F, 0.0F), Complex{}} * (t12 * x2)))
                + conjugate(a2)
                    * (imaginary
                       * (DualComplex{Complex(-1.0F, 0.0F), Complex{}} * (t20 * x0)
                          + t21 * x1))
            );

        fplus_bar[state] =
            conjugate(t00) * a0 + conjugate(t10) * a1 + conjugate(t20) * a2;
        fminus_bar[state] =
            conjugate(t01) * a0 + conjugate(t11) * a1 + conjugate(t21) * a2;
        longitudinal_bar[state] =
            conjugate(t02) * a0 + conjugate(t12) * a1 + conjugate(t22) * a2;
    }
    grad_alpha = grad_alpha + alpha_sum;
    grad_phi = grad_phi + phi_sum;
}

// The table read carrying a tangent in the flip angle. The pair's tangent is
// the stored slope times it, and the slope's own tangent is the Hermite
// segment's second derivative -- which its four coefficients give exactly,
// since the segment is a cubic.
inline void profile_pair_slope_dual(
    const Buffers& buffers,
    const std::int64_t location,
    const DualFloat theta,
    DualComplex& a,
    DualComplex& b,
    DualComplex& slope_a,
    DualComplex& slope_b
) {
    const float last = static_cast<float>(buffers.profile_bins - 1);
    const float step = buffers.profile_step;
    const float scaled = std::min(std::max(theta.value / step, 0.0F), last);
    const float lower = std::min(std::floor(scaled), last - 1.0F);
    const float u = scaled - lower;
    const float* const near = buffers.profile
        + (location * buffers.profile_bins + static_cast<std::int64_t>(lower))
            * PROFILE_STRIDE;
    const float* const far = near + PROFILE_STRIDE;

    const float u2 = u * u;
    const float u3 = u2 * u;
    const float h00 = 2.0F * u3 - 3.0F * u2 + 1.0F;
    const float h10 = (u3 - 2.0F * u2 + u) * step;
    const float h01 = -2.0F * u3 + 3.0F * u2;
    const float h11 = (u3 - u2) * step;
    const float g00 = (6.0F * u2 - 6.0F * u) / step;
    const float g10 = 3.0F * u2 - 4.0F * u + 1.0F;
    const float g01 = (-6.0F * u2 + 6.0F * u) / step;
    const float g11 = 3.0F * u2 - 2.0F * u;
    const float c00 = (12.0F * u - 6.0F) / (step * step);
    const float c10 = (6.0F * u - 4.0F) / step;
    const float c01 = (6.0F - 12.0F * u) / (step * step);
    const float c11 = (6.0F * u - 2.0F) / step;

    float value[4];
    float slope[4];
    float curve[4];
    for (std::size_t part = 0; part < 4; ++part) {
        value[part] = h00 * near[part] + h10 * near[part + 4]
            + h01 * far[part] + h11 * far[part + 4];
        slope[part] = g00 * near[part] + g10 * near[part + 4]
            + g01 * far[part] + g11 * far[part + 4];
        curve[part] = c00 * near[part] + c10 * near[part + 4]
            + c01 * far[part] + c11 * far[part + 4];
    }
    a = DualComplex{
        Complex(value[0], value[1]), theta.tangent * Complex(slope[0], slope[1])
    };
    b = DualComplex{
        Complex(value[2], value[3]), theta.tangent * Complex(slope[2], slope[3])
    };
    slope_a = DualComplex{
        Complex(slope[0], slope[1]), theta.tangent * Complex(curve[0], curve[1])
    };
    slope_b = DualComplex{
        Complex(slope[2], slope[3]), theta.tangent * Complex(curve[2], curve[3])
    };
}

// The spinor rotation's adjoint on dual numbers: the same closed form as
// `rotate_adjoint_spinor`, differentiated once more by the arithmetic itself.
inline void rotate_adjoint_spinor_dual(
    const DualState& fplus_in,
    const DualState& fminus_in,
    const DualState& longitudinal_in,
    DualState& fplus_bar,
    DualState& fminus_bar,
    DualState& longitudinal_bar,
    const DualComplex a,
    const DualComplex b,
    DualComplex& grad_a,
    DualComplex& grad_b
) {
    const DualComplex conj_a = conjugate(a);
    const DualComplex conj_b = conjugate(b);
    const DualComplex two{Complex(2.0F, 0.0F), Complex{}};
    const DualComplex minus_two{Complex(-2.0F, 0.0F), Complex{}};
    const DualComplex minus_one{Complex(-1.0F, 0.0F), Complex{}};

    const DualComplex t00 = conj_a * conj_a;
    const DualComplex t01 = minus_one * (conj_b * conj_b);
    const DualComplex t02 = minus_two * conjugate(a * b);
    const DualComplex t10 = minus_one * (b * b);
    const DualComplex t11 = a * a;
    const DualComplex t12 = minus_two * (a * b);
    const DualComplex t20 = conj_a * b;
    const DualComplex t21 = a * conj_b;
    const DualComplex t22 = (a * conj_a) - (b * conj_b);

    DualComplex m[3][3]{};
    for (std::size_t state = 0; state < fplus_bar.size(); ++state) {
        const DualComplex a0 = fplus_bar[state];
        const DualComplex a1 = fminus_bar[state];
        const DualComplex a2 = longitudinal_bar[state];
        const DualComplex x0 = fplus_in[state];
        const DualComplex x1 = fminus_in[state];
        const DualComplex x2 = longitudinal_in[state];

        const DualComplex s0 = conjugate(a0);
        const DualComplex s1 = conjugate(a1);
        const DualComplex s2 = conjugate(a2);
        m[0][0] = m[0][0] + s0 * x0;
        m[0][1] = m[0][1] + s0 * x1;
        m[0][2] = m[0][2] + s0 * x2;
        m[1][0] = m[1][0] + s1 * x0;
        m[1][1] = m[1][1] + s1 * x1;
        m[1][2] = m[1][2] + s1 * x2;
        m[2][0] = m[2][0] + s2 * x0;
        m[2][1] = m[2][1] + s2 * x1;
        m[2][2] = m[2][2] + s2 * x2;

        fplus_bar[state] =
            conjugate(t00) * a0 + conjugate(t10) * a1 + conjugate(t20) * a2;
        fminus_bar[state] =
            conjugate(t01) * a0 + conjugate(t11) * a1 + conjugate(t21) * a2;
        longitudinal_bar[state] =
            conjugate(t02) * a0 + conjugate(t12) * a1 + conjugate(t22) * a2;
    }

    const DualComplex holding_conj_a = two * (a * m[1][1])
        + minus_two * (b * m[1][2]) + conj_b * m[2][1] + conj_a * m[2][2];
    const DualComplex holding_a = two * (conj_a * m[0][0])
        + minus_two * (conj_b * m[0][2]) + b * m[2][0] + a * m[2][2];
    const DualComplex holding_conj_b = minus_two * (b * m[1][0])
        + minus_two * (a * m[1][2]) + conj_a * m[2][0]
        + minus_one * (conj_b * m[2][2]);
    const DualComplex holding_b = minus_two * (conj_b * m[0][1])
        + minus_two * (conj_a * m[0][2]) + a * m[2][1] + minus_one * (b * m[2][2]);
    grad_a = grad_a + conjugate(holding_conj_a) + holding_a;
    grad_b = grad_b + conjugate(holding_conj_b) + holding_b;
}

inline DualFloat dual_inverse_square(const DualFloat a) {
    const float inverse = 1.0F / (a.value * a.value);
    return {inverse, -2.0F * a.tangent * inverse / a.value};
}

inline void shift_real_dual(
    std::vector<DualFloat>& plus,
    std::vector<DualFloat>& minus,
    const std::size_t states
) {
    for (std::size_t state = 0; state + 1 < states; ++state) {
        minus[state] = minus[state + 1];
    }
    minus[states - 1] = DualFloat{0.0F, 0.0F};
    for (std::size_t state = states - 1; state > 0; --state) {
        plus[state] = plus[state - 1];
    }
    plus[0] = DualFloat{0.0F, 0.0F} - minus[0];
}

// Transpose of shift_real_dual. The a0 = -b0 coupling sends the incoming plus
// adjoint back onto minus, at the index the minus shift moves it to.
inline void shift_real_dual_adjoint(
    std::vector<DualFloat>& plus_bar,
    std::vector<DualFloat>& minus_bar,
    const std::size_t states
) {
    const DualFloat carry = DualFloat{0.0F, 0.0F} - plus_bar[0];
    for (std::size_t state = 0; state + 1 < states; ++state) {
        plus_bar[state] = plus_bar[state + 1];
    }
    plus_bar[states - 1] = DualFloat{0.0F, 0.0F};
    for (std::size_t state = states - 1; state > 0; --state) {
        minus_bar[state] = minus_bar[state - 1];
    }
    minus_bar[0] = DualFloat{0.0F, 0.0F};
    if (states > 1) {
        minus_bar[1] = minus_bar[1] + carry;
    }
}

// Forward-over-reverse through the real subspace.
//
// The forward is a chain of real linear maps, so each adjoint is a transpose
// rather than a conjugate transpose, and the state carries three reals instead
// of three complex numbers. Everything is dual-valued, giving the derivative of
// the forward-mode output with respect to the primal inputs.
//
// The RF phase does not appear: it divides out of the representation, so this
// kernel leaves grad_phase untouched. Perturbing a single pulse's phase is
// exactly the direction that leaves the subspace, so callers must not ask for
// that gradient.
void simulate_real_vjp_jvp_range(
    const VjpJvpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count,
    DualFloat* grad_flip_local,
    DualFloat* grad_phase_local,
    DualFloat* grad_duration_local,
    DualFloat* grad_tissue_local
) {
    (void)grad_phase_local;
    const Buffers& primal = buffers.primal;
    const std::size_t states = static_cast<std::size_t>(state_count);
    const std::size_t stride = 3U * states;

    std::vector<DualFloat> trajectory(
        static_cast<std::size_t>(event_count) * stride
    );
    std::vector<DualFloat> plus(states), minus(states), longitudinal(states);
    std::vector<DualFloat> plus_bar(states), minus_bar(states), longitudinal_bar(states);
    std::vector<DualFloat> plus_stage(states), minus_stage(states), longitudinal_stage(states);
    Damping<DualFloat> damping(states);

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const TrainView view = train_view(primal, work, event_count, output_count);
        const std::int64_t atom = view.atom;
        const float* const dot_duration = buffers.dot_duration + view.event_base;
        const float* const dot_flip = buffers.dot_flip + view.event_base;
        DualFloat* const grad_flip_train = grad_flip_local + view.event_base;
        DualFloat* const grad_duration_train = grad_duration_local + view.event_base;

        const DualFloat t1{primal.t1[atom], buffers.dot_t1[atom]};
        const DualFloat t2{primal.t2[atom], buffers.dot_t2[atom]};
        const DualFloat m0{primal.m0[atom], buffers.dot_m0[atom]};
        const DualFloat b1{primal.b1[atom], buffers.dot_b1[atom]};
        const DualFloat inversion{
            primal.inversion_efficiency[atom],
            buffers.dot_inversion_efficiency[atom]
        };
        const DualFloat r1{
            1000.0F / t1.value, -1000.0F * t1.tangent / (t1.value * t1.value)
        };
        const DualFloat r2{
            1000.0F / t2.value, -1000.0F * t2.tangent / (t2.value * t2.value)
        };
        const DualFloat damping_rate{
            primal.diffusion[atom], buffers.dot_diffusion[atom]
        };

        std::fill(plus.begin(), plus.end(), DualFloat{0.0F, 0.0F});
        std::fill(minus.begin(), minus.end(), DualFloat{0.0F, 0.0F});
        std::fill(longitudinal.begin(), longitudinal.end(), DualFloat{0.0F, 0.0F});
        longitudinal[0] = DualFloat{1.0F, 0.0F};

        // ---- forward, recording the state entering each event ----
        for (std::int64_t event = 0; event < event_count; ++event) {
            DualFloat* slot = trajectory.data()
                + static_cast<std::size_t>(event) * stride;
            std::copy(plus.begin(), plus.end(), slot);
            std::copy(minus.begin(), minus.end(), slot + states);
            std::copy(longitudinal.begin(), longitudinal.end(), slot + 2U * states);

            const DualFloat dt{view.duration[event], dot_duration[event]};
            damping.set(damping_rate, dt);
            const DualFloat e1 = dual_exp(DualFloat{0.0F, 0.0F} - r1 * dt);
            const DualFloat e2 = dual_exp(DualFloat{0.0F, 0.0F} - r2 * dt);
            for (std::size_t state = 0; state < states; ++state) {
                plus[state] = plus[state] * e2 * damping.transverse[state];
                minus[state] = minus[state] * e2 * damping.transverse[state];
                longitudinal[state] =
                    longitudinal[state] * e1 * damping.longitudinal[state];
            }
            longitudinal[0] = longitudinal[0] + (DualFloat{1.0F, 0.0F} - e1);

            const std::uint8_t action = primal.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift_real_dual(plus, minus, states);
            }
            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const DualFloat efficiency = DualFloat{0.0F, 0.0F} - inversion;
                    for (std::size_t state = 0; state < states; ++state) {
                        longitudinal[state] = longitudinal[state] * efficiency;
                    }
                } else {
                    const DualFloat alpha =
                        DualFloat{view.flip[event], dot_flip[event]} * b1;
                    const DualFloat cosine{
                        std::cos(alpha.value), -std::sin(alpha.value) * alpha.tangent
                    };
                    const DualFloat sine{
                        std::sin(alpha.value), std::cos(alpha.value) * alpha.tangent
                    };
                    const DualFloat cosine_half_sq =
                        0.5F * (DualFloat{1.0F, 0.0F} + cosine);
                    const DualFloat sine_half_sq =
                        0.5F * (DualFloat{1.0F, 0.0F} - cosine);
                    const DualFloat half_sine = 0.5F * sine;
                    for (std::size_t state = 0; state < states; ++state) {
                        const DualFloat p = plus[state];
                        const DualFloat m = minus[state];
                        const DualFloat z = longitudinal[state];
                        plus[state] = cosine_half_sq * p + sine_half_sq * m
                            - sine * z;
                        minus[state] = sine_half_sq * p + cosine_half_sq * m
                            + sine * z;
                        longitudinal[state] = half_sine * p - half_sine * m
                            + cosine * z;
                    }
                }
            }
            if ((action & POST_SHIFT) != 0) {
                shift_real_dual(plus, minus, states);
            }
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(plus.begin(), plus.end(), DualFloat{0.0F, 0.0F});
                std::fill(minus.begin(), minus.end(), DualFloat{0.0F, 0.0F});
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_real_dual(plus, minus, states);
            }
        }

        // ---- reverse ----
        std::fill(plus_bar.begin(), plus_bar.end(), DualFloat{0.0F, 0.0F});
        std::fill(minus_bar.begin(), minus_bar.end(), DualFloat{0.0F, 0.0F});
        std::fill(
            longitudinal_bar.begin(), longitudinal_bar.end(), DualFloat{0.0F, 0.0F}
        );
        DualFloat grad_t1{0.0F, 0.0F};
        DualFloat grad_t2{0.0F, 0.0F};
        DualFloat grad_m0{0.0F, 0.0F};
        DualFloat grad_b1{0.0F, 0.0F};
        DualFloat grad_inversion{0.0F, 0.0F};
        DualFloat grad_damping{0.0F, 0.0F};

        for (std::int64_t event = event_count - 1; event >= 0; --event) {
            const DualFloat* slot = trajectory.data()
                + static_cast<std::size_t>(event) * stride;
            const std::uint8_t action = primal.action[event];
            const DualFloat dt{view.duration[event], dot_duration[event]};
            damping.set(damping_rate, dt);
            const DualFloat e1 = dual_exp(DualFloat{0.0F, 0.0F} - r1 * dt);
            const DualFloat e2 = dual_exp(DualFloat{0.0F, 0.0F} - r2 * dt);

            // Replay the intra-event stages from the recorded entry state.
            for (std::size_t state = 0; state < states; ++state) {
                plus_stage[state] = slot[state] * e2 * damping.transverse[state];
                minus_stage[state] =
                    slot[states + state] * e2 * damping.transverse[state];
                longitudinal_stage[state] =
                    slot[2U * states + state] * e1 * damping.longitudinal[state];
            }
            longitudinal_stage[0] =
                longitudinal_stage[0] + (DualFloat{1.0F, 0.0F} - e1);
            if ((action & PRE_SHIFT) != 0) {
                shift_real_dual(plus_stage, minus_stage, states);
            }
            // plus_stage now holds the state entering the RF operator.

            // Undo the trailing spoil/shift.
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(plus_bar.begin(), plus_bar.end(), DualFloat{0.0F, 0.0F});
                std::fill(minus_bar.begin(), minus_bar.end(), DualFloat{0.0F, 0.0F});
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_real_dual_adjoint(plus_bar, minus_bar, states);
            }
            if ((action & POST_SHIFT) != 0) {
                shift_real_dual_adjoint(plus_bar, minus_bar, states);
            }

            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const DualFloat efficiency = DualFloat{0.0F, 0.0F} - inversion;
                    for (std::size_t state = 0; state < states; ++state) {
                        grad_inversion = grad_inversion
                            - longitudinal_bar[state] * longitudinal_stage[state];
                        longitudinal_bar[state] = longitudinal_bar[state] * efficiency;
                    }
                } else {
                    const DualFloat flip{view.flip[event], dot_flip[event]};
                    const DualFloat alpha = flip * b1;
                    const DualFloat cosine{
                        std::cos(alpha.value), -std::sin(alpha.value) * alpha.tangent
                    };
                    const DualFloat sine{
                        std::sin(alpha.value), std::cos(alpha.value) * alpha.tangent
                    };
                    const DualFloat cosine_half_sq =
                        0.5F * (DualFloat{1.0F, 0.0F} + cosine);
                    const DualFloat sine_half_sq =
                        0.5F * (DualFloat{1.0F, 0.0F} - cosine);
                    const DualFloat half_sine = 0.5F * sine;
                    DualFloat grad_alpha{0.0F, 0.0F};
                    for (std::size_t state = 0; state < states; ++state) {
                        const DualFloat p = plus_stage[state];
                        const DualFloat m = minus_stage[state];
                        const DualFloat z = longitudinal_stage[state];
                        const DualFloat pb = plus_bar[state];
                        const DualFloat mb = minus_bar[state];
                        const DualFloat zb = longitudinal_bar[state];
                        // d/dalpha of each output row, contracted with the adjoint.
                        grad_alpha = grad_alpha
                            + pb * (half_sine * m - half_sine * p - cosine * z)
                            + mb * (half_sine * p - half_sine * m + cosine * z)
                            + zb * (0.5F * cosine * p - 0.5F * cosine * m - sine * z);
                        // Transpose of the rotation.
                        plus_bar[state] = cosine_half_sq * pb + sine_half_sq * mb
                            + half_sine * zb;
                        minus_bar[state] = sine_half_sq * pb + cosine_half_sq * mb
                            - half_sine * zb;
                        longitudinal_bar[state] = (DualFloat{0.0F, 0.0F} - sine) * pb
                            + sine * mb + cosine * zb;
                    }
                    grad_flip_train[event] = grad_flip_train[event] + grad_alpha * b1;
                    grad_b1 = grad_b1 + grad_alpha * flip;
                }
            } else if (primal.kind[event] == 2 && (action & RECORD) != 0) {
                // The sample is i * m0 * plus[0]; only the imaginary seed acts.
                const std::int64_t index =
                    view.output_base + primal.output_index[event];
                const DualFloat seed{buffers.grad_output_imag[index], 0.0F};
                grad_m0 = grad_m0 + seed * plus_stage[0];
                plus_bar[0] = plus_bar[0] + seed * m0;
            }

            if ((action & PRE_SHIFT) != 0) {
                shift_real_dual_adjoint(plus_bar, minus_bar, states);
            }

            DualFloat grad_e1{0.0F, 0.0F};
            DualFloat grad_e2{0.0F, 0.0F};
            DualFloat grad_b_factor{0.0F, 0.0F};
            grad_e1 = grad_e1 - longitudinal_bar[0];
            for (std::size_t state = 0; state < states; ++state) {
                const DualFloat damp_transverse = damping.transverse[state];
                const DualFloat damp_longitudinal = damping.longitudinal[state];
                const DualFloat plus_term =
                    plus_bar[state] * slot[state] * damp_transverse;
                const DualFloat minus_term =
                    minus_bar[state] * slot[states + state] * damp_transverse;
                const DualFloat longitudinal_term = longitudinal_bar[state]
                    * slot[2U * states + state] * damp_longitudinal;
                grad_e2 = grad_e2 + plus_term + minus_term;
                grad_e1 = grad_e1 + longitudinal_term;
                grad_b_factor = grad_b_factor
                    - transverse_weight(state) * (e2 * (plus_term + minus_term))
                    - longitudinal_weight(state) * (e1 * longitudinal_term);
                plus_bar[state] = plus_bar[state] * e2 * damp_transverse;
                minus_bar[state] = minus_bar[state] * e2 * damp_transverse;
                longitudinal_bar[state] =
                    longitudinal_bar[state] * e1 * damp_longitudinal;
            }
            const DualFloat scale1 =
                e1 * dt * (1000.0F * dual_inverse_square(t1));
            const DualFloat scale2 =
                e2 * dt * (1000.0F * dual_inverse_square(t2));
            grad_t1 = grad_t1 + grad_e1 * scale1;
            grad_t2 = grad_t2 + grad_e2 * scale2;
            grad_damping = grad_damping + grad_b_factor * dt;
            grad_duration_train[event] = grad_duration_train[event]
                - grad_e1 * (r1 * e1) - grad_e2 * (r2 * e2)
                + grad_b_factor * damping_rate;
        }

        const std::int64_t atoms = primal.atom_count;
        const DualFloat contributions[TISSUE_COUNT] = {
            grad_t1, grad_t2, grad_m0, grad_b1, DualFloat{0.0F, 0.0F},
            DualFloat{0.0F, 0.0F}, grad_inversion, grad_damping,
            DualFloat{0.0F, 0.0F},
        };
        for (std::size_t parameter = 0; parameter < TISSUE_COUNT; ++parameter) {
            DualFloat& target = grad_tissue_local[parameter * atoms + atom];
            target = target + contributions[parameter];
        }
    }
}

// Lane-parallel dual: the value and tangent planes each hold one train per lane.
struct DualLane {
    LaneVector value;
    LaneVector tangent;
};

inline DualLane operator+(const DualLane left, const DualLane right) {
    return DualLane{left.value + right.value, left.tangent + right.tangent};
}

inline DualLane operator-(const DualLane left, const DualLane right) {
    return DualLane{left.value - right.value, left.tangent - right.tangent};
}

inline DualLane operator*(const DualLane left, const DualLane right) {
    return DualLane{
        left.value * right.value,
        left.value * right.tangent + left.tangent * right.value,
    };
}

inline DualLane operator-(const DualLane value) {
    return DualLane{-value.value, -value.tangent};
}

inline DualLane operator*(const float left, const DualLane right) {
    const LaneVector scale = lane_splat(left);
    return DualLane{scale * right.value, scale * right.tangent};
}

inline DualLane dual_lane_splat(const DualFloat value) {
    return DualLane{lane_splat(value.value), lane_splat(value.tangent)};
}

inline DualLane dual_lane_zero() {
    const LaneVector zero = lane_splat(0.0F);
    return DualLane{zero, zero};
}

inline DualLane dual_lane_exp(const DualLane argument) {
    float buffer[REAL_LANES];
    lane_store(buffer, argument.value);
    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
        buffer[lane] = std::exp(buffer[lane]);
    }
    const LaneVector value = lane_load(buffer);
    return DualLane{value, argument.tangent * value};
}

struct DualLaneTrig {
    DualLane cosine;
    DualLane sine;
};

inline DualLaneTrig dual_lane_trig(const DualLane angle) {
    float value[REAL_LANES], cosine[REAL_LANES], sine[REAL_LANES];
    lane_store(value, angle.value);
    for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
        cosine[lane] = std::cos(value[lane]);
        sine[lane] = std::sin(value[lane]);
    }
    const LaneVector c = lane_load(cosine);
    const LaneVector s = lane_load(sine);
    return DualLaneTrig{
        DualLane{c, -(s * angle.tangent)},
        DualLane{s, c * angle.tangent},
    };
}

// Inactive lanes repeat the block's first train, so a reduction that crosses the
// lane axis must stop at ``active`` or it would count that train twice.
inline float lane_sum(const LaneVector value, const std::int64_t active) {
    float buffer[REAL_LANES];
    lane_store(buffer, value);
    float total = 0.0F;
    for (std::int64_t lane = 0; lane < active; ++lane) {
        total += buffer[lane];
    }
    return total;
}

inline void shift_real_dual_lanes(
    DualLane* plus, DualLane* minus, const std::size_t states
) {
    for (std::size_t state = 0; state + 1 < states; ++state) {
        minus[state] = minus[state + 1];
    }
    minus[states - 1] = dual_lane_zero();
    for (std::size_t state = states - 1; state > 0; --state) {
        plus[state] = plus[state - 1];
    }
    plus[0] = -minus[0];
}

inline void shift_real_dual_lanes_adjoint(
    DualLane* plus_bar, DualLane* minus_bar, const std::size_t states
) {
    const DualLane carry = -plus_bar[0];
    for (std::size_t state = 0; state + 1 < states; ++state) {
        plus_bar[state] = plus_bar[state + 1];
    }
    plus_bar[states - 1] = dual_lane_zero();
    for (std::size_t state = states - 1; state > 0; --state) {
        minus_bar[state] = minus_bar[state - 1];
    }
    minus_bar[0] = dual_lane_zero();
    if (states > 1) {
        minus_bar[1] = minus_bar[1] + carry;
    }
}

void simulate_real_vjp_jvp_lane_range(
    const VjpJvpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count,
    DualFloat* grad_flip_local,
    DualFloat* grad_phase_local,
    DualFloat* grad_duration_local,
    DualFloat* grad_tissue_local
) {
    (void)grad_phase_local;
    const Buffers& primal = buffers.primal;
    const std::size_t states = static_cast<std::size_t>(state_count);
    const std::size_t stride = 3U * states;

    std::vector<DualLane> trajectory(
        static_cast<std::size_t>(event_count) * stride
    );
    std::vector<DualLane> plus(states), minus(states), longitudinal(states);
    std::vector<DualLane> plus_bar(states), minus_bar(states);
    std::vector<DualLane> longitudinal_bar(states);
    std::vector<DualLane> plus_stage(states), minus_stage(states);
    std::vector<DualLane> longitudinal_stage(states);

    float dt_value[REAL_LANES], dt_tangent[REAL_LANES];
    float flip_value[REAL_LANES], flip_tangent[REAL_LANES];
    float seed_value[REAL_LANES];
    float scatter_value[REAL_LANES], scatter_tangent[REAL_LANES];

    const DualLane zero = dual_lane_zero();
    const DualLane one = DualLane{lane_splat(1.0F), lane_splat(0.0F)};

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const LaneView view = lane_view(primal, work);
        const std::int64_t atom = view.atom;

        const DualFloat t1{primal.t1[atom], buffers.dot_t1[atom]};
        const DualFloat t2{primal.t2[atom], buffers.dot_t2[atom]};
        const DualLane m0 =
            dual_lane_splat(DualFloat{primal.m0[atom], buffers.dot_m0[atom]});
        const DualLane b1 =
            dual_lane_splat(DualFloat{primal.b1[atom], buffers.dot_b1[atom]});
        const DualLane inversion = dual_lane_splat(DualFloat{
            primal.inversion_efficiency[atom],
            buffers.dot_inversion_efficiency[atom],
        });
        const DualLane r1 = dual_lane_splat(DualFloat{
            1000.0F / t1.value, -1000.0F * t1.tangent / (t1.value * t1.value)
        });
        const DualLane r2 = dual_lane_splat(DualFloat{
            1000.0F / t2.value, -1000.0F * t2.tangent / (t2.value * t2.value)
        });
        const DualLane inverse_t1 =
            dual_lane_splat(1000.0F * dual_inverse_square(t1));
        const DualLane inverse_t2 =
            dual_lane_splat(1000.0F * dual_inverse_square(t2));

        std::fill(plus.begin(), plus.end(), zero);
        std::fill(minus.begin(), minus.end(), zero);
        std::fill(longitudinal.begin(), longitudinal.end(), zero);
        longitudinal[0] = one;

        // ---- forward, recording the state entering each event ----
        for (std::int64_t event = 0; event < event_count; ++event) {
            DualLane* slot = trajectory.data()
                + static_cast<std::size_t>(event) * stride;
            std::copy(plus.begin(), plus.end(), slot);
            std::copy(minus.begin(), minus.end(), slot + states);
            std::copy(longitudinal.begin(), longitudinal.end(), slot + 2U * states);

            gather_lanes(primal.duration, view, event, event_count, dt_value);
            gather_lanes(buffers.dot_duration, view, event, event_count, dt_tangent);
            const DualLane dt{lane_load(dt_value), lane_load(dt_tangent)};
            const DualLane e1 = dual_lane_exp(-(r1 * dt));
            const DualLane e2 = dual_lane_exp(-(r2 * dt));
            for (std::size_t state = 0; state < states; ++state) {
                plus[state] = plus[state] * e2;
                minus[state] = minus[state] * e2;
                longitudinal[state] = longitudinal[state] * e1;
            }
            longitudinal[0] = longitudinal[0] + (one - e1);

            const std::uint8_t action = primal.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift_real_dual_lanes(plus.data(), minus.data(), states);
            }
            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const DualLane efficiency = -inversion;
                    for (std::size_t state = 0; state < states; ++state) {
                        longitudinal[state] = longitudinal[state] * efficiency;
                    }
                } else {
                    gather_lanes(primal.flip, view, event, event_count, flip_value);
                    gather_lanes(
                        buffers.dot_flip, view, event, event_count, flip_tangent
                    );
                    const DualLane flip{
                        lane_load(flip_value), lane_load(flip_tangent)
                    };
                    const DualLaneTrig trig = dual_lane_trig(flip * b1);
                    const DualLane cosine_half_sq = 0.5F * (one + trig.cosine);
                    const DualLane sine_half_sq = 0.5F * (one - trig.cosine);
                    const DualLane half_sine = 0.5F * trig.sine;
                    for (std::size_t state = 0; state < states; ++state) {
                        const DualLane p = plus[state];
                        const DualLane m = minus[state];
                        const DualLane z = longitudinal[state];
                        plus[state] = cosine_half_sq * p + sine_half_sq * m
                            - trig.sine * z;
                        minus[state] = sine_half_sq * p + cosine_half_sq * m
                            + trig.sine * z;
                        longitudinal[state] = half_sine * p - half_sine * m
                            + trig.cosine * z;
                    }
                }
            }
            if ((action & POST_SHIFT) != 0) {
                shift_real_dual_lanes(plus.data(), minus.data(), states);
            }
            if ((action & SPOIL_AFTER) != 0) {
                std::fill(plus.begin(), plus.end(), zero);
                std::fill(minus.begin(), minus.end(), zero);
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_real_dual_lanes(plus.data(), minus.data(), states);
            }
        }

        // ---- reverse ----
        std::fill(plus_bar.begin(), plus_bar.end(), zero);
        std::fill(minus_bar.begin(), minus_bar.end(), zero);
        std::fill(longitudinal_bar.begin(), longitudinal_bar.end(), zero);
        DualLane grad_t1 = zero;
        DualLane grad_t2 = zero;
        DualLane grad_m0 = zero;
        DualLane grad_b1 = zero;
        DualLane grad_inversion = zero;
        DualLane grad_damping = zero;

        for (std::int64_t event = event_count - 1; event >= 0; --event) {
            const DualLane* slot = trajectory.data()
                + static_cast<std::size_t>(event) * stride;
            const std::uint8_t action = primal.action[event];
            gather_lanes(primal.duration, view, event, event_count, dt_value);
            gather_lanes(buffers.dot_duration, view, event, event_count, dt_tangent);
            const DualLane dt{lane_load(dt_value), lane_load(dt_tangent)};
            const DualLane e1 = dual_lane_exp(-(r1 * dt));
            const DualLane e2 = dual_lane_exp(-(r2 * dt));

            // Replay the intra-event stages from the recorded entry state.
            for (std::size_t state = 0; state < states; ++state) {
                plus_stage[state] = slot[state] * e2;
                minus_stage[state] = slot[states + state] * e2;
                longitudinal_stage[state] = slot[2U * states + state] * e1;
            }
            longitudinal_stage[0] = longitudinal_stage[0] + (one - e1);
            if ((action & PRE_SHIFT) != 0) {
                shift_real_dual_lanes(plus_stage.data(), minus_stage.data(), states);
            }

            if ((action & SPOIL_AFTER) != 0) {
                std::fill(plus_bar.begin(), plus_bar.end(), zero);
                std::fill(minus_bar.begin(), minus_bar.end(), zero);
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_real_dual_lanes_adjoint(
                    plus_bar.data(), minus_bar.data(), states
                );
            }
            if ((action & POST_SHIFT) != 0) {
                shift_real_dual_lanes_adjoint(
                    plus_bar.data(), minus_bar.data(), states
                );
            }

            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const DualLane efficiency = -inversion;
                    for (std::size_t state = 0; state < states; ++state) {
                        grad_inversion = grad_inversion
                            - longitudinal_bar[state] * longitudinal_stage[state];
                        longitudinal_bar[state] = longitudinal_bar[state] * efficiency;
                    }
                } else {
                    gather_lanes(primal.flip, view, event, event_count, flip_value);
                    gather_lanes(
                        buffers.dot_flip, view, event, event_count, flip_tangent
                    );
                    const DualLane flip{
                        lane_load(flip_value), lane_load(flip_tangent)
                    };
                    const DualLaneTrig trig = dual_lane_trig(flip * b1);
                    const DualLane cosine_half_sq = 0.5F * (one + trig.cosine);
                    const DualLane sine_half_sq = 0.5F * (one - trig.cosine);
                    const DualLane half_sine = 0.5F * trig.sine;
                    DualLane grad_alpha = zero;
                    for (std::size_t state = 0; state < states; ++state) {
                        const DualLane p = plus_stage[state];
                        const DualLane m = minus_stage[state];
                        const DualLane z = longitudinal_stage[state];
                        const DualLane pb = plus_bar[state];
                        const DualLane mb = minus_bar[state];
                        const DualLane zb = longitudinal_bar[state];
                        grad_alpha = grad_alpha
                            + pb * (half_sine * m - half_sine * p - trig.cosine * z)
                            + mb * (half_sine * p - half_sine * m + trig.cosine * z)
                            + zb * (0.5F * trig.cosine * p - 0.5F * trig.cosine * m
                                    - trig.sine * z);
                        plus_bar[state] = cosine_half_sq * pb + sine_half_sq * mb
                            + half_sine * zb;
                        minus_bar[state] = sine_half_sq * pb + cosine_half_sq * mb
                            - half_sine * zb;
                        longitudinal_bar[state] = -trig.sine * pb + trig.sine * mb
                            + trig.cosine * zb;
                    }
                    const DualLane contribution = grad_alpha * b1;
                    lane_store(scatter_value, contribution.value);
                    lane_store(scatter_tangent, contribution.tangent);
                    for (std::int64_t lane = 0; lane < view.active; ++lane) {
                        DualFloat& target = grad_flip_local
                            [(view.train_begin + lane) * event_count + event];
                        target.value += scatter_value[lane];
                        target.tangent += scatter_tangent[lane];
                    }
                    grad_b1 = grad_b1 + grad_alpha * flip;
                }
            } else if (primal.kind[event] == 2 && (action & RECORD) != 0) {
                // The sample is i * m0 * plus[0]; only the imaginary seed acts.
                const std::int64_t offset = primal.output_index[event];
                for (std::size_t lane = 0; lane < REAL_LANES; ++lane) {
                    seed_value[lane] = 0.0F;
                }
                for (std::int64_t lane = 0; lane < view.active; ++lane) {
                    const std::int64_t index =
                        ((view.train_begin + lane) * primal.atom_count + atom)
                            * output_count
                        + offset;
                    seed_value[lane] = buffers.grad_output_imag[index];
                }
                const DualLane seed{lane_load(seed_value), lane_splat(0.0F)};
                grad_m0 = grad_m0 + seed * plus_stage[0];
                plus_bar[0] = plus_bar[0] + seed * m0;
            }

            if ((action & PRE_SHIFT) != 0) {
                shift_real_dual_lanes_adjoint(
                    plus_bar.data(), minus_bar.data(), states
                );
            }

            DualLane grad_e1 = -longitudinal_bar[0];
            DualLane grad_e2 = zero;
            // The dispatch reaches this kernel only where the damping rate is
            // zero, so every per-order factor above is exactly one and the
            // b-factor's gradient is a weighted sum with no exponential in it.
            DualLane grad_b_factor = zero;
            for (std::size_t state = 0; state < states; ++state) {
                const DualLane plus_term = plus_bar[state] * slot[state];
                const DualLane minus_term =
                    minus_bar[state] * slot[states + state];
                const DualLane longitudinal_term =
                    longitudinal_bar[state] * slot[2U * states + state];
                grad_e2 = grad_e2 + plus_term + minus_term;
                grad_e1 = grad_e1 + longitudinal_term;
                grad_b_factor = grad_b_factor
                    - transverse_weight(state) * (e2 * (plus_term + minus_term))
                    - longitudinal_weight(state) * (e1 * longitudinal_term);
                plus_bar[state] = plus_bar[state] * e2;
                minus_bar[state] = minus_bar[state] * e2;
                longitudinal_bar[state] = longitudinal_bar[state] * e1;
            }
            grad_t1 = grad_t1 + grad_e1 * (e1 * dt * inverse_t1);
            grad_t2 = grad_t2 + grad_e2 * (e2 * dt * inverse_t2);
            grad_damping = grad_damping + grad_b_factor * dt;
            const DualLane grad_dt =
                -(grad_e1 * (r1 * e1)) - grad_e2 * (r2 * e2);
            lane_store(scatter_value, grad_dt.value);
            lane_store(scatter_tangent, grad_dt.tangent);
            for (std::int64_t lane = 0; lane < view.active; ++lane) {
                DualFloat& target = grad_duration_local
                    [(view.train_begin + lane) * event_count + event];
                target.value += scatter_value[lane];
                target.tangent += scatter_tangent[lane];
            }
        }

        const std::int64_t atoms = primal.atom_count;
        const DualLane contributions[TISSUE_COUNT] = {
            grad_t1, grad_t2, grad_m0, grad_b1, zero, zero, grad_inversion,
            grad_damping, zero,
        };
        for (std::size_t parameter = 0; parameter < TISSUE_COUNT; ++parameter) {
            DualFloat& target = grad_tissue_local[parameter * atoms + atom];
            target.value += lane_sum(contributions[parameter].value, view.active);
            target.tangent += lane_sum(contributions[parameter].tangent, view.active);
        }
    }
}

// One half of the transmit pair a pulse sees, with its tangent. ``voxel``
// carries the single-shim field, which is a property of the voxel and lifts
// out of the event loop; a shimmed sequence reads the row of the shim the
// pulse drives.
template <bool SHIMMED>
inline DualFloat shim_dual(
    const float* const value,
    const float* const tangent,
    const std::int64_t row,
    const DualFloat voxel
) {
    if constexpr (SHIMMED) {
        return DualFloat{value[row], tangent[row]};
    } else {
        (void)value;
        (void)tangent;
        (void)row;
        return voxel;
    }
}

template <bool SHIMMED, bool PROFILED>
#if defined(__GNUC__) && !defined(__clang__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target_clones("default", "sse4.2", "avx2", "avx512f")))
#endif
void simulate_vjp_jvp_range(
    const VjpJvpBuffers& buffers,
    const std::int64_t work_begin,
    const std::int64_t work_end,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count,
    DualFloat* grad_flip_local,
    DualFloat* grad_phase_local,
    DualFloat* grad_duration_local,
    // Per-atom accumulators laid out [row][atom]; see the first-order kernel
    // for the rows and for why these cannot be written directly.
    DualFloat* grad_tissue_local
) {
    const Buffers& primal = buffers.primal;
    const TissueLayout layout(primal.shim_count);
    const std::int64_t atoms = primal.atom_count;
    const std::size_t states = static_cast<std::size_t>(state_count);
    const std::size_t stride = 3U * states;

    std::vector<DualComplex> trajectory(
        static_cast<std::size_t>(event_count) * stride
    );
    DualState fplus(states);
    DualState fminus(states);
    DualState longitudinal(states);
    DualState fplus_bar(states);
    DualState fminus_bar(states);
    DualState longitudinal_bar(states);
    DualState fplus_relaxed(states);
    DualState fminus_relaxed(states);
    DualState longitudinal_relaxed(states);
    DualState fplus_shifted(states);
    DualState fminus_shifted(states);
    Damping<DualFloat> damping(states);

    for (std::int64_t work = work_begin; work < work_end; ++work) {
        const TrainView view = train_view(primal, work, event_count, output_count);
        const std::int64_t atom = view.atom;
        const std::int64_t location = slice_row<PROFILED>(primal, atom);
        const float* const dot_duration = buffers.dot_duration + view.event_base;
        const float* const dot_flip = buffers.dot_flip + view.event_base;
        const float* const dot_phase = buffers.dot_phase + view.event_base;
        DualFloat* const grad_flip_train = grad_flip_local + view.event_base;
        DualFloat* const grad_phase_train = grad_phase_local + view.event_base;
        DualFloat* const grad_duration_train = grad_duration_local + view.event_base;
        const DualFloat t1{primal.t1[atom], buffers.dot_t1[atom]};
        const DualFloat t2{primal.t2[atom], buffers.dot_t2[atom]};
        const DualFloat m0{primal.m0[atom], buffers.dot_m0[atom]};
        const DualFloat b1{primal.b1[atom], buffers.dot_b1[atom]};
        const DualFloat b1_phase{
            primal.b1_phase[atom], buffers.dot_b1_phase[atom]
        };
        const DualFloat b0{primal.b0[atom], buffers.dot_b0[atom]};
        const DualFloat efficiency{
            primal.inversion_efficiency[atom],
            buffers.dot_inversion_efficiency[atom],
        };
        const DualFloat r1 = 1000.0F * dual_reciprocal(t1);
        const DualFloat r2 = 1000.0F * dual_reciprocal(t2);
        const DualFloat damping_rate{
            primal.diffusion[atom], buffers.dot_diffusion[atom]
        };
        const float velocity = primal.velocity[atom];
        const float dot_velocity = buffers.dot_velocity[atom];
        const DualFloat flow_rate{
            velocity * primal.flow_scale, dot_velocity * primal.flow_scale
        };
        const DualFloat washout_rate{
            std::fabs(velocity) * primal.washout_scale,
            speed_direction(velocity) * dot_velocity * primal.washout_scale,
        };

        std::fill(fplus.begin(), fplus.end(), DualComplex{});
        std::fill(fminus.begin(), fminus.end(), DualComplex{});
        std::fill(longitudinal.begin(), longitudinal.end(), DualComplex{});
        longitudinal[0].value = Complex(1.0F, 0.0F);

        // ---- dual forward, recording the state entering each event ----
        for (std::int64_t event = 0; event < event_count; ++event) {
            DualComplex* slot = trajectory.data()
                + static_cast<std::size_t>(event) * stride;
            std::copy(fplus.begin(), fplus.end(), slot);
            std::copy(fminus.begin(), fminus.end(), slot + states);
            std::copy(longitudinal.begin(), longitudinal.end(), slot + 2U * states);

            const DualFloat dt{
                view.duration[event], dot_duration[event]
            };
            damping.set(damping_rate, dt);
            const DualFloat wout = washout_out(washout_rate, dt);
            const DualFloat dry1 = dual_exp(DualFloat{0.0F, 0.0F} - (r1 * dt));
            const DualFloat dry2 = dual_exp(DualFloat{0.0F, 0.0F} - (r2 * dt));
            const DualFloat e1 = dry1 * wout;
            const DualFloat e2 = dry2 * wout;
            const DualFloat angle = -2.0F * PI * (b0 * dt);
            const DualComplex off = e2 * dual_polar(angle);
            const DualComplex off_conj = conjugate(off);
            for (std::size_t state = 0; state < states; ++state) {
                const DualFloat damp_transverse = damping.transverse[state];
                DualFloat turn_longitudinal{};
                DualFloat turn_transverse{};
                flow_turn_dual(
                    flow_rate, dt, state, turn_longitudinal, turn_transverse
                );
                const DualComplex spin_transverse = dual_polar(turn_transverse);
                const DualComplex spin_longitudinal = dual_polar(turn_longitudinal);
                fplus[state] =
                    off * fplus[state] * damp_transverse * spin_transverse;
                fminus[state] = off_conj * fminus[state] * damp_transverse
                    * conjugate(spin_transverse);
                longitudinal[state] = e1 * longitudinal[state]
                    * damping.longitudinal[state] * spin_longitudinal;
            }
            longitudinal[0].value += Complex(1.0F - e1.value, 0.0F);
            longitudinal[0].tangent -= Complex(e1.tangent, 0.0F);

            const std::uint8_t action = primal.action[event];
            if ((action & PRE_SHIFT) != 0) {
                shift(fplus, fminus);
            }
            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const DualFloat negated = DualFloat{0.0F, 0.0F} - efficiency;
                    for (DualComplex& value : longitudinal) {
                        value = negated * value;
                    }
                } else {
                    const std::int64_t transmit =
                        transmit_row<SHIMMED>(primal, event, atom);
                    const DualFloat alpha =
                        DualFloat{view.flip[event], dot_flip[event]}
                        * shim_dual<SHIMMED>(
                            primal.b1, buffers.dot_b1, transmit, b1
                        );
                    const DualFloat phi =
                        DualFloat{view.phase[event], dot_phase[event]}
                        + shim_dual<SHIMMED>(
                            primal.b1_phase, buffers.dot_b1_phase, transmit,
                            b1_phase
                        );
                    if constexpr (PROFILED) {
                        DualComplex pair_a{};
                        DualComplex pair_b{};
                        DualComplex slope_a{};
                        DualComplex slope_b{};
                        profile_pair_slope_dual(
                            primal, location, alpha, pair_a, pair_b, slope_a,
                            slope_b
                        );
                        rotate_spinor(
                            fplus,
                            fminus,
                            longitudinal,
                            pair_a,
                            pair_b * dual_polar(DualFloat{0.0F, 0.0F} - phi)
                        );
                    } else {
                        rotate_dual(fplus, fminus, longitudinal, alpha, phi);
                    }
                }
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

        // ---- dual reverse ----
        std::fill(fplus_bar.begin(), fplus_bar.end(), DualComplex{});
        std::fill(fminus_bar.begin(), fminus_bar.end(), DualComplex{});
        std::fill(longitudinal_bar.begin(), longitudinal_bar.end(), DualComplex{});
        DualFloat grad_t1{0.0F, 0.0F};
        DualFloat grad_t2{0.0F, 0.0F};
        DualFloat grad_m0{0.0F, 0.0F};
        DualFloat grad_b1{0.0F, 0.0F};
        DualFloat grad_b1_phase{0.0F, 0.0F};
        // Flushed to its shim's row whenever the walk back reaches a pulse on
        // a different one; see the first-order kernel.
        std::int64_t held = 0;
        DualFloat grad_b0{0.0F, 0.0F};
        DualFloat grad_efficiency{0.0F, 0.0F};
        DualFloat grad_damping{0.0F, 0.0F};
        DualFloat grad_flow{0.0F, 0.0F};
        DualFloat grad_washout{0.0F, 0.0F};

        for (std::int64_t event = event_count - 1; event >= 0; --event) {
            const DualComplex* slot = trajectory.data()
                + static_cast<std::size_t>(event) * stride;
            std::copy(slot, slot + states, fplus.begin());
            std::copy(slot + states, slot + 2U * states, fminus.begin());
            std::copy(slot + 2U * states, slot + 3U * states, longitudinal.begin());

            const DualFloat dt{
                view.duration[event], dot_duration[event]
            };
            damping.set(damping_rate, dt);
            const DualFloat wout = washout_out(washout_rate, dt);
            const DualFloat dry1 = dual_exp(DualFloat{0.0F, 0.0F} - (r1 * dt));
            const DualFloat dry2 = dual_exp(DualFloat{0.0F, 0.0F} - (r2 * dt));
            const DualFloat e1 = dry1 * wout;
            const DualFloat e2 = dry2 * wout;
            const DualFloat angle = -2.0F * PI * (b0 * dt);
            const DualComplex phase = dual_polar(angle);
            const DualComplex off = e2 * phase;
            const DualComplex off_conj = conjugate(off);
            const std::uint8_t action = primal.action[event];

            for (std::size_t state = 0; state < states; ++state) {
                const DualFloat damp_transverse = damping.transverse[state];
                DualFloat turn_longitudinal{};
                DualFloat turn_transverse{};
                flow_turn_dual(
                    flow_rate, dt, state, turn_longitudinal, turn_transverse
                );
                const DualComplex spin_transverse = dual_polar(turn_transverse);
                const DualComplex spin_longitudinal = dual_polar(turn_longitudinal);
                fplus_relaxed[state] =
                    off * fplus[state] * damp_transverse * spin_transverse;
                fminus_relaxed[state] = off_conj * fminus[state] * damp_transverse
                    * conjugate(spin_transverse);
                longitudinal_relaxed[state] = e1 * longitudinal[state]
                    * damping.longitudinal[state] * spin_longitudinal;
            }
            longitudinal_relaxed[0].value += Complex(1.0F - e1.value, 0.0F);
            longitudinal_relaxed[0].tangent -= Complex(e1.tangent, 0.0F);

            fplus_shifted = fplus_relaxed;
            fminus_shifted = fminus_relaxed;
            if ((action & PRE_SHIFT) != 0) {
                shift(fplus_shifted, fminus_shifted);
            }

            if ((action & SPOIL_AFTER) != 0) {
                std::fill(fplus_bar.begin(), fplus_bar.end(), DualComplex{});
                std::fill(fminus_bar.begin(), fminus_bar.end(), DualComplex{});
            } else if ((action & SHIFT_AFTER) != 0) {
                shift_adjoint(fplus_bar, fminus_bar);
            }
            if ((action & POST_SHIFT) != 0) {
                shift_adjoint(fplus_bar, fminus_bar);
            }

            if (primal.kind[event] == 2 && (action & RECORD) != 0) {
                const std::int64_t output = primal.output_index[event];
                const std::int64_t index = view.output_base + output;
                const DualComplex seed{
                    Complex(
                        buffers.grad_output_real[index],
                        buffers.grad_output_imag[index]
                    ),
                    Complex{},
                };
                const DualFloat adc_phase{
                    view.phase[event], dot_phase[event]
                };
                const DualComplex demodulation =
                    dual_polar(DualFloat{0.0F, 0.0F} - adc_phase);
                const DualComplex recorded = fplus_shifted[0];
                grad_m0 = grad_m0
                    + real_part(conjugate(seed) * recorded * demodulation);
                grad_phase_train[event] = grad_phase_train[event]
                    + real_part(
                        conjugate(seed) * m0 * recorded
                        * DualComplex{Complex(0.0F, -1.0F), Complex{}}
                        * demodulation
                    );
                fplus_bar[0] =
                    fplus_bar[0] + conjugate(m0 * demodulation) * seed;
            }

            if (primal.kind[event] == 1) {
                if ((action & INVERSION) != 0) {
                    const DualFloat negated = DualFloat{0.0F, 0.0F} - efficiency;
                    for (std::size_t state = 0; state < states; ++state) {
                        grad_efficiency = grad_efficiency
                            + real_part(
                                conjugate(longitudinal_bar[state])
                                * (DualComplex{Complex(-1.0F, 0.0F), Complex{}}
                                   * longitudinal_relaxed[state])
                            );
                        longitudinal_bar[state] = negated * longitudinal_bar[state];
                    }
                } else {
                    const DualFloat flip_value{
                        view.flip[event], dot_flip[event]
                    };
                    const std::int64_t transmit =
                        transmit_row<SHIMMED>(primal, event, atom);
                    const DualFloat pulse_b1 = shim_dual<SHIMMED>(
                        primal.b1, buffers.dot_b1, transmit, b1
                    );
                    const DualFloat alpha = flip_value * pulse_b1;
                    const DualFloat phi =
                        DualFloat{view.phase[event], dot_phase[event]}
                        + shim_dual<SHIMMED>(
                            primal.b1_phase, buffers.dot_b1_phase, transmit,
                            b1_phase
                        );
                    if constexpr (SHIMMED) {
                        const std::int64_t row = primal.shim_index[event];
                        if (row != held) {
                            DualFloat& magnitude = grad_tissue_local[
                                (layout.base[B1_INDEX] + held) * atoms + atom
                            ];
                            DualFloat& angle = grad_tissue_local[
                                (layout.base[B1_PHASE_INDEX] + held) * atoms + atom
                            ];
                            magnitude = magnitude + grad_b1;
                            angle = angle + grad_b1_phase;
                            grad_b1 = DualFloat{0.0F, 0.0F};
                            grad_b1_phase = DualFloat{0.0F, 0.0F};
                            held = row;
                        }
                    }
                    DualFloat grad_alpha{0.0F, 0.0F};
                    DualFloat grad_phi{0.0F, 0.0F};
                    if constexpr (PROFILED) {
                        DualComplex pair_a{};
                        DualComplex pair_b{};
                        DualComplex slope_a{};
                        DualComplex slope_b{};
                        profile_pair_slope_dual(
                            primal, location, alpha, pair_a, pair_b, slope_a,
                            slope_b
                        );
                        const DualComplex turn =
                            dual_polar(DualFloat{0.0F, 0.0F} - phi);
                        const DualComplex spun = pair_b * turn;
                        DualComplex grad_a{};
                        DualComplex grad_b{};
                        rotate_adjoint_spinor_dual(
                            fplus_shifted,
                            fminus_shifted,
                            longitudinal_relaxed,
                            fplus_bar,
                            fminus_bar,
                            longitudinal_bar,
                            pair_a,
                            spun,
                            grad_a,
                            grad_b
                        );
                        grad_alpha = real_part(conjugate(grad_a) * slope_a)
                            + real_part(conjugate(grad_b) * (slope_b * turn));
                        grad_phi = real_part(
                            conjugate(grad_b) * (Complex(0.0F, -1.0F) * spun)
                        );
                    } else {
                        rotate_adjoint_dual(
                            fplus_shifted,
                            fminus_shifted,
                            longitudinal_relaxed,
                            fplus_bar,
                            fminus_bar,
                            longitudinal_bar,
                            alpha,
                            phi,
                            grad_alpha,
                            grad_phi
                        );
                    }
                    grad_flip_train[event] =
                        grad_flip_train[event] + grad_alpha * pulse_b1;
                    grad_b1 = grad_b1 + grad_alpha * flip_value;
                    grad_phase_train[event] = grad_phase_train[event] + grad_phi;
                    grad_b1_phase = grad_b1_phase + grad_phi;
                }
            }

            if ((action & PRE_SHIFT) != 0) {
                shift_adjoint(fplus_bar, fminus_bar);
            }

            DualFloat grad_e1{0.0F, 0.0F};
            DualFloat grad_e2{0.0F, 0.0F};
            DualFloat grad_angle{0.0F, 0.0F};
            DualFloat grad_b_factor{0.0F, 0.0F};
            DualFloat grad_turn{0.0F, 0.0F};
            const Complex imaginary(0.0F, 1.0F);
            grad_e1 = grad_e1 - real_part(longitudinal_bar[0]);
            for (std::size_t state = 0; state < states; ++state) {
                const DualComplex ap = fplus_bar[state];
                const DualComplex am = fminus_bar[state];
                const DualComplex az = longitudinal_bar[state];
                const DualFloat damp_transverse = damping.transverse[state];
                const DualFloat damp_longitudinal = damping.longitudinal[state];
                DualFloat turn_longitudinal{};
                DualFloat turn_transverse{};
                flow_turn_dual(
                    flow_rate, dt, state, turn_longitudinal, turn_transverse
                );
                const DualComplex spin_transverse = dual_polar(turn_transverse);
                const DualComplex spin_longitudinal = dual_polar(turn_longitudinal);
                const DualComplex carried =
                    phase * damp_transverse * spin_transverse;
                const DualComplex full_transverse =
                    off * damp_transverse * spin_transverse;
                const DualComplex full_longitudinal =
                    e1 * damp_longitudinal * spin_longitudinal;
                const DualFloat transverse_term = real_part(
                    conjugate(ap) * carried * fplus[state]
                    + conjugate(am) * conjugate(carried) * fminus[state]
                );
                grad_e2 = grad_e2 + transverse_term;
                // A turn of the transverse states and the off-resonance angle
                // are the same derivative; only the weight they carry differs.
                const DualFloat angle_term =
                    real_part(conjugate(ap) * (imaginary * (full_transverse * fplus[state])))
                    - real_part(
                        conjugate(am)
                        * (imaginary * (conjugate(full_transverse) * fminus[state]))
                    );
                grad_angle = grad_angle + angle_term;
                const DualFloat longitudinal_term = real_part(
                    conjugate(az) * damp_longitudinal * spin_longitudinal
                        * longitudinal[state]
                );
                grad_e1 = grad_e1 + longitudinal_term;
                const DualFloat longitudinal_angle_term = real_part(
                    conjugate(az) * (imaginary * (full_longitudinal * longitudinal[state]))
                );
                grad_b_factor = grad_b_factor
                    - transverse_weight(state) * (e2 * transverse_term)
                    - longitudinal_weight(state) * (e1 * longitudinal_term);
                grad_turn = grad_turn
                    - (static_cast<float>(state) + 0.5F) * angle_term
                    - static_cast<float>(state) * longitudinal_angle_term;
                fplus_bar[state] = conjugate(full_transverse) * ap;
                fminus_bar[state] = full_transverse * am;
                longitudinal_bar[state] = conjugate(full_longitudinal) * az;
            }

            const DualFloat inverse_t1_squared =
                dual_reciprocal(t1 * t1);
            const DualFloat inverse_t2_squared =
                dual_reciprocal(t2 * t2);
            grad_t1 = grad_t1 + grad_e1 * e1 * (1000.0F * (dt * inverse_t1_squared));
            grad_t2 = grad_t2 + grad_e2 * e2 * (1000.0F * (dt * inverse_t2_squared));
            grad_b0 = grad_b0 + grad_angle * (-2.0F * PI * dt);
            const DualFloat grad_wout =
                washout_rate.value * dt.value < 1.0F
                    ? DualFloat{0.0F, 0.0F} - (dry1 * grad_e1 + dry2 * grad_e2)
                    : DualFloat{0.0F, 0.0F};
            grad_damping = grad_damping + grad_b_factor * dt;
            grad_flow = grad_flow + grad_turn * dt;
            grad_washout = grad_washout + grad_wout * dt;
            grad_duration_train[event] = grad_duration_train[event]
                + (DualFloat{0.0F, 0.0F} - (grad_e1 * (r1 * e1)))
                - (grad_e2 * (r2 * e2))
                + grad_angle * (-2.0F * PI * b0)
                + grad_b_factor * damping_rate
                + grad_turn * flow_rate
                + grad_wout * washout_rate;
        }

        const DualFloat contributions[TISSUE_COUNT] = {
            grad_t1, grad_t2, grad_m0, grad_b1, grad_b1_phase, grad_b0,
            grad_efficiency, grad_damping,
            primal.flow_scale * grad_flow
                + (speed_direction(velocity) * primal.washout_scale)
                    * grad_washout,
        };
        for (std::size_t parameter = 0; parameter < TISSUE_COUNT; ++parameter) {
            const std::size_t plane = layout.base[parameter]
                + (parameter == B1_INDEX || parameter == B1_PHASE_INDEX
                       ? static_cast<std::size_t>(held)
                       : 0U);
            DualFloat& slot = grad_tissue_local[plane * atoms + atom];
            slot = slot + contributions[parameter];
        }
    }
}

// The packed pointers arrive in the order the parameter registry declares:
// every tissue property, then duration, kind, flip, phase, action and the
// output index. Naming them once here means a new parameter moves one table
// rather than every entry point's aggregate initializer.
inline Buffers packed_buffers(
    void* const* raw,
    float* const output_real,
    float* const output_imag,
    const std::int64_t atom_count,
    const std::int64_t train_count,
    const float flow_scale,
    const float washout_scale,
    const std::int64_t shim_count,
    const void* const profile = nullptr,
    const std::int64_t profile_bins = 0,
    const std::int64_t locations = 1,
    const float profile_step = 1.0F
) {
    Buffers buffers{};
    const float** tissue[TISSUE_COUNT] = {
        &buffers.t1, &buffers.t2, &buffers.m0, &buffers.b1,
        &buffers.b1_phase, &buffers.b0, &buffers.inversion_efficiency,
        &buffers.diffusion, &buffers.velocity,
    };
    for (std::size_t index = 0; index < TISSUE_COUNT; ++index) {
        *tissue[index] = static_cast<const float*>(raw[index]);
    }
    buffers.duration = static_cast<const float*>(raw[TISSUE_COUNT]);
    buffers.kind = static_cast<const std::int32_t*>(raw[TISSUE_COUNT + 1]);
    buffers.flip = static_cast<const float*>(raw[TISSUE_COUNT + 2]);
    buffers.phase = static_cast<const float*>(raw[TISSUE_COUNT + 3]);
    buffers.action = static_cast<const std::uint8_t*>(raw[TISSUE_COUNT + 4]);
    buffers.output_index =
        static_cast<const std::int32_t*>(raw[TISSUE_COUNT + 5]);
    buffers.shim_index = static_cast<const std::int32_t*>(raw[TISSUE_COUNT + 6]);
    buffers.output_real = output_real;
    buffers.output_imag = output_imag;
    buffers.atom_count = atom_count;
    buffers.train_count = train_count;
    buffers.flow_scale = flow_scale;
    buffers.washout_scale = washout_scale;
    buffers.shim_count = shim_count;
    buffers.profile = static_cast<const float*>(profile);
    buffers.profile_bins = profile_bins;
    buffers.locations = locations;
    buffers.profile_step = profile_step;
    return buffers;
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


// ---------------------------------------------------------------------------
// Fused A-optimal T2 optimizer.
//
// The per-iteration work outside the kernels -- rebuilding the flip buffer,
// forming the objective and its cotangent, the penalties, the Adam step -- is
// small but was costing more than the kernels themselves once it went through
// Python and autograd. Running the whole loop here leaves the interpreter out
// of it entirely.
//
// This is one specific recipe, not a general optimizer: the data term is the
// relative Cramer-Rao bound on T2 and the parameter is an FSE refocusing train
// laid out as [excitation, (refocus, adc) * etl]. Anything else belongs on the
// Python path.
// ---------------------------------------------------------------------------

struct FseT2Weights {
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    double smoothness;
    double curvature;
    double rf_power;
};

constexpr float DEGREES_TO_RADIANS = 3.14159265358979323846F / 180.0F;
constexpr float INFORMATION_FLOOR = 1e-12F;

// Writes the refocusing flip angles into the event buffer. Every other slot --
// the excitation and the readouts -- was filled when the plan was built and
// does not change with the parameter.
void write_refocusing_flips(
    float* const flip,
    const float* const flip_deg,
    const std::int64_t train_count,
    const std::int64_t event_count,
    const std::int64_t echo_train_length
) {
    for (std::int64_t train = 0; train < train_count; ++train) {
        const float* const source = flip_deg + train * echo_train_length;
        float* const target = flip + train * event_count;
        for (std::int64_t echo = 0; echo < echo_train_length; ++echo) {
            target[1 + 2 * echo] = source[echo] * DEGREES_TO_RADIANS;
        }
    }
}

// The data term and its cotangent on the jacobian.
//
// Fisher information per (train, atom) is the squared norm of dS/dT2 along the
// echo axis; the relative Cramer-Rao bound is its reciprocal over T2 squared,
// and the objective is the logarithm of that averaged over design points.
double t2_precision_cotangent(
    const float* const jacobian_real,
    const float* const jacobian_imag,
    const float* const t2_ms,
    float* const grad_real,
    float* const grad_imag,
    const std::int64_t train_count,
    const std::int64_t atom_count,
    const std::int64_t output_count
) {
    const std::int64_t points = train_count * atom_count;
    const double scale = 1.0 / static_cast<double>(points);
    std::vector<double> information(static_cast<std::size_t>(points), 0.0);
    double precision = 0.0;
    for (std::int64_t point = 0; point < points; ++point) {
        const std::int64_t base = point * output_count;
        double total = 0.0;
        for (std::int64_t echo = 0; echo < output_count; ++echo) {
            const double real = jacobian_real[base + echo];
            const double imag = jacobian_imag[base + echo];
            total += real * real + imag * imag;
        }
        information[static_cast<std::size_t>(point)] = total;
        const double t2 = t2_ms[point % atom_count];
        precision += scale / (std::max(total, static_cast<double>(INFORMATION_FLOOR)) * t2 * t2);
    }

    for (std::int64_t point = 0; point < points; ++point) {
        const std::int64_t base = point * output_count;
        const double total = information[static_cast<std::size_t>(point)];
        const double t2 = t2_ms[point % atom_count];
        // A floored information carries no derivative, matching a clamp.
        double sensitivity = 0.0;
        if (total >= static_cast<double>(INFORMATION_FLOOR)) {
            sensitivity = -scale / (total * total * t2 * t2) / precision;
        }
        for (std::int64_t echo = 0; echo < output_count; ++echo) {
            grad_real[base + echo] =
                static_cast<float>(2.0 * sensitivity * jacobian_real[base + echo]);
            grad_imag[base + echo] =
                static_cast<float>(2.0 * sensitivity * jacobian_imag[base + echo]);
        }
    }
    return std::log(precision);
}

// Penalties on flip angles normalized by 180 degrees, and their gradients.
// Returns the penalty contribution to the loss and adds to ``gradient``, which
// is expressed per degree.
double flip_penalties(
    const float* const flip_deg,
    double* const gradient,
    const std::int64_t train_count,
    const std::int64_t echo_train_length,
    const FseT2Weights& weights
) {
    const double normalize = 1.0 / 180.0;
    const std::int64_t slopes = echo_train_length > 1 ? echo_train_length - 1 : 0;
    const std::int64_t bends = echo_train_length > 2 ? echo_train_length - 2 : 0;
    double smoothness = 0.0;
    double curvature = 0.0;
    double power = 0.0;

    const double slope_scale =
        slopes > 0 ? 1.0 / static_cast<double>(train_count * slopes) : 0.0;
    const double bend_scale =
        bends > 0 ? 1.0 / static_cast<double>(train_count * bends) : 0.0;
    const double power_scale =
        1.0 / static_cast<double>(train_count * echo_train_length);

    for (std::int64_t train = 0; train < train_count; ++train) {
        const std::int64_t base = train * echo_train_length;
        for (std::int64_t echo = 0; echo + 1 < echo_train_length; ++echo) {
            const double difference =
                (flip_deg[base + echo + 1] - flip_deg[base + echo]) * normalize;
            smoothness += slope_scale * difference * difference;
            const double sensitivity =
                weights.smoothness * 2.0 * difference * slope_scale * normalize;
            gradient[base + echo + 1] += sensitivity;
            gradient[base + echo] -= sensitivity;
        }
        for (std::int64_t echo = 0; echo + 2 < echo_train_length; ++echo) {
            const double bend = (flip_deg[base + echo + 2]
                                 - 2.0 * flip_deg[base + echo + 1]
                                 + flip_deg[base + echo]) * normalize;
            curvature += bend_scale * bend * bend;
            const double sensitivity =
                weights.curvature * 2.0 * bend * bend_scale * normalize;
            gradient[base + echo + 2] += sensitivity;
            gradient[base + echo + 1] -= 2.0 * sensitivity;
            gradient[base + echo] += sensitivity;
        }
        for (std::int64_t echo = 0; echo < echo_train_length; ++echo) {
            const double value = flip_deg[base + echo] * normalize;
            power += power_scale * value * value;
            gradient[base + echo] +=
                weights.rf_power * 2.0 * value * power_scale * normalize;
        }
    }
    return weights.smoothness * smoothness + weights.curvature * curvature
        + weights.rf_power * power;
}

// One Adam step, matching torch.optim.Adam's bias correction.
void adam_step(
    float* const parameter,
    const double* const gradient,
    float* const moment,
    float* const velocity,
    const std::int64_t count,
    const std::int64_t step,
    const FseT2Weights& weights
) {
    const double correction1 = 1.0 - std::pow(weights.beta1, static_cast<double>(step));
    const double correction2 = 1.0 - std::pow(weights.beta2, static_cast<double>(step));
    const double step_size = weights.learning_rate / correction1;
    const double correction2_root = std::sqrt(correction2);
    for (std::int64_t index = 0; index < count; ++index) {
        const double value = gradient[index];
        const double first =
            weights.beta1 * moment[index] + (1.0 - weights.beta1) * value;
        const double second =
            weights.beta2 * velocity[index] + (1.0 - weights.beta2) * value * value;
        moment[index] = static_cast<float>(first);
        velocity[index] = static_cast<float>(second);
        const double denominator = std::sqrt(second) / correction2_root + weights.epsilon;
        parameter[index] =
            static_cast<float>(parameter[index] - step_size * first / denominator);
    }
}

PyObject* simulate(PyObject*, PyObject* arguments) {
    PyObject* pointers = nullptr;
    long long atom_count = 0;
    long long train_count = 0;
    long long event_count = 0;
    long long state_count = 0;
    long long output_count = 0;
    int requested_threads = 0;
    int real_axis = -1;
    double flow_scale = 0.0;
    double washout_scale = 0.0;
    long long shim_count = 1;
    long long locations = 1;
    long long profile_bins = 0;
    double profile_step = 1.0;
    if (!PyArg_ParseTuple(
            arguments,
            "OLLLLLiiddLLLd",
            &pointers,
            &atom_count,
            &train_count,
            &event_count,
            &state_count,
            &output_count,
            &requested_threads,
            &real_axis,
            &flow_scale,
            &washout_scale,
            &shim_count,
            &locations,
            &profile_bins,
            &profile_step
        )) {
        return nullptr;
    }
    // The packed buffers, the two output planes, then the transition table --
    // null when the sequence has none.
    constexpr Py_ssize_t expected = PACKED_COUNT + 3;
    if (!PySequence_Check(pointers) || PySequence_Size(pointers) != expected) {
        PyErr_SetString(PyExc_ValueError, "wrong number of buffer pointers");
        return nullptr;
    }
    if (atom_count < 0 || train_count < 1 || event_count < 0 || state_count < 1
        || output_count < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid EPG buffer dimensions");
        return nullptr;
    }

    void* raw[expected]{};
    for (Py_ssize_t index = 0; index < expected; ++index) {
        if (!parse_pointer(pointers, index, &raw[index])) {
            return nullptr;
        }
    }
    const Buffers buffers = packed_buffers(
        raw,
        static_cast<float*>(raw[PACKED_COUNT]),
        static_cast<float*>(raw[PACKED_COUNT + 1]),
        static_cast<std::int64_t>(atom_count),
        static_cast<std::int64_t>(train_count),
        static_cast<float>(flow_scale),
        static_cast<float>(washout_scale),
        static_cast<std::int64_t>(shim_count),
        raw[PACKED_COUNT + 2],
        static_cast<std::int64_t>(profile_bins),
        static_cast<std::int64_t>(locations),
        static_cast<float>(profile_step)
    );

    // TORCHSIM_LANES=1 selects a lane-vectorized forward that walks the
    // (atom, train block) product instead of (atom, train), putting a block of
    // trains in the SIMD lanes. It is opt-in: it is measurably faster only for
    // very wide batches, and it does not agree bit for bit with the scalar
    // kernel, so enabling it changes results in the last place.
    const char* const lane_override = std::getenv("TORCHSIM_LANES");
    const bool lanes_enabled = lane_override != nullptr && lane_override[0] == '1';
    const bool vectorize = lanes_enabled && train_count >= 4
        && buffers.shim_count == 1
        && buffers.profile == nullptr
        && !any_diffusion(buffers.diffusion, buffers.atom_count)
        && !any_diffusion(buffers.velocity, buffers.atom_count);
    const std::int64_t lane_blocks =
        (static_cast<std::int64_t>(train_count) + static_cast<std::int64_t>(LANES) - 1)
        / static_cast<std::int64_t>(LANES);
    const std::int64_t work_count = static_cast<std::int64_t>(atom_count)
        * (vectorize ? lane_blocks : static_cast<std::int64_t>(train_count));
    void (*kernel)(
        const Buffers&, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t
    ) = vectorize ? &simulate_lane_range : &simulate_range<false, false>;
    if (!vectorize) {
        const bool shimmed = buffers.shim_count > 1;
        const bool profiled = buffers.profile != nullptr;
        if (shimmed) {
            kernel = profiled ? &simulate_range<true, true>
                              : &simulate_range<true, false>;
        } else if (profiled) {
            kernel = &simulate_range<false, true>;
        }
    }
    // The caller establishes the real-subspace conditions; axis 1 puts the
    // signal on the imaginary axis, which is the representation below.
    if (real_axis == 1) {
        kernel = &simulate_real_range;
    }
    const unsigned int thread_count = worker_count(requested_threads, work_count);

    Py_BEGIN_ALLOW_THREADS
    {
        const std::int64_t block = (work_count + thread_count - 1) / thread_count;
        WorkerPool::instance().run(thread_count, [&](const unsigned int slot) {
            const std::int64_t begin = static_cast<std::int64_t>(slot) * block;
            const std::int64_t end = std::min<std::int64_t>(work_count, begin + block);
            if (begin < end) {
                kernel(
                    buffers, begin, end, event_count, state_count, output_count
                );
            }
        });
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

// Spread one forward-mode pass over the pool. Shared by the standalone entry
// point and by the fused optimizer, so both pick the same kernel for a given
// subspace verdict.
void dispatch_jvp(
    const JvpBuffers& buffers,
    const std::int64_t atom_count,
    const std::int64_t train_count,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count,
    const int requested_threads,
    const int real_axis
) {
    const bool lanes = real_axis == 1 && lane_kernels_enabled()
        && !any_diffusion(buffers.primal.diffusion, atom_count)
        && !any_diffusion(buffers.diffusion, atom_count);
    void (*kernel)(
        const JvpBuffers&, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t
    ) = &simulate_jvp_range<false, false>;
    if (lanes) {
        kernel = &simulate_real_jvp_lane_range;
    } else if (real_axis == 1) {
        kernel = &simulate_real_jvp_range;
    } else {
        const bool shimmed = buffers.primal.shim_count > 1;
        const bool profiled = buffers.primal.profile != nullptr;
        if (shimmed) {
            kernel = profiled ? &simulate_jvp_range<true, true>
                              : &simulate_jvp_range<true, false>;
        } else if (profiled) {
            kernel = &simulate_jvp_range<false, true>;
        }
    }
    // A lane kernel's work item covers a block of trains rather than one train.
    const std::int64_t work_count =
        atom_count * (lanes ? lane_blocks(train_count) : train_count);
    const unsigned int thread_count = worker_count(requested_threads, work_count);
    const std::int64_t block = (work_count + thread_count - 1) / thread_count;
    WorkerPool::instance().run(thread_count, [&](const unsigned int slot) {
        const std::int64_t begin = static_cast<std::int64_t>(slot) * block;
        const std::int64_t end = std::min<std::int64_t>(work_count, begin + block);
        if (begin < end) {
            kernel(buffers, begin, end, event_count, state_count, output_count);
        }
    });
}

PyObject* simulate_jvp(PyObject*, PyObject* arguments) {
    PyObject* pointers = nullptr;
    long long atom_count = 0;
    long long train_count = 0;
    long long event_count = 0;
    long long state_count = 0;
    long long output_count = 0;
    int requested_threads = 0;
    int real_axis = -1;
    double flow_scale = 0.0;
    double washout_scale = 0.0;
    long long shim_count = 1;
    long long locations = 1;
    long long profile_bins = 0;
    double profile_step = 1.0;
    if (!PyArg_ParseTuple(
            arguments,
            "OLLLLLiiddLLLd",
            &pointers,
            &atom_count,
            &train_count,
            &event_count,
            &state_count,
            &output_count,
            &requested_threads,
            &real_axis,
            &flow_scale,
            &washout_scale,
            &shim_count,
            &locations,
            &profile_bins,
            &profile_step
        )) {
        return nullptr;
    }
    // The packed buffers, one tangent per differentiable input, the two output
    // planes, then the transition table -- null when there is none.
    constexpr Py_ssize_t expected = PACKED_COUNT + FLOAT_COUNT + 3;
    if (!PySequence_Check(pointers) || PySequence_Size(pointers) != expected) {
        PyErr_SetString(PyExc_ValueError, "wrong number of buffer pointers");
        return nullptr;
    }
    if (atom_count < 0 || train_count < 1 || event_count < 0 || state_count < 1
        || output_count < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid EPG buffer dimensions");
        return nullptr;
    }

    void* raw[expected]{};
    for (Py_ssize_t index = 0; index < expected; ++index) {
        if (!parse_pointer(pointers, index, &raw[index])) {
            return nullptr;
        }
    }
    constexpr std::size_t tangents = PACKED_COUNT;
    constexpr std::size_t outputs = tangents + FLOAT_COUNT;
    const Buffers primal = packed_buffers(
        raw,
        static_cast<float*>(raw[outputs]),
        static_cast<float*>(raw[outputs + 1]),
        static_cast<std::int64_t>(atom_count),
        static_cast<std::int64_t>(train_count),
        static_cast<float>(flow_scale),
        static_cast<float>(washout_scale),
        static_cast<std::int64_t>(shim_count),
        raw[expected - 1],
        static_cast<std::int64_t>(profile_bins),
        static_cast<std::int64_t>(locations),
        static_cast<float>(profile_step)
    );
    const JvpBuffers buffers{
        primal,
        static_cast<const float*>(raw[tangents]),
        static_cast<const float*>(raw[tangents + 1]),
        static_cast<const float*>(raw[tangents + 2]),
        static_cast<const float*>(raw[tangents + 3]),
        static_cast<const float*>(raw[tangents + 4]),
        static_cast<const float*>(raw[tangents + 5]),
        static_cast<const float*>(raw[tangents + 6]),
        static_cast<const float*>(raw[tangents + 7]),
        static_cast<const float*>(raw[tangents + 8]),
        static_cast<const float*>(raw[tangents + 9]),
        static_cast<const float*>(raw[tangents + 10]),
        static_cast<const float*>(raw[tangents + 11]),
    };

    Py_BEGIN_ALLOW_THREADS
    dispatch_jvp(
        buffers, atom_count, train_count, event_count, state_count, output_count,
        requested_threads, real_axis
    );
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

PyObject* simulate_vjp(PyObject*, PyObject* arguments) {
    PyObject* pointers = nullptr;
    long long atom_count = 0;
    long long train_count = 0;
    long long event_count = 0;
    long long state_count = 0;
    long long output_count = 0;
    int requested_threads = 0;
    double flow_scale = 0.0;
    double washout_scale = 0.0;
    long long shim_count = 1;
    long long locations = 1;
    long long profile_bins = 0;
    double profile_step = 1.0;
    if (!PyArg_ParseTuple(
            arguments,
            "OLLLLLiddLLLd",
            &pointers,
            &atom_count,
            &train_count,
            &event_count,
            &state_count,
            &output_count,
            &requested_threads,
            &flow_scale,
            &washout_scale,
            &shim_count,
            &locations,
            &profile_bins,
            &profile_step
        )) {
        return nullptr;
    }
    // The packed buffers, the two seed planes, one gradient per differentiable
    // input, then the transition table -- null when there is none.
    constexpr Py_ssize_t expected = PACKED_COUNT + 2 + FLOAT_COUNT + 1;
    if (!PySequence_Check(pointers) || PySequence_Size(pointers) != expected) {
        PyErr_SetString(PyExc_ValueError, "wrong number of buffer pointers");
        return nullptr;
    }
    if (atom_count < 0 || train_count < 1 || event_count < 0 || state_count < 1
        || output_count < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid EPG buffer dimensions");
        return nullptr;
    }

    void* raw[expected]{};
    for (Py_ssize_t index = 0; index < expected; ++index) {
        if (!parse_pointer(pointers, index, &raw[index])) {
            return nullptr;
        }
    }
    constexpr std::size_t seed = PACKED_COUNT;
    constexpr std::size_t grads = seed + 2;
    const Buffers primal = packed_buffers(
        raw, nullptr, nullptr,
        static_cast<std::int64_t>(atom_count),
        static_cast<std::int64_t>(train_count),
        static_cast<float>(flow_scale),
        static_cast<float>(washout_scale),
        static_cast<std::int64_t>(shim_count),
        raw[expected - 1],
        static_cast<std::int64_t>(profile_bins),
        static_cast<std::int64_t>(locations),
        static_cast<float>(profile_step)
    );
    VjpBuffers buffers{};
    buffers.primal = primal;
    buffers.grad_output_real = static_cast<const float*>(raw[seed]);
    buffers.grad_output_imag = static_cast<const float*>(raw[seed + 1]);
    float** const grad_slots[] = {
        &buffers.grad_t1, &buffers.grad_t2, &buffers.grad_m0, &buffers.grad_b1,
        &buffers.grad_b1_phase, &buffers.grad_b0,
        &buffers.grad_inversion_efficiency, &buffers.grad_diffusion,
        &buffers.grad_velocity,
        &buffers.grad_flip, &buffers.grad_phase, &buffers.grad_duration,
    };
    for (std::size_t index = 0; index < FLOAT_COUNT; ++index) {
        *grad_slots[index] = static_cast<float*>(raw[grads + index]);
    }

    const std::int64_t work_count =
        static_cast<std::int64_t>(atom_count) * static_cast<std::int64_t>(train_count);
    const unsigned int thread_count = worker_count(requested_threads, work_count);

    const std::size_t events =
        static_cast<std::size_t>(event_count) * static_cast<std::size_t>(train_count);
    const std::size_t atoms = static_cast<std::size_t>(atom_count);
    const TissueLayout layout(buffers.primal.shim_count);
    std::vector<float> shared(3U * events * thread_count, 0.0F);
    std::vector<float> shared_tissue(layout.rows * atoms * thread_count, 0.0F);

    Py_BEGIN_ALLOW_THREADS
    auto slice = [&](unsigned int thread, std::size_t which) {
        return shared.data() + (static_cast<std::size_t>(thread) * 3U + which) * events;
    };
    auto tissue_slice = [&](unsigned int thread) {
        return shared_tissue.data()
            + static_cast<std::size_t>(thread) * layout.rows * atoms;
    };
    {
        void (*kernel)(
            const VjpBuffers&, std::int64_t, std::int64_t, std::int64_t,
            std::int64_t, std::int64_t, float*, float*, float*, float*
        ) = &simulate_vjp_range<false, false>;
        const bool shimmed = buffers.primal.shim_count > 1;
        const bool profiled = buffers.primal.profile != nullptr;
        if (shimmed) {
            kernel = profiled ? &simulate_vjp_range<true, true>
                              : &simulate_vjp_range<true, false>;
        } else if (profiled) {
            kernel = &simulate_vjp_range<false, true>;
        }
        const std::int64_t block = (work_count + thread_count - 1) / thread_count;
        WorkerPool::instance().run(thread_count, [&](const unsigned int slot) {
            const std::int64_t begin = static_cast<std::int64_t>(slot) * block;
            const std::int64_t end = std::min<std::int64_t>(work_count, begin + block);
            if (begin < end) {
                kernel(
                    buffers, begin, end, event_count, state_count, output_count,
                    slice(slot, 0), slice(slot, 1), slice(slot, 2),
                    tissue_slice(slot)
                );
            }
        });
    }
    // Tissue gradients sum over every train, reduced in ascending thread order
    // for the same bitwise reproducibility as the per-event buffers.
    {
        float* const destinations[TISSUE_COUNT] = {
            buffers.grad_t1, buffers.grad_t2, buffers.grad_m0, buffers.grad_b1,
            buffers.grad_b1_phase, buffers.grad_b0,
            buffers.grad_inversion_efficiency, buffers.grad_diffusion,
            buffers.grad_velocity,
        };
        for (std::size_t parameter = 0; parameter < TISSUE_COUNT; ++parameter) {
            const std::size_t rows = tissue_rows(parameter, buffers.primal.shim_count);
            for (std::size_t row = 0; row < rows; ++row) {
                const std::size_t plane = layout.base[parameter] + row;
                for (std::size_t atom = 0; atom < atoms; ++atom) {
                    float total = 0.0F;
                    for (unsigned int thread = 0; thread < thread_count; ++thread) {
                        total += shared_tissue[
                            (static_cast<std::size_t>(thread) * layout.rows + plane)
                                * atoms + atom
                        ];
                    }
                    destinations[parameter][row * atoms + atom] = total;
                }
            }
        }
    }
    // Reduce in ascending thread order so the sum is bitwise reproducible and
    // independent of how the workers were scheduled.
    for (std::size_t event = 0; event < events; ++event) {
        float flip = 0.0F;
        float phase = 0.0F;
        float duration = 0.0F;
        for (unsigned int thread = 0; thread < thread_count; ++thread) {
            flip += shared[(static_cast<std::size_t>(thread) * 3U + 0U) * events + event];
            phase += shared[(static_cast<std::size_t>(thread) * 3U + 1U) * events + event];
            duration += shared[(static_cast<std::size_t>(thread) * 3U + 2U) * events + event];
        }
        buffers.grad_flip[event] = flip;
        buffers.grad_phase[event] = phase;
        buffers.grad_duration[event] = duration;
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

// Spread one forward-over-reverse pass over the pool, then fold the workers'
// partial results into the destination buffers.
void dispatch_second_order(
    const VjpJvpBuffers& buffers,
    const std::int64_t atom_count,
    const std::int64_t train_count,
    const std::int64_t event_count,
    const std::int64_t state_count,
    const std::int64_t output_count,
    const int requested_threads,
    const int real_axis
) {
    const bool lanes = real_axis == 1 && lane_kernels_enabled()
        && !any_diffusion(buffers.primal.diffusion, atom_count)
        && !any_diffusion(buffers.dot_diffusion, atom_count);
    void (*kernel)(
        const VjpJvpBuffers&, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
        std::int64_t, DualFloat*, DualFloat*, DualFloat*, DualFloat*
    ) = &simulate_vjp_jvp_range<false, false>;
    if (lanes) {
        kernel = &simulate_real_vjp_jvp_lane_range;
    } else if (real_axis == 1) {
        kernel = &simulate_real_vjp_jvp_range;
    } else {
        const bool shimmed = buffers.primal.shim_count > 1;
        const bool profiled = buffers.primal.profile != nullptr;
        if (shimmed) {
            kernel = profiled ? &simulate_vjp_jvp_range<true, true>
                              : &simulate_vjp_jvp_range<true, false>;
        } else if (profiled) {
            kernel = &simulate_vjp_jvp_range<false, true>;
        }
    }
    const std::int64_t work_count =
        atom_count * (lanes ? lane_blocks(train_count) : train_count);
    const unsigned int thread_count = worker_count(requested_threads, work_count);

    const std::size_t events =
        static_cast<std::size_t>(event_count) * static_cast<std::size_t>(train_count);
    const std::size_t atoms = static_cast<std::size_t>(atom_count);
    // Workers take disjoint trains, and an event gradient belongs to exactly one
    // train, so all of them accumulate into one buffer without sharing a slot.
    // Only the tissue gradients, which every train contributes to, need a copy
    // per worker and a reduction.
    const TissueLayout layout(buffers.primal.shim_count);
    std::vector<DualFloat> shared(3U * events, DualFloat{0.0F, 0.0F});
    std::vector<DualFloat> shared_tissue(
        layout.rows * atoms * thread_count, DualFloat{0.0F, 0.0F}
    );
    auto slice = [&](std::size_t which) { return shared.data() + which * events; };
    auto tissue_slice = [&](unsigned int slot) {
        return shared_tissue.data()
            + static_cast<std::size_t>(slot) * layout.rows * atoms;
    };
    {
        // Round the block up to a whole number of trains so no two workers hold
        // the same train's event gradients.
        const std::int64_t trains_per_block =
            (work_count / atom_count + thread_count - 1) / thread_count;
        const std::int64_t block = trains_per_block * atom_count;
        WorkerPool::instance().run(thread_count, [&](const unsigned int slot) {
            const std::int64_t begin = static_cast<std::int64_t>(slot) * block;
            const std::int64_t end = std::min<std::int64_t>(work_count, begin + block);
            if (begin < end) {
                kernel(
                    buffers, begin, end, event_count, state_count, output_count,
                    slice(0), slice(1), slice(2), tissue_slice(slot)
                );
            }
        });
    }
    {
        float* const value_destinations[TISSUE_COUNT] = {
            buffers.grad_dot_t1, buffers.grad_dot_t2, buffers.grad_dot_m0,
            buffers.grad_dot_b1, buffers.grad_dot_b1_phase, buffers.grad_dot_b0,
            buffers.grad_dot_inversion_efficiency, buffers.grad_dot_diffusion,
            buffers.grad_dot_velocity,
        };
        float* const tangent_destinations[TISSUE_COUNT] = {
            buffers.grad_t1, buffers.grad_t2, buffers.grad_m0, buffers.grad_b1,
            buffers.grad_b1_phase, buffers.grad_b0,
            buffers.grad_inversion_efficiency, buffers.grad_diffusion,
            buffers.grad_velocity,
        };
        for (std::size_t parameter = 0; parameter < TISSUE_COUNT; ++parameter) {
            const std::size_t rows = tissue_rows(parameter, buffers.primal.shim_count);
            for (std::size_t row = 0; row < rows; ++row) {
                const std::size_t plane = layout.base[parameter] + row;
                for (std::size_t atom = 0; atom < atoms; ++atom) {
                    DualFloat total{0.0F, 0.0F};
                    for (unsigned int slot = 0; slot < thread_count; ++slot) {
                        total = total + shared_tissue[
                            (static_cast<std::size_t>(slot) * layout.rows + plane)
                                * atoms + atom
                        ];
                    }
                    value_destinations[parameter][row * atoms + atom] = total.value;
                    tangent_destinations[parameter][row * atoms + atom] = total.tangent;
                }
            }
        }
    }
    for (std::size_t event = 0; event < events; ++event) {
        const DualFloat flip = shared[event];
        const DualFloat phase = shared[events + event];
        const DualFloat duration = shared[2U * events + event];
        buffers.grad_dot_flip[event] = flip.value;
        buffers.grad_flip[event] = flip.tangent;
        buffers.grad_dot_phase[event] = phase.value;
        buffers.grad_phase[event] = phase.tangent;
        buffers.grad_dot_duration[event] = duration.value;
        buffers.grad_duration[event] = duration.tangent;
    }
}

PyObject* simulate_vjp_jvp(PyObject*, PyObject* arguments) {
    PyObject* pointers = nullptr;
    long long atom_count = 0;
    long long train_count = 0;
    long long event_count = 0;
    long long state_count = 0;
    long long output_count = 0;
    int requested_threads = 0;
    int real_axis = -1;
    double flow_scale = 0.0;
    double washout_scale = 0.0;
    long long shim_count = 1;
    long long locations = 1;
    long long profile_bins = 0;
    double profile_step = 1.0;
    if (!PyArg_ParseTuple(
            arguments,
            "OLLLLLiiddLLLd",
            &pointers,
            &atom_count,
            &train_count,
            &event_count,
            &state_count,
            &output_count,
            &requested_threads,
            &real_axis,
            &flow_scale,
            &washout_scale,
            &shim_count,
            &locations,
            &profile_bins,
            &profile_step
        )) {
        return nullptr;
    }
    // The packed buffers, the tangents, the two seed planes, both gradient
    // blocks, then the transition table -- null when there is none.
    constexpr Py_ssize_t expected = PACKED_COUNT + 3 * FLOAT_COUNT + 2 + 1;
    if (!PySequence_Check(pointers) || PySequence_Size(pointers) != expected) {
        PyErr_SetString(PyExc_ValueError, "wrong number of buffer pointers");
        return nullptr;
    }
    if (atom_count < 0 || train_count < 1 || event_count < 0 || state_count < 1
        || output_count < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid EPG buffer dimensions");
        return nullptr;
    }

    void* raw[expected]{};
    for (Py_ssize_t index = 0; index < expected; ++index) {
        if (!parse_pointer(pointers, index, &raw[index])) {
            return nullptr;
        }
    }
    const Buffers primal = packed_buffers(
        raw, nullptr, nullptr,
        static_cast<std::int64_t>(atom_count),
        static_cast<std::int64_t>(train_count),
        static_cast<float>(flow_scale),
        static_cast<float>(washout_scale),
        static_cast<std::int64_t>(shim_count),
        raw[expected - 1],
        static_cast<std::int64_t>(profile_bins),
        static_cast<std::int64_t>(locations),
        static_cast<float>(profile_step)
    );
    VjpJvpBuffers buffers{};
    buffers.primal = primal;
    constexpr std::size_t tangents = PACKED_COUNT;
    constexpr std::size_t seed = tangents + FLOAT_COUNT;
    constexpr std::size_t values = seed + 2;
    constexpr std::size_t tangent_grads = values + FLOAT_COUNT;
    const float** tangent_slots[] = {
        &buffers.dot_t1, &buffers.dot_t2, &buffers.dot_m0, &buffers.dot_b1,
        &buffers.dot_b1_phase, &buffers.dot_b0,
        &buffers.dot_inversion_efficiency, &buffers.dot_diffusion,
        &buffers.dot_velocity,
        &buffers.dot_duration, &buffers.dot_flip, &buffers.dot_phase,
    };
    for (std::size_t index = 0; index < FLOAT_COUNT; ++index) {
        *tangent_slots[index] = static_cast<const float*>(raw[tangents + index]);
    }
    buffers.grad_output_real = static_cast<const float*>(raw[seed]);
    buffers.grad_output_imag = static_cast<const float*>(raw[seed + 1]);
    float** value_slots[] = {
        &buffers.grad_dot_t1, &buffers.grad_dot_t2, &buffers.grad_dot_m0,
        &buffers.grad_dot_b1, &buffers.grad_dot_b1_phase, &buffers.grad_dot_b0,
        &buffers.grad_dot_inversion_efficiency, &buffers.grad_dot_diffusion,
        &buffers.grad_dot_velocity,
        &buffers.grad_dot_duration, &buffers.grad_dot_flip,
        &buffers.grad_dot_phase,
    };
    float** tangent_grad_slots[] = {
        &buffers.grad_t1, &buffers.grad_t2, &buffers.grad_m0, &buffers.grad_b1,
        &buffers.grad_b1_phase, &buffers.grad_b0,
        &buffers.grad_inversion_efficiency, &buffers.grad_diffusion,
        &buffers.grad_velocity,
        &buffers.grad_duration, &buffers.grad_flip, &buffers.grad_phase,
    };
    for (std::size_t index = 0; index < FLOAT_COUNT; ++index) {
        *value_slots[index] = static_cast<float*>(raw[values + index]);
        *tangent_grad_slots[index] =
            static_cast<float*>(raw[tangent_grads + index]);
    }

    Py_BEGIN_ALLOW_THREADS
    dispatch_second_order(
        buffers, atom_count, train_count, event_count, state_count, output_count,
        requested_threads, real_axis
    );
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}


PyObject* optimize_fse_t2(PyObject*, PyObject* arguments) {
    PyObject* pointers = nullptr;
    PyObject* integers = nullptr;
    PyObject* reals = nullptr;
    if (!PyArg_ParseTuple(arguments, "OOO", &pointers, &integers, &reals)) {
        return nullptr;
    }
    // The packed buffers, one tangent each, the signal and cotangent pairs,
    // both gradient groups, then the optimizer's flip angles, moments,
    // velocities and T2 design points.
    constexpr Py_ssize_t expected = PACKED_COUNT + 3 * FLOAT_COUNT + 4 + 4;
    if (!PySequence_Check(pointers) || PySequence_Size(pointers) != expected) {
        PyErr_SetString(PyExc_ValueError, "wrong number of buffer pointers");
        return nullptr;
    }
    if (!PySequence_Check(integers) || PySequence_Size(integers) != 9
        || !PySequence_Check(reals) || PySequence_Size(reals) != 7) {
        PyErr_SetString(PyExc_ValueError, "expected nine sizes and seven weights");
        return nullptr;
    }

    void* raw[expected]{};
    for (Py_ssize_t index = 0; index < expected; ++index) {
        if (!parse_pointer(pointers, index, &raw[index])) {
            return nullptr;
        }
    }
    std::int64_t sizes[9]{};
    for (Py_ssize_t index = 0; index < 9; ++index) {
        PyObject* item = PySequence_GetItem(integers, index);
        if (item == nullptr) {
            return nullptr;
        }
        sizes[index] = PyLong_AsLongLong(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            return nullptr;
        }
    }
    double values[7]{};
    for (Py_ssize_t index = 0; index < 7; ++index) {
        PyObject* item = PySequence_GetItem(reals, index);
        if (item == nullptr) {
            return nullptr;
        }
        values[index] = PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            return nullptr;
        }
    }

    const std::int64_t atom_count = sizes[0];
    const std::int64_t train_count = sizes[1];
    const std::int64_t event_count = sizes[2];
    const std::int64_t state_count = sizes[3];
    const std::int64_t output_count = sizes[4];
    const std::int64_t echo_train_length = sizes[5];
    const std::int64_t iterations = sizes[6];
    const int requested_threads = static_cast<int>(sizes[7]);
    const int real_axis = static_cast<int>(sizes[8]);
    if (atom_count < 1 || train_count < 1 || event_count < 1 || state_count < 1
        || output_count < 1 || echo_train_length < 1 || iterations < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid optimizer dimensions");
        return nullptr;
    }
    const FseT2Weights weights{
        values[0], values[1], values[2], values[3], values[4], values[5], values[6],
    };

    // The caller lays these out as: the packed buffers, one tangent per
    // differentiable input, the signal, the cotangent, then the two gradient
    // groups and the optimizer's own state.
    constexpr std::size_t tangents = PACKED_COUNT;
    constexpr std::size_t signal = tangents + FLOAT_COUNT;
    constexpr std::size_t cotangent = signal + 2;
    constexpr std::size_t value_grads = cotangent + 2;
    constexpr std::size_t tangent_grads = value_grads + FLOAT_COUNT;
    constexpr std::size_t optimizer = tangent_grads + FLOAT_COUNT;
    Buffers primal = packed_buffers(
        raw,
        static_cast<float*>(raw[signal]),
        static_cast<float*>(raw[signal + 1]),
        atom_count,
        train_count,
        // This recipe declares no unbalanced gradient and no voxel size, so the
        // order-weighted terms have no geometry to act through. The Python side
        // refuses a tissue that asks for them.
        0.0F,
        0.0F,
        1
    );
    const JvpBuffers forward{
        primal,
        static_cast<const float*>(raw[tangents]),
        static_cast<const float*>(raw[tangents + 1]),
        static_cast<const float*>(raw[tangents + 2]),
        static_cast<const float*>(raw[tangents + 3]),
        static_cast<const float*>(raw[tangents + 4]),
        static_cast<const float*>(raw[tangents + 5]),
        static_cast<const float*>(raw[tangents + 6]),
        static_cast<const float*>(raw[tangents + 7]),
        static_cast<const float*>(raw[tangents + 8]),
        static_cast<const float*>(raw[tangents + 9]),
        static_cast<const float*>(raw[tangents + 10]),
        static_cast<const float*>(raw[tangents + 11]),
    };
    VjpJvpBuffers second{};
    second.primal = primal;
    const float** dot_slots[] = {
        &second.dot_t1, &second.dot_t2, &second.dot_m0, &second.dot_b1,
        &second.dot_b1_phase, &second.dot_b0,
        &second.dot_inversion_efficiency, &second.dot_diffusion,
        &second.dot_velocity,
        &second.dot_duration, &second.dot_flip, &second.dot_phase,
    };
    for (std::size_t index = 0; index < FLOAT_COUNT; ++index) {
        *dot_slots[index] = static_cast<const float*>(raw[tangents + index]);
    }
    second.grad_output_real = static_cast<const float*>(raw[cotangent]);
    second.grad_output_imag = static_cast<const float*>(raw[cotangent + 1]);
    float** value_slots[] = {
        &second.grad_dot_t1, &second.grad_dot_t2, &second.grad_dot_m0,
        &second.grad_dot_b1, &second.grad_dot_b1_phase, &second.grad_dot_b0,
        &second.grad_dot_inversion_efficiency, &second.grad_dot_diffusion,
        &second.grad_dot_velocity,
        &second.grad_dot_duration, &second.grad_dot_flip,
        &second.grad_dot_phase,
    };
    float** tangent_slots[] = {
        &second.grad_t1, &second.grad_t2, &second.grad_m0, &second.grad_b1,
        &second.grad_b1_phase, &second.grad_b0, &second.grad_inversion_efficiency,
        &second.grad_diffusion, &second.grad_velocity,
        &second.grad_duration, &second.grad_flip,
        &second.grad_phase,
    };
    for (std::size_t index = 0; index < FLOAT_COUNT; ++index) {
        *value_slots[index] = static_cast<float*>(raw[value_grads + index]);
        *tangent_slots[index] = static_cast<float*>(raw[tangent_grads + index]);
    }

    float* const flip_events = static_cast<float*>(raw[TISSUE_COUNT + 2]);
    float* const flip_deg = static_cast<float*>(raw[optimizer]);
    float* const moment = static_cast<float*>(raw[optimizer + 1]);
    float* const velocity = static_cast<float*>(raw[optimizer + 2]);
    const float* const t2_ms = static_cast<const float*>(raw[optimizer + 3]);
    float* const jacobian_real = static_cast<float*>(raw[signal]);
    float* const jacobian_imag = static_cast<float*>(raw[signal + 1]);
    float* const grad_real = static_cast<float*>(raw[cotangent]);
    float* const grad_imag = static_cast<float*>(raw[cotangent + 1]);
    // The flip-angle half of the second-order tangent gradients.
    const float* const grad_flip_events =
        static_cast<const float*>(raw[tangent_grads + TISSUE_COUNT + 1]);

    const std::int64_t parameters = train_count * echo_train_length;
    std::vector<double> gradient(static_cast<std::size_t>(parameters), 0.0);
    double loss = 0.0;

    Py_BEGIN_ALLOW_THREADS
    for (std::int64_t iteration = 1; iteration <= iterations; ++iteration) {
        write_refocusing_flips(
            flip_events, flip_deg, train_count, event_count, echo_train_length
        );
        dispatch_jvp(
            forward, atom_count, train_count, event_count, state_count,
            output_count, requested_threads, real_axis
        );
        loss = t2_precision_cotangent(
            jacobian_real, jacobian_imag, t2_ms, grad_real, grad_imag,
            train_count, atom_count, output_count
        );
        dispatch_second_order(
            second, atom_count, train_count, event_count, state_count,
            output_count, requested_threads, real_axis
        );
        // The kernel differentiates the event buffer; only the refocusing slots
        // trace back to a parameter, and the chain rule to degrees is constant.
        for (std::int64_t train = 0; train < train_count; ++train) {
            const float* const source = grad_flip_events + train * event_count;
            double* const target = gradient.data() + train * echo_train_length;
            for (std::int64_t echo = 0; echo < echo_train_length; ++echo) {
                target[echo] =
                    static_cast<double>(source[1 + 2 * echo]) * DEGREES_TO_RADIANS;
            }
        }
        loss += flip_penalties(
            flip_deg, gradient.data(), train_count, echo_train_length, weights
        );
        adam_step(
            flip_deg, gradient.data(), moment, velocity, parameters, iteration,
            weights
        );
    }
    Py_END_ALLOW_THREADS

    return PyFloat_FromDouble(loss);
}

PyMethodDef methods[] = {
    {"simulate", simulate, METH_VARARGS, "Run a fused CPU EPG state machine."},
    {"simulate_jvp", simulate_jvp, METH_VARARGS, "Run a fused CPU EPG JVP."},
    {"simulate_vjp", simulate_vjp, METH_VARARGS, "Run a fused CPU EPG VJP."},
    {"simulate_vjp_jvp", simulate_vjp_jvp, METH_VARARGS,
     "Run a fused CPU EPG forward-over-reverse pass."},
    {"optimize_fse_t2", optimize_fse_t2, METH_VARARGS,
     "Run an A-optimal T2 refocusing-train optimization to completion."},
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
