# The fingerprinting dictionary, on BlochSimulators.jl's EPG simulator.
#
# `FISP2D` is the sequence this benchmark's task already is: an inversion, a
# variable flip-angle train at fixed TR, one sample per repetition at the echo
# time, and a spoiling shift at the end of each. Setting the echo time to zero
# puts the sample immediately after the excitation, which is where the other
# three backends take theirs.
#
# Two differences from the Python backends are structural rather than
# incidental, and are recorded in the note of every record written here:
#
#   * `FISP2D` requires the number of configuration states to be a multiple of
#     32, so the shared task carries 32 orders rather than 20.
#   * A real `RF_train` keeps the configuration states real, which is the same
#     specialization TorchSim applies when a sequence earns it. The train here
#     has no phase, so both packages take their real path.
#
# Run as (from the repository root):
#
#     julia -t 4 --project=benchmarks/julia benchmarks/julia/bench_blochsimulators.jl --atoms 10000
#
# Derivatives are `--mode jacobian`, which is what MR-STAT does: a forward
# difference per parameter, so three passes for two parameters.

using ComputationalResources
using BlochSimulators
using Pkg

include(joinpath(@__DIR__, "common.jl"))

function main()
    options = arguments(Dict(
        "atoms" => "1000",
        "length" => "500",
        "states" => "32",
        "repeats" => "3",
        "mode" => "forward",
        "precision" => "f32",
        "rf" => "real",
        "json" => "",
        "tissues" => "",
        "dump" => "",
    ))
    atoms = parse(Int, options["atoms"])
    length_ = parse(Int, options["length"])
    states = parse(Int, options["states"])
    repeats = parse(Int, options["repeats"])
    mode = options["mode"]
    single = options["precision"] == "f32"

    baseline = peak_rss_mib()

    # Seconds and milliseconds: BlochSimulators takes times in seconds,
    # TorchSim in milliseconds, and the task is stated in milliseconds.
    TR = 0.010
    TE = 0.0
    TI = 0.0
    spoil = 0.0  # no diffusion is declared, so the spoiler area is unused

    flip = flip_train(length_)
    if isempty(options["tissues"])
        T1, T2 = tissue_grid(atoms)
    else
        # "T1:T2,T1:T2,..." in milliseconds, for the cross-implementation check
        pairs = [parse.(Float64, split(pair, ":")) for pair in split(options["tissues"], ",")]
        T1 = [pair[1] for pair in pairs]
        T2 = [pair[2] for pair in pairs]
        atoms = length(T1)
    end
    # A real RF train keeps the configuration states real, which is the same
    # saving TorchSim's real-subspace kernels make. Asking for a complex train
    # is how the two halves of the difference are told apart: what the
    # specialization buys, and what is left over.
    train = options["rf"] == "complex" ? complex.(flip) : flip
    sequence = FISP2D(train, TR, TE, states, TI, spoil)
    parameters = map(T₁T₂, T1 ./ 1000, T2 ./ 1000)
    if single
        sequence = f32(sequence)
        parameters = f32(parameters)
    end

    resource = Threads.nthreads() > 1 ? CPUThreads() : CPU1()

    forward() = simulate_magnetization(resource, sequence, parameters)

    # What MR-STAT does for its Jacobian: perturb one parameter, simulate
    # again, difference. Two parameters, so three passes in all.
    function jacobian()
        step = single ? 1.0f-4 : 1e-4
        signal = simulate_magnetization(resource, sequence, parameters)
        shifted = map(p -> T₁T₂(p.T₁ + step, p.T₂), parameters)
        single && (shifted = f32(shifted))
        by_t1 = (simulate_magnetization(resource, sequence, shifted) .- signal) ./ step
        shifted = map(p -> T₁T₂(p.T₁, p.T₂ + step), parameters)
        single && (shifted = f32(shifted))
        by_t2 = (simulate_magnetization(resource, sequence, shifted) .- signal) ./ step
        return by_t1 .+ by_t2
    end

    setup, seconds, signal = timed(mode == "jacobian" ? jacobian : forward; repeats)

    installed = Pkg.dependencies()
    version = string(only(v.version for v in values(installed) if v.name == "BlochSimulators"))
    record = Dict{String,Any}(
        "backend" => "BlochSimulators.jl",
        "mode" => mode == "jacobian" ? "jacobian(T1,T2)" :
                  options["rf"] == "complex" ? "forward(complex)" : "forward",
        "atoms" => atoms,
        "length" => length_,
        "states" => states,
        "device" => "cpu",
        "threads" => Threads.nthreads(),
        "repeats" => repeats,
        "seconds" => seconds,
        "setup_seconds" => setup,
        "baseline_rss_mib" => baseline,
        "peak_rss_mib" => peak_rss_mib(),
        "peak_device_mib" => 0.0,
        "checksum" => [real(sum(signal)), imag(sum(signal))],
        "versions" => Dict("BlochSimulators" => version, "julia" => string(VERSION)),
        "machine" => machine(),
        "note" => string(
            single ? "float32" : "float64",
            options["rf"] == "complex" ? "; complex RF train, complex states; " :
                "; real RF train, so the states stay real; ",
            "max_state is a multiple of 32 by construction",
            mode == "jacobian" ? "; forward differences, three passes" : "",
        ),
    )
    report(record, options["json"])

    # The signal itself, for `validate.py` to hold against the other three.
    if !isempty(options["dump"])
        open(options["dump"], "w") do stream
            for row in axes(signal, 2), column in axes(signal, 1)
                @printf(stream, "%d,%d,%.9e,%.9e\n",
                    row, column, real(signal[column, row]), imag(signal[column, row]))
            end
        end
    end
end

main()
