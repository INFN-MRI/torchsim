# The same fingerprinting dictionary, on KomaMRI's isochromat solver.
#
# This is not the same computation as the other four backends and is not meant
# to be read as a like-for-like race. KomaMRI carries isochromats through real
# gradient waveforms at coordinates, which is a strictly larger model: it needs
# no ideal-spoiling assumption, and it is what you reach for when the answer
# depends on the waveform. The price is that a spoiled sequence has to be
# resolved by *spreading spins through the spoiler*: a voxel is a bundle of
# isochromats spanning one dephasing cycle, and the signal is their average.
#
# That makes the benchmark say something the throughput number alone does not:
# an extended phase graph carrying `states` orders and an isochromat bundle of
# `spins` spins converge to the same answer once `spins` is about twice
# `states` -- so the cost of the isochromat picture on this sequence is the
# `spins` factor, and `--spins` is the axis to sweep.
#
# Run as (from the repository root):
#
#     julia -t 4 --project=benchmarks/julia benchmarks/julia/bench_koma.jl --atoms 100 --spins 64
#
# `--dump` writes the per-tissue signal for `validate.py` to hold against the
# EPG implementations.

using KomaMRICore
using KomaMRIBase
using Pkg

include(joinpath(@__DIR__, "common.jl"))

"""
The MRF train as a Koma sequence: an inversion, then one hard excitation, one
sample and one spoiler per repetition.

The spoiler area is what makes `cycle_m` of space carry one full turn of
dephasing, which is the extent the spins are spread over.
"""
function mrf_sequence(flip, TR, system; rf_s=10e-6, adc_s=1e-6, spoil_s=1e-3, cycle_m=1e-3)
    hard(angle_rad, duration_s) =
        PulseDesigner.RF_hard(angle_rad / (2π * KomaMRIBase.γ * duration_s), duration_s, system)
    sample = Sequence(
        [Grad(0.0, adc_s); Grad(0.0, adc_s); Grad(0.0, adc_s);;],
        [RF(0.0, adc_s);;],
        [ADC(1, adc_s)],
    )
    amplitude = 1 / (KomaMRIBase.γ * spoil_s * cycle_m)
    spoiler = Sequence([Grad(0.0, spoil_s); Grad(0.0, spoil_s); Grad(amplitude, spoil_s);;])
    rest = TR - rf_s - adc_s - spoil_s
    rest > 0 || error("TR is shorter than one repetition's blocks")

    sequence = hard(π, rf_s)  # inversion, TI = 0
    for angle in flip
        sequence += hard(deg2rad(angle), rf_s) + sample + spoiler + Delay(rest)
    end
    return sequence
end

"One bundle of `spins` isochromats per tissue, spread through one spoiler cycle."
function bundles(T1_ms, T2_ms, spins; cycle_m=1e-3)
    atoms = length(T1_ms)
    offsets = [(index - 0.5) / spins * cycle_m for index in 1:spins]
    z = repeat(offsets, atoms)
    T1 = vec(repeat(T1_ms' ./ 1000, spins, 1))
    T2 = vec(repeat(T2_ms' ./ 1000, spins, 1))
    return Phantom{Float64}(
        x=zeros(atoms * spins), y=zeros(atoms * spins), z=z, T1=T1, T2=T2
    )
end

function main()
    options = arguments(Dict(
        "atoms" => "100",
        "length" => "500",
        "spins" => "64",
        "repeats" => "3",
        "mode" => "forward",
        "json" => "",
        "tissues" => "",
        "dump" => "",
    ))
    atoms = parse(Int, options["atoms"])
    length_ = parse(Int, options["length"])
    spins = parse(Int, options["spins"])
    repeats = parse(Int, options["repeats"])

    baseline = peak_rss_mib()

    TR = 0.010
    flip = flip_train(length_)
    if isempty(options["tissues"])
        T1, T2 = tissue_grid(atoms)
    else
        pairs = [parse.(Float64, split(pair, ":")) for pair in split(options["tissues"], ",")]
        T1 = [pair[1] for pair in pairs]
        T2 = [pair[2] for pair in pairs]
        atoms = length(T1)
    end

    system = Scanner()
    sequence = mrf_sequence(flip, TR, system)
    phantom = bundles(T1, T2, spins)
    parameters = Dict{String,Any}(
        "sim_method" => BlochDict(),
        "return_type" => "mat",
        "gpu" => false,
        "Nthreads" => Threads.nthreads(),
    )

    # (samples, spins * atoms, 1) out; the bundle of each tissue averages to
    # the signal an extended phase graph reports for it.
    run() = simulate(phantom, sequence, system; sim_params=parameters, verbose=false)
    setup, seconds, raw = timed(run; repeats)
    signal = dropdims(sum(reshape(raw[:, :, 1], :, spins, atoms), dims=2), dims=2) ./ spins

    installed = Pkg.dependencies()
    version = string(only(v.version for v in values(installed) if v.name == "KomaMRICore"))
    record = Dict{String,Any}(
        "backend" => "KomaMRI.jl",
        "mode" => "forward",
        "atoms" => atoms,
        "length" => length_,
        "states" => spins,
        "device" => "cpu",
        "threads" => Threads.nthreads(),
        "repeats" => repeats,
        "seconds" => seconds,
        "setup_seconds" => setup,
        "baseline_rss_mib" => baseline,
        "peak_rss_mib" => peak_rss_mib(),
        "peak_device_mib" => 0.0,
        "checksum" => [real(sum(signal)), imag(sum(signal))],
        "versions" => Dict("KomaMRICore" => version, "julia" => string(VERSION)),
        "machine" => machine(),
        "note" => "isochromat: $(spins) spins per tissue through one spoiler cycle, $(atoms * spins) spins in all",
    )
    report(record, options["json"])

    if !isempty(options["dump"])
        open(options["dump"], "w") do stream
            for tissue in axes(signal, 2), sample in axes(signal, 1)
                @printf(stream, "%d,%d,%.9e,%.9e\n",
                    tissue, sample, real(signal[sample, tissue]), imag(signal[sample, tissue]))
            end
        end
    end
end

main()
