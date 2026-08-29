# What the Julia backends share with the Python ones: the task, the timing
# discipline, and the record written out at the end.
#
# The task is defined here exactly as it is in `benchmarks/_common.py`, in the
# units each package takes. Nothing is read across from the Python side at run
# time, so either half can be run on its own; the two definitions have to agree
# and `validate.py` is what checks that they do.

using Printf

"The flip-angle pattern the dictionary is simulated over, in degrees."
function flip_train(length_::Int)
    up = range(5.0, 60.0, length=round(Int, 0.375 * length_))
    down = range(60.0, 2.0, length=round(Int, 0.375 * length_))
    tail = fill(2.0, length_ - Base.length(up) - Base.length(down))
    return vcat(collect(up), collect(down), tail)
end

"`atoms` (T1, T2) pairs spanning the range a brain dictionary covers, in ms."
function tissue_grid(atoms::Int)
    logspace(a, b, n) = n == 1 ? [a] : exp.(range(log(a), log(b), length=n))
    return logspace(200.0, 3000.0, atoms), logspace(10.0, 300.0, atoms)
end

"Peak resident set size of this process so far, in MiB."
peak_rss_mib() = Sys.maxrss() / 2^20

"""
Run `f` once to warm up -- which in Julia is where the compilation goes -- then
`repeats` times, timing each. Returns `(setup_seconds, seconds, result)`.
"""
function timed(f; repeats::Int)
    start = time_ns()
    result = f()
    setup = (time_ns() - start) / 1e9
    seconds = Float64[]
    for _ in 1:repeats
        start = time_ns()
        result = f()
        push!(seconds, (time_ns() - start) / 1e9)
    end
    return setup, seconds, result
end

"The command line every backend script takes, as a dictionary of strings."
function arguments(defaults::Dict{String,String})
    options = copy(defaults)
    index = 1
    while index <= length(ARGS)
        key = ARGS[index]
        startswith(key, "--") || error("unexpected argument $(key)")
        options[key[3:end]] = ARGS[index+1]
        index += 2
    end
    return options
end

# A record is a flat mapping of numbers, strings and vectors of those, which is
# little enough JSON to write by hand rather than take a dependency for.
json(value::AbstractString) = "\"" * replace(value, "\\" => "\\\\", "\"" => "\\\"") * "\""
json(value::Bool) = value ? "true" : "false"
json(value::Integer) = string(value)
json(value::Real) = isfinite(value) ? @sprintf("%.10g", value) : "null"
json(value::AbstractVector) = "[" * join(json.(value), ", ") * "]"
json(value::AbstractDict) =
    "{" * join(["  " * json(string(k)) * ": " * json(v) for (k, v) in value], ",\n") * "}"

"""
Print the line a human reads, and write the record a table is built from.

The keys are the ones `benchmarks/_common.py` writes, so `summarize.py` and
`make_figures.py` read a Julia record without knowing it is one.
"""
function report(record::AbstractDict, path::AbstractString)
    best = minimum(record["seconds"])
    median = sort(record["seconds"])[cld(length(record["seconds"]), 2)]
    over = record["peak_rss_mib"] - record["baseline_rss_mib"]
    @printf("%12s %-9s atoms=%-7d best=%9.2f ms  median=%9.2f ms  peak_rss=%7.1f MiB  (+%6.1f)\n",
        record["backend"], record["mode"], record["atoms"], best * 1e3, median * 1e3,
        record["peak_rss_mib"], over)
    if !isempty(path)
        full = merge(record, Dict(
            "best" => best,
            "median" => median,
            "atoms_per_second" => record["atoms"] / best,
        ))
        open(path, "w") do stream
            write(stream, json(full), "\n")
        end
    end
end

"Enough about where this ran that a number can be read later."
machine() = Dict(
    "platform" => string(Sys.KERNEL, " ", Sys.MACHINE),
    "processor" => Sys.cpu_info()[1].model,
    "julia" => string(VERSION),
)
