# ===================================================================================================
# Classical ODE Model for Malware Dynamics
# ===================================================================================================

using DifferentialEquations
using CSV, DataFrames, Interpolations, Dates, Statistics
using Plots, Random
Random.seed!(12345)

# Model parameters
const ODE_PARAMS = (α=0.0501, β=0.0001, κ=0.005, K=100000.0, p_decay=0.48)

# Data loading function
function load_data(filepath)
    if !isfile(filepath)
        error("Data file not found: $filepath")
    end
    
    println("Loading data from: $filepath")
    df = CSV.read(filepath, DataFrame)
    
    # Process timestamps
    timestamps = [DateTime(split(ts, '.')[1], dateformat"yyyy-mm-dd HH:MM:SS") for ts in df.timestamp]
    t0 = minimum(timestamps)
    tsec = Float64.((timestamps .- t0) ./ Millisecond(1000))
    
    # Get and smooth data
    η_raw = Float64.(df.intensity)
    η_smoothed = [mean(η_raw[max(1,i-1):min(end,i+1)]) for i in eachindex(η_raw)]
    η_interp = LinearInterpolation(tsec, η_smoothed, extrapolation_bc=Line())
    
    return tsec, η_raw, η_smoothed, η_interp, timestamps
end

# ODE model implementation
function run_ode_model(tsec, η_raw, η_smoothed, η_interp)
    (; α, β, κ, K, p_decay) = ODE_PARAMS
    
    println("Running ODE model: dM/dt = α(t)·M·(1-M/K) + η(t) - βM² + κM·log(1+M)")
    
    u0 = [max(η_smoothed[1], 1.0)]
    tspan = (tsec[1], tsec[end])

    function malware_ode!(du, u, p, t)
        M = max(u[1], 0.0)
        α_t = α * exp(-p_decay * t / maximum(tsec))
        du[1] = α_t * M * (1 - M / K) + η_interp(t) - β * M^2 + κ * M * log(1 + M)
    end

    prob = ODEProblem(malware_ode!, u0, tspan)
    sol = solve(prob, Rodas5(), saveat=tsec, abstol=1e-6, reltol=1e-6)
    M_pred = sol[1, :]
    
    rmse = sqrt(mean((M_pred .- η_smoothed).^2))
    println("ODE RMSE: $(round(rmse, digits=2))")
    
    return M_pred, rmse
end

# Main execution function
function run_ode(filepath="codered_processed.csv")
    tsec, η_raw, η_smoothed, η_interp, timestamps = load_data(filepath)
    ode_pred, ode_rmse = run_ode_model(tsec, η_raw, η_smoothed, η_interp)
    
    # Create visualization
    p = plot(timestamps, η_smoothed, label="Observed", lw=2, color=:black,
             title="ODE Model vs Data", xlabel="Time", ylabel="Intensity")
    plot!(p, timestamps, ode_pred, label="ODE (RMSE: $(round(ode_rmse, digits=2)))", 
          lw=2, color=:blue)
    
    # Save results
    mkpath("results")
    savefig(p, "results/ode_comparison.png")
    
    df = DataFrame(timestamp=timestamps, observed=η_smoothed, ode=ode_pred)
    CSV.write("results/ode_predictions.csv", df)
    
    println("Results saved to results/ directory")
    return ode_pred, ode_rmse
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    filepath = length(ARGS) > 0 ? ARGS[1] : "codered_processed.csv"
    run_ode(filepath)
end