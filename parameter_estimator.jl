# ===================================================================================================
# Parameter Estimator for Malware Dynamics Models
# ===================================================================================================

using DifferentialEquations
using CSV, DataFrames, Dates, Statistics
using Optimization, OptimizationOptimJL
using Interpolations, Plots, Random

Random.seed!(12345)

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
    intensity = Float64.(df.intensity)
    intensity_smoothed = [mean(intensity[max(1,i-1):min(end,i+1)]) for i in eachindex(intensity)]
    
    return tsec, intensity, intensity_smoothed, timestamps
end

# Parameter estimation function
function estimate_parameters(filepath="codered_processed.csv")
    tsec, intensity, intensity_smoothed, timestamps = load_data(filepath)
    
    # Create interpolation
    η_interp = LinearInterpolation(tsec, intensity_smoothed, extrapolation_bc=Line())
    
    # Initial parameter estimates
    max_intensity = maximum(intensity_smoothed)
    growth_rates = diff(intensity_smoothed) ./ intensity_smoothed[1:end-1]
    positive_growth = growth_rates[growth_rates .> 0]
    mean_growth = isempty(positive_growth) ? 0.1 : mean(positive_growth)
    
    α_initial = min(0.2, max(0.05, mean_growth * 2))
    β_initial = 1.0 / (max_intensity * 15)
    κ_initial = 0.05 * α_initial
    K_initial = max_intensity * 3
    p_decay_initial = 0.2
    
    println("Initial parameter estimates:")
    println("α = $α_initial, β = $β_initial, κ = $κ_initial")
    println("K = $K_initial, p_decay = $p_decay_initial")
    
    # ODE model
    function malware_ode!(du, u, p, t)
        α, β, κ, K, p_decay = p
        M = max(u[1], 0.0)
        α_t = α * exp(-p_decay * t / maximum(tsec))
        du[1] = α_t * M * (1 - M / K) + η_interp(t) - β * M^2 + κ * M * log(1 + M)
    end
    
    # Simulation function
    function simulate(params)
        u0 = [max(intensity_smoothed[1], 1.0)]
        tspan = (tsec[1], tsec[end])
        prob = ODEProblem(malware_ode!, u0, tspan, params)
        sol = solve(prob, Rodas5(), saveat=tsec, abstol=1e-6, reltol=1e-6)
        return sol[1, :]
    end
    
    # Objective function
    function objective(params, _)
        if any(params .≤ 0)
            return 1e10
        end
        
        try
            M_sim = simulate(params)
            return mean((M_sim .- intensity_smoothed).^2)
        catch
            return 1e10
        end
    end
    
    # Parameter optimization
    initial_params = [α_initial, β_initial, κ_initial, K_initial, p_decay_initial]
    lower_bounds = [0.01, 1e-6, 0.001, max_intensity * 1.5, 0.05]
    upper_bounds = [0.5, 1e-3, 0.05, max_intensity * 10, 0.5]
    
    println("\nOptimizing parameters...")
    optf = OptimizationFunction(objective, Optimization.AutoForwardDiff())
    optprob = OptimizationProblem(optf, initial_params, lb=lower_bounds, ub=upper_bounds)
    result = solve(optprob, OptimizationOptimJL.LBFGS(), maxiters=100)
    
    # Results
    optimized_params = result.u
    α_opt, β_opt, κ_opt, K_opt, p_decay_opt = optimized_params
    
    sim_optimized = simulate(optimized_params)
    rmse = sqrt(mean((sim_optimized .- intensity_smoothed).^2))
    
    println("\nOptimized parameters:")
    println("α = $α_opt")
    println("β = $β_opt") 
    println("κ = $κ_opt")
    println("K = $K_opt")
    println("p_decay = $p_decay_opt")
    println("RMSE = $rmse")
    
    # Create visualization
    p = plot(timestamps, intensity_smoothed, label="Observed", lw=2, color=:black,
             title="Parameter Estimation Results", xlabel="Time", ylabel="Intensity")
    plot!(p, timestamps, sim_optimized, label="Fitted (RMSE: $(round(rmse, digits=2)))", 
          lw=2, color=:blue)
    
    # Save results
    mkpath("results")
    savefig(p, "results/parameter_estimation.png")
    
    df = DataFrame(timestamp=timestamps, observed=intensity_smoothed, fitted=sim_optimized)
    CSV.write("results/parameter_estimation.csv", df)
    
    println("Results saved to results/ directory")
    
    return (α=α_opt, β=β_opt, κ=κ_opt, K=K_opt, p_decay=p_decay_opt, rmse=rmse)
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    filepath = length(ARGS) > 0 ? ARGS[1] : "codered_processed.csv"
    params = estimate_parameters(filepath)
    
    # Print Julia constant format
    println("\nJulia constant format:")
    println("const ODE_PARAMS = (α=$(round(params.α, digits=4)), β=$(round(params.β, sigdigits=2)), κ=$(round(params.κ, digits=4)), K=$(round(params.K, sigdigits=2)), p_decay=$(round(params.p_decay, digits=2)))")
end