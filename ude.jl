# ===================================================================================================
# Universal Differential Equation Model for Malware Dynamics
# ===================================================================================================

using DifferentialEquations, Lux, ComponentArrays
using CSV, DataFrames, Interpolations, Dates, Statistics
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Plots, Random
Random.seed!(12345)

# Model parameters
const UDE_PARAMS = (α=0.0501, β=0.0001, K=100000.0, p_decay=0.48)
const MAXITERS_ADAM = 300
const MAXITERS_LBFGS = 200

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

# UDE model implementation
function run_ude_model(tsec, η_raw, η_smoothed, η_interp)
    (; α, β, K, p_decay) = UDE_PARAMS
    
    println("Running UDE model: dM/dt = α(t)·M·(1-M/K) + η(t) - βM² + NN(M)")
    
    u0 = [max(η_smoothed[1], 1.0)]
    tspan = (tsec[1], tsec[end])
    max_η = maximum(η_smoothed)

    # Define neural network
    nn_model = Lux.Chain(
        Lux.Dense(1, 10, relu),
        Lux.Dense(10, 1)
    )
    
    rng = MersenneTwister(12345)
    nn_params, nn_state = Lux.setup(rng, nn_model)

    # Combined parameters
    p_combined = ComponentArray(
        ode=[α, β, K, p_decay], 
        nn=nn_params
    )

    # UDE function
    function ude_hybrid!(du, u, p, t)
        M = max(min(u[1], 5 * max_η), 0.0)
        α, β, K, p_decay = abs.(p.ode)
        α_t = α * exp(-p_decay * t / maximum(tsec))
        growth = α_t * M * (1 - M / K)
        suppression = -β * M^2
        external = η_interp(t)
        M_norm = M / max_η
        nn_input = reshape([M_norm], :, 1)
        nn_out, _ = nn_model(nn_input, p.nn, nn_state)
        nn_term = clamp(nn_out[1], -1000.0, 1000.0)
        du[1] = growth + suppression + external + nn_term
    end

    prob_ude = ODEProblem(ude_hybrid!, u0, tspan, p_combined)

    # Training functions
    function predict(θ)
        sol = solve(prob_ude, Rodas5(), p=θ, saveat=tsec, abstol=1e-3, reltol=1e-3, maxiters=10^6)
        if sol.retcode != :Success
            return fill(mean(η_smoothed), 1, length(tsec))
        end
        return Array(sol)
    end

    function loss(θ)
        pred = predict(θ)
        return sum(abs2, η_smoothed .- pred[1, :]) / length(η_smoothed)
    end

    # Training callback
    iter_count = Ref(0)
    callback = function (θ, l)
        iter_count[] += 1
        if iter_count[] % 50 == 0
            println("UDE Iter $(iter_count[]) | Loss = $(round(l, digits=4))")
        end
        return false
    end

    # Train with Adam
    println("Training UDE with Adam...")
    optf = OptimizationFunction((x, p) -> loss(x), Optimization.AutoFiniteDiff())
    optprob = OptimizationProblem(optf, p_combined)
    result1 = Optimization.solve(optprob, OptimizationOptimisers.Adam(5e-4), callback=callback, maxiters=MAXITERS_ADAM)

    # Fine-tune with LBFGS
    println("Fine-tuning with LBFGS...")
    result2 = Optimization.solve(remake(optprob, u0=result1.u), OptimizationOptimJL.LBFGS(), callback=callback, maxiters=MAXITERS_LBFGS)

    # Final prediction
    p_final = result2.u
    sol = solve(prob_ude, Rodas5(), p=p_final, saveat=tsec, abstol=1e-4, reltol=1e-4)
    M_ude = sol[1, :]
    
    rmse = sqrt(mean((M_ude .- η_smoothed).^2))
    println("UDE RMSE: $(round(rmse, digits=2))")
    
    return M_ude, rmse, p_final, nn_model, nn_state
end

# Main execution function
function run_ude(filepath="codered_processed.csv")
    tsec, η_raw, η_smoothed, η_interp, timestamps = load_data(filepath)
    ude_pred, ude_rmse, ude_params, nn_model, nn_state = run_ude_model(tsec, η_raw, η_smoothed, η_interp)
    
    # Create visualization
    p = plot(timestamps, η_smoothed, label="Observed", lw=2, color=:black,
             title="UDE Model vs Data", xlabel="Time", ylabel="Intensity")
    plot!(p, timestamps, ude_pred, label="UDE (RMSE: $(round(ude_rmse, digits=2)))", 
          lw=2, color=:green)
    
    # Save results
    mkpath("results")
    savefig(p, "results/ude_comparison.png")
    
    df = DataFrame(timestamp=timestamps, observed=η_smoothed, ude=ude_pred)
    CSV.write("results/ude_predictions.csv", df)
    
    println("Results saved to results/ directory")
    return ude_pred, ude_rmse
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    filepath = length(ARGS) > 0 ? ARGS[1] : "codered_processed.csv"
    run_ude(filepath)
end