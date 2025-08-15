# ===================================================================================================
# Neural ODE Model for Malware Dynamics
# ===================================================================================================

using DifferentialEquations, Lux, ComponentArrays
using CSV, DataFrames, Interpolations, Dates, Statistics
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Plots, Random
Random.seed!(12345)

# Training parameters
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

# Neural ODE model implementation
function run_neural_ode_model(tsec, η_raw, η_smoothed, η_interp)
    println("Running Neural ODE model: dM/dt = NN(M, t)")
    
    u0 = [max(η_smoothed[1], 1.0)]
    tspan = (tsec[1], tsec[end])
    max_η = maximum(η_smoothed)

    # Define neural network
    neural_ode_model = Lux.Chain(
        Lux.Dense(2, 16, relu),   # Input: M and t
        Lux.Dense(16, 16, relu),
        Lux.Dense(16, 1)          # Output: dM/dt
    )
    
    rng = MersenneTwister(12345)
    nn_params, nn_state = Lux.setup(rng, neural_ode_model)
    p_flat = ComponentArray(nn_params)

    # Neural ODE function
    function neural_ode!(du, u, p, t)
        M = max(min(u[1], 5 * max_η), 0.0)
        M_norm = M / max_η
        t_norm = t / maximum(tsec)
        nn_input = reshape([M_norm, t_norm], :, 1)
        nn_out, _ = neural_ode_model(nn_input, p, nn_state)
        du[1] = clamp(nn_out[1], -1000.0, 1000.0)
    end

    prob_neural_ode = ODEProblem(neural_ode!, u0, tspan, p_flat)

    # Training functions
    function predict(θ)
        sol = solve(prob_neural_ode, Rodas5(), p=θ, saveat=tsec, abstol=1e-3, reltol=1e-3, maxiters=10^6)
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
            println("Neural ODE Iter $(iter_count[]) | Loss = $(round(l, digits=4))")
        end
        return false
    end

    # Train with Adam
    println("Training Neural ODE with Adam...")
    optf = OptimizationFunction((x, p) -> loss(x), Optimization.AutoFiniteDiff())
    optprob = OptimizationProblem(optf, p_flat)
    result1 = Optimization.solve(optprob, OptimizationOptimisers.Adam(5e-4), callback=callback, maxiters=MAXITERS_ADAM)

    # Fine-tune with LBFGS
    println("Fine-tuning with LBFGS...")
    result2 = Optimization.solve(remake(optprob, u0=result1.u), OptimizationOptimJL.LBFGS(), callback=callback, maxiters=MAXITERS_LBFGS)

    # Final prediction
    p_final = result2.u
    sol = solve(prob_neural_ode, Rodas5(), p=p_final, saveat=tsec, abstol=1e-4, reltol=1e-4)
    M_neural_ode = sol[1, :]
    
    rmse = sqrt(mean((M_neural_ode .- η_smoothed).^2))
    println("Neural ODE RMSE: $(round(rmse, digits=2))")
    
    return M_neural_ode, rmse, p_final, neural_ode_model, nn_state
end

# Main execution function
function run_neural_ode(filepath="codered_processed.csv")
    tsec, η_raw, η_smoothed, η_interp, timestamps = load_data(filepath)
    node_pred, node_rmse, node_params, nn_model, nn_state = run_neural_ode_model(tsec, η_raw, η_smoothed, η_interp)
    
    # Create visualization
    p = plot(timestamps, η_smoothed, label="Observed", lw=2, color=:black,
             title="Neural ODE Model vs Data", xlabel="Time", ylabel="Intensity")
    plot!(p, timestamps, node_pred, label="Neural ODE (RMSE: $(round(node_rmse, digits=2)))", 
          lw=2, color=:red)
    
    # Save results
    mkpath("results")
    savefig(p, "results/neural_ode_comparison.png")
    
    df = DataFrame(timestamp=timestamps, observed=η_smoothed, neural_ode=node_pred)
    CSV.write("results/neural_ode_predictions.csv", df)
    
    println("Results saved to results/ directory")
    return node_pred, node_rmse
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    filepath = length(ARGS) > 0 ? ARGS[1] : "codered_processed.csv"
    run_neural_ode(filepath)
end