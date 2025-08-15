# ===================================================================================================
# Simplified Analysis for Malware Dynamics Models
# ===================================================================================================

using CSV, DataFrames, Dates, Statistics
using Plots, Random
Random.seed!(12345)

# Load all model results
function load_all_results()
    # Try to load individual model results
    ode_file = "results/ode_predictions.csv"
    ude_file = "results/ude_predictions.csv" 
    node_file = "results/neural_ode_predictions.csv"
    
    results = Dict()
    
    if isfile(ode_file)
        df_ode = CSV.read(ode_file, DataFrame)
        results["ode"] = df_ode
        println("Loaded ODE results")
    end
    
    if isfile(ude_file)
        df_ude = CSV.read(ude_file, DataFrame)
        results["ude"] = df_ude
        println("Loaded UDE results")
    end
    
    if isfile(node_file)
        df_node = CSV.read(node_file, DataFrame)
        results["neural_ode"] = df_node
        println("Loaded Neural ODE results")
    end
    
    return results
end

# Calculate performance metrics
function calculate_metrics(observed, predicted, model_name)
    rmse = sqrt(mean((predicted .- observed).^2))
    mae = mean(abs.(predicted .- observed))
    correlation = cor(observed, predicted)
    
    println("$model_name Performance:")
    println("  RMSE: $(round(rmse, digits=2))")
    println("  MAE: $(round(mae, digits=2))")
    println("  Correlation: $(round(correlation, digits=3))")
    
    return (rmse=rmse, mae=mae, correlation=correlation)
end

# Create comparison visualization
function create_comparison_plot(results)
    if isempty(results)
        println("No results to plot")
        return
    end
    
    # Use first available dataset for timestamps and observed data
    first_key = first(keys(results))
    df_ref = results[first_key]
    timestamps = df_ref.timestamp
    observed = df_ref.observed
    
    # Create plot
    p = plot(timestamps, observed, label="Observed", lw=2, color=:black,
             title="Model Comparison", xlabel="Time", ylabel="Intensity",
             size=(1000, 600))
    
    colors = [:blue, :green, :red]
    color_idx = 1
    
    # Plot each model
    for (model_name, df) in results
        if model_name == "ode" && "ode" in names(df)
            plot!(p, timestamps, df.ode, label="ODE", lw=2, color=colors[color_idx])
        elseif model_name == "ude" && "ude" in names(df)
            plot!(p, timestamps, df.ude, label="UDE", lw=2, color=colors[color_idx])
        elseif model_name == "neural_ode" && "neural_ode" in names(df)
            plot!(p, timestamps, df.neural_ode, label="Neural ODE", lw=2, color=colors[color_idx])
        end
        color_idx = min(color_idx + 1, length(colors))
    end
    
    # Save plot
    mkpath("results")
    savefig(p, "results/model_comparison.png")
    println("Comparison plot saved to results/model_comparison.png")
    
    return p
end

# Main analysis function
function run_analysis()
    println("=== Malware Dynamics Model Analysis ===")
    
    # Load results
    results = load_all_results()
    
    if isempty(results)
        println("No model results found. Run the individual models first.")
        return
    end
    
    # Calculate metrics for each model
    metrics = Dict()
    first_key = first(keys(results))
    observed = results[first_key].observed
    
    for (model_name, df) in results
        if model_name == "ode" && "ode" in names(df)
            metrics[model_name] = calculate_metrics(observed, df.ode, "ODE")
        elseif model_name == "ude" && "ude" in names(df)
            metrics[model_name] = calculate_metrics(observed, df.ude, "UDE")
        elseif model_name == "neural_ode" && "neural_ode" in names(df)
            metrics[model_name] = calculate_metrics(observed, df.neural_ode, "Neural ODE")
        end
    end
    
    # Create comparison plot
    create_comparison_plot(results)
    
    # Summary
    println("\n=== SUMMARY ===")
    best_rmse = Inf
    best_model = ""
    
    for (model_name, metric) in metrics
        if metric.rmse < best_rmse
            best_rmse = metric.rmse
            best_model = model_name
        end
    end
    
    println("Best performing model: $best_model (RMSE: $(round(best_rmse, digits=2)))")
    
    return metrics
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_analysis()
end