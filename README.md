# Universal Differential Equations for Malware Propagation Dynamics

[![Julia](https://img.shields.io/badge/Julia-1.9.4-9558B2?style=flat&logo=julia&logoColor=white)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains clean, working implementations of three complementary approaches for modeling malware propagation dynamics: classical Ordinary Differential Equations (ODEs), Universal Differential Equations (UDEs), and Neural ODEs. The code accompanies the paper "Universal Differential Equations Outperform Traditional Models for Malware Propagation Dynamics" published in Scientific Reports.

## Key Features

- **Three modeling approaches**: Traditional ODE, hybrid UDE (physics + neural), and pure Neural ODE
- **44% improvement**: UDE approach reduces prediction error by 44% compared to traditional models
- **Interpretable AI**: Symbolic recovery transforms neural components into explicit mathematical expressions
- **Real-world validation**: Tested on Code Red worm outbreak data from CAIDA
- **Production ready**: Clean, documented code suitable for research and operational use

## Repository Structure

```
├── preprocess.py             # Data preprocessing for CAIDA Code Red dataset
├── parameter_estimator.jl    # Parameter optimization utility
├── ode.jl                    # Classical ODE model implementation
├── ude.jl                    # Universal Differential Equation m  
├── neural_ode.jl             # Pure neural ODE model
├── analysis.jl               # Model comparison and analysis
└── README.md                 # This file
```

## Installation

1. **Install Julia** (version 1.9.4):
   ```bash
   # Download from https://julialang.org/downloads/
   ```

2. **Clone this repository**:
   ```bash
   git clone https://github.com/kp7425/sciml_codered.git
   cd sciml_codered
   ```

3. **Install Python dependencies** (for data preprocessing):
   ```bash
   pip install pandas numpy matplotlib
   ```

4. **Install Julia dependencies**:
   
   The following package versions are tested and confirmed working:
   
   ```julia
   using Pkg
   
   # Core packages
   Pkg.add(name="CSV", version="0.10.15")
   Pkg.add(name="DataFrames", version="1.7.0")
   Pkg.add(name="DifferentialEquations", version="7.10.0")
   Pkg.add(name="Lux", version="0.5.16")
   Pkg.add(name="ComponentArrays", version="0.15.17")
   Pkg.add(name="Plots", version="1.40.13")
   
   # Optimization packages
   Pkg.add(name="Optimization", version="3.19.3")
   Pkg.add(name="OptimizationOptimJL", version="0.1.14")
   Pkg.add(name="OptimizationOptimisers", version="0.1.6")
   Pkg.add(name="Optimisers", version="0.3.4")
   
   # Additional packages
   Pkg.add(name="Interpolations", version="0.15.1")
   Pkg.add(name="Distributions", version="0.25.118")
   Pkg.add(name="StatsBase", version="0.34.4")
   Pkg.add(name="StatsPlots", version="0.15.7")
   Pkg.add(name="Symbolics", version="5.5.3")
   Pkg.add(name="SciMLSensitivity", version="7.51.0")
   Pkg.add(name="ChainRules", version="1.44.7")
   Pkg.add(name="ColorSchemes", version="3.29.0")
   Pkg.add(name="Sundials", version="4.20.1")
   ```
   
   Or for quick installation (may use newer versions):
   ```julia
   julia -e 'using Pkg; Pkg.add(["CSV", "DataFrames", "DifferentialEquations", "Lux", "ComponentArrays", "Plots", "Optimization", "OptimizationOptimJL", "OptimizationOptimisers", "Interpolations", "Distributions", "StatsBase", "StatsPlots", "Symbolics"])'
   ```

## Usage

### Complete Workflow

Follow this sequence for full analysis:

1. **Download CAIDA Code Red dataset**:
   - Get `codered-august.table.txt` from [CAIDA Code Red Worm Dataset](https://catalog.caida.org/dataset/telescope_codered_worm)
   - Place it in the repository directory

2. **Preprocess the data**:
   ```bash
   python preprocess.py
   ```
   This creates `codered_processed.csv` with 30-minute time bins.

3. **Estimate optimal parameters**:
   ```bash
   julia parameter_estimator.jl codered_processed.csv
   ```

4. **Run individual models**:
   ```bash
   # Classical ODE model
   julia ode.jl codered_processed.csv

   # Universal Differential Equation model (recommended)
   julia ude.jl codered_processed.csv

   # Neural ODE model
   julia neural_ode.jl codered_processed.csv
   ```

5. **Compare all models**:
   ```bash
   julia analysis.jl
   ```

### Quick Start (with preprocessed data)

If you already have preprocessed data:

```bash
julia ode.jl your_data.csv
julia ude.jl your_data.csv
julia neural_ode.jl your_data.csv
julia analysis.jl
```

### Programmatic Usage

```julia
include("ude.jl")

# Load your data
tsec, η_raw, η_smoothed, η_interp, timestamps = load_data("your_data.csv")

# Run UDE model
ude_pred, ude_rmse, ude_params, nn_model, nn_state = run_ude_model(tsec, η_raw, η_smoothed, η_interp)

println("UDE RMSE: $ude_rmse")
```

## Data Format

### CAIDA Raw Data
The raw CAIDA dataset `codered-august.table.txt` is a tab-separated file with columns:
- `start_time`: Unix timestamp
- `end_time`: Unix timestamp  
- `tld`: Top-level domain
- `country`: Country code
- `latitude`: Geographic latitude
- `longitude`: Geographic longitude
- `as_number`: Autonomous System number

### Processed Data
After running `preprocess.py`, the output CSV file contains:
- `timestamp`: Date/time in format "YYYY-MM-DD HH:MM:SS"
- `intensity`: Number of infection attempts per time bin
- `cumulative`: Cumulative infection count

Example:
```csv
timestamp,intensity,cumulative
2001-07-19 18:30:00,1250,1250
2001-07-19 19:00:00,2100,3350
2001-07-19 19:30:00,3800,7150
```

### Custom Data Format
For your own malware data, ensure CSV format with:
- `timestamp`: Date/time in format "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD HH:MM:SS.fff"
- `intensity`: Malware intensity/infection count values

## Model Comparison

| Model | RMSE | MAE | Correlation | Parameters |
|-------|------|-----|-------------|------------|
| **UDE** | **1281.8** | **883.9** | **0.946** | 31 |
| ODE | 2289.1 | 2174.8 | 0.948 | 5 |
| Neural ODE | 2036.8 | 1304.0 | 0.687 | 337 |

*Results on Code Red dataset. Bold indicates best performance.*

## Key Results

- **Superior accuracy**: UDE achieves 44% lower prediction error than traditional approaches
- **Data efficiency**: Maintains performance with only 25% of training data
- **Noise robustness**: Outperforms alternatives across 0-20% noise levels
- **Interpretability**: Symbolic recovery reveals cybersecurity mechanisms:
  - Network saturation effects
  - Security response mechanisms  
  - Malware variant evolution
  - Peer-to-peer propagation

## Output

Models save results to `results/` directory:
- **Visualizations**: High-quality plots showing model fit and predictions
- **Predictions**: CSV files with timestamps and model outputs
- **Metrics**: Performance statistics (RMSE, MAE, correlation)

## Advanced Usage

### Parameter Optimization
```julia
# Estimate optimal parameters for your dataset
julia parameter_estimator.jl your_data.csv

# Output provides optimized parameters in Julia format
# const ODE_PARAMS = (α=0.0501, β=0.0001, κ=0.005, K=100000.0, p_decay=0.48)
```

### Forecasting with Limited Data
```julia
# Train on first 25% of data, predict remainder
# Modify train_pct in the model files
```

### Custom Neural Architecture
```julia
# In ude.jl, modify the neural network:
nn_model = Lux.Chain(
    Lux.Dense(1, 20, relu),    # Increase neurons
    Lux.Dense(20, 10, tanh),   # Add layer
    Lux.Dense(10, 1)
)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{pappu2024universal,
    title={Universal Differential Equations Outperform Traditional Models for Malware Propagation Dynamics},
    author={Pappu, Karthik and Joshi, Prathamesh Dinesh and Dandekar, Raj Abhijit and Dandekar, Rajat and Panat, Sreedath},
    journal={Scientific Reports},
    year={2024},
    publisher={Nature Publishing Group}
}
```

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [CAIDA](https://catalog.caida.org/dataset/telescope_codered_worm) for the Code Red worm dataset
- Julia community for excellent scientific computing packages
- [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) ecosystem
- [Lux.jl](https://github.com/avik-pal/Lux.jl) for neural network implementations

## Contact

- **Karthik Pappu** - karthik.pappu@trojans.dsu.edu
- **Project Link**: https://github.com/kp7425/sciml_codered

## Related Work

- [Scientific Machine Learning](https://sciml.ai/)
- [Physics-Informed Neural Networks](https://github.com/maziarraissi/PINNs)
- [Universal Differential Equations](https://github.com/SciML/UniversalDiffEq.jl)
