# convergent-neural-dynamical-systems
Analysis code accompanying '[Convergent neural dynamical systems for task control in artificial networks and human brains](https://www.biorxiv.org/content/10.1101/2024.09.29.615736v5)'


For a more general SSM fitting pipeline, refer to [StateSpaceAnalysis.jl](https://github.com/harrisonritz/StateSpaceAnalysis.jl)


## Repository structure

- `data/`
	- Placeholder for datasets (very large; hosting TBD). See `data/README.md` for updates.

- `src/`
	- `em/`: Expectation–Maximization implementation and helpers for task switching models
	- `likelihoods/`
		- Filtering/smoothing primitives and likelihood computation
	- `simulators/`
		- Generative simulators for benchmarking/validation
	- `_wrappers/`
		- Ready-to-run scripts to fit models to specific datasets or configurations and to batch jobs
	- `_analysis/`
		- MATLAB-based RNN & EEG analysis and plotting functions
	- `utils/`
		- Shared analysis and plotting utilities
	- `validation/`
		- parameter recovery and code optimization