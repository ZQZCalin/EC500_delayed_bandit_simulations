# EC500: Delayed Multi-armed Bandits

Simulation of EC500 class project on delayed multi-armed bandit algorithm based on [Zimmert and Seldin's work](https://arxiv.org/abs/1910.06054).
The bandit algorithm receives losses from local online sub-gradient descent (OSD) algorithms with different learning rates.

### Instruction

1. Run `generate_OSD_data.py --data_dir [data] --text_series [time-series]` to get generate OSD losses using `[times-series]` data and then save generated losses and performance plots to `[data]`.
2. Run `experiment.py --data_dir [data] --regularizer [...] --tuning [...]` to run the experiment on the bandit algorithm and the generated losses.
    1. Regularizer: `negative_entropy`, or `Tsallis_entropy`, or `mixed_entropy`
    2. Tuning method: `simple`, or `advanced`

### Experiment Results
