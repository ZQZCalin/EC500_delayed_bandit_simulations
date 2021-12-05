# EC500: Delayed Multi-armed Bandits

Simulation of EC500 class project on delayed multi-armed bandit algorithm based on [Zimmert and Seldin's work](https://arxiv.org/abs/1910.06054).
The bandit algorithm receives losses from local online sub-gradient descent (OSD) algorithms with different learning rates.

### Instruction

1. Run `generate_OSD_data.py --data_dir [data] --text_series [time-series]` to get generate OSD losses using `[times-series]` data and then save generated losses and performance plots to `[data]`.
2. Run `experiment.py --regularizer --tuning --delay --data_dir --result_dir --name --rewrite` to run the experiment. For each argument, see report for details.
    1. `--data_dir`: folder path of local losses, default=`data`
    2. `--result_dir`: folder path of experiments, default=`results`
    3. `--name`: experiment name, results will be saved to `result_dir/name`, default=`experiment`
    4. `--rewrite`: overwrite the existing results or not, default=`False`

##### Regularizers
- `negative_entropy`
- `Tsallis_entropy`
- `mixed_entropy`: hybrid of negative entropy and Tsallis entropy
- `no_name`: Orabona entropy

##### Tuning method
- `simple`
- `advanced` (*not in use*)

##### Delays
- `non`: no delay
- `uniform_a_b`: Uniform(a, b)
- `Gaussian_m_s`: Gaussian(m, s)
