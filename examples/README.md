# Examples

This folder contains the original generic UKTL usage/demo code and a controlled
synthetic benchmark comparing KTL and UKTL.

## Files

- `api_demo.py`: generic API usage example showing how to apply the task-agnostic
  UKTL core to user-provided tensor data. Unchanged.
- `synthetic_data.py`: the original generic Tucker-style dataset used by the
  existing demo/training scripts. Unchanged.
- `mode_noise_data.py`: controlled data generation for the clean and
  unequal-mode-reliability benchmark.
- `toy_experiments.py`: runs matched KTL and UKTL experiments on 4- and 10-class
  versions of both tasks.
- `toy_results.txt`: recorded results from the current benchmark over seeds
  0, 1, and 2.

## Controlled benchmark

The benchmark is designed to test the setting that motivates UKTL: different
tensor modes can have different reliability.

### Clean subspace task

All three tensor modes contain class-specific low-rank structure. No nuisance
component is added, so all modes are treated as reliable in the data
generation process. This is the control condition.

### Nuisance-mode task

The same class structure is used, but exactly half of the samples receive a
strong, sample-dependent higher-rank component varying across mode 3. The
nuisance is constructed from the class-related mode-1 and mode-2 factors, so
modes 1 and 2 retain class information while mode 3 becomes less reliable
relative to the retained rank `p=2` signal.

This creates the intended unequal-reliability setting for UKTL: the model can
learn mode-wise uncertainty and reduce the influence of a less reliable mode.

## Experimental setup

The benchmark uses 400 samples for each run with a class-balanced
50% / 20% / 30% train/validation/test split, giving 200 training, 80
validation, and 120 test samples.

For the nuisance task, clean and corrupted samples are split separately within
each class, so train, validation, and test each contain exactly 50% nuisance
samples.

KTL and UKTL use the same tensor shape, subspace rank, number of Nyström
pivots, kernel bandwidth, mixture weight, optimizer, learning rate, weight
decay, gradient clipping, random seeds, and 40 training epochs. Nyström pivots
are initialized from the training split only. Validation accuracy is used only
for checkpoint selection, and the held-out test set is evaluated after model
selection.

The methods differ in whether the UKTL uncertainty network (MSN) is enabled.
Because UKTL contains additional learned MSN parameters, the results should
be interpreted as a controlled comparison of the complete KTL and UKTL
methods, rather than as a capacity-matched isolation of the uncertainty
mechanism.

## Results

Results are mean +/- population standard deviation over seeds 0, 1, and 2.

| Task | Classes | Method | Val | Overall test | Clean-subset test | Nuisance-subset test |
|---|---:|---|---:|---:|---:|---:|
| Clean subspace | 4 | KTL | 97.08 +/- 2.57% | 97.22 +/- 1.42% | 97.22 +/- 1.42% | -- |
| Clean subspace | 4 | UKTL | 99.58 +/- 0.59% | 99.17 +/- 0.00% | 99.17 +/- 0.00% | -- |
| Nuisance mode | 4 | KTL | 90.42 +/- 13.55% | 87.22 +/- 16.31% | 85.00 +/- 17.69% | 89.44 +/- 14.93% |
| Nuisance mode | 4 | UKTL | 95.42 +/- 6.48% | 95.83 +/- 3.54% | 95.56 +/- 2.83% | 96.11 +/- 4.37% |
| Clean subspace | 10 | KTL | 73.33 +/- 16.53% | 70.56 +/- 16.31% | 70.56 +/- 16.31% | -- |
| Clean subspace | 10 | UKTL | 88.33 +/- 8.25% | 85.56 +/- 8.17% | 85.56 +/- 8.17% | -- |
| Nuisance mode | 10 | KTL | 55.83 +/- 11.47% | 54.17 +/- 11.24% | 57.22 +/- 13.63% | 51.11 +/- 10.03% |
| Nuisance mode | 10 | UKTL | 71.67 +/- 16.40% | 71.67 +/- 16.72% | 71.67 +/- 18.00% | 71.67 +/- 15.46% |

In the clean task, `Overall test` and `Clean-subset test` are identical because
the held-out test set contains no nuisance samples.

For the nuisance task, `Overall test` is the accuracy over all held-out test
samples, while `Clean-subset test` and `Nuisance-subset test` report accuracy
on the corresponding subsets of that same held-out test set.

The corresponding mean UKTL - KTL differences in overall test accuracy are
+1.94 points (4 classes, clean), +8.61 points (4 classes, nuisance),
+15.00 points (10 classes, clean), and +17.50 points (10 classes, nuisance).

For the nuisance task, the mean UKTL - KTL differences on the
Nuisance-subset test are +6.67 points for 4 classes and +20.56 points for
10 classes.

UKTL mean sigma values are:

- 4-class clean: mode 1 = 1.292, mode 2 = 1.292, mode 3 = 1.297.
- 4-class nuisance: mode 1 = 1.176, mode 2 = 1.176, mode 3 = 1.229.
- 10-class clean: mode 1 = 1.333, mode 2 = 1.404, mode 3 = 1.341.
- 10-class nuisance: mode 1 = 1.238, mode 2 = 1.265, mode 3 = 1.343.

In the nuisance condition, mode 3 receives the largest mean sigma for both
class counts, which is consistent with the intended mode-reliability
diagnostic.

## Interpretation

The clean task is a control rather than the main uncertainty test. Its purpose
is to provide a reference case where all modes are informative. The nuisance
task is the more direct mechanism-focused test because it deliberately makes
one mode less reliable while retaining useful class information in the other
modes.

The results show improved UKTL robustness in this controlled stress setting,
especially on the nuisance subset. They do not establish universal superiority
over KTL. In particular, the clean-task improvement should not be attributed
solely to uncertainty because UKTL has additional learned MSN parameters. A
capacity-matched ablation would be needed to isolate the contribution of the
uncertainty mechanism itself.

## Run

From the repository root:

```bash
python -m examples.toy_experiments
```

The script reports validation accuracy, overall test accuracy, clean-subset
test accuracy, nuisance-subset test accuracy when applicable, and UKTL mean
sigma per mode.
