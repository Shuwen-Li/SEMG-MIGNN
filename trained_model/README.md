# Trained models

This directory stores the early-stopping-restored TensorFlow weights generated
by `reproduce_semg.py` or the training notebooks.

- `data1`: data1 model weights.
- `data2`: data2 model weights.

Each TensorFlow checkpoint consists of an `.index` file and one or more `.data`
files. The adjacent JSON file records the configuration and validation metrics.
