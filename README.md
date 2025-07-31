This is a simple implementation of an image classification task. The whole implementation is based on the Jax eco-system.

## Data loading
The data loading is written to be more framework-agnostic through JSON files. In particular, each JSON file is a list of "dictionary" objects, each will have the following format:
```bash
[
    {
        "file": <path_to_one_image_file>,
        "label": <true_label_in_int>
    },
]
```

## Experiment tracking and management
The experiment is managed through *MLFlow*. See the script `mlflow_server.sh` for further details.