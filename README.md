This is a simple implementation of an image classification task. The whole implementation is based on the Jax eco-system.

## Python packages
Instead of providing a `requirements.txt` through `pip freeze`, the main packages used in the implementation are listed here. The reason is to avoid notification from Github bots about vulnerabilities. The options `--no-compile` and `--no-cache-dir` are to reduce the size of the container running the implementation. You can ignore those options if you do not use containers (e.g., Docker or Apptainer).
```bash
pip3 install -U "jax[cuda12]" --no-compile --no-cache-dir
pip3 install flax --no-compile --no-cache-dir
pip3 install optax --no-compile --no-cache-dir
pip3 install orbax-checkpoint --no-compile --no-cache-dir
pip3 install hydra-core --no-compile --no-cache-dir
pip3 install mlflow --no-compile --no-cache-dir
pip3 install tqdm --no-compile --no-cache-dir
pip3 install grain --no-compile --no-cache-dir
pip3 install albumentations --no-compile --no-cache-dir
pip3 install Pillow --no-compile --no-cache-dir
```

## Data loading
The data loading is written to be more framework-agnostic through JSON files. In particular, each JSON file is a list of "dictionary" objects, each will have the following format:
```json
[
    {
        "file": "<path_to_one_image_file>",
        "label": <true_label_in_int>
    },
]
```

## Experiment tracking and management
The experiment is managed through *MLFlow*. See the script `mlflow_server.sh` for further details.