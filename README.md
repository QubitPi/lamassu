---
title: Lamassu
emoji: 🤗
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 5.1.0
app_file: app.py
pinned: false
license: apache-2.0
---

[![Hugging Face space badge]][Hugging Face space URL]
[![Hugging Face sync status badge]][Hugging Face sync status URL]
[![MLflow badge]][MLflow URL]
[![MLflow build status badge]][MLflow build status URL]
[![Apache License Badge]][Apache License, Version 2.0]

Lamassu is a Named Entity Extraction service that is capable of running on [Hugging Face][Hugging Face space URL] and
MLflow managed environment. It is the service backing the [Nexus Graph](https://paion-data.github.io/nexusgraph.com/)

Hugging Face
------------

Lamassu is directly available on [Hugging Face space][Hugging Face space URL]. Please check it out.

MLflow
------

![Python Version Badge]

### Getting Source Code

```console
git clone git@github.com:QubitPi/lamassu.git
```

### Running Locally

Create virtual environment and install dependencies:

```console
cd lamassu/mlflow
python3 -m venv .venv
. .venv/bin/activate
pip3 install -r requirements.txt
```

Generate Model with

```console
python3 HanLPner.py
```

A model directory called "HanLPner" appears under `mlflow/models`. Then build Docker image

```console
mlflow models build-docker --name "entity-extraction"
```

and run container with

```console
cp parser.py models/HanLPner/
export ML_MODEL_PATH=/absolute/path/to/models/HanLPner

docker run --rm \
  --memory=4000m \
  -p 8080:8080 \
  -v $ML_MODEL_PATH:/opt/ml/model \
  -e PYTHONPATH="/opt/ml/model:$PYTHONPATH" \
  -e GUNICORN_CMD_ARGS="--timeout 60 -k gevent" \
  "entity-extraction"
```

> [!TIP]
> If `docker.errors.DockerException: Error while fetching server API version: ('Connection aborted.', FileNotFoundError(2, 'No such file or directory'))`
> error is seen, refer to
> https://forums.docker.com/t/docker-errors-dockerexception-error-while-fetching-server-api-version-connection-aborted-filenotfounderror-2-no-such-file-or-directory-error-in-python/135637/5

The container runs Gunicorn server inside to serve incoming requests.

Example query:

```bash
curl -X POST -H "Content-Type:application/json" \
  --data '{"dataframe_split": {"columns":["text"], "data":[["我爱中国"], ["世界会变、科技会变，但「派昂」不会变，它不会向任何人低头，不会向任何困难低头，甚至不会向「时代」低头。「派昂」，永远引领对科技的热爱。只有那些不向梦想道路上的阻挠认输的人，才配得上与我们一起追逐梦想"]]}}' \
  http://127.0.0.1:8080/invocations
```

[Note the JSON schema of the `--data` value](https://stackoverflow.com/a/75104855)

License
-------

The use and distribution terms for [lamassu]() are covered by the [Apache License, Version 2.0].

[Apache License Badge]: https://img.shields.io/badge/Apache%202.0-F25910.svg?style=for-the-badge&logo=Apache&logoColor=white
[Apache License, Version 2.0]: https://www.apache.org/licenses/LICENSE-2.0

[Hugging Face space badge]: https://img.shields.io/badge/Hugging%20Face%20Space-lamassu-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white
[Hugging Face space URL]: https://huggingface.co/spaces/QubitPi/lamassu

[Hugging Face sync status badge]: https://img.shields.io/github/actions/workflow/status/QubitPi/lamassu/ci-cd.yaml?branch=master&style=for-the-badge&logo=github&logoColor=white&label=Hugging%20Face%20Sync%20Up
[Hugging Face sync status URL]: https://github.com/QubitPi/lamassu/actions/workflows/ci-cd.yaml

[MLflow badge]: https://img.shields.io/badge/MLflow%20Supported-0194E2?style=for-the-badge&logo=mlflow&logoColor=white
[MLflow URL]: https://mlflow.qubitpi.org/
[MLflow build status badge]: https://img.shields.io/github/actions/workflow/status/QubitPi/lamassu/ci-cd.yaml?branch=master&style=for-the-badge&logo=github&logoColor=white&label=MLflow%20Build
[MLflow build status URL]: https://github.com/QubitPi/lamassu/actions/workflows/ci-cd.yaml

[Python Version Badge]: https://img.shields.io/badge/Python-3.10-brightgreen?style=for-the-badge&logo=python&logoColor=white
