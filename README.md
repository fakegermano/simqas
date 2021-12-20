# Setup

Build and run the image for the server and the monitoring containers using docker-compose

``` shell
docker-compose up --build -d
```

- The `simqas` service will be available at port `8000`
- The `cadvisor` service will collect data from the running containers and be available at port `8080`
- The `prometheus` service will gather the metrics from `cadvisor` and be available at port `9090`

You need to install the tool [`pyenv`](https://github.com/pyenv/pyenv) in your system.

Withe the tools installed you should do from the root of the project folder:

``` shell
pyenv install 3.8.12
pyenv local 3.8.12
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```

# Running the simulations:
## Running normal mode:

``` shell
# on one terminal:
poetry run python monitoring/extractor.py normal

# on another:
poetry run python modes/normal.py
```

## Running in DoS simulated mode:

``` shell
# on one terminal:
poetry run python monitoring/extractor.py dos

# on another:
poetry run python modes/dos.py
# slowloris with nmap
```

## Running in Malware simulated mode:

``` shell
# on one terminal:
poetry run python monitoring/extractor.py malware

# on another:
poetry run python modes/malware.py
# inside container: uses base python to connect to a server and send files
# outside container: setup a netcat listen: nc -l <port> | tar -x -C extracted
```

## Running in Ransomware simulated mode:

``` shell
# on one terminal:
poetry run python monitoring/extractor.py ransom

# on another:
poetry run python modes/ransom.py
# read and write whole files from disk
```
# Running the data processing and analysis

``` shell
poetry run python processing.py
```
