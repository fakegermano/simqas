#!/usr/bin/env python3
import docker
import time
import signal
import requests
import csv
import sys
import shutil

run = True


def signal_handler(signal, frame):
    print("Stopping...")
    global run
    run = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_keys(obj, prefix=""):
    keys = set()
    for k, v in obj.items():
        if isinstance(v, dict):
            keys |= get_keys(v, prefix=prefix + f"{k}_")
        elif isinstance(v, list):
            if isinstance(v[0], dict):
                for i, _v in enumerate(v):
                    keys |= get_keys(_v, prefix=prefix + f"{k}_{i}_")
        elif not isinstance(v, bool):
            keys.add(prefix + k)
    return keys


def extract_values(obj, prefix=""):
    values = dict()
    for k, v in obj.items():
        if isinstance(v, dict):
            values.update(extract_values(v, prefix=prefix + f"{k}_"))
        elif isinstance(v, list):
            if isinstance(v[0], dict):
                for i, _v in enumerate(v):
                    values.update(extract_values(_v, prefix=prefix + f"{k}_{i}_"))
        elif not isinstance(v, bool):
            values[prefix + k] = v
    return values


LABEL = sys.argv[1]

if __name__ == "__main__":
    shutil.copyfile("saved.csv", "saved.csv.bk")
    client = docker.from_env()
    container = client.containers.get("simqas_server_1")
    headers = set()

    response = requests.get(
        f"http://localhost:8080/api/v2.0/stats/{container.short_id}",
        params={"type": "docker"},
    )
    assert response.status_code == 200
    data = list(response.json().values())[0]

    for v in data:
        headers |= get_keys(v)
    headers.add("label")
    headers = list(headers)
    headers.sort()
    while run:
        values = {}
        with open("saved.csv", "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                values[row["timestamp"]] = row
        print(f"Read {len(values)} from csv")
        time.sleep(1)
        response = requests.get(
            f"http://localhost:8080/api/v2.0/stats/{container.short_id}",
            params={"type": "docker"},
        )
        assert response.status_code == 200
        stats = list(response.json().values())[0]
        for entry in stats:
            if entry["timestamp"] not in values:
                values[entry["timestamp"]] = extract_values(entry)
                values[entry["timestamp"]]["label"] = LABEL

        with open("saved.csv", "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for v in values.values():
                writer.writerow(v)
        print(f"Wrote {len(values)} to csv")
