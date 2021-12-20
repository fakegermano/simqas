#!/usr/bin/env python3

import asyncio
import httpx
import logging
import signal
import time
import socket
import random
import os
import base64
import docker
from random import random
from datetime import datetime

run = True


def handler_stop_signal(signum, frame):
    print("Stopping...")
    global run
    run = False


signal.signal(signal.SIGTERM, handler_stop_signal)
signal.signal(signal.SIGINT, handler_stop_signal)


async def call(client):
    await asyncio.sleep(random())
    url = "http://localhost:8000"
    response = await client.get(url)
    response.raise_for_status()
    return response


async def client(start):
    logging.info("Starting client and Ransom attack")

    num_clients = 2
    num_requests = 3
    lock = asyncio.Lock()
    while run:
        asyncio.create_task(ransom(lock))
        for i in range(num_clients):
            logging.info(f"client #{i}")
            async with httpx.AsyncClient(timeout=10.0) as client:
                logging.info(f"#{i} launching requests")
                results = await asyncio.gather(
                    *[call(client) for _ in range(num_requests)], return_exceptions=True
                )
                logging.info(results)
        await asyncio.sleep(random() + 1)


async def ransom(lock):
    async with lock:
        with open("modes/ransom_code.py", "rb") as mfile:
            mdata = base64.b64encode(mfile.read())

        client = docker.from_env()
        container = client.containers.get("simqas_server_1")
        await asyncio.sleep(1)

        logging.info("Running ransom on container")
        code, output = container.exec_run(
            "/bin/sh -c 'echo ${RANSOM} | base64 -d | python -'",
            environment={"RANSOM": mdata},
        )
        logging.info("%d: %s", code, output.decode("utf-8"))
        await asyncio.sleep(5)


def main():
    logging.info("Starting experiment in ransom mode")

    elapsed = time.perf_counter()
    start = datetime.now().astimezone().replace(microsecond=0).isoformat()
    asyncio.run(client(elapsed))
    end = datetime.now().astimezone().replace(microsecond=0).isoformat()
    elapsed = time.perf_counter() - elapsed

    logging.info(f"Normal mode executed in {elapsed:.2f} seconds")
    logging.info(f"Start time: {start} End time: {end}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
