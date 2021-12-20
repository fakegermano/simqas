#!/usr/bin/env python3

import asyncio
import httpx
import signal
import time
import socket
import random
from loguru import logger as logging
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
    logging.info("Starting client and DoS attack")
    asyncio.create_task(attack())
    num_clients = 1
    num_requests = 1
    while run:
        for i in range(num_clients):
            logging.info(f"client #{i}")
            async with httpx.AsyncClient(timeout=10.0) as client:
                logging.info(f"#{i} launching requests")
                results = await asyncio.gather(
                    *[call(client) for _ in range(num_requests)], return_exceptions=True
                )
                logging.info(results)
        await asyncio.sleep(random() + 1)


async def attack():
    while run:
        nmap_proc = await asyncio.create_subprocess_shell(
            "nmap -Pn -p8000 --script http-slowloris --max-parallelism 750 localhost",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        child_pid = nmap_proc.pid

        await nmap_proc.wait()


def main():
    logging.info("Starting experiment in dos mode")

    elapsed = time.perf_counter()
    start = datetime.now().astimezone().replace(microsecond=0).isoformat()
    asyncio.run(client(elapsed))
    end = datetime.now().astimezone().replace(microsecond=0).isoformat()
    elapsed = time.perf_counter() - elapsed

    logging.info(f"Dos mode executed in {elapsed:.2f} seconds")
    logging.info(f"Start time: {start} End time: {end}")


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    main()
