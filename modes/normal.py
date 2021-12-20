#!/usr/bin/env python3

import asyncio
import httpx
import logging
import signal
import time
from random import random, randint
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
    logging.info("Starting clients")
    while run:
        num_clients = randint(1, 4)
        for i in range(num_clients):
            logging.info(f"client #{i}")
            num_requests = randint(1, 5)
            async with httpx.AsyncClient(timeout=10.0) as client:
                logging.info(f"#{i} launching requests")
                results = await asyncio.gather(
                    *[call(client) for _ in range(num_requests)], return_exceptions=True
                )
                logging.info(results)
        await asyncio.sleep(random() + 1)


def main():
    logging.info("Starting experiment in normal mode")
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
