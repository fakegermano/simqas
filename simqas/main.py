import random
import uvicorn
import numpy as np
from fastapi import FastAPI
from fastapi.logger import logger
from pydantic import BaseModel
from typing import List

app = FastAPI()


class Output(BaseModel):
    first: List[float]
    last: List[float]


def get_data(index):
    try:
        array = np.load(f"data/{index}.npz", mmap_mode="r")
        logger.info(array["arr_0"].nbytes)
        return {
            "first": list(array["arr_0"].flatten()[: (index + 1) * 5]),
            "last": list(array["arr_0"].flatten()[-(index + 1) * 5 :]),
        }
    finally:
        array.close()


@app.get("/", response_model=Output)
async def index():
    index = random.randint(0, 9)
    return get_data(index)


if __name__ == "__main__":
    uvicorn.run(app)
