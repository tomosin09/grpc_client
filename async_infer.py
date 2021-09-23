import asyncio
import cv2
import os
from time import time, sleep
from triton_client import TritonClient
import concurrent.futures


def block_function():
    count = 0
    while True:
        sleep(0.4)
        print('count is', count)
        count += 1
        if count == 2:
            break


class AsyncClient:
    def __init__(self):
        self.client = TritonClient(url='localhost:8001', model_name='extractor_onnx')
        if self.client.is_alive():
            self.inp_data, self.out_data = self.client.get_metadata()
        self.images = []
        self.tasks = []

    async def generate_request(self):
        running_loop = asyncio.get_running_loop()
        if len(images) > 0:
            for i in range(len(self.images)):
                response = self.client.get_response(self.images.pop(0), self.inp_data, self.out_data, i)
                self.tasks.append(asyncio.create_task(response))
        with concurrent.futures.ThreadPoolExecutor() as pool:
            await running_loop.run_in_executor(pool, block_function)
        await asyncio.gather(*self.tasks)

    def run(self):
        asyncio.run(self.generate_request())


if __name__ == "__main__":
    async_client = AsyncClient()
    images = []
    dir_images = 'test_image'
    names = os.listdir(dir_images)
    for i in names:
        images.append(os.path.join(dir_images, i))
    t0 = time()
    for i in images:
        img = cv2.imread(i)
        async_client.images.append(img)
    async_client.run()
    print('inference time is', time() - t0)
