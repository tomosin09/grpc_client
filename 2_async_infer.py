import asyncio
import cv2
import os
from time import time, sleep
from triton_client import TritonClient


# Такой себе пример
class AsyncClient:
    def __init__(self):
        self.client = TritonClient(url='localhost:8001', model_name='extractor_onnx')
        if self.client.is_alive():
            self.inp_data, self.out_data = self.client.get_metadata()
        self.images = []
        self.tasks = []
        self.responses = []

    async def generate_request(self):
        if len(images) > 0:
            for i in range(len(self.images)):
                response = self.client.get_response(self.images.pop(0), self.inp_data, self.out_data)
                self.tasks.append(asyncio.create_task(response))
        await asyncio.gather(*self.tasks)
        for i in range(len(self.tasks)):
            if self.tasks[i].done:
                self.responses.append(self.tasks[i].result())
                self.tasks[i].cancel()

    def run(self):
        asyncio.run(self.generate_request())


if __name__ == "__main__":
    async_client = AsyncClient()
    images = []
    dir_images = 'test_image'
    names = os.listdir(dir_images)
    for i in names:
        images.append(os.path.join(dir_images, i))
    for i in images:
        img = cv2.imread(i)
        async_client.images.append(img)
    count = 0
    while True:
        print('count is', count)
        sleep(1)
        async_client.run()
        count += 1
        if count == 10:
            break
