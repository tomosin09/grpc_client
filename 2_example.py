from triton_client import TritonClient
from time import sleep, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def working():
    count = 0
    client = TritonClient(url='localhost:8001', model_name='extractor_onnx')
    executor = ThreadPoolExecutor(max_workers=1)
    message = None
    future = None
    while 1:
        print(f'count is {count}')
        image = np.random.random((120, 120, 3))
        if image is not None and future is None:
            message = image
            future = executor.submit(client.gen_response, message)
        sleep(1)
        if future is not None:
            if future.running():
                print('future still running...')
            if future.done():
                print('-------FUTURE IS DONE-------')
                print('GOT VECTOR WHICH HAS SHAPE', future.result().shape)
                future.cancel()
                future = None
                diff = np.subtract(message, image)
                diff = np.mean(diff)
                if diff:
                    message = image
                    future = executor.submit(client.gen_response, message)
        count += 1
        if count == 50:
            break


if __name__ == '__main__':
    working()
