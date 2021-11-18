from functools import partial
import argparse
import numpy as np
import time
import sys
import queue

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

FLAGS = None


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


def async_stream_send(triton_client, images, sequence_id, model_name):
    count = 1
    for image in images:
        inputs = [grpcclient.InferInput('input0', [1, 3, 112, 112], 'FP32')]
        # Initialize the data
        inputs[0].set_data_from_numpy(image)
        outputs = [grpcclient.InferRequestedOutput('output0')]
        # Issue the asynchronous sequence inference.
        triton_client.async_stream_infer(model_name=model_name,
                                         inputs=inputs,
                                         outputs=outputs,
                                         request_id='{}_{}'.format(
                                             sequence_id, count),
                                         sequence_id=sequence_id,
                                         sequence_start=(count == 1),
                                         sequence_end=(count == len(images)))
        count = count + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False,
                        default=False, help='Enable verbose output')
    parser.add_argument('-t', '--stream-timeout', type=float, required=False,
                        default=None, help='Stream timeout in seconds. Default is None.')
    parser.add_argument("-u", "--url", default='0.0.0.0:8001', help='path to project dir')
    parser.add_argument("-n", "--model_name", default='model_name', help='path to project dir')
    parser.add_argument("--max_number", default=3, help='maximum numbers images')
    parser.add_argument("--numbers_requests", default=5, help='numbers requests for one image')
    args = parser.parse_args()
    diff = 0
    for num in range(args.max_number):
        num += 1
        average_time = 0
        results = []
        images = []
        for i in range(num):
            images.append(np.random.random((1, 3, 112, 112)).astype(np.float32))
        user_data = UserData()
        for r in range(args.numbers_requests):
            t0 = time.time()
            with grpcclient.InferenceServerClient(
                    url=args.url, verbose=args.verbose) as triton_client:
                try:
                    # Establish stream
                    triton_client.start_stream(callback=partial(callback, user_data),
                                               stream_timeout=args.stream_timeout)
                    async_stream_send(triton_client, images, 1, args.model_name)
                except InferenceServerException as error:
                    print(error)
                    sys.exit(1)
            for i in range(len(images)):
                data_item = user_data._completed_requests.get()
                if type(data_item) == InferenceServerException:
                    print(data_item)
                    sys.exit(1)
                else:
                    this_id = data_item.get_response().id.split('_')[0]
                    if int(this_id) == 1:
                        results.append(data_item.as_numpy('output0'))
                    else:
                        print("unexpected sequence id returned by the server: {}".format(this_id))
                        sys.exit(1)
            t1 = time.time() - t0
            average_time += t1
        average_time = average_time / args.numbers_requests
        print('-'*30 + f'\ncompleted {args.numbers_requests} requests with {num} images')
        print(f'average time = {round(average_time, 3)}')
        print(f'got {len(results)} responses from {args.url}')