import time
import sys
import tritonclient.grpc as grpcclient
import cv2
import numpy as np

from transform_image import preprocess


class TritonClient:
    def __init__(self, url, model_name, verbose=False):
        self.model_name = model_name
        self.verbose = verbose
        try:
            self.triton_client = grpcclient.InferenceServerClient(
                url=url,
                verbose=self.verbose)
        except Exception as e:
            print("context creation failed: " + str(e))
            sys.exit()

    def is_alive(self):
        condition = self.triton_client.is_server_live() \
                    and self.triton_client.is_server_ready() \
                    and self.triton_client.is_model_ready(self.model_name)
        if not condition:
            print('Impossible states onimage the server: server live state = {}, server ready state = {}, model state '
                  '= {} '
                  .format(self.triton_client.is_server_live(),
                          self.triton_client.is_server_ready(),
                          self.triton_client.is_model_ready(self.model_name)))
            sys.exit(1)
        return condition

    def get_metadata(self):
        metadata = self.triton_client.get_model_metadata(self.model_name)
        if len(metadata.inputs) != 1:
            raise Exception("expecting 1 input, got {}".format(
                len(metadata.inputs)))
        if len(metadata.outputs) != 1:
            raise Exception("expecting 1 output, got {}".format(
                len(metadata.outputs)))
        return metadata.inputs[0], metadata.outputs[0]

    def gen_response(self, frame):
        # time.sleep(2)
        infer_input = grpcclient.InferInput('input0', [1, 3, 112, 112], 'FP32')
        infer_output = grpcclient.InferRequestedOutput('output0')
        image_buf = preprocess(frame, 3, 112, 112)
        infer_input.set_data_from_numpy(image_buf)
        response = self.triton_client.infer(model_name='extractor_onnx',
                                            inputs=[infer_input],
                                            outputs=[infer_output])
        response = response.as_numpy('output0')
        return response


if __name__ == "__main__":
    t0 = time.time()
    images = []
    numbers_images = 10
    for i in range(numbers_images):
        images.append(np.random.random((112, 112, 3)).astype(np.float32))
    client = TritonClient(url='0.0.0.0:8001', model_name='model_name')
    if client.is_alive():
        for i, img in enumerate(images):
            t1 = time.time()
            result = client.gen_response(img)
            print(result.shape)
            print("-"*30+'\ninference time is', time.time() - t1)
    print("#"*30 + '\ninference time is', time.time() - t0)
