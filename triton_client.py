import asyncio
import time
import sys
import tritonclient.grpc as grpcclient
import random
import cv2
import os

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

    def gen_response(self, frame, input_data, output_data):
        infer_input = grpcclient.InferInput(input_data.name, input_data.shape, input_data.datatype)
        infer_output = grpcclient.InferRequestedOutput(output_data.name)
        image_buf = preprocess(frame, input_data.shape[1],
                               input_data.shape[2],
                               input_data.shape[3])
        infer_input.set_data_from_numpy(image_buf)
        infer_result = self.triton_client.infer(model_name=self.model_name,
                                                inputs=[infer_input],
                                                outputs=[infer_output])
        return infer_result.as_numpy(output_data.name)

    async def get_response(self, frame, input_data, output_data):
        print('request image')
        infer_input = grpcclient.InferInput(input_data.name, input_data.shape, input_data.datatype)
        infer_output = grpcclient.InferRequestedOutput(output_data.name)
        image_buf = preprocess(frame, input_data.shape[1],
                               input_data.shape[2],
                               input_data.shape[3])
        infer_input.set_data_from_numpy(image_buf)
        infer_result = self.triton_client.infer(model_name=self.model_name,
                                                inputs=[infer_input],
                                                outputs=[infer_output])
        return infer_result.as_numpy(output_data.name)


if __name__ == "__main__":
    t0 = time.time()
    images = []
    dir_images = 'test_image'
    names = os.listdir(dir_images)
    for i in names:
        images.append(os.path.join(dir_images, i))
    client = TritonClient(url='localhost:8001', model_name='extractor_onnx')
    if client.is_alive():
        input_metadata, output_metadata = client.get_metadata()
        for i, img in enumerate(images):
            image = cv2.imread(img)
            result = client.gen_response(image, input_metadata, output_metadata)
            print(result.shape)
    print('inference time is', time.time() - t0)
