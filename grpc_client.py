import sys
import time
import logging
import cv2
import numpy as np
import tritonclient.grpc as grpcclient


def preprocess(image, channels, h, w):
    if image.shape[:2] != (128, 128):
        image = cv2.resize(image, (128, 128))
        # center crop image
        a = int((128 - w) / 2)  # x start
        b = int((128 - w) / 2 + 112)  # x end
        c = int((128 - h) / 2)  # y start
        d = int((128 - h) / 2 + 112)  # y end
        cropped = image[a:b, c:d]  # center crop the image
        cropped = cropped[..., ::-1]  # BGR to RGB
        # flip image horizontally
        flipped = cv2.flip(cropped, 1)

        def to_format(image):
            image = image.swapaxes(1, 2).swapaxes(0, 1)
            image = np.reshape(image, [1, channels, w, h])
            image = np.array(image, dtype=np.float32)
            image = (image - 127.5) / 128.0
            return image

        return (to_format(cropped) + to_format(flipped)).astype(np.float32)


class gRPCClient:
    def __init__(self, url, model_name, verbose=False, ssl=False, root_certificates=None,
                 private_key=None, certificate_chain=None):
        self.url = url
        self.model_name = model_name
        self.verbose = verbose
        self.ssl = ssl
        self.root_certificates = root_certificates
        self.private_key = private_key
        self.certificate_chain = certificate_chain
        self.triton_client = None
        self.input_metadata = None
        self.output_metadata = None

    def set_triton_client(self):
        try:
            self.triton_client = grpcclient.InferenceServerClient(
                url=self.url,
                verbose=self.verbose,
                ssl=self.ssl,
                root_certificates=self.root_certificates,
                private_key=self.private_key,
                certificate_chain=self.certificate_chain)
        except Exception as e:
            logging.error("context creation failed: " + str(e))
            sys.exit()

    def is_alive(self):
        if self.triton_client is not None:
            condition = self.triton_client.is_server_live() \
                        and self.triton_client.is_server_ready() \
                        and self.triton_client.is_model_ready(self.model_name)
            if not condition:
                logging.error('Impossible states: server live state = {}, server ready state = {}, model state = {}'
                              .format(self.triton_client.is_server_live(),
                                      self.triton_client.is_server_ready(),
                                      self.triton_client.is_model_ready(self.model_name)))
                sys.exit(1)
            return condition
        else:
            logging.error('client is None')
            sys.exit(1)

    def get_metadata(self):
        metadata = self.triton_client.get_model_metadata(self.model_name)
        if len(metadata.inputs) != 1:
            raise Exception("expecting 1 input, got {}".format(
                len(metadata.inputs)))
        if len(metadata.outputs) != 1:
            raise Exception("expecting 1 output, got {}".format(
                len(metadata.outputs)))
        self.input_metadata = metadata.inputs[0]
        self.output_metadata = metadata.outputs[0]

    def do_request(self, image):
        inputs = []
        outputs = []
        inputs.append(
            grpcclient.InferInput(self.input_metadata.name, self.input_metadata.shape, self.input_metadata.datatype))
        outputs.append(grpcclient.InferRequestedOutput(self.output_metadata.name))
        if image is None:
            print(f'image is not received')
            sys.exit(1)
        input_image_buffer = preprocess(image, self.input_metadata.shape[1],
                                        self.input_metadata.shape[2],
                                        self.input_metadata.shape[3])
        inputs[0].set_data_from_numpy(input_image_buffer)
        result = self.triton_client.infer(model_name=self.model_name,
                                          inputs=inputs,
                                          outputs=outputs,
                                          client_timeout=None)
        return result.as_numpy(self.output_metadata.name)


if __name__ == "__main__":
    client = gRPCClient(url='localhost:8001', model_name='extractor_onnx')
    client.set_triton_client()
    if client.is_alive():
        client.get_metadata()
    path_image = 'test_image/track-3.png'
    image = cv2.imread(path_image)
    start = time.time()
    response = client.do_request(image).reshape((1, 512))
    print('infernce time is {}'.format(time.time() - start))
    print(response.shape)
