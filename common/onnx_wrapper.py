"""
Simple ONNXRuntime helper wrapper used by the pipeline.
Provides a thin interface to load sessions and run batched inputs.
"""
import os
try:
    import onnxruntime as ort
    HAS_ONNXRT = True
except Exception:
    ort = None
    HAS_ONNXRT = False

import numpy as np


class ONNXWrapper:
    def __init__(self, path, providers=None):
        if not HAS_ONNXRT:
            raise RuntimeError('onnxruntime not installed')
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.sess = ort.InferenceSession(path, providers=providers)
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def run(self, inputs: dict):
        # inputs: name->numpy array
        return self.sess.run(self.output_names, inputs)


if __name__ == '__main__':
    print('ONNXWrapper module')
