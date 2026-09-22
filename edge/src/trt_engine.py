"""
trt_engine.py — Thin wrapper around a TensorRT engine for Jetson Nano.

Uses the "binding index" TensorRT Python API (execute_async_v2 + a flat
bindings list), which is the API version shipped with JetPack 4.6.x
(TensorRT ~8.2) on the original Jetson Nano. If you're on a newer Jetson
(Orin, JetPack 5/6 with TensorRT 10+), the named-tensor API
(execute_async_v3, set_tensor_address) replaces this — check `trtexec
--version` if you're unsure which one you have.

pycuda and tensorrt are NOT pip-installable in the normal sense on Jetson —
they come with the JetPack OS image / apt packages
(`sudo apt install python3-libnvinfer python3-pycuda` or similar depending
on JetPack version). Do not try to `pip install tensorrt` on a Nano.
"""

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401 — initializes the CUDA context for this process
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTModel:
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"Failed to load TensorRT engine at {engine_path}. This usually means it "
                "was built on a different GPU/TensorRT version than what's running on this "
                "Nano right now — rebuild it here with build_tensorrt_engine.sh."
            )
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.bindings = []
        self.input_spec = None
        self.output_spec = None

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = trt.volume(shape)

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            spec = {"name": name, "host": host_mem, "device": device_mem, "shape": tuple(shape), "dtype": dtype}
            if self.engine.binding_is_input(i):
                self.input_spec = spec
            else:
                self.output_spec = spec

        # (N, H, W, C) input expected upstream — pull H, W out for preprocessing.
        _, self.in_h, self.in_w, _ = self.input_spec["shape"]
        self.in_dtype = self.input_spec["dtype"]

    def infer(self, input_array):
        """input_array: preprocessed (1, H, W, 3) array matching self.in_dtype."""
        np.copyto(self.input_spec["host"], input_array.ravel())
        cuda.memcpy_htod_async(self.input_spec["device"], self.input_spec["host"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.output_spec["host"], self.output_spec["device"], self.stream)
        self.stream.synchronize()
        return np.array(self.output_spec["host"]).reshape(self.output_spec["shape"])
