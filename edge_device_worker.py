"""edge_device_worker.py - the program that runs *inside* the simulated edge device.

The notebook launches this script in a sandbox limited to 256 MB of RAM and one
CPU core. Only NumPy and the stand-alone LiteRT runtime (``ai_edge_litert``) are
imported: full TensorFlow is never loaded on the "device", exactly as on a real
board where only the TFLite/LiteRT interpreter is installed.

Usage
-----
  python edge_device_worker.py bench <model.tflite> <image.npy> <n_warmup> <n_runs>
  python edge_device_worker.py eval  <model.tflite> <images.npy> <labels.npy>
  python edge_device_worker.py alloc <megabytes>            # sandbox self-test

Each mode prints a single JSON line on stdout.
"""
import json
import os
import resource
import sys
import time


def proc_status(key):
    """Read a field such as VmRSS / VmHWM (peak RSS) from /proc/self/status, in bytes."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024
    return None


def device_info():
    return {
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "os_threads": len(os.listdir("/proc/self/task")),
        "pid": os.getpid(),
    }


if sys.argv[1] == "alloc":
    # Self-test: try to allocate more memory than the device has.
    mb = int(sys.argv[2])
    block = bytearray(b"\x01") * (mb * 2**20)   # every page is written, so it is really resident
    print(json.dumps({"allocated_mb": mb, "peak_rss": proc_status("VmHWM")}))
    sys.exit(0)

import numpy as np                                   # noqa: E402
from ai_edge_litert.interpreter import Interpreter   # noqa: E402

RSS_RUNTIME = proc_status("VmRSS")   # Python + NumPy + LiteRT, before any model is loaded


def load_interpreter(path):
    t0 = time.perf_counter()
    interp = Interpreter(model_path=path, num_threads=1)
    interp.allocate_tensors()
    return interp, (time.perf_counter() - t0) * 1e3


def to_model_input(pixels_uint8, inp):
    """Raw RGB pixels (0..255) -> whatever dtype the model's input tensor expects."""
    if inp["dtype"] == np.uint8:
        return pixels_uint8.astype(np.uint8)
    if inp["dtype"] == np.int8:
        scale, zero_point = inp["quantization"]
        q = np.round(pixels_uint8.astype(np.float32) / scale + zero_point)
        return np.clip(q, -128, 127).astype(np.int8)
    return pixels_uint8.astype(np.float32)


def bench(model_path, image_path, n_warmup, n_runs):
    interp, load_ms = load_interpreter(model_path)
    rss_after_load = proc_status("VmRSS")
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    x = to_model_input(np.load(image_path)[None], inp)

    def run_once():
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        return interp.get_tensor(out["index"])

    t0 = time.perf_counter()
    run_once()                                     # cold first inference
    first_ms = (time.perf_counter() - t0) * 1e3
    for _ in range(n_warmup):
        run_once()

    cpu0 = resource.getrusage(resource.RUSAGE_SELF)
    wall0 = time.perf_counter()
    lat = np.empty(n_runs)
    for i in range(n_runs):
        t0 = time.perf_counter()
        run_once()
        lat[i] = time.perf_counter() - t0
    wall = time.perf_counter() - wall0
    cpu1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu_time = (cpu1.ru_utime - cpu0.ru_utime) + (cpu1.ru_stime - cpu0.ru_stime)

    return {
        "load_ms": load_ms,
        "first_ms": first_ms,
        "lat_ms": (lat * 1e3).round(4).tolist(),
        "rss_runtime": RSS_RUNTIME,
        "rss_after_load": rss_after_load,
        "rss_peak": proc_status("VmHWM"),
        "cores_used": cpu_time / wall,     # ~1.0 means exactly one core was busy
        "input_dtype": str(np.dtype(inp["dtype"])),
        **device_info(),
    }


def iter_npy_chunks(path, chunk=32):
    """Stream a large .npy array from storage a few images at a time (never the whole file in RAM)."""
    with open(path, "rb") as f:
        major, _ = np.lib.format.read_magic(f)
        read_header = (np.lib.format.read_array_header_1_0 if major == 1
                       else np.lib.format.read_array_header_2_0)
        shape, _, dtype = read_header(f)
        per_item = int(np.prod(shape[1:]))
        for start in range(0, shape[0], chunk):
            n = min(chunk, shape[0] - start)
            yield np.fromfile(f, dtype=dtype, count=n * per_item).reshape((n,) + tuple(shape[1:]))


def evaluate(model_path, images_path, labels_path):
    interp, _ = load_interpreter(model_path)
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    labels = np.load(labels_path)
    top5 = np.empty((len(labels), 5), dtype=np.int32)
    i = 0
    t0 = time.perf_counter()
    for batch in iter_npy_chunks(images_path):
        for img in batch:                           # batch size 1, as on the device
            interp.set_tensor(inp["index"], to_model_input(img[None], inp))
            interp.invoke()
            scores = interp.get_tensor(out["index"])[0]
            # stable sort so that ties between equal uint8 scores are broken deterministically
            top5[i] = np.argsort(-scores.astype(np.int32 if scores.dtype == np.uint8 else np.float32),
                                 kind="stable")[:5]
            i += 1
    assert i == len(labels), (i, len(labels))
    return {
        "n": int(len(labels)),
        "top1": float(np.mean(top5[:, 0] == labels)),
        "top5": float(np.mean((top5 == labels[:, None]).any(axis=1))),
        "pred_top1": top5[:, 0].tolist(),
        "eval_seconds": time.perf_counter() - t0,
        "rss_peak": proc_status("VmHWM"),
        **device_info(),
    }


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "bench":
        result = bench(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    elif mode == "eval":
        result = evaluate(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    print(json.dumps(result))
