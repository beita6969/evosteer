"""Keep this pod's own GPU at a memory floor, yielding to our training as it grows.

Same design as the other project's memory_floor: hold 256 MiB blocks up to a target
fraction, recompute every loop from what other processes use, and free blocks as soon
as they need the memory. UUID-scoped to this pod's two H200s. usage: memory_floor2.py <logical_index>
"""
import ctypes as C, json, math, os, signal, sys, time
import pynvml as N

ALLOWED = {
    "GPU-00000000-0000-0000-0000-000000000000": 0,
    "GPU-00000000-0000-0000-0000-000000000001": 1,
}
uuid = os.environ["CUDA_VISIBLE_DEVICES"]
assert uuid in ALLOWED
N.nvmlInit()
handle = N.nvmlDeviceGetHandleByUUID(uuid)
driver = C.CDLL("libcuda.so.1")
def ok(code):
    if code:
        raise RuntimeError("CUDA driver status " + str(code))
ok(driver.cuInit(0))
device = C.c_int(); ok(driver.cuDeviceGet(C.byref(device), 0))
context = C.c_void_p(); ok(driver.cuCtxCreate_v2(C.byref(context), 4, device))
driver.cuMemAlloc_v2.argtypes = [C.POINTER(C.c_uint64), C.c_size_t]
driver.cuMemFree_v2.argtypes = [C.c_uint64]
chunk = 256 * 1024 ** 2
blocks, running, last = [], True, 0.0
def stop(*_):
    global running
    running = False
signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
try:
    while running:
        memory = N.nvmlDeviceGetMemoryInfo(handle)
        held = len(blocks) * chunk
        other = max(0, memory.used - held)
        desired = max(0, math.ceil((0.62 * memory.total - other) / chunk))
        while len(blocks) > desired:
            ok(driver.cuMemFree_v2(blocks.pop()))
        if len(blocks) < desired and memory.free > 20 * 1024 ** 3:
            pointer = C.c_uint64()
            if driver.cuMemAlloc_v2(C.byref(pointer), C.c_size_t(chunk)) == 0:
                blocks.append(pointer.value)
            else:
                time.sleep(0.2)
            continue
        now = time.monotonic()
        if now - last >= 60:
            print(json.dumps({"gpu_uuid": uuid, "used_MiB": round(memory.used / 1024 ** 2),
                              "fraction": round(memory.used / memory.total, 4),
                              "own_reserve_MiB": len(blocks) * 256}), flush=True)
            last = now
        time.sleep(0.05)
finally:
    for pointer in blocks:
        driver.cuMemFree_v2(pointer)
    driver.cuCtxDestroy_v2(context)
    N.nvmlShutdown()
