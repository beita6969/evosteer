"""UUID-scoped elastic VRAM floor. No compute loops, no foreign process control.

Holds the card at TARGET occupancy so idle VRAM does not look free to a squatter,
and gives memory back the moment our own training wants it:
  - `desired` is recomputed from live NVML every loop, so as our usage ("other")
    grows the floor shrinks by exactly that much.
  - it never allocates while free VRAM is under HEADROOM, so a concurrent model
    load always has room.
Env: CUDA_VISIBLE_DEVICES = the GPU UUID. FLOOR_TARGET / FLOOR_HEADROOM_GIB optional.
"""
import ctypes as C, json, math, os, signal, time
import pynvml as N

uuid = os.environ['CUDA_VISIBLE_DEVICES']
TARGET = float(os.environ.get('FLOOR_TARGET', '0.82'))
HEADROOM = float(os.environ.get('FLOOR_HEADROOM_GIB', '28')) * 1024 ** 3

N.nvmlInit(); h = N.nvmlDeviceGetHandleByUUID(uuid)
drv = C.CDLL('libcuda.so.1')
def ok(code):
    if code: raise RuntimeError('CUDA driver status ' + str(code))
ok(drv.cuInit(0)); dev = C.c_int(); ok(drv.cuDeviceGet(C.byref(dev), 0))
ctx = C.c_void_p(); ok(drv.cuCtxCreate_v2(C.byref(ctx), 4, dev))
drv.cuMemAlloc_v2.argtypes = [C.POINTER(C.c_uint64), C.c_size_t]
drv.cuMemFree_v2.argtypes = [C.c_uint64]

chunk = 256 * 1024 ** 2
blocks = []; running = True; last_log = 0.
def stop(*_):
    global running
    running = False
signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
print(json.dumps({'event': 'start', 'pid': os.getpid(), 'gpu_uuid': uuid,
                  'target_fraction': TARGET, 'headroom_GiB': HEADROOM / 1024 ** 3}), flush=True)
try:
    while running:
        mem = N.nvmlDeviceGetMemoryInfo(h)
        held = len(blocks) * chunk
        other = max(0, mem.used - held)                 # everything that is not ours
        desired = max(0, math.ceil((TARGET * mem.total - other) / chunk))
        while len(blocks) > desired:                    # give it back immediately
            ok(drv.cuMemFree_v2(blocks.pop()))
        if len(blocks) < desired and mem.free > HEADROOM:
            ptr = C.c_uint64()
            code = drv.cuMemAlloc_v2(C.byref(ptr), C.c_size_t(chunk))
            if code == 0:
                blocks.append(ptr.value)
            else:
                print(json.dumps({'event': 'allocation_failure', 'cuda_status': code,
                                  'held_MiB': len(blocks) * 256}), flush=True)
                time.sleep(.2)
            continue
        now = time.monotonic()
        if now - last_log >= 60:
            mem = N.nvmlDeviceGetMemoryInfo(h)
            print(json.dumps({'pid': os.getpid(), 'gpu_uuid': uuid,
                              'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                              'used_MiB': round(mem.used / 1024 ** 2),
                              'free_MiB': round(mem.free / 1024 ** 2),
                              'fraction': round(mem.used / mem.total, 4),
                              'own_reserve_MiB': len(blocks) * 256,
                              'target_fraction': TARGET}), flush=True)
            last_log = now
        time.sleep(.05)
finally:
    for ptr in blocks:
        drv.cuMemFree_v2(ptr)
    drv.cuCtxDestroy_v2(ctx)
    N.nvmlShutdown()
