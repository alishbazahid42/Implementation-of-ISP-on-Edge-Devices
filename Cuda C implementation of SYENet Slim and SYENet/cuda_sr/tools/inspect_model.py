"""Inspect an unzipped PyTorch checkpoint (data.pkl + data/ storages) without torch."""
import pickle, io, struct, json, sys

DTYPE = {
    'FloatStorage': ('float32', 4), 'HalfStorage': ('float16', 2),
    'DoubleStorage': ('float64', 8), 'LongStorage': ('int64', 8),
    'IntStorage': ('int32', 4), 'ByteStorage': ('uint8', 1),
    'BoolStorage': ('bool', 1), 'BFloat16Storage': ('bfloat16', 2),
}

class FakeStorage:
    def __init__(self, cls_name, key, device, numel):
        self.dtype, self.itemsize = DTYPE[cls_name]
        self.key, self.device, self.numel = key, device, numel

def rebuild_tensor_v2(storage, offset, size, stride, requires_grad, hooks, *extra):
    return {'storage_key': storage.key, 'dtype': storage.dtype,
            'itemsize': storage.itemsize, 'device': storage.device,
            'storage_numel': storage.numel, 'offset': offset,
            'shape': list(size), 'stride': list(stride)}

class StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name in DTYPE:
            return name  # marker string; persistent_load builds FakeStorage
        if name == '_rebuild_tensor_v2':
            return rebuild_tensor_v2
        if (module, name) == ('collections', 'OrderedDict'):
            import collections
            return collections.OrderedDict
        raise pickle.UnpicklingError(f'unexpected global {module}.{name}')
    def persistent_load(self, pid):
        kind, cls_name, key, device, numel = pid
        assert kind == 'storage'
        return FakeStorage(cls_name, key, device, numel)

with open('data.pkl', 'rb') as f:
    sd = StubUnpickler(f).load()

total = 0
for k, v in sd.items():
    n = 1
    for d in v['shape']:
        n *= d
    total += n
    print(f"{k:35s} shape={str(v['shape']):20s} dtype={v['dtype']:8s} params={n:6d} storage=data/{v['storage_key']}")
print(f"\nTotal parameters: {total}")
json.dump({k: v for k, v in sd.items()}, open('state_dict_meta.json', 'w'), indent=1)
