"""Print input/output tensor descriptors of .cix graphs (loads the graph, no inference)."""

import sys

import libnoe
from libnoe import NOE_TENSOR_TYPE_INPUT, NOE_TENSOR_TYPE_OUTPUT, NPU

print("noe_data_type_t:", [n for n in dir(libnoe.noe_data_type_t) if n.startswith("NOE_DATA_TYPE")])
for path in sys.argv[1:]:
    npu = NPU()
    npu.noe_init_context()
    status, graph = npu.noe_load_graph(path)
    print(f"\n{path.rsplit('/', 2)[-2:]} status={status}")
    for kind, label in ((NOE_TENSOR_TYPE_INPUT, "in "), (NOE_TENSOR_TYPE_OUTPUT, "out")):
        _, count = npu.noe_get_tensor_count(graph, kind)
        for i in range(count):
            d = npu.noe_get_tensor_descriptor(graph, kind, i)
            print(f"  {label}[{i}] type={d.data_type} size={d.size} scale={d.scale:.6g} zp={d.zero_point}")
    npu.noe_unload_graph(graph)
    npu.noe_deinit_context()
