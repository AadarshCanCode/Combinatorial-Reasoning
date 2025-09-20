import sys
import types


def _inject_module(name, attrs=None):
    m = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(m, k, v)
    sys.modules[name] = m
    return m


# Lightweight stubs for optional heavy dependencies used in optimization/quantum code
_inject_module('qiskit_aer', {'Aer': type('Aer', (), {})})

_inject_module('qiskit_algorithms', {'QAOA': type('QAOA', (), {})})

# qiskit_optimization package and its submodules
qo = _inject_module('qiskit_optimization', {'QuadraticProgram': type('QuadraticProgram', (), {})})
alg = types.ModuleType('qiskit_optimization.algorithms')
alg.RecursiveMinimumEigenOptimizer = type('RecursiveMinimumEigenOptimizer', (), {})
sys.modules['qiskit_optimization.algorithms'] = alg
conv = types.ModuleType('qiskit_optimization.converters')
conv.QuadraticProgramToQubo = type('QuadraticProgramToQubo', (), {})
sys.modules['qiskit_optimization.converters'] = conv

# qiskit top-level minimal stubs
qiskit = _inject_module('qiskit')
qiskit.utils = types.ModuleType('qiskit.utils')
sys.modules['qiskit.utils'] = qiskit.utils

# D-Wave system stubs
dw = _inject_module('dwave.system', {
    'DWaveSampler': type('DWaveSampler', (), {}),
    'EmbeddingComposite': type('EmbeddingComposite', (), {}),
})
