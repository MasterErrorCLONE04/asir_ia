Correcto. Con esos dos ajustes, **yo daría el plan por aprobado y pasaría a implementación**. Los dos puntos que señalas son importantes porque afectan la mantenibilidad del repo y la reproducibilidad del protocolo R1.

Haría estos cambios definitivos antes del primer commit.

## 1. Estructura definitiva

No crearíamos nombres que compitan semánticamente con los módulos existentes.

```text
adaptive-inference/
│
├── analysis/
│   ├── evaluator.py
│   ├── moe_profiler.py
│   ├── quantization_eval.py       # NUEVO
│   └── ...
│
├── training/
│   ├── models/
│   │   ├── transformer.py
│   │   ├── attention.py           # NUEVO
│   │   └── ...
│   │
│   ├── moe/
│   │   ├── reference.py           # implementación actual/oráculo
│   │   ├── router.py
│   │   ├── dispatcher.py          # NUEVO
│   │   └── combine.py             # NUEVO
│   │
│   ├── profiling/
│   │   ├── memory.py              # NUEVO
│   │   └── dispatch_timing.py     # NUEVO
│   │
│   ├── optim/
│   │   ├── factory.py             # NUEVO
│   │   ├── adamw.py               # NUEVO
│   │   └── adam8bit.py            # NUEVO
│   │
│   ├── memory/
│   │   ├── manager.py             # NUEVO
│   │   └── checkpoint.py          # NUEVO
│   │
│   └── train.py
│
├── runtime/
│   ├── core/
│   ├── experts/
│   ├── cache/
│   ├── paging/
│   ├── quantization/
│   └── benchmarks/
│
├── tests/
│   ├── test_moe_integration.py
│   ├── test_r13_real_moe_smoke.py
│   ├── test_moe_dispatch.py       # NUEVO
│   ├── test_memory.py             # NUEVO
│   ├── test_cache.py              # NUEVO
│   └── test_quantization.py       # NUEVO
│
├── requirements.txt               # R1 CPU — NO TOCAR
├── requirements-gpu.txt           # NUEVO
├── Dockerfile                     # R1 CPU — NO TOCAR
└── Dockerfile.gpu                 # NUEVO
```

La separación queda bastante limpia:

### `analysis/`

**¿Qué significa científicamente el resultado?**

```text
moe_profiler.py
evaluator.py
quantization_eval.py
```

### `training/profiling/`

**¿Qué está haciendo físicamente el entrenamiento?**

```text
memory.py
dispatch_timing.py
```

### `runtime/`

**¿Cómo ejecutamos el modelo con memoria limitada?**

---

# 2. Tests: CPU-first

Esto también lo establecería como una regla de arquitectura.

Los tests del dispatcher **no necesitan CUDA**.

El test debe construir algo pequeño:

```python
d_model = 64
n_layers = 2
num_experts = 8
top_k = 2
```

y comparar:

```text
ReferenceMoE
      │
      ├── output
      ├── loss
      └── gradients
             │
             ▼
      OptimizedMoE
```

Con:

```python
torch.testing.assert_close(...)
```

y tolerancias explícitas.

El patrón debe seguir el existente:

```python
@unittest.skipUnless(HAS_TORCH, ...)
class TestMoEDispatch(unittest.TestCase):
    ...
```

De esa forma:

```bash
pytest
```

continúa funcionando en el **R1 CPU-only environment**.

CUDA solamente será necesario para:

```text
GPU integration tests
GPU benchmarks
training benchmarks
```

y esos pueden estar separados/condicionados.

---

# 3. Importante: `reference.py` no debe ser simplemente borrado

Yo haría una pequeña modificación conceptual.

Actualmente tenemos el MoE existente.

Lo convertimos en:

```text
training/moe/reference.py
```

y lo dejamos **deliberadamente sencillo**.

Su función pasa a ser:

> implementación de referencia utilizada para validar cualquier optimización posterior.

Por ejemplo:

```python
reference_output = reference_moe(x, ...)
optimized_output = optimized_moe(x, ...)
```

Entonces podemos hacer:

```text
                  Reference
                     │
                     │
              ┌──────┴──────┐
              │             │
              ▼             ▼
           logits         grads
              │             │
              └──────┬──────┘
                     │
                  compare
                     │
                     ▼
                 PASS/FAIL
```

Esto será extremadamente útil cuando empecemos a tocar el dispatcher.

---

# 4. La regla de oro del dispatcher

Antes de optimizar:

```text
Correctness
```

Después:

```text
Performance
```

Nunca al revés.

El nuevo dispatcher tiene que demostrar:

```text
reference ≈ optimized
```

para:

* forward;
* loss;
* backward;
* gradients.

Solo entonces medimos:

* latency;
* memory;
* throughput;
* dispatch overhead.

---

# 5. Y haría una tercera corrección al plan

No pondría `attention.py` demasiado pronto.

El orden real debería ser:

```text
GPU environment
        ↓
Memory profiling
        ↓
BF16
        ↓
Optimizer abstraction
        ↓
Adam8bit
        ↓
Checkpointing
        ↓
M2 benchmark
        ↓
MoE dispatcher
        ↓
M2 validation
        ↓
M3/M4/M5
```

**Después**, si el profiling demuestra que la atención es un cuello de botella:

```text
SDPA
```

Esto evita optimizar componentes sin evidencia.

La regla debería ser:

> **profile → identify bottleneck → optimize → re-profile.**

---

# 6. Primer milestone

Por tanto, el primer milestone oficial de implementación sería:

## ASIR-TR-0 — GPU Training Infrastructure

Debe entregar:

```text
requirements-gpu.txt
Dockerfile.gpu

--device
--precision

training/profiling/memory.py
training/profiling/dispatch_timing.py
```

Y producir un experimento reproducible:

```bash
python training/train.py \
    --model M2 \
    --device cuda \
    --batch_size 1 \
    --max_steps 10
```

con información suficiente para contestar:

```text
¿Cuánto consume M2 realmente?

¿Dónde está el consumo?

¿Cuánto corresponde a:
    weights?
    gradients?
    optimizer?
    activations?
    allocator?
```

**Todavía sin modificar el MoE.**

---

# 7. Segundo milestone

## ASIR-TR-1 — Memory Efficient Training

```text
BF16
+
optimizer abstraction
+
Adam8bit
+
checkpointing
```

Objetivo:

```text
M2
32 experts
K=2
≈140M active
12 GB GPU
batch=1

→ 10 steps
→ no OOM
→ finite loss
→ finite gradients
```

Y aquí hay una condición importante:

**Si BF16 + Adam8bit + checkpointing todavía no caben en 12 GB, no empezamos a mutilar M2.**

El siguiente candidato sería:

```text
CPU optimizer offload
```

o una estrategia equivalente de memory management.

---

# 8. Tercer milestone

## ASIR-TR-2 — Sparse MoE Dispatcher

Aquí ocurre el cambio arquitectónico fuerte:

```text
O(K·E) Python control flow
              ↓
top-k indices
              ↓
grouped dispatch
              ↓
only selected experts
```

Con:

```text
reference.py
      ↓
correctness oracle

dispatcher.py
      ↓
optimized implementation
```

Y entonces:

```text
M2
 ↓
M3
 ↓
M4
 ↓
M5
```

---

# 9. Cuarto milestone

## ASIR-TR-3 — Expert Telemetry

Aquí aprovechamos el trabajo R1 existente.

No sustituimos:

```text
analysis/moe_profiler.py
```

Lo complementamos.

Tendríamos:

```text
R1 profiler
───────────
¿Qué expertos selecciona el router?

ASIR telemetry
──────────────
¿Cómo se comportan temporalmente esos expertos?
```

Y añadimos:

```text
frequency
reuse distance
transition probability
co-occurrence
locality
load
```

Esto alimentará directamente el runtime.

---

# 10. Quinto milestone

## ASIR-RT-0 — CPU Reference Runtime

Antes de INT4.

Antes de C++.

Antes de K3.

Simplemente:

```text
safetensors
     ↓
ExpertStore
     ↓
load expert
     ↓
execute
     ↓
unload
```

Esto nos proporciona el primer runtime funcional.

---

# 11. Sexto milestone

## ASIR-RT-1 — Memory-Aware Runtime

```text
ExpertStore
     +
LRU
     +
memory budget
```

Ejemplo:

```bash
python -m runtime \
    --model M2 \
    --memory-limit 7
```

Y debe garantizar:

```text
Peak RAM <= 7 GB
```

---

# 12. Séptimo milestone

## ASIR-RT-2 — Adaptive Runtime

Aquí empiezan las ideas realmente interesantes:

```text
routing
   ↓
prediction
   ↓
prefetch
   ↓
cache
   ↓
paging
```

Y podremos comparar:

```text
Random cache
LRU
LFU
Routing-aware
Routing-aware + prefetch
```

Eso ya es investigación de ASIR propiamente dicha.

---

# 13. Octavo milestone

## ASIR-RT-3 — Quantized Runtime

```text
FP16
 ↓
INT8
 ↓
INT4
```

con el `analysis/quantization_eval.py` como evaluador.

Cada resultado debe tener:

```text
quality
RAM
I/O
cache
tok/s
TTFT
```

Nunca solamente:

```text
"INT4 ocupa menos"
```

---

# 14. Noveno milestone

## ASIR-RT-4 — 8 GB Challenge

Condición:

```text
RAM = 8 GB
GPU = none
```

Y nuestro target final:

```text
resident RAM < 8 GB
```

idealmente:

```text
≤ 7 GB
```

dejando margen para el SO.

Entonces podemos empezar a perseguir:

```text
40 tok/s
```

pero **no asumir que 40 tok/s es alcanzable desde el principio**.

Lo tratamos como benchmark target:

```text
5
10
20
30
40
```

y documentamos dónde está el techo.

---

# 15. Décimo milestone

## ASIR-K3

Solo cuando exista:

```text
M1–M5 validated
+
runtime validated
+
INT4 validated
+
8 GB benchmark
```

hacemos:

```text
Kimi K3
   ↓
architecture analysis
   ↓
expert extraction
   ↓
quantization
   ↓
ASIR loader
   ↓
cache/paging
   ↓
CPU
```

con el Go/No-Go que planteaste.

---

## En definitiva

Estoy de acuerdo con tus dos correcciones y las incorporaría como reglas permanentes del proyecto:

### Regla 1

**R1 CPU no se toca.**

```text
requirements.txt
Dockerfile
R1 protocol
existing analysis
existing statistical tests
```

permanecen reproducibles.

### Regla 2

**Todo optimizador tiene un reference implementation.**

Especialmente:

```text
MoE
quantization
cache
runtime
```

### Regla 3

**M3/M4/M5 están bloqueados hasta tener dispatcher optimizado.**

### Regla 4

**No diseñamos un formato `.asir` hasta demostrar que `safetensors` no es suficiente.**

### Regla 5

**No optimizamos sin profiling.**

Y con eso, el primer trabajo de código queda muy concreto:

```text
PR #1
────────────────────────────────────
feat(training): GPU development profile

+ requirements-gpu.txt
+ Dockerfile.gpu
+ device selection
+ GPU environment logging
+ memory profiler
+ dispatch timing profiler
+ CPU-safe tests
────────────────────────────────────
NO cambio de arquitectura M1-M5
NO cambio del protocolo R1
NO dispatcher todavía
NO quantization todavía
```

Después de ese PR ya tendremos algo que ahora mismo nos falta: **datos duros del consumo de M2 en el entorno GPU reproducible**. Con esos datos podremos decidir exactamente qué combinación de BF16, optimizer states, checkpointing y eventualmente offload necesitamos, en lugar de introducir optimizaciones a ciegas.
