# R1 — Adaptive Sparse Inference Runtime
## Research Specification v0.7 (congelado — contrato definitivo, inicio de implementación)

**Cambios respecto a v0.6 (cierre de cuatro puntos de auditoría antes de implementar):**
1. §2: la frase conceptual "N_eff aumenta de forma significativa" contradecía el criterio formal `ΔN_eff > 0` (sin test estadístico definido en §11). Se fija `ΔN_eff > 0` como condición mecanística/descriptiva (chequeo de signo, sin IC95%/Wilcoxon propio); la significancia estadística se reserva exclusivamente para `ΔQ`.
2. §2/§4: `H4-primary` (v0.6) se probaba indistintamente sobre `M2→M3→M4→M5` (EXP-B) o `C1→C2→C3→C4` (EXP-C), pero en EXP-B `K` y `E` co-varían con `N_total` — no aísla el efecto de `N_total` por sí solo. Se separa en **H4-primary-B** (EXP-B: escalado de capacidad latente bajo `A` constante, con `K/E` co-variando — efecto conjunto) y **H4-primary-C** (EXP-C: efecto aislado de `N_total`, único experimento con `K`/`E` fijos que permite esa atribución causal).
3. §11: se congela la distinción entre **contrastes confirmatorios** (pairwise, `M2→M3`, `M3→M4`, `M4→M5`, `C1→C2`, `C2→C3`, `C3→C4`, cada uno con protocolo estadístico completo de 10 seeds) y **tendencia descriptiva** (`N_total` vs `Q`/`N_eff`/`η_cap`, exploratoria, nunca citable como evidencia por sí sola — evita un grado de libertad estadístico no previsto).
4. §4: la nota "arquitectura comparable salvo routing" para `Dense-A` se convierte en un checklist concreto de qué debe compartir (tokenizer, embedding, hidden size, capas, activation, normalization, attention, positional encoding, datos y presupuesto de entrenamiento, optimizer, LR schedule, seed, eval set) y en qué debe diferir exclusivamente (estructura densa vs. backbone+expertos enrutados).

**Cambios heredados de v0.6 (respecto a v0.5):**
1. §2: `ΔA ≈ 0` carecía de tolerancia operacional. Se congela `ΔA_rel = |A_model−A_baseline|/A_baseline ≤ 1%` como tolerancia general; para M1–M5/EXP-B/EXP-C, donde `A=140M` es exactamente construible, se exige igualdad exacta `A_model = A_baseline`, sin `≈`. El 1% queda reservado a casos donde `A` no es exactamente construible (p. ej. `Dense-A` si el backbone denso no admite 140M nativamente).
2. §2/§4: H4 mezclaba dos preguntas experimentales distintas — escalado de `N_total` dentro de la familia MoE (comparación entre configuraciones) y comparación MoE vs `Dense-A` (comparación contra baseline). Se separan formalmente en **H4-primary** (M2→M3→M4→M5 o C1→C2→C3→C4, A constante, nunca contra baseline denso) y **H4-mechanistic** (MoE(A=140M) vs Dense-A, contraste único por tamaño). Una mejora frente a `Dense-A` nunca se cita como evidencia de `H4-primary`, y viceversa.

**Cambios heredados de v0.5 (respecto a v0.4):**
1. §2: la "evidencia de H4" definida como `ΔN_eff>0 ∧ ΔQ>0` no era coherente con el umbral de relevancia de §11 (`|ΔQ|≥2pp`, IC95%, Wilcoxon). Se introducen tres niveles: evidencia de capacidad, evidencia nominal de H4, evidencia estadística y prácticamente relevante de H4 — esta última es la única citable como conclusión.
2. §6: el umbral de Specialization Density ("definido por fase, documentado") quedaba abierto a ajuste post-hoc. Se congela `τ_spec = 0.50` como métrica primaria, con `τ = 0.30/0.50/0.70` como análisis de sensibilidad obligatorio.
3. §6: la afirmación "Hungarian se reserva para M0/M1" es matemáticamente incorrecta (con 3 roles, ni N_total=1 ni N_total=8 permiten biyección). Hungarian se retira de la ruta principal; queda como método auxiliar opcional solo cuando `N_total == N_roles` exactamente.
4. §6: se define explícitamente el paso de distribución de expertos a distribución de roles (`Q_role`), el tratamiento de la masa de expertos sin asignar (renormalización excluyendo `∅`, con `unassigned routing mass` reportada aparte), que antes no estaba especificado.
5. §9: se aclara que `η_cap` es un ratio de capacidad efectiva nominalizada, no una fracción de expertos con tráfico no-nulo — evita una lectura incorrecta ("25% de expertos utilizados").
6. §9: se congela explícitamente que `N_eff` se calcula sobre conteos de routing agregados de todo el dataset de evaluación, nunca como promedio de `N_eff` por batch/muestra (evita el sesgo de Jensen, `E[N_eff(p)] ≠ N_eff(E[p])`).
7. Confirmado sin cambios: diseño de EXP-C (§3/§4), ya correcto desde v0.4.

**Cambios heredados de v0.4 (respecto a v0.3):**
1. §3: la fila de EXP-C decía "N_total ↑, expert size ↓", contradiciendo la tabla real de §4 donde `E` es constante (50M) en C1–C4. Corregido a "N_total ↑, expert size constante".
2. §2/§9: `ΔN_eff > 0` se separa explícitamente en **evidencia de capacidad** (condición necesaria pero no suficiente) vs **evidencia de H4** (que exige además `ΔQ > 0` bajo `ΔA ≈ 0`), resolviendo la contradicción entre la definición de H4 en §2 (que ya exigía ambas cosas) y la regla de invalidación de §9 (que solo pedía `ΔN_eff > 0`). Se añade tabla de interpretación 2×2.
3. §15/§16: referencias residuales a "Hungarian matching" como mecanismo general de alineación experto↔rol, ya reformulado como many-to-one desde §6. Corregidas para reflejar `alignment.py` (many-to-one) con `hungarian.py` como caso particular (solo M0/M1).
4. Se añade §17: contrato de fases final (R1.0–R1.9), con B0 (Fareed) declarado explícitamente fuera de la secuencia.

**Cambios heredados de v0.3 (respecto a v0.2):** condición de validez de H4 relajada a `ΔN_eff > 0` + `η_cap = N_eff/N_total`; Hungarian matching del Oracle reformulado como many-to-one (`f: E → R ∪ {∅}`) con `Specialization Density`; M0 declarado baseline denso explícito, con `Dense-A` añadido como baseline adicional obligatorio.

**Cambios heredados de v0.2 (respecto a v0.1):** corrección del Routing Oracle (simetría de permutación), definición exacta del presupuesto de parámetros activos, separación training stack / inference runtime, protocolo estadístico pre-registrado, `N_eff`, reformulación operacional de H4, EXP-C.

**Baseline externo:** B0 = [`kimi-k3-in-c`](https://github.com/FareedKhan-dev/kimi-k3-in-c) (Fareed Khan).

---

## 1. Research Question

(sin cambios respecto a v0.1)

> ¿Puede un sistema de inferencia sparse con routing adaptativo y memoria/cache externa mantener una calidad determinada usando menos cómputo activo y memoria residente que un modelo denso equivalente, y puede su eficiencia mejorar cuando el presupuesto de recursos es restringido?

---

## 2. Hipótesis

H1, H2 (H2a/H2b), H3 sin cambios respecto a v0.1.

**H4 — reformulada operacionalmente, condición de validez relajada (v0.3), tres niveles de evidencia (v0.4/v0.5), separación primary/mechanistic + tolerancia de A (v0.6), separación B/C + fijación mecanística de N_eff (v0.7)**

```
H4:  Q(N_total ↑, A_active ≈ const) ↑
     acompañado de un aumento de N_effective (condición mecanística, no un segundo test de significancia — ver corrección v0.7 abajo)
```

**Corrección v0.6 — tolerancia operacional de `ΔA ≈ 0`:**

Hasta v0.5, `ΔA ≈ 0` no tenía una tolerancia numérica definida, lo cual es un problema porque el criterio estadístico de §11 es binario mientras que `A` puede variar por redondeos, tratamiento de embeddings, parámetros del router, etc. Queda congelado:

- **Caso general (tolerancia por defecto):**
```
ΔA_rel = |A_model − A_baseline| / A_baseline
A ≈ const  ⇔  ΔA_rel ≤ 1%
```
- **Caso M1–M5, EXP-B y EXP-C:** dado que `A = B + K·E` es exactamente construible por diseño (`A = 140M` en todas las configuraciones de la tabla de §4), aquí se exige igualdad exacta, sin el `≈`:
```
A_model = A_baseline   (140M = 140M, sin tolerancia)
```
El `1%` de `ΔA_rel` queda reservado exclusivamente para comparaciones donde `A` no puede construirse a igualdad exacta por razones de implementación (p. ej. `Dense-A` frente a la familia MoE, si el backbone denso no admite exactamente `140M` de forma nativa). Toda tabla de resultados debe reportar `A_model`, `A_baseline` y `ΔA_rel` explícitamente — nunca solo la palabra "constante".

**Corrección v0.6 — H4 se separa en dos hipótesis distintas, no una:**

v0.5 usaba `Dense-A` como parte del mismo argumento que la comparación de escalado `N_total ↑` dentro de la familia MoE, pero son dos preguntas experimentales distintas que no deben mezclarse:

```
H4-primary:      dentro de la familia MoE, con A constante (§4), N_total ↑ ⇒ Q ↑, junto con N_eff ↑.

H4-mechanistic:  MoE(A=140M) supera a Dense-A (mismo presupuesto activo A, arquitectura
                  comparable salvo la existencia de routing/sparsity).
```

`H4-primary` responde "¿escalar N_total dentro de la familia MoE mejora Q?" — la comparación es **entre configuraciones MoE**, nunca contra un baseline denso. `H4-mechanistic` responde una pregunta distinta: "¿la arquitectura sparse en sí misma aporta algo frente a un denso con el mismo presupuesto?" — esta comparación es **contra baseline**, y una victoria aquí no implica nada sobre si `N_total` importa dentro de la familia MoE. Ambas hipótesis se prueban y reportan por separado; una mejora frente a `Dense-A` **nunca** se cita como evidencia de `H4-primary`, y viceversa.

**Corrección v0.7 — H4-primary en v0.6 pedía más de lo que EXP-B puede sostener; se separa en H4-primary-B y H4-primary-C:**

v0.6 formulaba `H4-primary` como una sola afirmación probada indistintamente sobre `M2→M3→M4→M5` (EXP-B) o `C1→C2→C3→C4` (EXP-C). Esto es un problema porque en EXP-B, `K` y `E` cambian junto con `N_total`:

| Modelo | N_total | K | E |
|---|---:|---:|---:|
| M2 | 32  | 2 | 50M |
| M3 | 128 | 4 | 25M |
| M4 | 512 | 8 | 12.5M |
| M5 | 896 | 8 | 12.5M |

Entre `M2→M3`, por ejemplo, `N_total ↑`, `K ↑` y `E ↓` simultáneamente (con `A` constante). No se puede atribuir una mejora de `Q` exclusivamente a `N_total` en esta secuencia — se está midiendo el efecto conjunto de escalar la capacidad latente bajo presupuesto activo constante, con granularidad del expert pool y `top-k` co-variando según la tabla de §4. Eso es una pregunta legítima, pero distinta de "¿el número de expertos disponibles, por sí mismo, mejora `Q`?" — esa pregunta solo la aísla `EXP-C`, donde `K=2` y `E=50M` permanecen fijos mientras `N_total` varía (§3, §4).

Se corrige por tanto:

```
H4-primary-B (capacidad escalable, EXP-B):
    Dentro de la familia EXP-B (M2→M3→M4→M5), al aumentar N_total bajo A constante
    (con K/E co-variando según la tabla de §4), Q aumenta junto con N_eff.
    → No se atribuye la mejora a N_total en aislamiento; se atribuye a "escalar
      capacidad latente bajo presupuesto activo constante" como fenómeno conjunto.

H4-primary-C (efecto aislado de disponibilidad, EXP-C):
    Dentro de la familia EXP-C (C1→C2→C3→C4), con K y E fijos, al aumentar N_total,
    Q aumenta junto con N_eff.
    → Esta es la única secuencia que permite atribuir el efecto a N_total en
      aislamiento, porque K y E no cambian.
```

`H4-primary-B` y `H4-primary-C` se prueban, reportan y citan por separado. Un resultado positivo en `H4-primary-B` sin un resultado correspondiente en `H4-primary-C` es evidencia de que "más capacidad latente ayuda", pero no de que "más expertos disponibles per se ayuden" — esa distinción causal solo la resuelve `EXP-C`. `EXP-C` es, por diseño, el experimento que permite hablar de un efecto de `N_total` en sentido estricto; `EXP-B` demuestra escalabilidad de capacidad, pero bajo un cambio conjunto de tres variables.

**Corrección v0.7 — "ΔN_eff aumenta significativamente" no está operacionalizado; se fija como condición mecanística sin test de significancia propio:**

El enunciado original de H4 (heredado de v0.1, arriba) decía que `N_effective` debía "aumentar de forma significativa", pero §11 nunca definió un test estadístico para `ΔN_eff` — el criterio formal usado en todas las versiones fue simplemente `ΔN_eff > 0`. Esta contradicción (texto conceptual pidiendo significancia, criterio formal sin test) queda resuelta así:

- `ΔN_eff > 0` es una condición **mecanística/descriptiva**, no un test de hipótesis. No se le exige IC95%, Wilcoxon, ni un umbral de magnitud — es un chequeo de signo.
- La significancia estadística y la relevancia práctica (`|ΔQ|≥2pp`, IC95%, Wilcoxon) se evalúan **exclusivamente sobre `ΔQ`**, nunca sobre `ΔN_eff`.
- Esto es intencional y consistente con la filosofía del contrato: `N_eff` es un mecanismo explicativo, `Q` es el resultado principal (outcome) sobre el que se prueban hipótesis. Introducir un segundo test de significancia sobre `ΔN_eff` crearía un grado de libertad estadístico no previsto en el protocolo pre-registrado de §11.

**Por qué `ΔN_eff > 0` sigue siendo insuficiente por sí solo como evidencia de H4-primary-B/C:**

`ΔN_eff > 0` es la condición mecanística mínima para hablar de un aumento de *capacidad efectiva*. No es, por sí sola, evidencia de H4-primary-B ni de H4-primary-C. Considérese:

```
32  → Q=82%, N_eff=28
128 → Q=82%, N_eff=31
512 → Q=82%, N_eff=32
```

Aquí `ΔN_eff > 0` en cada paso, pero `ΔQ ≈ 0` — la capacidad efectiva aumentó sin traducirse en calidad. Citar esto como "evidencia de H4-primary-B/C" sería incorrecto: es evidencia de aumento de capacidad, nada más.

**Tres niveles de evidencia, no dos (v0.5), aplicados a H4-primary-B y H4-primary-C por separado (v0.7):**

Bajo la definición de v0.4, un resultado como:
```
N_eff: 30 → 31
Q:     82.00% → 82.01%
```
técnicamente cumplía `ΔN_eff > 0 ∧ ΔQ > 0` y podía citarse como "H4 respaldada" — lo cual es absurdo dado que `ΔQ` es indistinguible de ruido. Se distinguen por tanto **tres** afirmaciones, no dos, que nunca deben mezclarse en el reporte de resultados (aplicadas independientemente a `H4-primary-B` sobre EXP-B, y a `H4-primary-C` sobre EXP-C):

| Nivel | Condición |
|---|---|
| **Evidencia de capacidad** | `ΔN_eff > 0` (chequeo de signo, sin test estadístico — ver corrección v0.7 arriba) |
| **Evidencia nominal de H4-primary-{B,C}** | `ΔN_eff > 0 ∧ ΔQ > 0 ∧ A_model = A_baseline` (igualdad exacta, ver arriba) |
| **Evidencia estadística y prácticamente relevante de H4-primary-{B,C}** | `ΔN_eff > 0 ∧ ΔQ ≥ 2pp ∧ IC95%(ΔQ) excluye 0 ∧ p_Wilcoxon < 0.05 ∧ A_model = A_baseline` |

Solo el tercer nivel es citable como conclusión del proyecto (consistente con el criterio de relevancia de §11). El primer y segundo nivel son hallazgos exploratorios o diagnósticos, nunca conclusiones. `H4-mechanistic` se cita como respaldada únicamente si el contraste MoE(A) vs Dense-A cumple el mismo tercer nivel (con `ΔA_rel ≤ 1%` si `Dense-A` no admite igualdad exacta).

Y se reporta siempre junto a los tres:
```
η_cap = N_eff / N_total
```
para que `N_eff` no se convierta accidentalmente en el objetivo de la investigación en sí mismo (el objetivo es `Q`; `N_eff` es un mecanismo explicativo).

**Tabla de interpretación (aplica a H4-primary-B y H4-primary-C por separado; obligatoria en todo reporte de escalado):**

| ΔN_eff | ΔQ | Interpretación |
|---|---|---|
| > 0 | ≥ 2pp, significativo (IC95%, Wilcoxon) | Evidencia de H4-primary-{B/C} |
| > 0 | > 0 pero < 2pp o no significativo | Compatible con H4-primary-{B/C} (no citable como conclusión) |
| > 0 | ≈ 0 | Capacidad efectiva aumentó sin beneficio de calidad |
| ≈ 0 | > 0 | Mejora por otro mecanismo; no atribuir a capacidad latente |
| < 0 | < 0 | Collapse / regresión |

El patrón de `η_cap` (constante, decreciente, creciente) y de `Specialization Density` (§6) se documenta como parte de la conclusión en todos los casos, no se usa como criterio binario adicional de invalidación.

Ejemplo de lectura correcta bajo esta tabla (secuencia EXP-B, M2→M3→M4→M5 — se lee bajo `H4-primary-B`; la misma tabla se aplica de forma independiente a C1→C2→C3→C4 bajo `H4-primary-C`):
```
32  experts → Q=82%, N_eff=28,  η_cap=0.88
128 experts → Q=86%, N_eff=31,  η_cap=0.24
512 experts → Q=86%, N_eff=32,  η_cap=0.06
```
`ΔQ > 0` y `ΔN_eff > 0` entre 32→128 → celda "Evidencia de H4-primary-B" (sujeto a que también cumpla el umbral estadístico de §11). Entre 128→512, `ΔQ ≈ 0` y `ΔN_eff ≈ 0` → celda "collapse/saturación", concretamente saturación de capacidad efectiva alrededor de ~31–32, con rendimientos decrecientes ya visibles desde 128.

---

## 3. Variables experimentales

| Experimento | Qué varía | Qué se mantiene fijo | Qué aísla |
|---|---|---|---|
| EXP-A | N_total ↑, K ↑ (tamaño experto fijo) | expert size = 20M | trade-off capacidad/compute |
| EXP-B | N_total ↑, expert size ↓ | A (active params) fijo, K ↑ | capacidad latente bajo A constante |
| EXP-C | N_total ↑, expert size constante | A fijo, **K fijo** | ¿la ganancia viene de tener más expertos disponibles, o de repartir A entre más expertos activos (K↑)? |

*(Corrección v0.4: la fila de EXP-C decía antes "expert size ↓", lo cual contradecía la tabla de §4 — ahí `E=50M` es constante en las cuatro configuraciones C1–C4. El tamaño de cada experto individual no cambia; lo que cambia es cuántos expertos hay disponibles en total mientras `K` permanece fijo en 2. Este es, de hecho, el diseño correcto para aislar la pregunta de EXP-C: "¿qué ocurre cuando hay más expertos disponibles pero solo se activan dos?")*

EXP-C es necesario porque en la matriz M0–M5 de v0.1, `top-k` también cambia junto con `N_total` — sin EXP-C no se puede distinguir "más expertos disponibles" de "más expertos activos simultáneamente".

Resto de variables independientes/dependientes/controladas: sin cambios respecto a v0.1 §3.

---

## 4. Arquitectura M0–M5 y definición exacta del presupuesto activo

### Definición formal de A

```
A = B + K·E
```
donde:
- `A` = presupuesto total de parámetros activos por token (constante entre M0–M5).
- `B` = parámetros activos del backbone compartido.
- `K` = número de expertos activos (top-k).
- `E` = parámetros por experto.

**Regla:** cualquier tabla de configuraciones M0–M5 debe reportar `B`, `K`, `E` por separado, no solo el total `A`.

### Tabla M0–M5 (EXP-B, top-k variable)

| Modelo | N_total | K (top-k) | B (backbone activo) | E (por experto) | A = B+K·E |
|---|---:|---:|---:|---:|---:|
| M0 | 1   | 1 | — (single expert = todo el modelo) | 140M | 140M |
| M1 | 8   | 2 | 40M | 50M  | 140M |
| M2 | 32  | 2 | 40M | 50M  | 140M |
| M3 | 128 | 4 | 40M | 25M  | 140M |
| M4 | 512 | 8 | 40M | 12.5M | 140M |
| M5 | 896 | 8 | 40M | 12.5M | 140M |

### M0 es baseline denso, no miembro de la familia MoE

```
M0            = Dense baseline (arquitectura densa estándar, ~140M parámetros totales = activos)
M1 – M5       = Sparse MoE family (backbone compartido B=40M + expertos)
```

**Requisito:** el set de baselines para evaluar H4 (`H4-primary-B`, `H4-primary-C` y `H4-mechanistic`, §2) debe incluir, como mínimo:
```
Dense-140M      (= M0, por continuidad con v0.1/v0.2)
Dense-A         (modelo denso con el mismo presupuesto A que M1–M5)
MoE-32   (M2)
MoE-128  (M3)
MoE-512  (M4)
MoE-896  (M5)
```
`Dense-A` es el baseline que aísla el efecto de sparsity/routing: superar a `Dense-A` (mismo `A`, misma familia arquitectónica salvo el routing) es el contraste correcto para **H4-mechanistic** (§2) — evidencia sobre si la arquitectura sparse aporta algo frente a un denso equivalente, no sobre si escalar `N_total` dentro de la familia MoE mejora `Q` (eso es `H4-primary-B`/`H4-primary-C`, que se prueban entre configuraciones MoE, nunca contra `Dense-A` ni contra M0). Superar solamente a M0 no aísla ninguna de las dos cosas con limpieza, dado que M0 no comparte backbone ni presupuesto `A` con la familia MoE (ver más arriba).

**Corrección v0.7 — checklist concreto de construcción de `Dense-A` (antes, "arquitectura comparable salvo routing" no era una especificación auditable):**

Para que la comparación `MoE(A) vs Dense-A` de `H4-mechanistic` sea limpia, `Dense-A` debe compartir con la familia MoE, en la medida físicamente posible:
```
tokenizer, embedding, hidden size, número de capas, activation,
normalization, attention, positional encoding,
training data, training budget, optimizer, learning-rate schedule,
seed, evaluation set
```
y diferir **únicamente** en la estructura del bloque de cómputo por token:
```
Dense-A:  B_total ≈ 140M (todo el modelo es un único bloque denso)
MoE:      B_shared + K·E = 140M (backbone compartido + expertos enrutados)
```
Cualquier divergencia adicional entre `Dense-A` y la familia MoE (más allá de esta estructura densa-vs-routed) debe documentarse explícitamente en el checkpoint de `Dense-A`, ya que reduce la limpieza con la que una victoria de MoE puede atribuirse a sparsity/routing en vez de a diferencias arquitectónicas incidentales.

### Tabla EXP-C (N_total ↑, K fijo, expert size constante)

| Config | N_total | K | E | A |
|---|---:|---:|---:|---:|
| C1 | 32  | 2 | 50M | 140M |
| C2 | 128 | 2 | 50M | 140M |
| C3 | 512 | 2 | 50M | 140M |
| C4 | 896 | 2 | 50M | 140M |

`K=2` y `E=50M` se mantienen constantes en las cuatro configuraciones; solo `N_total` cambia (y por tanto la fracción `K/N_total` cae). Esto aísla el efecto de "expertos disponibles pero no simultáneamente activos" de forma limpia.

---

## 5. Generador de tareas sintéticas

Sin cambios respecto a v0.1.

---

## 6. Routing Oracle

### Oracle por rol/subdominio
```
Oracle(x) = {
  arithmetic: 0.40,
  logic:      0.35,
  language:   0.25
}
```

### Alineación experto↔rol vía matching many-to-one

```
f: E → R ∪ {∅}
```
Múltiples expertos pueden mapear al mismo rol; expertos sin correlación significativa quedan sin asignar (`∅`).

**Corrección v0.5 — umbral de especialización congelado:**

v0.4 dejaba el umbral "definido por fase, documentado", lo cual reabre exactamente el problema que el resto del contrato busca evitar: el criterio podría fijarse después de observar los resultados (con `τ=0.30` una config puede parecer altamente especializada y con `τ=0.50` no, o viceversa). Se congela:
```
τ_spec = 0.50   (métrica primaria)
```
```
f(e) = argmax_r corr(e, r)     si max_r corr(e, r) ≥ τ_spec
       ∅                        en otro caso
```
```
Specialization Density = |{ e : max_r corr(e, r) ≥ 0.50 }| / N_total
```
Se reporta obligatoriamente un análisis de sensibilidad secundario con `τ = 0.30, 0.50, 0.70`, pero `τ_spec = 0.50` es la métrica primaria citable en conclusiones — no puede sustituirse retrospectivamente por el valor que favorezca al modelo.

**Corrección v0.5 — Hungarian retirado de la ruta principal:**

v0.4 afirmaba que el algoritmo húngaro (biyectivo) se reservaba para `M0/M1` "donde `N_total` es comparable al número de roles". Esto es matemáticamente incorrecto: con 3 roles, ni `M0` (`N_total=1`) ni `M1` (`N_total=8`) admiten una asignación biyectiva 1:1 contra 3 roles — ningún caso de la tabla M0–M5 cumple `N_total == N_roles`. Se corrige:

- El algoritmo principal, único usado en la ruta estándar para toda M0–M5 y EXP-A/B/C, es el `argmax` con umbral de arriba — trivial, auditable, y ya compatible con `f: E → R ∪ {∅}`:
```
for expert in experts:
    role = argmax(correlation[expert, :])
    if correlation[expert, role] >= tau_spec:
        assignment[expert] = role
    else:
        assignment[expert] = None
```
- Hungarian (biyectivo) queda como **método auxiliar opcional**, aplicable únicamente cuando exista un subproblema exactamente cuadrado, es decir `N_total == N_roles` de forma literal — no como caso general de ningún modelo de la matriz actual. Si en el futuro se añade una configuración con `N_total == N_roles`, puede usarse como comparación diagnóstica, nunca como el método primario de alineación.

```
Specialization Density = |{ e : max_r corr(e, r) ≥ τ_spec }| / N_total
```
Se reporta junto a `N_eff` y `η_cap` en toda tabla de escalado.

### De distribución de expertos a distribución de roles — Q_role_aligned (nuevo en v0.5)

v0.4 no especificaba cómo pasar de la distribución del router sobre `N_total` expertos a una distribución sobre los roles del Oracle (3 en el ejemplo estándar). Queda definido:

**Paso 1 — agregación por rol:**
```
Q_role(r | x) = Σ_{e : f(e)=r} Q(e | x)
Q_role(∅ | x) = Σ_{e : f(e)=∅} Q(e | x)
```

**Paso 2 — el Oracle no tiene categoría `∅`** (`P_oracle = [0.40, 0.35, 0.25]` suma 1 sobre los roles reales), así que la masa de expertos sin asignar no puede compararse directamente contra `P_oracle`. Para el KL primario, se renormaliza excluyendo `∅`:
```
Q_role'(r | x) = Q_role(r | x) / Σ_{e : f(e)≠∅} Q(e | x)
```
y el KL primario se calcula como:
```
D_KL( P_oracle_role || Q_role' )
```

**Paso 3 — la masa excluida se reporta aparte, nunca se descarta silenciosamente:**
```
unassigned routing mass = Σ_{e : f(e)=∅} Q(e | x)     (promedio sobre el eval set)
```
Esto evita esconder la sub-especialización dentro de un KL "limpio" — un modelo con alto `unassigned routing mass` puede tener un KL primario bajo y aun así estar mal especializado; ambas cifras se reportan siempre juntas.

### Consecuencias
- Alineación recalculada **por seed** y reportada junto con `Specialization Density` y `unassigned routing mass`.
- Routing accuracy top-k se recalcula sobre la alineación many-to-one, no sobre IDs crudos.
- Expertos sin asignación se documentan como "sub-especialización" y se conectan con `N_effective`, `η_cap` y `Specialization Density` en una sola tabla de diagnóstico de capacidad.

---

## 7. Definición de calidad

Sin cambios respecto a v0.1 (`Q = exact-match accuracy`, held-out fijo, perplexity solo como diagnóstico secundario).

---

## 8. Protocolo de presupuesto de entrenamiento

Sin cambios respecto a v0.1 (`T-fixed` y `Expert-exposure-fixed`, nunca mezclados), aplicado también a EXP-C.

---

## 9. Control de expert collapse — capacidad nominal vs efectiva

```
N_effective = 1 / Σ_i p_i²
```

**Corrección v0.5 — método de agregación congelado:**

`p_i` se calcula **una sola vez**, sobre los conteos de routing agregados de todo el dataset de evaluación:
```
p_i = (tokens enrutados al experto i en todo el eval set) / (total de asignaciones de expertos en todo el eval set)
N_eff = 1 / Σ_i p_i²
```
Queda explícitamente prohibido calcular `N_eff` por batch o por muestra y luego promediar (`mean(N_eff(batch))`), porque `N_eff` es una función no lineal de `p` y `E[N_eff(p)] ≠ N_eff(E[p])` — promediar por batch produce un número sesgado y no reproducible frente al método agregado. El flujo obligatorio es: agregar todos los conteos de routing del eval set → normalizar → calcular `N_eff` una vez.

- `N_eff` se reporta en **toda** tabla de resultados de escalado, junto a `N_total`, utilización por experto, entropía de routing y coeficiente de variación de carga.
- **Regla v0.7 (refina la regla de v0.6):** se reportan siempre `ΔN_eff`, `η_cap = N_eff/N_total`, `Specialization Density` y `unassigned routing mass` juntos. La cita de un resultado como evidencia de capacidad, evidencia nominal de H4-primary-{B,C}, o evidencia estadísticamente relevante de H4-primary-{B,C} sigue los tres niveles definidos en §2 — nunca `ΔN_eff > 0` por sí solo, y nunca con un test de significancia aplicado a `ΔN_eff` (ese test se reserva para `ΔQ`, ver §2). Esta regla aplica a comparaciones **entre configuraciones MoE** (`H4-primary-B` sobre EXP-B, `H4-primary-C` sobre EXP-C, reportadas por separado); el contraste contra `Dense-A` (`H4-mechanistic`) se reporta aparte y no se mezcla en la misma tabla de escalado.
- **Aclaración sobre `η_cap` (v0.5):** `η_cap = N_eff / N_total` es un *ratio de capacidad efectiva nominalizada*, no la fracción de expertos que reciben tráfico no-nulo. Por ejemplo, `N_total=512, N_eff=128 → η_cap=0.25` **no** significa "el 25% de los expertos fue utilizado"; significa que la distribución de utilización tiene una capacidad efectiva equivalente a 128 categorías uniformes frente a 512 disponibles. Esta distinción debe mantenerse en cualquier documentación o nombre de variable en código (`η_cap` o, alternativamente, `capacity_utilization_efficiency`, pero nunca `pct_experts_used` o similar).
- Comparación con/sin auxiliary load-balancing loss: sin cambios respecto a v0.1.

---

## 10. Métricas de compute / RAM / I/O — ECD

Sin cambios respecto a v0.1.

---

## 11. Protocolo estadístico — pre-registrado

### Comparación primaria: pareada por seed
```
Δ_s = Q_model,s − Q_baseline,s      para s = seed 1..N
```
Se reporta: media de Δ, mediana de Δ, IC 95%, Cohen's d pareado, y Wilcoxon signed-rank como test robusto principal.

### Criterio de relevancia (pre-registrado, no ajustable post-hoc)
1. IC 95% de Δ excluye 0.
2. p < 0.05 (Wilcoxon signed-rank pareado).
3. `|Δ| ≥ 2 puntos porcentuales de Q`.

### Escalón de seeds
- **Exploración**: 5 seeds mínimo.
- **Confirmación**: 10 seeds totales antes de declarar definitivo.
- Ningún resultado con 5 seeds se cita como conclusión final.

### Contrastes confirmatorios vs tendencia descriptiva (nuevo en v0.7)

`H4-primary-B` y `H4-primary-C` describen conceptualmente una secuencia (`M2→M3→M4→M5`, `C1→C2→C3→C4`), pero la comparación primaria de §11 es pareada por seed entre dos configuraciones, no una regresión sobre la secuencia completa. Queda congelado cuáles contrastes son confirmatorios (citables como evidencia de H4) y cuál lectura es solo descriptiva:

**Contrastes confirmatorios (cada uno con su propio protocolo completo — 10 seeds, `ΔQ`, IC95%, Wilcoxon, Cohen's d, `ΔN_eff`, `η_cap`, `Specialization Density`, `unassigned routing mass`):**
```
EXP-B (H4-primary-B):   M2→M3,  M3→M4,  M4→M5
EXP-C (H4-primary-C):   C1→C2,  C2→C3,  C3→C4
```
Cada contraste se evalúa independientemente contra los tres niveles de evidencia de §2. `H4-primary-B`/`H4-primary-C` se declaran respaldadas por la secuencia completa solo si la mayoría de los contrastes pairwise alcanzan al menos "evidencia estadística y prácticamente relevante"; un solo contraste positivo dentro de una secuencia donde el resto no significa no basta para declarar la hipótesis respaldada en general — se reporta como respaldo parcial, específico al tramo de escala donde ocurrió.

**Contraste descriptivo (nunca confirmatorio, no se le aplica el criterio de relevancia de §11):**
```
N_total vs Q
N_total vs N_eff
N_total vs η_cap
```
Estas relaciones se grafican y describen como contexto exploratorio de toda la secuencia (tendencia, forma de la curva, rendimientos decrecientes), pero no se tratan como un test de hipótesis adicional ni se citan por sí solas como evidencia de H4 — hacerlo introduciría un grado de libertad estadístico no previsto en el protocolo pre-registrado. Toda cita de "H4 respaldada" debe remitir a uno o más de los contrastes confirmatorios pairwise de arriba, nunca únicamente a la tendencia global.

---

## 12. Matriz de ablación

EXP-C como tercer eje de escalado junto a EXP-A y EXP-B. Resto sin cambios respecto a v0.1 §12.

---

## 13. Requisitos de reproducibilidad

La asignación experto↔rol (§6) se serializa junto con cada checkpoint y resultado. Resto sin cambios respecto a v0.1 §13.

---

## 14. B0 — Baseline Fareed

Sin cambios respecto a v0.1 §14.

---

## 15. Estructura de repositorio — separación training/runtime

```
adaptive-inference/
├── training/                   # Python / PyTorch — responde "¿la arquitectura funciona?"
│   ├── models/
│   ├── router/
│   │   ├── oracle.py            # roles + generación de distribución objetivo
│   │   └── alignment.py         # many-to-one expert↔role alignment
│   │       └── hungarian.py     # auxiliar opcional: solo si N_total == N_roles exactamente
│   ├── losses/                  # task loss, load-balancing aux loss, cost-aware loss
│   └── train.py
│
├── runtime/                     # C/C++ — responde "¿podemos ejecutarla eficientemente?"
│   ├── core/
│   │   ├── tensor/
│   │   ├── memory/
│   │   ├── scheduler/
│   │   └── runtime/
│   ├── routing/
│   │   └── policies.c           # ejecuta política ya entrenada, no la aprende
│   ├── experts/
│   │   ├── expert_store.c
│   │   ├── cache.c               # LRU-N
│   │   └── prefetch.c
│   └── memory/
│       ├── ram.c
│       ├── mmap.c
│       └── nvme.c
│
├── kernels/
│   ├── scalar/
│   ├── avx2/
│   ├── avx512/
│   └── asm/
│
├── tasks/
│   ├── generator.py
│   └── eval_set/
│
├── experiments/
│   ├── m0_m5/
│   ├── exp_a_expert_size/
│   ├── exp_b_active_compute/
│   ├── exp_c_topk_fixed/
│   ├── routing/
│   ├── cache/
│   └── kernels/
│
├── analysis/
│   ├── stats.py
│   ├── n_effective.py
│   └── ecd_sensitivity.py
│
├── checkpoints/
│   └── <config_hash>/
│
├── docs/
│   ├── R1_Research_Specification_v0.1.md
│   ├── R1_Research_Specification_v0.2.md
│   ├── R1_Research_Specification_v0.3.md
│   ├── R1_Research_Specification_v0.4.md
│   ├── R1_Research_Specification_v0.5.md
│   ├── R1_Research_Specification_v0.6.md
│   └── R1_Research_Specification_v0.7.md   # este documento
│
└── results/
    └── <fase>/<config_hash>/
```

*(Corrección v0.4: `alignment.py` ya no se describe como "Hungarian matching" — ese mecanismo pasa a `hungarian.py`. Corrección v0.5: `hungarian.py` no está reservado a M0/M1 — ver §6, esa afirmación era matemáticamente incorrecta — sino que es un método auxiliar opcional solo cuando `N_total == N_roles` exactamente, algo que ninguna configuración de la matriz actual cumple.)*

**Consecuencia metodológica explícita:** el training stack en Python resuelve H1–H4 a nivel de arquitectura. El runtime en C resuelve las preguntas de sistema (cache, NVMe, kernels, latencia real). Un checkpoint es el contrato de datos entre ambos stacks — el runtime nunca re-entrena, solo ejecuta.

---

## 16. Criterios de aceptación por fase

| Fase | Criterio de aceptación |
|---|---|
| R1.1 Expert scaling | EXP-C ejecutado junto a EXP-A/EXP-B; `N_effective`, `η_cap` y `Specialization Density` reportados en toda celda; ningún resultado de escalado se declara "confirmado" con menos de 10 seeds. |
| R1.2 Discrete routing | Alineación experto↔rol **many-to-one** documentada y serializada por seed antes de calcular routing accuracy. |

*(Corrección v0.4: la fila de R1.2 decía "Hungarian matching" — corregido para reflejar la reformulación many-to-one de §6.)*

---

## 17. Contrato de fases — secuencia final (nuevo en v0.4)

```
R1.0  Controlled MoE
R1.1  Expert Scaling
R1.2  Discrete Routing
R1.3  Probabilistic Routing
R1.4  Cost-aware Routing
R1.5  External Storage
R1.6  Cache + Prefetch
R1.7  Hardware-aware Routing
R1.8  SIMD / Assembly
R1.9  128–896 Experts
```

B0 (Fareed, `kimi-k3-in-c`) queda completamente separado de esta secuencia — es un baseline externo de referencia, no una fase del contrato.

**Nota sobre R1.8 (SIMD/Assembly):** esta fase se ordena deliberadamente al final. La pregunta de arquitectura (¿MoE mejora `Q`?) precede a la de routing (¿el routing sparse reduce compute activo?), que precede a la de memoria/cache (¿el almacenamiento externo preserva viabilidad?), que precede a la de kernels. Optimizar en Assembly antes de medir dónde está el cuello de botella real es prematuro: si el runtime gasta 78% del tiempo en I/O y 7% en compute, optimizar ese 7% con Assembly produce una mejora marginal; si tras R1.5–R1.7 el compute pasa a ser 75% del tiempo, SIMD/Assembly se vuelve mucho más relevante. R1.8 se ejecuta con esos datos ya en mano, no antes.

---

## Regla metodológica general

Ninguna técnica se declara superior sin control aislado del efecto atribuido, y ningún cambio de hipótesis, métrica o protocolo ocurre sin una nueva versión de este documento.

**v0.7 se considera el contrato definitivo de congelación.** A partir de aquí, implementación — sin más experimentos añadidos al diseño.