# R1 — Adaptive Sparse Inference Runtime

## Research Specification v0.12-final (CONGELADA — blinda RSE@k frente a denominador inestable, precisa la incertidumbre de RSE@k, ajusta nomenclatura de bootstrap condicionado, formaliza S_random, y congela algoritmo de near-duplicates; R1.0–R1.0.1 permanecen congelados y no se reabren)

**Estado de este documento:** esta es la versión **oficialmente congelada (v0.12-final)**. Tras incorporar las correcciones de v0.12 (terminología de `S_search-oracle` y regla de estabilidad de `RSE@k`) y las 5 precisiones finales de limpieza metodológica (estimador pre-registrado de incertidumbre de `RSE@k`, manejo de denominadores \(\le 0\) sin exclusión post-hoc, nomenclatura de bootstrap condicionado a la referencia congelada, formalización i.i.d. de \(S_{\text{random}}\), y congelamiento del algoritmo de detección de near-duplicates), este documento constituye el contrato metodológico inflexible que queda vigente para R1.3/R1.4 en adelante. No se modificarán las reglas durante la experimentación.

**Naturaleza de este cambio de versión:** v0.11 introdujo `RSE@k` con una definición matemáticamente correcta pero sin blindaje frente a un denominador `Q_oracle − Q_random` pequeño o estadísticamente indistinguible de cero, caso en el que la métrica se vuelve numéricamente inestable e interpretable en direcciones falsas. v0.12 corrige eso, y de paso corrige un uso de lenguaje impreciso heredado desde v0.10: `S_oracle` nunca fue el óptimo global, sino el mejor subset *encontrado por un procedimiento de búsqueda concreto*, y llamarlo "oracle" a secas invita a sobreinterpretar `ESP@k` como distancia al óptimo real.

---

## Cambios respecto a v0.11

**1. §3.2 — `S_oracle` se renombra a `S_search-oracle` (o "empirical oracle"), y se exige documentar el procedimiento de búsqueda para que `ESP@k` sea comparable entre versiones.**

En todo el documento, donde v0.10/v0.11 decían `S_oracle`, se lee ahora `S_search-oracle`. La definición de v0.11 §3.2 no cambia en su procedimiento (Dataset B, búsqueda dirigida por calidad, aproximada); cambia el nombre y se añade una obligación de reporte:

```
S_search-oracle = Search(B, k)
```

donde `Search` es el procedimiento concreto usado (greedy, beam search, u otro), nunca `argmax` exacto salvo que `k` sea lo bastante pequeño para permitir enumeración completa y documentada como tal. Para `ESP@k` y `RSE@k`, `Q` se interpreta siempre como métrica orientada a calidad donde mayor es mejor. Si la métrica primaria original es una loss `L`, la regla canónica es `Q = -L`; no se permiten transformaciones no afines de la métrica primaria para construir `Q`, salvo que la transformación haya sido pre-registrada antes del experimento y constituya explícitamente una métrica distinta. La métrica debe registrarse con nombre, transformación y unidad para evitar confundir, por ejemplo, negative loss con puntos porcentuales:

```
quality_metric:
  name:                    nombre reportado de Q (p. ej. negative_cross_entropy)
  source_metric:           métrica primaria original (p. ej. cross_entropy)
  transform:               [identity | negate | pre_registered_metric]
  direction:               maximize
  unit:                    unidad reportable de Q (p. ej. pp, nats, normalized_score)
```

Todo resultado que reporte `ESP@k` o `RSE@k` debe acompañarse de una ficha del procedimiento de búsqueda:

```
oracle_search:
  method:                  [greedy | beam | exhaustive | otro]
  objective:
    metric:                Q
    direction:             maximize
    dataset_split:         B
  budget:
    unit:                  candidate_subset_evaluations
    value:                 número de subsets candidatos evaluados
  beam_width:              (si aplica)
  greedy_iterations:       (si aplica)
  stopping_criterion:      criterio de parada usado
  search_determinism:
    deterministic:         [true | false]
    seeds:                 semillas usadas si deterministic=false
  tie_breaking:
    policy:                [deterministic | stochastic]  (obligatorio)
    rule:                  regla usada ante empates de Q (p. ej. lexicographic_expert_id)
  dataset_split:           B
  k:                       tamaño del subset buscado
  candidate_pool:          conjunto de expertos candidatos disponible para Search(B,k)
  implementation_version:  versión/hash del código de búsqueda usado
```

El espacio factible de selección también debe quedar congelado como artefacto común:

```
candidate_space:
  expert_pool:               conjunto idéntico para selector/search/random
  subset_size:               k
  feasibility_constraints:   restricciones idénticas de factibilidad del subset
  sampling_distribution:     uniform_over_feasible_subsets
```

Formalmente, para cada dominio/tarea `D` y tamaño `k`, el espacio factible queda definido como:

```
S_space(D,k) = { S : |S| = k, S ⊆ expert_pool(D), S cumple feasibility_constraints(D,k) }

S_search-oracle = Search(B, k; S_space(D,k))

S_random^{(i)} \overset{iid}{\sim} Uniform(S_space(D,k)), \quad i = 1, \ldots, N
```

`S_space(D,k)` es el conjunto de subsets factibles; `S_search-oracle` es un único elemento encontrado por el procedimiento de búsqueda documentado; y cada `S_random^(i)` es un único elemento muestreado desde ese mismo espacio factible.

Esta ficha debe guardarse como artefacto reproducible junto al resultado, no solo describirse en prosa en el reporte. Sin esta ficha, un `ESP@k` reportado en una versión futura con presupuesto de búsqueda distinto (p. ej. beam search con 10⁶ evaluaciones frente a greedy con 100) no es comparable al de v0.12, y no debe presentarse como si lo fuera. La ausencia de esta ficha es un error de reporte, en el mismo sentido que v0.9 §12 exige etiquetar el protocolo A/B/C de EXP-D.

`tie_breaking` es obligatorio conceptualmente, no un detalle opcional de implementación. Si el procedimiento de búsqueda encuentra varios subsets empatados dentro de la resolución de `Q_B`, el `S_search-oracle` final queda determinado por:

```
S_search-oracle = TieBreak(argmax_{S ∈ S_space(D,k)} Q_B(S))
```

o por el análogo aproximado del conjunto de mejores candidatos encontrados por `Search`. La regla de desempate forma parte de la configuración/hash de `oracle_search`, porque distintos desempates pueden producir subsets con distinto rendimiento en `C`.

Para comparaciones entre selectores dentro de un mismo estudio, `S_selector`, `S_search-oracle` y `S_random` deben definirse sobre el mismo espacio factible de subsets (`candidate_space`), salvo que el experimento declare explícitamente otra condición. Además, `S_search-oracle` debe construirse bajo idéntico `dataset_split`, `k`, método de búsqueda, objetivo, presupuesto de búsqueda, criterio de parada, semillas/protocolo estocástico, `candidate_pool` e `implementation_version`. Un cambio en cualquiera de estos elementos constituye un cambio del procedimiento `oracle_search` y requiere etiquetado independiente; de lo contrario, las diferencias podrían atribuirse erróneamente al selector cuando en realidad cambió el techo empírico usado como referencia.

**Consecuencia terminológica:** `ESP@k` pasa a leerse formalmente como *distancia al mejor subset encontrado por el procedimiento de búsqueda documentado*, no como distancia al óptimo global. Esto no cambia la fórmula (`ESP@k = Q_test(S_selector,D) − Q_test(S_search-oracle,D)`), solo la interpretación que se le puede dar por escrito. Cuando `Q` es una métrica de calidad donde mayor es mejor, `ESP@k = 0` es el caso ideal; valores negativos indican distancia respecto al mejor subset encontrado en B; valores positivos no implican superioridad para H5 y deben tratarse como diagnóstico de generalización/evaluación.

`ESP@k > 0` no se interpretará como evidencia de H5 ni como superioridad global del selector sobre el óptimo. La regla de estado para `ESP@k` es:

```
ESP@k < 0 y el IC95% no contiene 0:
  → selector_below_search_oracle

IC95%(ESP@k) contiene 0:
  → statistically_indistinguishable_from_zero

LCI95%(ESP@k) > 0:
  → selector_search_inversion_on_C
```

Solo `LCI95%(ESP@k) > 0` se marca como `selector_search_inversion_on_C`: una inversión selector–`S_search-oracle` en C que debe investigarse como posible sobreajuste del procedimiento de búsqueda a B, varianza estadística o diferencia real de generalización entre el selector perfilado en A y el subset encontrado por búsqueda en B.

**2. §3.3 — `RSE@k` se blinda frente a denominador pequeño o inestable; se reemplaza el umbral implícito por una regla de estabilidad sin introducir un `δ` arbitrario.**

La definición de `RSE@k` (v0.11 §3.3) no cambia en su fórmula:

```
RSE@k(D) = ( Q_test(S_selector,D) − Q̄_test(S_random,D) )
           ─────────────────────────────────────────────
           ( Q_test(S_search-oracle,D) − Q̄_test(S_random,D) )
```

pero se añade una regla de estabilidad, obligatoria para su reporte:

```
Si | Q_test(S_search-oracle,D) − Q̄_test(S_random,D) | no está separado de cero
con una incertidumbre estadística compatible con una diferencia real:

  Regla operacional: el denominador se considera interpretable para `RSE@k`
  únicamente cuando el límite inferior del intervalo de confianza del 95% de
  Δ_search-random = Q_test(S_search-oracle,D) − Q̄_test(S_random,D)
  es estrictamente mayor que 0:

    LCI95%(Δ_search-random) > 0

Si el IC95% de Δ_search-random contiene 0:

  → RSE@k(D) se reporta como `unstable_denominator` para ese (D,k),
    y el valor numérico NO se presenta como resultado principal interpretable,
    aunque puede incluirse entre paréntesis por transparencia.

Si el IC95% de Δ_search-random está completamente por debajo de 0:

  → RSE@k(D) se reporta como `oracle_below_random` para ese (D,k),
    y el valor numérico NO se presenta como resultado principal interpretable,
    aunque puede incluirse entre paréntesis por transparencia.
```

No se fija un umbral numérico (`δ_RSE`) para decidir "suficientemente separado de cero", siguiendo la misma disciplina que v0.9 §2.2 aplicó a `knee_point`: el criterio obligatorio no es que el denominador sea meramente distinto de cero, sino que el margen `random → search-oracle` sea positivo con soporte estadístico (`LCI95% > 0`). Si el denominador es interpretable, `RSE@k` puede reportarse como fracción normalizada de ese margen, pero la incertidumbre debe propagarse al cociente mediante un procedimiento que preserve la dependencia entre la variabilidad de `S_random` y la variabilidad sobre `C` (p. ej. bootstrap jerárquico u otro método equivalente). Debe reportarse el intervalo de confianza del denominador, el `denominator_status`, el `ratio_status`, y debe reportarse también el intervalo de confianza de `RSE@k` en sí cuando sea estimable:

```
Δ_search-random = 4.2 pp   [CI 95%: 2.8, 5.6]      (denominador interpretable)
RSE@k = 0.84               [CI 95%: 0.76, 0.91]    (interpretable)
denominator_status = interpretable
ratio_status = estimable

Δ_search-random = 0.7 pp   [CI 95%: -0.8, 2.1]     (denominador inestable)
RSE@k = —                  denominator_status = unstable_denominator
ratio_status = non_estimable

Δ_search-random = -4.0 pp  [CI 95%: -5.5, -2.5]    (search-oracle por debajo de random)
RSE@k = —                  denominator_status = oracle_below_random
ratio_status = non_estimable
```

`denominator_status = interpretable` y `ratio_status = estimable` son condiciones distintas. Si el procedimiento de propagación de incertidumbre no produce un intervalo de confianza válido para `RSE@k`, entonces `RSE@k = —` y `ratio_status = non_estimable`, aunque el denominador puntual esté marcado como interpretable. Solo la combinación `denominator_status = interpretable` y `ratio_status = estimable` permite presentar `RSE@k` como resultado principal.

**Regla pre-registrada para incertidumbre del cociente (`RSE@k`) y réplicas con denominador \(\le 0\):**

La incertidumbre de `RSE@k` debe estimarse propagando la variabilidad mediante el procedimiento pre-registrado antes de ejecutar el experimento principal. `RSE@k` se considera estimable únicamente si el procedimiento pre-registrado de bootstrap produce una distribución del cociente válida y finita.

1. **Estimador pre-registrado:**
   ```yaml
   ratio_uncertainty:
     estimator:              paired_bootstrap
     interval:               percentile
     confidence_level:       0.95
     invalid_ratio_replicates:
       denominator_nonpositive_rule: mark_ratio_non_estimable
     posthoc_exclusion:      false
     diagnostics_reporting:
       required_fields:
         - total_replicates
         - invalid_denominator_replicates
         - invalid_ratio_replicates
         - invalid_fraction
   ```
2. **Réplicas con denominador \(\le 0\) y diagnóstico auditable:** Si durante el remuestreo bootstrap alguna réplica produce un denominador \(\Delta_{\text{search-random}} \le 0\) (o indeterminado), la regla pre-registrada prohíbe taxativamente la exclusión post-hoc de réplicas (`posthoc_exclusion: false`). En tal situación, el procedimiento debe reportar:
   ```text
   denominator_status = interpretable   (si LCI95%(Δ_search-random) > 0 puntual en C)
   ratio_status = non_estimable
   RSE@k = —
   ```
   junto con el bloque `bootstrap_diagnostics` (ej. `invalid_denominator_replicates: 17`), haciendo auditable y transparente exactamente por qué `ratio_status` pasó a `non_estimable` sin distorsionar el resultado principal.

**Aclaración de nombre (sin cambiar el acrónimo):** `RSE@k` se documenta explícitamente como *normalized quality-gap capture*, no como una medida de eficiencia computacional o de runtime, para evitar que se confunda con las métricas de throughput que v0.9 §10 y v0.10 §10 desacoplan de H5/H6. El acrónimo "Routing Selection Efficiency" se mantiene por continuidad con v0.10, pero cualquier descripción en prosa de esta métrica debe usar la aclaración de arriba.

**3. §3.2 — `S_random` se especifica como distribución, no como muestra única, con recomendación (no obligación) de 10 semillas.**

Se formaliza lo que v0.11 §3.3 ya daba por hecho implícitamente:

```
S_random^{(i)} \overset{iid}{\sim} Uniform(S_space(D,k)), \quad i = 1, \ldots, N   (muestreo uniforme e independiente sobre subsets factibles)

Q̄_test(S_random,D) = (1/N) \sum_{i=1}^{N} Q_test(S_random^{(i)}, D)
```

`N_random ≥ 5` es el mínimo operativo para producir un reporte exploratorio (escalón exploratorio de v0.7 §11, sin cambios). `N_random = 10` es recomendado para el reporte principal cuando el presupuesto de cómputo lo permita, dado que `S_random` sirve como baseline de referencia y sanity check para `ESP@k` (desacoplado matemáticamente) y entra directamente en la fórmula de `RSE@k` (como término del denominador y del numerador). `N_random` no constituye una garantía de precisión estadística, ni se convierte en criterio obligatorio de H5 o de ninguna fase; la incertidumbre de `S_random` debe incorporarse al procedimiento de estimación del IC.

La referencia random se congela antes de la evaluación principal:

```
random_reference:
  sampling_distribution: uniform_over_feasible_subsets
  N_random: N
  random_seeds: [...]
  subsets_frozen_for_evaluation: true
  resample_random_subsets_in_bootstrap: false
```

Es decir, v0.12 estima el rendimiento medio de los `N_random` subsets pre-muestreados y congelados, no la expectativa de toda la distribución poblacional de subsets factibles con nuevos samples en cada réplica bootstrap. Si un experimento futuro quiere estimar esa expectativa poblacional, debe declararlo como una condición metodológica distinta antes de ejecutar el experimento.

Para estimar ICs del denominador y de `RSE@k`, el procedimiento debe respetar las dos fuentes de variabilidad: las semillas/muestras aleatorias que producen `S_random^(1), ..., S_random^(N)` y la incertidumbre del rendimiento sobre los ejemplos de `C`. Además, como los mismos ejemplos de `C` evalúan `S_selector`, `S_search-oracle` y cada subset random, el remuestreo debe preservar la estructura pareada por ejemplo.

Protocolo pre-registrado obligatorio para incertidumbre (Bootstrap pareado por ejemplo, condicionado a la referencia random congelada):

```yaml
uncertainty:
  method: paired_example_bootstrap_conditional_on_frozen_random_reference
  resample_unit: example
  preserve_paired_evaluation: true
  hierarchy:
    level_1: evaluation_examples_C
    level_2: frozen_random_reference
    random_subset_resampling: false
  random_subset_variability:
    subsets_frozen: true
    resample_new_subsets: false
    resample_frozen_subset_indices: false
  confidence_level: 0.95
  interval_type: percentile
  bootstrap_replicates: [valor pre-registrado antes del experimento principal]
```

**Aclaración terminológica de la incertidumbre:** Se renombra el método de `hierarchical_bootstrap` a `paired_example_bootstrap_conditional_on_frozen_random_reference` para evitar que se interprete erróneamente como si el bootstrap estuviese estimando también la variabilidad de la distribución poblacional de subsets. El procedimiento es un bootstrap pareado sobre los ejemplos de `C`, condicionado a la referencia `S_random` congelada previa al experimento.

Para cada réplica bootstrap, se remuestrean ejemplos de `C` con reemplazo, se mantiene el mismo conjunto bootstrap de ejemplos para selector, search y cada random, se conserva intacto el conjunto de `N_random` subsets congelados, y se recalculan `Δ_search-random` y `RSE@k`. Bajo v0.12 no se muestrean nuevos subsets random dentro de cada réplica bootstrap ni se remuestrean índices de subsets random congelados; `random_reference` es literalmente un artefacto experimental fijo. Por tanto, los ICs principales representan la incertidumbre de evaluación sobre `C` condicionada a esa referencia random congelada. Si un experimento futuro quiere representar incertidumbre Monte Carlo de la referencia random mediante remuestreo de índices de subsets congelados, o estimar la expectativa poblacional mediante nuevos subsets en cada réplica, debe declararlo como condición metodológica distinta antes de ejecutar el experimento principal.

Si `denominator_status = interpretable` y `ratio_status = estimable`, `RSE@k` puede tomar cualquier valor real: `RSE@k < 0` indica que el selector queda por debajo del baseline random; `0 <= RSE@k <= 1` indica captura parcial o completa del margen random→search-oracle; `RSE@k > 1` indica que el selector supera a `S_search-oracle` en C, no constituye evidencia de H5 y debe diagnosticarse junto con `ESP@k`. Si además `LCI95%(ESP@k) > 0`, el estado diagnóstico canónico es `selector_search_inversion_on_C`.

**4. §3.2 — la partición A/B/C se formaliza a nivel de ejemplo, no de batch, con salvaguarda explícita contra near-duplicates.**

Se añade, como precisión del procedimiento de v0.11 §3.2:

```
Dataset D (dominio/tarea) se particiona a nivel de EJEMPLO:

  A ∩ B = ∅      A ∩ C = ∅      B ∩ C = ∅      (por ejemplo individual, no por batch)
```

Si el dominio contiene ejemplos derivados de una misma fuente (paráfrasis, aumentos de datos, variantes casi idénticas del mismo ítem base), el particionado debe agrupar esas variantes en el mismo split — nunca repartir ejemplos casi idénticos entre A/B/C, porque eso reintroduce leakage de forma indirecta (el selector o el oracle podrían "ver" en A o B una variante casi idéntica de algo evaluado en C). Esta salvaguarda se implementa como `group-level split`: si `g(x)` es el identificador de fuente/grupo de un ejemplo, entonces:

```
g(x_i) = g(x_j)  ⇒  split(x_i) = split(x_j)
```

cuando exista una relación de derivación, paráfrasis, augmentación o near-duplicate entre ejemplos.

**Congelamiento obligatorio del algoritmo de deduplicación / near-duplicates:**
Dado que el propio algoritmo de deduplicación puede modificar la composición de los splits A/B/C, no basta con declarar `group_split: true`. Es obligatorio registrar y congelar la ficha del procedimiento de agrupación como artefacto reproducible antes de generar la partición:

```yaml
grouping:
  method:                  [source_id | exact_hash | semantic_dedup | otro]
  implementation_version:  versión o commit del código/biblioteca de deduplicación
  threshold:               umbral numérico de similitud (si aplica, p. ej. 0.92)
  seed:                    semilla usada en la deduplicación (si aplica)
```

Se exige documentar, para cada dominio, el criterio y la ficha `grouping` usados. Si no aplica, se declara explícitamente `method: none` ("sin near-duplicates conocidos") en el reporte.

**5. §3.2 — se fija la semántica de `D` en `ESP@k(D)` / `RSE@k(D)`: dominio/tarea de evaluación, con sus propios splits A/B/C.**

Se congela: `D` denota un dominio o tarea de evaluación (p. ej. "Coding", "Math"), y cada `D` posee su propia partición A/B/C independiente:

```
Domain D1 (p. ej. Coding)
  ├── A1 (selection/profiling)
  ├── B1 (oracle search)
  └── C1 (test)

Domain D2 (p. ej. Math)
  ├── A2
  ├── B2
  └── C2
```

Queda explícitamente prohibido, y se documenta como error de reporte, seleccionar o buscar el oracle sobre un dominio y evaluar sobre otro (p. ej. `S_selector` perfilado sobre A1 pero `Q_test` medido sobre C2) bajo la etiqueta `ESP@k(D1)` o `RSE@k(D1)` — eso mide transferencia entre dominios, una pregunta distinta y legítima, pero que requeriría su propia notación (`ESP@k(D1→D2)`) si alguna vez se investiga, fuera del alcance de v0.12.

---

## 0. Qué NO cambia respecto a v0.11

- Todo lo listado en v0.11 §0: H5 (§2.1/§2.2 de v0.9), H6 (v0.8 §2), los protocolos EXP-D-A/B/C (v0.9 §12), la separación selector/evaluador (v0.10 §3.1), la distinción subset lógico/físico (v0.10 §9), la nota terminológica de memoria nominal vs. residente (v0.11 §9, sin fórmula modificada), y R1.5 como fase candidata no comprometida (v0.11 §R1.5).
- El desacoplamiento de throughput e I/O respecto a H5/H6 (v0.9 §10, v0.10 §10): sin cambios.
- La regla de que `ESP@k` y `RSE@k` son diagnósticas del método de selección, nunca evidencia de H5: se mantiene y se refuerza explícitamente en este documento (ver Cambio 2, aclaración de nombre).
- R1.0/R1.0.1: permanecen congelados desde v0.7, no se reabren por este documento.
- Todo resultado ya reportado bajo v0.7–v0.11 sigue siendo válido. **Excepción a vigilar (extiende la de v0.11 §0):** cualquier `ESP@k` o `RSE@k` calculado bajo v0.11 sin la ficha de procedimiento de búsqueda (Cambio 1) o sin la regla de estabilidad (Cambio 2) no se invalida retroactivamente, pero debe re-etiquetarse como "pre-v0.12 (ficha de oracle no documentada / estabilidad no verificada)" antes de compararse con resultados nuevos.

---

## 16. Criterios de aceptación por fase — ajuste

Se añade una precisión a la fila de R1.3 de v0.11 §16 (el resto de la tabla, incluidas R1.4 y R1.5, no cambia):

| Fase | Criterio de aceptación |
|---|---|
| R1.3 Selector Design | Sin cambios respecto a v0.11 §16, con las siguientes precisiones de v0.12-final: (a) todo `ESP@k` reportado debe incluir la ficha de procedimiento de búsqueda del `S_search-oracle` (§3.2 de este documento, Cambio 1), con `Q` orientada a calidad (`maximize`), `Q = -L` para losses salvo métrica distinta pre-registrada, `budget.unit`, `tie_breaking` y determinismo documentados; (b) `S_selector`, `S_search-oracle` y `S_random` deben compartir el mismo `candidate_space = S_space(D,k)` con $P = N_{\text{experts}}(\text{model})$ (el tamaño del espacio coincide exactamente con los expertos físicos del modelo MoE evaluado, p. ej. $P=8$ para M1, $P=32$ para M2); (c) solo `LCI95%(ESP@k) > 0` se trata como inversión selector–`S_search-oracle` en C (`selector_search_inversion_on_C`), nunca como evidencia de H5 ni superioridad global; (d) todo `RSE@k` reportado debe aplicar la regla operacional de denominador semánticamente válido (`LCI95%(Δ_search-random) > 0`), estimador de incertidumbre pre-registrado (`percentile` bootstrap pareado por ejemplo condicionado a la referencia congelada), regla para denominadores \(\le 0\) sin exclusión post-hoc (`posthoc_exclusion: false`), y `ratio_status = estimable` para resultado principal (Cambios 2 y 3), o marcarse explícitamente como `unstable_denominator`, `oracle_below_random` o `non_estimable`; (e) `S_random` se formaliza i.i.d. y se congela como `random_reference` antes de la evaluación principal; (f) la partición A/B/C debe ser a nivel de ejemplo con `group-level split` y la ficha `grouping` (algoritmo, versión, umbral, semilla) congelada (Cambio 4), y el perfilado en A para $S_{\text{selector}}$ se restringe estrictamente a la ventana del prompt (`prompt_ids + sep_id`) omitiendo tokens futuros/target para prevenir fuga de datos; (g) `D` se usa exclusivamente en el sentido de dominio/tarea con splits propios (Cambio 5). |
| R1.5 Physical subset validation | Sin cambios respecto a v0.11 §16: no aplica, fase candidata sin criterio de aceptación. |


---

## Regla metodológica general (sin cambios respecto a v0.7–v0.11)

Ninguna técnica se declara superior sin control aislado del efecto atribuido, y ningún cambio de hipótesis, métrica o protocolo ocurre sin una nueva versión de este documento.

**v0.12-final congela formalmente el contrato:** el renombramiento de `S_oracle` a `S_search-oracle`/empirical oracle con ficha de procedimiento obligatoria, reproducible y comparable dentro de cada estudio, `Q` orientada a calidad (`maximize`), regla canónica `Q = -L` para losses, `budget.unit`, `tie_breaking`, determinismo de búsqueda y `candidate_space = S_space(D,k)` común para selector/search/random (Cambio 1), la regla operacional de `RSE@k` basada en margen `random → search-oracle` positivo (`LCI95%(Δ_search-random) > 0`) y propagación de incertidumbre al cociente con el estimador pre-registrado (`ratio_uncertainty`), prohibiendo exclusión post-hoc de réplicas con denominador \(\le 0\) y separando `ratio_status` de `denominator_status` (Cambio 2), la formalización i.i.d. de \(S_{\text{random}}^{(i)} \overset{iid}{\sim} Uniform(S_{space}(D,k))\) congelado como `random_reference` previa evaluación y el renombrado de la incertidumbre a `paired_example_bootstrap_conditional_on_frozen_random_reference` (Cambio 3), la partición A/B/C a nivel de ejemplo con `group-level split` y el congelamiento obligatorio de la ficha del algoritmo de deduplicación/near-duplicates (`grouping`) (Cambio 4), y la semántica fija de `D` como dominio/tarea con splits propios (Cambio 5). No se congela ninguna fase R1.5 (sigue candidata, sin protocolo). No se congela ningún contenido de H5/H6 en sí, ni de R2–R5. R1.0/R1.0.1 permanecen congelados desde v0.7 y no se reabren por este documento.

## Resumen de la arquitectura metodológica vigente tras v0.12-final

```
H5            → ¿el subset preserva calidad bajo zero-adaptation?           (EXP-D-A, único protocolo citable)
knee_point(D) → ¿dónde está el quiebre calidad/compresión?                  (descriptivo, v0.9 §2.2)
ESP@k(D)      → ¿qué tan lejos está el selector del mejor subset ENCONTRADO
                 por un procedimiento de búsqueda documentado y comparable? (diagnóstico del selector, nunca H5)
RSE@k(D)      → ¿qué fracción del margen random→search-oracle captura
                 el selector, si LCI95%(Δ_search-random) > 0
                 y propagación de incertidumbre al cociente?                 (diagnóstico del selector, nunca H5)
EXP-D-B       → ¿cuánto recupera el router mediante recalibración?          (nunca H5)
EXP-D-C       → ¿cuánto recupera el subset mediante distillation?           (fuera de alcance de v0.9, sin cambios)
R1.5          → ¿la compresión lógica se materializa en memoria física?     (candidata, no comprometida)
```

Ninguna de estas métricas sustituye a otra como evidencia de H5; cada una responde una pregunta distinta y se reporta con su propia etiqueta.
