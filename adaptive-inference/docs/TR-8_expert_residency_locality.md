# TR-8: Expert Residency & Locality Analysis

This technical report investigates the sparse routing characteristics of the MoE model architecture to evaluate the feasibility of an **Expert Cache** offloading system, targeting local deployment on consumer devices with limited RAM (~8 GB).

The analysis was performed on the trained **M1 MoE model** (8 experts, Top-$K=2$, $L=5$ layers, ~400M total parameters, 140M active per token).

---

## Q1: Is There Expert Selection Concentration?

Yes, but it is highly dependent on the layer depth. 

- **Global effective experts ($N_{\text{eff}}$)**: **7.54** (out of 8 experts available).
- **Global η_cap**: **0.9431**
- **Global Gini Index**: **0.1352** (low global inequality).

However, looking at the per-layer metrics reveals a distinct shift in concentration as tokens propagate deeper:

| Layer Index | Layer Type / Depth | $N_{\text{eff}}$ | $\eta_{\text{cap}}$ | Gini Index |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 0** | Early Layer | 7.68 | 0.960 | 0.114 |
| **Layer 1** | Early-Mid Layer | 6.07 | 0.759 | 0.314 |
| **Layer 2** | Mid Layer | 6.55 | 0.818 | 0.262 |
| **Layer 3** | Mid-Late Layer | **3.51** | **0.438** | **0.583** |
| **Layer 4** | Late Layer | **3.34** | **0.417** | **0.593** |

### Analysis
Early layers act as general feature extractors, routing tokens uniformly across all experts. Later layers (Layers 3 & 4) show a high concentration ($N_{\text{eff}} \approx 3.3$, Gini $\approx 0.59$), indicating that later routing choices specialize strongly based on semantic/logic task domains.

---

## Q2: Does Temporal Token Locality Exist?

Yes. The temporal token-to-token transition matrix $T[i, j] = P(E_{t+1}=j \mid E_t=i)$ shows strong self-transition probabilities on the diagonal:

- $P(E^0_{t+1} = 0 \mid E^0_t = 0) = \mathbf{36.9\%}$
- $P(E^3_{t+1} = 3 \mid E^3_t = 3) = \mathbf{30.4\%}$
- $P(E^4_{t+1} = 4 \mid E^4_t = 4) = \mathbf{32.8\%}$
- $P(E^7_{t+1} = 7 \mid E^7_t = 7) = \mathbf{26.9\%}$

Compared to a uniform random transition probability of $12.5\%$ (1/8), tokens exhibit a **$2.1\times$ to $3.0\times$ higher probability of re-activating the same expert** on the subsequent step. This confirms temporal locality.

---

## Q3: Does Inter-Layer Locality Exist?

Yes. The layer-to-layer transition matrix $L[i, j] = P(E^{(l+1)}=j \mid E^{(l)}=i)$ shows significant routing correlations. For example:
- Tokens routed to Expert 5 at Layer 2 have a **$43.2\%$ probability of routing to Expert 5 at Layer 3**.
- Tokens routed to Expert 3 at Layer 1 have a **$26.1\%$ probability of routing to Expert 3 at Layer 2**.

This correlation between layer paths within the same token can be exploited for **layer-by-layer prefetching**.

---

## Q4: Cache Capacity ($C$) vs Hit Rate & NVMe Traffic

We simulated Expert Caches with capacity $C$ experts (where $C \in [2, 8]$ is relevant for the 8-expert model) using different eviction policies.

### Cache Hit Rates (%)

| Capacity ($C$) | Static Top-$C$ | LRU | LFU | Markov Prefetch |
| :---: | :---: | :---: | :---: | :---: |
| **C = 2** | 31.77% | **56.36%** | 50.49% | 55.69% |
| **C = 4** | 61.59% | **80.52%** | 78.05% | 80.28% |
| **C = 6** | 82.53% | **94.14%** | 92.85% | 94.10% |
| **C = 8** | 100.00% | 99.99% | 99.99% | 99.99% |

### NVMe Traffic (MB / Token) for LRU

Assuming different expert parameter sizes:

| Capacity ($C$) | LRU Hit Rate | 50 MB Expert | 200 MB Expert | 500 MB Expert |
| :---: | :---: | :---: | :---: | :---: |
| **C = 2** | 56.36% | 218.18 MB / tok | 872.74 MB / tok | 2,181.84 MB / tok |
| **C = 4** | 80.52% | 97.41 MB / tok | 389.63 MB / tok | 974.08 MB / tok |
| **C = 6** | **94.14%** | **29.32 MB / tok** | **117.29 MB / tok** | **293.23 MB / tok** |
| **C = 8** | 99.99% | 0.03 MB / tok | 0.12 MB / tok | 0.29 MB / tok |

### Sliding Window Working Set Sizes $W(w)$
- **$W(100)$**: Mean = **7.13** experts.
- **$W(500)$**: Mean = **7.96** experts.
- **$W(1000)$**: Mean = **8.00** experts.

---

## Q5: Can the NVMe Architecture Reach 20 tok/s?

We modeled per-token latency budget (Target: **$\le 50 \text{ ms}$ / token**):
- **Baseline Compute Latency ($T_{\text{compute}} + T_{\text{router}} + T_{\text{KV}}$)**: Estimated at **15 ms**.
- **NVMe Bandwidth**: PCIe Gen4 NVMe estimated at **5.0 GB/s** ($5 \text{ MB/ms}$).
- **Expert Size**: **50 MB** (quantized model expert).

$$T_{\text{NVMe}} = \text{Expected Misses per token} \times \frac{\text{Expert Size MB}}{\text{NVMe BW (MB/ms)}}$$
$$\text{Expected Misses per token} = (1 - \text{Hit Rate}) \times K \times L$$

Using the LRU hit rates:

| Capacity ($C$) | Hit Rate | Expected Misses / Token | NVMe Latency | Total Latency | Est. Throughput | Meets Target? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C = 2** | 56.36% | 4.364 | 43.64 ms | 58.64 ms | 17.1 tok/s | No |
| **C = 4** | **80.52%** | **1.948** | **19.48 ms** | **34.48 ms** | **29.0 tok/s** | **YES** |
| **C = 6** | **94.14%** | **0.586** | **5.86 ms** | **20.86 ms** | **47.9 tok/s** | **YES** |
| **C = 8** | 99.99% | 0.001 | 0.01 ms | 15.01 ms | 66.6 tok/s | YES |

### Conclusion
- At **$C=4$** (keeping 4 experts resident in RAM), the system achieves **29.0 tok/s**, satisfying the 20 tok/s budget with headroom.
- At **$C=6$**, the system achieves **47.9 tok/s** with negligible NVMe overhead (~5.9 ms).
- **FEASIBILITY DEMONSTRATED**: An expert offloading architecture is technically viable for meeting local consumer laptop constraints when keeping at least 50% to 75% of experts resident.
