# Pairwise LambdaRank: learn preferences, spend effort where ranking quality moves

Our implementation trains on ordered document pairs, weights every correction by its top-K NDCG impact, and maintains a latent document ranking.

## 1. Pair evidence

Each training row represents an ordered pair `(i, j)` with first-item features, second-item features, and optionally their differences. The tree emits `f(i,j)`.

```text
d_ij = mean{ f(i,j), -f(j,i) }
m_ij = ((1-alpha)d_ij + alpha*a_ij) / available_weight
```

`a_ij` averages observed two-edge paths through an auxiliary document, for example `f(i,k) + f(k,j)`, including the code's signed orientation variants.

## 2. NDCG-aware LambdaRank

For unequal-label documents, where at least one current rank is inside the truncation level `K`:

```text
DeltaNDCG_ij = |G(y_i)-G(y_j)| |D(r_i)-D(r_j)| / IDCG@K
q_ij          = 1 / (1 + exp(sigma*m_ij))
lambda_ij     = -sigma*q_ij*DeltaNDCG_ij
h_ij          = sigma^2*q_ij*(1-q_ij)*DeltaNDCG_ij
```

By default, `G(y)=2^y-1` and `D(r)=1/log2(r+2)`. The signed lambda and positive curvature are distributed back to all direct, inverse, and indirect pair rows that contributed to the margin.

## 3. Latent ranking state

The objective keeps pointwise scores `s_i` and iteratively aligns `sigmoid(sigma*(s_i-s_j))` with the preference probability implied by `m_ij`. A damped Newton-style update changes `s_i`, clipped to `[-0.3, 0.3]`, then current ranks and NDCG swap weights are recomputed.

**Manager takeaway:** The model learns which document should win, but concentrates learning on swaps that would change top-K NDCG the most.

Implementation anchors: `src/objective/rank_objective.hpp`, `src/objective/rank_objective.cpp`, `src/io/metadata.cpp`, and `src/io/dataset.cpp`.
