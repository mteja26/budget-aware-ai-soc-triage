# Research Charter v0.3

## Working title

**Budget-Aware AI Decision Intelligence from Cybersecurity Incident Data: Cost–Safety–Priority Trade-offs in SOC Queue Triage**

Author: **Manas Teja Maruri**

The earlier wording, *network alert triage*, is not retained as the preferred
title because the strongest available ground truth in Microsoft GUIDE is at the
incident and organization-queue level. Calling these records network alerts
would misstate the unit of analysis.

## Publication objective

Develop and evaluate a reproducible decision-intelligence method that allocates
limited analyst attention across security incidents. The paper is intended as a
full empirical journal article. It is not a detector-comparison paper and it
will not claim that ordinary classification accuracy constitutes decision
intelligence.

## Hard novelty boundary

The following claims are already occupied and cannot be presented as the main
contribution:

| Occupied contribution | Closest work | Consequence |
|---|---|---|
| Rank SOC incidents by urgency | Microsoft AIP (2026) | Generic risk ranking is not novel. |
| Adaptively defer SOC alerts to analysts | L2DHF (2025) | Deferral alone is not novel. |
| Optimize deferral under expert workload constraints | DeCCaF (2024) | A capacity constraint alone is not novel. |
| Use calibration, costs, and an alert budget | CALIBURN (2026) | Cost-sensitive thresholding plus alert-rate control is not novel. |
| Use limited analyst queries for adaptive SOC screening | PACT (2026) | Budgeted active learning is not novel. |

Primary sources:

- AIP: https://arxiv.org/abs/2607.16963
- L2DHF: https://arxiv.org/abs/2506.18462
- DeCCaF: https://arxiv.org/abs/2403.06906
- CALIBURN: https://arxiv.org/abs/2605.24696
- PACT: https://arxiv.org/abs/2605.22324

## Proposed research gap

Existing systems largely separate two questions:

1. **Triage outcome:** Is an incident a true positive, benign positive, or
   false positive?
2. **Queue priority:** Which incident should receive analyst attention first?

Microsoft AIP explicitly treats prioritization as distinct from triage. The
proposed study investigates the missing operational decision layer: how to
jointly use calibrated triage probabilities and queue-relative priority under a
finite review budget, while remaining robust to uncertain decision costs and
probability shift.

The novelty claim is therefore the **joint, constrained, ambiguity-aware action
policy**, not the underlying classifier or ranking model.

## Data decision

### Primary dataset: Microsoft GUIDE v2

The official release contains:

- `GUIDE_Train.csv`: real-world incident/evidence records with triage outcomes;
- `GUIDE_Test.csv`: held-out real-world incident/evidence records;
- `GUIDE_Test_Queue_Rankings.csv`: 9,980 expert-ranked incidents from 499
  organization queues.

The prioritization labels identify the expert-selected order within each
organization queue. They do not mean that every highly ranked incident is more
likely to be a true positive than every lower-ranked incident.

The received ranking file has been structurally verified: 9,980 rows, 499
organizations, exactly 20 ranks per organization, complete ranks 1--20, and no
missing or duplicate `(OrgId, IncidentId)` keys. It contains 9,739 distinct
`IncidentId` values; 237 incident IDs occur in more than one organization.
Therefore, every join and split must use the composite key
`(OrgId, IncidentId)`. Joining on `IncidentId` alone is prohibited.

License reported by the official data card: CDLA-Permissive-2.0.

Official dataset: https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction/data

### Secondary dataset: AIT-ADS

AIT-ADS may be used only for a secondary transfer or stress experiment. Its
attack-phase intervals are not human-confirmed per-alert verdicts. Temporal
overlap labels must not be represented as analyst triage ground truth.

## Unit of analysis

The decision unit is a deduplicated security **incident** within an
organization-specific queue. Evidence-level rows must be aggregated without
allowing records from the same incident to cross training, calibration,
validation, bootstrap, or evaluation boundaries. In the audited samples,
incidents contain up to 17 evidence rows in training and 29 in testing.
Separate organization-grouped analyses will be required before making any
claim about transfer to previously unseen organizations.

`ActionGrouped` and `ActionGranular` are post-response information and excluded
from predictors. `SuspicionLevel` and `LastVerdict` have ambiguous temporal
availability; the primary model excludes them unless documentation proves they
were available at the triage decision time. `Usage` is test-partition metadata
and excluded from predictors. `Id`, `OrgId`, `IncidentId`, and `AlertId` are
used only for joins, grouping, splitting, and evaluation; they are not
predictive features.

## Decision actions

For incident `i` in organization queue `q`, the policy selects one action:

1. `close`: automatically close only when the incident is sufficiently safe to
   treat as a false positive;
2. `defer`: retain in the queue for later attention;
3. `review_now`: consume one unit of the current analyst-review budget.

The action vocabulary will be revised if the GUIDE license or schema does not
support a defensible mapping. No response or containment action will be claimed
without corresponding outcome data.

## Formal decision problem

Let `p_i` be the calibrated probability vector for triage outcomes, `r_i` the
queue-priority signal, `a_i` an action, `B_q` the current analyst budget, `C`
an uncertain action-cost matrix, and `P` a plausible deployment distribution.
The policy solves:

\[
\pi^* = \arg\min_{a_1,\ldots,a_n}
\sup_{P\in\mathcal{U}(P_0),\,C\in\mathcal{C}}
\mathbb{E}_{P}\left[
\sum_{i\in q} L(a_i,Y_i,r_i;C)
\right]
\]

subject to:

\[
\sum_{i\in q}\mathbf{1}(a_i=\text{review\_now})\le B_q.
\]

A safety constraint will limit automatic closure of incidents with material
true-positive risk. Its threshold must be selected on validation organizations,
not on the test queues.

## Research questions

**RQ1.** At identical review budgets, does a joint triage-priority policy reduce
worst-case normalized decision loss relative to severity ordering, public AIP
baselines, triage-risk ordering, and nominal cost optimization?

**RQ2.** How much does probability calibration affect budget allocation and
unsafe automatic closure under organization and time shift?

**RQ3.** Does ambiguity-aware optimization reduce cost regret when the deployed
cost matrix differs from the validation cost matrix?

**RQ4.** Which component contributes measurable value: triage probability,
queue priority, capacity optimization, or robustness?

**RQ5.** How does performance change across analyst budgets and organization
queue sizes?

## Preregistered hypotheses

- **H1:** The joint robust policy has lower worst-case normalized decision loss
  than the nominal capacity optimizer at the same budget.
- **H2:** The joint policy recovers more expert-priority incidents than a pure
  triage-probability policy at the same budget.
- **H3:** Calibration reduces unsafe automatic closure under held-out
  organizations.
- **H4:** Removing either triage probability or queue priority significantly
  worsens the primary decision outcome.

Hypotheses will be rejected when confidence intervals include no improvement.
The method will not be rewritten after test evaluation solely to make the
hypotheses pass.

## Post-v0.5 audit status

The v0.5 results rejected H1 and supported H2. H3 was only partially supported,
and H4 could not be interpreted because the original `priority_only` policy
still used triage probabilities for automatic close/defer actions. These are
results, not defects to be hidden.

Version 0.6 is a protocol-correction run. It does not change the prespecified
cost matrices, ambiguity radii, closure thresholds, primary priority weight,
or hypotheses. It makes the preregistered equal-queue macro estimand primary,
adds a true no-probability/defer-only ablation, reports precision and NDCG on
the expert queues, adds a largest-remainder pooled-capacity sensitivity, and
places paired organization-bootstrap intervals directly in the main figures.
The old `priority_only` policy remains as a review-ordering ablation so that
the v0.5 comparison is reproducible.

## Baselines

1. Random within-queue selection.
2. Arrival-time ordering.
3. Maximum product severity.
4. Alert-count ordering.
5. Triage true-positive probability.
6. Uncertainty/entropy ordering.
7. Public AIP variants: TF-IDF, BM25, and BM25 plus component cap.
8. Nominal capacity-constrained cost optimizer.
9. Proposed ambiguity-aware joint optimizer.

## Evaluation

### Primary outcomes

- Worst-case normalized decision loss per organization queue.
- Regret relative to an oracle with known outcomes and costs.
- Expert-priority coverage at budget `B`.
- Unsafe automatic-closure rate for true-positive incidents.

### Secondary outcomes

- Precision@5, Precision@10, and Precision@20.
- NDCG@K when complete rank information supports it.
- Macro-F1 and per-class recall for triage outcomes.
- Log loss, multiclass Brier score, and calibration error.
- Review-budget utilization and capacity violations.
- Inference time and peak memory.

### Statistical protocol

- Organization-level partitions only.
- Macro-average queue metrics so large organizations do not dominate.
- Paired bootstrap resampling over organization queues.
- Report 95% confidence intervals and effect sizes.
- Correct families of multiple comparisons.
- Freeze costs, ambiguity radii, and thresholds using training/validation only.

## Required ablations

1. No probability calibration.
2. No queue-priority signal.
3. No triage-probability signal.
4. Point cost instead of cost intervals.
5. No distributional ambiguity.
6. Independent thresholds instead of queue-level optimization.

## Claims that are prohibited

- Publication or acceptance is guaranteed.
- GUIDE incidents are individual network alerts.
- Normalized experimental costs are real dollar losses.
- Expert queue rank is the same as true-positive probability.
- AIT-ADS temporal overlap is human-confirmed alert ground truth.
- Synthetic or smoke-test results are empirical evidence for the paper.
- Better accuracy automatically means better SOC decision quality.

## Current data gate

The author has the official archive locally, and the queue-ranking file has been
verified. The two full incident files exceed 1 GB each and will remain local.
Modeling begins after their headers and small quote-safe samples are audited;
the final pipeline will stream or chunk the complete local files.
