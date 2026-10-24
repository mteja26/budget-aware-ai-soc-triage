import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_guide_experiment as guide


def test_tv_radius_zero_matches_nominal_expectation():
    probability = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
    cost = np.array([100.0, 4.0, 0.0])
    observed = guide.worst_case_tv_expectation(probability, cost, 0.0)
    assert np.allclose(observed, probability @ cost)


def test_tv_robust_expectation_is_never_smaller():
    probability = np.array([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
    cost = np.array([100.0, 4.0, 0.0])
    nominal = probability @ cost
    robust = guide.worst_case_tv_expectation(probability, cost, 0.05)
    assert np.all(robust >= nominal - 1e-12)


def test_fractional_budget_is_a_true_upper_bound():
    assert guide.budget_for_queue(9, 0.10, exact=False) == 0
    assert guide.budget_for_queue(10, 0.10, exact=False) == 1
    assert guide.budget_for_queue(99, 0.10, exact=False) == 9


def test_largest_remainder_preserves_pooled_fractional_target():
    groups = [
        ("a", np.arange(9)),
        ("b", np.arange(9)),
        ("c", np.arange(82)),
    ]
    floor = guide.allocate_queue_budgets(
        groups, 0.10, exact=False, allocation_mode="per_queue_floor"
    )
    pooled = guide.allocate_queue_budgets(
        groups, 0.10, exact=False, allocation_mode="largest_remainder"
    )
    assert floor == {"a": 0, "b": 0, "c": 8}
    assert pooled == {"a": 1, "b": 1, "c": 8}
    assert sum(pooled.values()) == 10


def test_policy_uses_exact_ranked_budget():
    probability = np.array(
        [
            [0.01, 0.09, 0.90],
            [0.10, 0.30, 0.60],
            [0.40, 0.20, 0.40],
            [0.05, 0.20, 0.75],
        ]
    )
    result = guide.apply_policy_to_queue(
        probability,
        priority=np.array([0.1, 0.2, 0.3, 0.4]),
        alert_count=np.ones(4),
        timestamps=np.arange(4).astype("datetime64[D]"),
        budget=2,
        policy="joint_robust",
        costs=guide.COST_SCENARIOS["medium"],
        tv_radius=0.025,
        automation_cost_width=0.25,
        close_threshold=0.10,
        closure_safety="tp",
        priority_lambda=10.0,
        seed=2026,
    )
    assert int(np.sum(result.actions == guide.REVIEW)) == 2


def test_tp_safety_rule_blocks_high_risk_closure():
    probability = np.array(
        [
            [0.01, 0.09, 0.90],  # eligible for closure under TP-only safety
            [0.20, 0.05, 0.75],  # ineligible for closure
        ]
    )
    result = guide.apply_policy_to_queue(
        probability,
        priority=np.zeros(2),
        alert_count=np.ones(2),
        timestamps=np.arange(2).astype("datetime64[D]"),
        budget=0,
        policy="robust_no_priority",
        costs=guide.COST_SCENARIOS["medium"],
        tv_radius=0.025,
        automation_cost_width=0.25,
        close_threshold=0.10,
        closure_safety="tp",
        priority_lambda=0.0,
        seed=2026,
    )
    assert result.actions[0] == guide.CLOSE
    assert result.actions[1] != guide.CLOSE


def test_true_no_probability_ablation_ignores_probability_values():
    common = dict(
        priority=np.array([0.1, 0.9, 0.5]),
        alert_count=np.ones(3),
        timestamps=np.arange(3).astype("datetime64[D]"),
        budget=1,
        policy="priority_defer_only",
        costs=guide.COST_SCENARIOS["medium"],
        tv_radius=0.025,
        automation_cost_width=0.25,
        close_threshold=0.10,
        closure_safety="tp",
        priority_lambda=0.0,
        seed=2026,
    )
    first = guide.apply_policy_to_queue(
        np.array([[0.9, 0.05, 0.05], [0.2, 0.3, 0.5], [0.1, 0.1, 0.8]]),
        **common,
    )
    second = guide.apply_policy_to_queue(
        np.array([[0.01, 0.01, 0.98], [0.98, 0.01, 0.01], [0.2, 0.7, 0.1]]),
        **common,
    )
    assert np.array_equal(first.actions, second.actions)
    assert first.actions.tolist() == [guide.DEFER, guide.REVIEW, guide.DEFER]
    assert np.isnan(first.predicted_risk).all()


def test_expert_ndcg_is_one_for_ideal_policy_order():
    metrics = guide.expert_ranking_metrics(
        ranks=np.array([1, 2, 3, 4]),
        reviewed=np.array([True, True, False, False]),
        review_scores=np.array([0.9, 0.8, 0.2, 0.1]),
        budget=2,
    )
    assert np.isclose(metrics["expert_precision_at_budget"], 1.0)
    assert np.isclose(metrics["expert_ndcg_at_budget"], 1.0)


def test_priority_score_does_not_read_expert_rank():
    frame = pd.DataFrame(
        {
            "OrgId": ["a", "a", "a"],
            "evidence_rows": [1, 3, 2],
            "alert_count": [1, 2, 1],
            "first_detector_id": ["x", "y", "x"],
            "first_alert_title": ["m", "n", "m"],
            "QueueRank": [1, 2, 3],
        }
    )
    builder = guide.IncidentFeatureBuilder()
    builder.n_fit = len(frame)
    builder.high_counts = {
        "first_detector_id": {"x": 2, "y": 1},
        "first_alert_title": {"m": 2, "n": 1},
    }
    before = builder.priority_score(frame)
    frame["QueueRank"] = [3, 1, 2]
    after = builder.priority_score(frame)
    assert np.allclose(before, after)


def test_temperature_scaling_returns_simplex_probabilities():
    probability = np.array(
        [[0.95, 0.04, 0.01], [0.70, 0.20, 0.10], [0.10, 0.20, 0.70]]
    )
    labels = np.array([1, 1, 2])
    scaler = guide.TemperatureScaler().fit(probability, labels)
    transformed = scaler.transform(probability)
    assert np.isfinite(transformed).all()
    assert np.allclose(transformed.sum(axis=1), 1.0)


def test_automation_cost_width_leaves_review_cost_fixed():
    probability = np.array([[0.2, 0.3, 0.5]])
    costs = guide.COST_SCENARIOS["medium"]
    risks = guide.action_risks(
        probability,
        costs,
        robust=True,
        tv_radius=0.0,
        automation_cost_width=0.25,
    )
    expected = probability @ costs.T
    assert np.allclose(risks[0, :2], expected[0, :2] * 1.25)
    assert np.isclose(risks[0, guide.REVIEW], expected[0, guide.REVIEW])


def test_probability_variant_selection_uses_held_out_log_loss():
    raw_fit = np.array([[0.99, 0.005, 0.005], [0.005, 0.99, 0.005], [0.005, 0.005, 0.99]])
    labels_fit = np.array([0, 1, 2])
    raw_selection = np.array([[0.90, 0.05, 0.05], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90]])
    labels_selection = np.array([0, 1, 2])
    choice = guide.select_probability_variant(
        raw_fit,
        labels_fit,
        raw_selection,
        labels_selection,
        raw_selection,
    )
    expected = (
        "temperature"
        if choice.selection_log_loss_temperature < choice.selection_log_loss_raw
        else "raw"
    )
    assert choice.selected_variant == expected
    assert np.allclose(choice.selected_test.sum(axis=1), 1.0)


def test_cluster_bootstrap_point_matches_incident_weighted_difference():
    rows = []
    for policy, cost_sums, tp_closes in (
        ("joint_robust_lambda_10", [10.0, 10.0], [0, 1]),
        ("nominal_optimizer", [0.0, 0.0], [1, 1]),
    ):
        for organization, queue_size, cost_sum, tp_close in zip(
            ["a", "b"], [1, 9], cost_sums, tp_closes
        ):
            rows.append(
                {
                    "organization": organization,
                    "policy": policy,
                    "queue_size": queue_size,
                    "realized_cost_sum": cost_sum,
                    "realized_cost_mean": cost_sum / queue_size,
                    "regret_sum": cost_sum,
                    "regret_mean": cost_sum / queue_size,
                    "tp_close_count": tp_close,
                    "tp_count": 1,
                    "non_fp_close_count": tp_close,
                    "non_fp_count": 1,
                }
            )
    results = guide.paired_cluster_bootstrap(
        pd.DataFrame(rows),
        "joint_robust_lambda_10",
        "nominal_optimizer",
        repetitions=50,
        seed=2026,
    )
    by_metric = {row["metric"]: row for row in results}
    assert np.isclose(by_metric["incident_weighted_cost"]["mean_difference"], 2.0)
    assert np.isclose(by_metric["tp_close_rate"]["mean_difference"], -0.5)


def test_policy_grid_records_corrected_uncertainty_and_probability_variant():
    frame = pd.DataFrame(
        {
            "OrgId": ["a", "a", "a", "b", "b", "b"],
            "IncidentGrade": [
                "TruePositive",
                "BenignPositive",
                "FalsePositive",
                "TruePositive",
                "BenignPositive",
                "FalsePositive",
            ],
            "alert_count": [1, 2, 1, 1, 2, 1],
            "first_timestamp": pd.date_range("2025-01-01", periods=6, freq="h"),
            "QueueRank": [1, 2, 3, 1, 2, 3],
        }
    )
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    _, aggregate, _ = guide.evaluate_policy_grid(
        frame,
        probabilities,
        priority=np.linspace(0.0, 1.0, len(frame)),
        scope="ranked",
        budget_values=[1],
        exact_budgets=True,
        priority_lambdas=[10.0],
        cost_scenario="medium",
        tv_radius=0.025,
        automation_cost_width=0.25,
        close_threshold=0.10,
        closure_safety="tp",
        seed=2026,
        probability_variant="raw",
    )
    robust = aggregate[aggregate["policy"].eq("robust_no_priority")].iloc[0]
    nominal = aggregate[aggregate["policy"].eq("nominal_optimizer")].iloc[0]
    assert robust["automation_cost_width"] == 0.25
    assert nominal["automation_cost_width"] == 0.0
    assert aggregate["probability_variant"].eq("raw").all()
    assert aggregate["reviews"].eq(2).all()


def test_oracle_cost_is_a_lower_bound_under_same_capacity():
    labels = np.array([0, 1, 2, 0], dtype=np.int8)
    costs = guide.COST_SCENARIOS["medium"]
    oracle, reviews = guide.oracle_cost_for_queue(labels, costs, review_budget=2)
    arbitrary_actions = np.array(
        [guide.DEFER, guide.CLOSE, guide.REVIEW, guide.CLOSE], dtype=np.int8
    )
    observed = float(costs[arbitrary_actions, labels].sum())
    assert oracle <= observed
    assert reviews <= 2


def test_deployment_cost_width_changes_only_automatic_actions():
    base = guide.COST_SCENARIOS["medium"]
    deployed = guide.deployment_cost_matrix(base, 0.50)
    assert np.allclose(deployed[:2], base[:2] * 1.50)
    assert np.allclose(deployed[guide.REVIEW], base[guide.REVIEW])


def test_global_threshold_is_fit_without_test_data_and_may_violate_queue_cap():
    calibration_probabilities = np.array(
        [[0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.2, 0.3, 0.5], [0.1, 0.2, 0.7]]
    )
    thresholds = guide.fit_global_tp_thresholds(calibration_probabilities, [0.5])
    assert np.isfinite(thresholds[0.5])

    frame = pd.DataFrame(
        {
            "OrgId": ["a", "a", "b", "b"],
            "IncidentGrade": ["TruePositive", "FalsePositive"] * 2,
            "alert_count": [1, 1, 1, 1],
            "first_timestamp": pd.date_range("2025-01-01", periods=4, freq="h"),
        }
    )
    probabilities = np.array(
        [[0.95, 0.03, 0.02], [0.90, 0.05, 0.05], [0.85, 0.10, 0.05], [0.80, 0.10, 0.10]]
    )
    _, aggregate, _ = guide.evaluate_global_threshold_baseline(
        frame,
        probabilities,
        priority=np.zeros(4),
        scope="test",
        budget_values=[0.5],
        exact_budgets=False,
        thresholds=thresholds,
        cost_scenario="medium",
        tv_radius=0.025,
        automation_cost_width=0.25,
        close_threshold=0.10,
        closure_safety="tp",
        seed=2026,
        probability_variant="raw",
    )
    assert aggregate.iloc[0]["capacity_violation_queues"] == 2


def test_holm_adjustment_is_monotone_and_not_smaller_than_raw_p():
    frame = pd.DataFrame(
        {
            "family": ["a", "a", "a"],
            "bootstrap_p_two_sided": [0.01, 0.03, 0.20],
        }
    )
    adjusted = guide.add_holm_adjustment(frame, ["family"])
    assert np.all(adjusted["holm_adjusted_p"] >= adjusted["bootstrap_p_two_sided"])
    ordered = adjusted.sort_values("bootstrap_p_two_sided")
    assert np.all(np.diff(ordered["holm_adjusted_p"]) >= -1e-12)


def test_worst_case_bootstrap_reports_macro_and_weighted_estimands_separately():
    rows = []
    for policy, means in (
        ("joint_robust_lambda_10", [1.0, 0.0]),
        ("nominal_optimizer", [0.4, 0.4]),
    ):
        for organization, queue_size, mean in zip(["a", "b"], [1, 9], means):
            rows.append(
                {
                    "organization": organization,
                    "policy": policy,
                    "deployment_cost_scenario": "medium",
                    "deployment_automation_cost_width": 0.25,
                    "queue_size": queue_size,
                    "normalized_realized_cost_sum": mean * queue_size,
                    "normalized_realized_cost_mean": mean,
                    "normalized_regret_sum": mean * queue_size,
                    "normalized_regret_mean": mean,
                }
            )
    result = guide.paired_worst_case_stress_bootstrap(
        pd.DataFrame(rows),
        "joint_robust_lambda_10",
        "nominal_optimizer",
        repetitions=100,
        seed=2026,
    )
    by_metric = {row["metric"]: row for row in result}
    assert np.isclose(
        by_metric["worst_case_macro_queue_normalized_cost"]["mean_difference"],
        0.1,
    )
    assert np.isclose(
        by_metric["worst_case_incident_weighted_normalized_cost"][
            "mean_difference"
        ],
        -0.3,
    )
