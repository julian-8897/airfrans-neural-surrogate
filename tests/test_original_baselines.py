import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest

from airfoil_surrogate.data.staged_original import case_order_sha256
from airfoil_surrogate.eval.freeze_original import freeze_evaluation
from airfoil_surrogate.train_original import OriginalTrainConfig


def test_ridge_is_family_specific_and_leaves_intercept_unpenalised():
    module = pytest.importorskip("airfoil_surrogate.eval.original_baselines")
    four = "airFoil2D_SST_40_2_1_4_11"
    five = "airFoil2D_SST_40_2_2_3_1_11"
    four_family, four_inputs = module.naca_design(four)
    five_family, five_inputs = module.naca_design(five)
    assert (four_family, five_family) == ("naca4", "naca5")
    assert four_inputs.tolist() == [1, 4, 11, 40 / module.AIRFRANS_VISCOSITY, 2]
    assert five_inputs.tolist() == [2, 3, 1, 11, 40 / module.AIRFRANS_VISCOSITY, 2]

    lower = four_inputs.copy()
    lower[0] -= 2
    four_model = module.CoefficientRidge.fit(
        np.stack([lower, four_inputs]), np.array([[2.0, 1.0], [6.0, 1.0]])
    )
    np.testing.assert_array_equal(
        four_model.input_mask, [True, False, False, False, False]
    )
    np.testing.assert_array_equal(four_model.term_mask, [True, False])
    # The varying input standardises to (-1, 1). With unit ridge the slope is 4/3.
    np.testing.assert_allclose(four_model.intercept, [4, 1])
    np.testing.assert_allclose(four_model.predict(lower), [4 - 4 / 3, 1])
    quadratic = module.CoefficientRidge.fit(
        np.array([[-1.0], [0], [1]]), np.array([[1.0], [0], [1]])
    )
    # The centred quadratic column (1/2, -1, 1/2) has squared norm 3/2.
    np.testing.assert_allclose(quadratic.weights[:, 0], [0, 2 / 5], atol=1e-15)
    np.testing.assert_allclose(quadratic.predict(np.array([1.0])), [13 / 15])
    five_model = module.CoefficientRidge.fit(five_inputs[None], np.array([[9.0, 0.2]]))
    field_means = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    baseline = module.OriginalBaselines(
        field_means, {"naca4": four_model, "naca5": five_model}
    )
    restored = module.OriginalBaselines.from_dict(
        json.loads(json.dumps(baseline.to_dict()))
    )
    np.testing.assert_allclose(restored.predict_coefficients(four), [4 + 4 / 3, 1])
    np.testing.assert_allclose(restored.predict_coefficients(five), [9, 0.2])
    np.testing.assert_array_equal(
        restored.predict_fields(np.array([True, False, True])), field_means[[1, 0, 1]]
    )


def test_freeze_binds_pinned_training_artifacts_and_rejects_a_later_tie(tmp_path):
    fit_cases = ["airFoil2D_SST_40_2_1_4_11", "airFoil2D_SST_40_2_2_3_1_11"]
    pin = {
        "partitions": {
            "train": {"size": 2, "order_sha256": case_order_sha256(fit_cases)}
        }
    }
    fitting = {"fit_cases": fit_cases, "fit_stage_sha256": "training-stage-digest"}
    training = tmp_path / "training"
    training.mkdir()
    (training / "model.pt").write_bytes(b"checkpoint")
    normaliser = training / "normaliser.json"
    normaliser.write_text(json.dumps(fitting))
    baselines = tmp_path / "baselines.json"
    baselines.write_text(json.dumps(fitting))

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    config = OriginalTrainConfig(epochs=3, val_every=2)
    metadata = {
        "inputs": {
            "train_stage_sha256": fitting["fit_stage_sha256"],
            "qualification": {
                "parity_sha256": "parity",
                "preflight_sha256": "preflight",
                "force_audit_sha256": "force-audit",
            },
        },
        "normaliser_sha256": digest(normaliser),
        "config": asdict(config),
    }
    run_file = training / "run.json"
    run_file.write_text(json.dumps(metadata))
    metrics = [
        {"epoch": 1},
        {"epoch": 2, "validation_score": 1.0},
        {"epoch": 3, "validation_score": 1.0},
    ]
    (training / "metrics.jsonl").write_text("\n".join(map(json.dumps, metrics)))
    selection = {
        "epoch": 2,
        "score": 1.0,
        "checkpoint_sha256": digest(training / "model.pt"),
        "validation_scores": [{"epoch": i, "score": 1.0} for i in (2, 3)],
    }
    selection_file = training / "selection.json"
    selection_file.write_text(json.dumps(selection))
    frozen = tmp_path / "frozen.json"
    freeze_evaluation(training, baselines, pin, frozen, config=config)
    initial = json.loads(frozen.read_text())
    assert initial["selected"] == {"epoch": 2, "score": 1.0}
    assert initial["sha256"]["model.pt"] == digest(training / "model.pt")
    assert initial["sha256"]["baselines"] == digest(baselines)
    assert initial["sha256"]["selection.json"] == digest(selection_file)

    with pytest.raises(FileExistsError):
        freeze_evaluation(training, baselines, pin, frozen, config=config)
    selection_file.write_text(json.dumps(selection | {"epoch": 3}))
    retry = tmp_path / "retry.json"
    with pytest.raises(ValueError, match="recorded validation rule"):
        freeze_evaluation(training, baselines, pin, retry, config=config)

    selection_file.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="complete prespecified training run"):
        freeze_evaluation(training, baselines, pin, retry, config=OriginalTrainConfig())
    metadata["inputs"]["timing_subset"] = fit_cases
    run_file.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="complete prespecified training run"):
        freeze_evaluation(training, baselines, pin, retry, config=config)
    del metadata["inputs"]["timing_subset"]
    proof = metadata["inputs"].pop("qualification")
    run_file.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="provenance from the training qualification"):
        freeze_evaluation(training, baselines, pin, retry, config=config)
    metadata["inputs"]["qualification"] = proof
    run_file.write_text(json.dumps(metadata))
    baselines.write_text(json.dumps(fitting | {"fit_cases": fit_cases[::-1]}))
    with pytest.raises(ValueError, match="pinned training stage"):
        freeze_evaluation(training, baselines, pin, retry, config=config)
    assert not retry.exists()
