"""Class-name resolution for predictions (crash found in manual use).

A model trained in-app is made the active prediction model with no separate
load step -- but ``class_names`` was only ever populated by *loading* a model
through its yaml, so straight after training it was still None. The first
prediction then did ``class_names[class_id]`` and raised "'NoneType' object is
not subscriptable", which the caller reported as a probable mismatch between
the model and the YAML file classes: the one explanation that was certainly
wrong, since no YAML was involved.

The model always knows its own names. That is the fallback.
"""

import pytest

from src.digitalsreeni_image_annotator.dialogs.yolo_trainer import YOLOTrainer


def _trainer(class_names=None, model_names=None):
    trainer = YOLOTrainer.__new__(YOLOTrainer)
    trainer.class_names = class_names
    trainer.model = type("M", (), {"names": model_names})() if model_names else None
    return trainer


def test_a_freshly_trained_model_resolves_from_its_own_names():
    """The crash: nothing loaded, so class_names is None."""
    trainer = _trainer(class_names=None, model_names={0: "bee"})
    assert trainer.class_name_for(0) == "bee"


def test_loaded_names_win_over_the_models_own():
    trainer = _trainer(class_names={0: "from-yaml"}, model_names={0: "from-model"})
    assert trainer.class_name_for(0) == "from-yaml"


def test_a_list_of_names_works_too():
    """A hand-written yaml may carry `names: [bee]` rather than a mapping."""
    assert _trainer(class_names=["bee", "wasp"]).class_name_for(1) == "wasp"


def test_an_unknown_index_raises_IndexError_not_KeyError():
    """Both callers catch IndexError to report a genuine class mismatch; a
    dict's native KeyError would sail past them and reach the user as an
    unhandled crash instead."""
    trainer = _trainer(class_names={0: "bee"})
    with pytest.raises(IndexError):
        trainer.class_name_for(7)


def test_no_names_anywhere_raises_rather_than_returning_none():
    trainer = _trainer(class_names=None, model_names=None)
    with pytest.raises(IndexError):
        trainer.class_name_for(0)
