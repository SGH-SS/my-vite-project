from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, conint, confloat


class ParamSpec(BaseModel):
    type: Literal['int', 'float', 'categorical']
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[str]] = None


class HpSpace(BaseModel):
    params: Dict[str, ParamSpec]


class SimilarityGuard(BaseModel):
    mode: Literal['corr_threshold'] = 'corr_threshold'
    threshold: confloat(ge=0.0, le=1.0)
    reference_nodes: List[str] = Field(default_factory=list)


class DataWindow(BaseModel):
    lookback_years: conint(ge=1) = 5
    test_window_days: conint(ge=1) = 35


class ValidationHoldout(BaseModel):
    method: Literal['holdout_time_ordered']
    train_ratio: confloat(gt=0.0, lt=1.0)
    val_ratio: confloat(gt=0.0, lt=1.0)


class ValidationPurgedCV(BaseModel):
    method: Literal['purged_time_series_cv']
    n_splits: conint(ge=2, le=20) = 5
    gap_days: conint(ge=0, le=30) = 5
    recency_weighting: Optional[Dict[str, Optional[confloat(gt=0)]]] = None  # {'enabled': True, 'tau_bars': 180}


class ThresholdGrid(BaseModel):
    metric: Literal['accuracy']
    low: confloat(ge=0.0, le=1.0)
    high: confloat(ge=0.0, le=1.0)
    step: confloat(gt=0.0, le=1.0)


class ThresholdSelection(BaseModel):
    metric: Literal['F1']
    aggregation: Literal['median_across_folds']
    low: confloat(ge=0.0, le=1.0)
    high: confloat(ge=0.0, le=1.0)
    step: confloat(gt=0.0, le=1.0)


class Costs(BaseModel):
    bps_each_way: confloat(ge=0.0, le=50.0) = 1.5
    borrow_bps_per_day: confloat(ge=0.0, le=10.0) = 0.0


class Tests(BaseModel):
    spa: bool = False
    mcs: bool = False


class EvalProtocol(BaseModel):
    data_window: DataWindow
    validation: Optional[ValidationHoldout | ValidationPurgedCV] = None
    threshold_grid: Optional[ThresholdGrid] = None
    threshold_selection: Optional[ThresholdSelection] = None
    costs: Costs
    tests: Tests
    turnover_penalty: confloat(ge=0.0, le=0.05) = 0.0
    slippage_model: Literal['bps_linear']


class RegistryTags(BaseModel):
    family_bucket: Optional[str] = None
    feature_bucket: Optional[str] = None
    validation_style: Optional[str] = None
    thresholding: Optional[str] = None
    early_stopping: Optional[str] = None
    class_weight: Optional[str] = None
    notes: Optional[str] = None


class LightGBMSpec(BaseModel):
    node_id: str
    family: Literal['LightGBMClassifier']
    target: str
    featureset: List[str]
    description: str

    hp_space: HpSpace

    change_budget_per_loop: conint(ge=0) = 1
    trials_budget_per_loop: conint(ge=0) = 0
    time_budget_minutes: conint(ge=1, le=720) = 30

    similarity_guard: SimilarityGuard
    eval_protocol: EvalProtocol
    registry_tags: Optional[RegistryTags] = None


