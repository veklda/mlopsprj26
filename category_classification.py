"""
Shared utilities for the category classification FTI pipeline.

Extracted from:
  - category_classification_f.ipynb  (Feature Pipeline)
  - category_classification_t.ipynb  (Training Pipeline)
  - category_classification_i.ipynb  (Inference Pipeline)

This module provides helper functions and classes used across the Feature,
Training, and Inference (FTI) stages of the category classification workflow.
"""
from __future__ import annotations

# Standard library imports for parsing, I/O, and type hints.
import ast
import io
from pathlib import Path
from typing import TYPE_CHECKING

# Third-party scientific computing and ML libraries.
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin

if TYPE_CHECKING:
    import pyarrow as pa


# ── Distribution helpers ───────────────────────────────────────────────────────

def make_dist(d):
    """
    Convert a TOML distribution dict → scipy distribution object.

    Parameters
    ----------
    d : dict
        TOML distribution specification with a "type" key (randint, loguniform,
        or uniform) plus the corresponding distribution parameters.

    Returns
    -------
    scipy.stats distribution
        A frozen scipy distribution ready for use in hyper-parameter search.
    """
    from scipy.stats import loguniform, randint, uniform

    # Make a shallow copy so we can mutate without side effects.
    d = dict(d)
    # Pop the distribution type so only parameters remain.
    t = d.pop("type")

    # Dispatch to the matching scipy distribution constructor.
    if t == "randint":
        return randint(**d)
    if t == "loguniform":
        return loguniform(d["low"], d["high"])
    if t == "uniform":
        low, high = d["low"], d["high"]
        return uniform(loc=low, scale=high - low)

    # Reject unsupported distribution types early.
    raise ValueError(f"Unknown distribution type: {t}")


def tfidf_kw(d: dict) -> dict:
    """
    Normalise a TOML TF-IDF param dict: convert ngram_range list → tuple.

    Parameters
    ----------
    d : dict
        Raw parameter dictionary, typically loaded from TOML.

    Returns
    -------
    dict
        A new dict where list-valued ``ngram_range`` has been replaced by a
        tuple (required by scikit-learn vectorizers).
    """
    d = dict(d)
    # scikit-learn expects ngram_range as a tuple, but TOML stores lists.
    if isinstance(d.get("ngram_range"), list):
        d["ngram_range"] = tuple(d["ngram_range"])
    return d


# ── LightGBM device detection ──────────────────────────────────────────────────

def detect_lgb_device() -> str:
    """
    Return 'cuda', 'gpu', or 'cpu' depending on what LightGBM can use.

    This performs a tiny training smoke-test for each accelerator backend and
    falls back to CPU if none succeed.

    Returns
    -------
    str
        One of "cuda", "gpu", or "cpu".
    """
    import lightgbm as _lgb

    # Dummy data just large enough for LightGBM to accept.
    _X, _y = np.zeros((10, 2)), np.zeros(10)

    # Try the fastest backends first; silence output with verbose=-1.
    for dev in ("cuda", "gpu"):
        try:
            _lgb.train(
                {"device": dev, "num_leaves": 4, "verbose": -1},
                _lgb.Dataset(_X, _y),
                num_boost_round=1,
            )
            return dev
        except Exception:
            # If the backend is unavailable, continue to the next candidate.
            pass

    return "cpu"


# ── PyArrow schema helpers ─────────────────────────────────────────────────────

def unify_arrow_type(t1, t2):
    """
    Return the broadest PyArrow type that can hold both t1 and t2.

    Parameters
    ----------
    t1, t2 : pyarrow.DataType
        Two PyArrow types to reconcile.

    Returns
    -------
    pyarrow.DataType
        The wider of the two types, or pa.string() when no common numeric
        supertype exists.
    """
    import pyarrow as pa

    # Identical types need no widening.
    if t1 == t2:
        return t1
    # A null column safely adopts the other column's type.
    if pa.types.is_null(t1):
        return t2
    if pa.types.is_null(t2):
        return t1
    # Float beats integer when mixed.
    if pa.types.is_integer(t1) and pa.types.is_floating(t2):
        return t2
    if pa.types.is_floating(t1) and pa.types.is_integer(t2):
        return t1
    # Fallback: string can represent (almost) anything.
    return pa.string()


def merge_arrow_schema(base, new):
    """
    Merge two PyArrow schemas, widening column types where they differ.

    Columns present in *new* but not in *base* are included.

    Parameters
    ----------
    base, new : pyarrow.Schema
        The existing schema and the incoming schema to merge.

    Returns
    -------
    pyarrow.Schema
        A unified schema containing the union of columns, with types widened
        via :func:`unify_arrow_type` when both schemas define a column.
    """
    import pyarrow as pa

    # Index fields by name for O(1) lookup.
    base_map = {f.name: f for f in base}
    new_map = {f.name: f for f in new}

    # Preserve base column order first, then append new columns.
    all_names = sorted(
        set(base_map) | set(new_map),
        key=lambda n: (n not in base_map, n),
    )

    # Build the unified schema field by field.
    return pa.schema([
        pa.field(
            name,
            unify_arrow_type(base_map[name].type, new_map[name].type)
            if name in base_map and name in new_map
            else (base_map[name].type if name in base_map else new_map[name].type)
        )
        for name in all_names
    ])


# ── Feature value parsers ──────────────────────────────────────────────────────

def parse_first(val):
    """
    Return the first item of a stringified Python list, or NaN.

    Parameters
    ----------
    val : any
        A value that may be a string representation of a list.

    Returns
    -------
    any
        The stripped first element, or ``np.nan`` on failure.
    """
    # Missing values propagate as NaN.
    if pd.isna(val):
        return np.nan
    try:
        # Safely evaluate the string as a Python literal.
        items = ast.literal_eval(str(val))
        # Ensure we have a non-empty list before indexing.
        if isinstance(items, list) and items:
            return items[0].strip()
    except (ValueError, SyntaxError):
        # Malformed strings are treated as missing.
        pass
    return np.nan


def parse_count(val) -> int:
    """
    Return the length of a stringified Python list, or 0.

    Parameters
    ----------
    val : any
        A value that may be a string representation of a list.

    Returns
    -------
    int
        The list length, or 0 on failure / missing input.
    """
    if pd.isna(val):
        return 0
    try:
        items = ast.literal_eval(str(val))
        if isinstance(items, list):
            return len(items)
    except (ValueError, SyntaxError):
        pass
    return 0


# ── Rolling platform feature worker ───────────────────────────────────────────

def fp3_platform_worker(plat_grp, win_ns: np.int64, test_history: str = "train_only"):
    """
    Compute 3-hour rolling volume, automation fraction, and content-type diversity
    for one platform group.

    Parameters
    ----------
    plat_grp : tuple (platform_name, group_df)
        group_df must have columns: _ts, _auto, _ctp, _istr (bool), _pos (int64).
    win_ns : np.int64
        Window width in nanoseconds (e.g. 3 * 3600 * 1_000_000_000).
    test_history : str
        "train_only" — test rows look up windows against train timestamps only.
        "all_prior"  — test rows look up windows against train + earlier-test
                       timestamps, simulating what the model would see at
                       deployment time.

    Returns
    -------
    tuple
        8-tuple: (tr_pos, vol_tr, af_tr, div_tr, te_pos, vol_te, af_te, div_te)
    """
    # Unpack the group tuple; the platform name is not used directly here.
    _, _grp = plat_grp

    # Boolean mask separating train and test rows within this platform.
    _tr_mask = _grp["_istr"].values

    # Split and sort each partition by timestamp to prepare for windowing.
    _gr_tr = _grp[_tr_mask].sort_values("_ts")
    _gr_te = _grp[~_tr_mask].sort_values("_ts")

    # Remember original row positions so results can be aligned later.
    _tr_pos = _gr_tr["_pos"].values.astype(np.int64)
    _te_pos = _gr_te["_pos"].values.astype(np.int64) if len(_gr_te) else np.array([], np.int64)

    # Cache partition sizes for quick empty checks.
    _n_tr = len(_gr_tr)
    _n_te = len(_te_pos)

    # Without training data there is no history to compute rolling statistics.
    if not _n_tr:
        return (
            _tr_pos,
            np.zeros(0, np.float32), np.zeros(0, np.float32), np.ones(0, np.float32),
            _te_pos,
            np.zeros(_n_te, np.float32), np.zeros(_n_te, np.float32), np.ones(_n_te, np.float32),
        )

    # Cast timestamps and automation flags to numeric arrays.
    _tr_ts_ns = _gr_tr["_ts"].values.astype(np.int64)
    _tr_auto = _gr_tr["_auto"].values.astype(np.float32)

    # Build a cumulative sum of automation flags; prepend 0 for easy interval diffs.
    _auto_cs = np.empty(_n_tr + 1, np.float32)
    _auto_cs[0] = 0.0
    np.cumsum(_tr_auto, out=_auto_cs[1:])

    # Encode content types as integer codes and build a one-hot cumulative matrix.
    _, _ctp_codes = np.unique(_gr_tr["_ctp"].values, return_inverse=True)
    _n_cats = int(_ctp_codes.max()) + 1
    _onehot = (
        _ctp_codes[np.newaxis, :] == np.arange(_n_cats, dtype=np.int32)[:, np.newaxis]
    ).astype(np.int32)
    # Cumulative counts per category; shape (n_cats, n_tr + 1).
    _cat_cs = np.hstack([np.zeros((_n_cats, 1), np.int32), np.cumsum(_onehot, axis=1)])
    del _onehot

    # Train rows: window is [lo, i) — closed on the left, open on the right.
    _lo_tr = np.searchsorted(_tr_ts_ns, _tr_ts_ns - win_ns, side="left").astype(np.int64)
    _hi_tr = np.arange(_n_tr, dtype=np.int64)
    _vol_tr = (_hi_tr - _lo_tr).astype(np.float32)
    # Automation fraction inside each window.
    _af_tr = np.where(
        _vol_tr > 0,
        (_auto_cs[_hi_tr] - _auto_cs[_lo_tr]) / _vol_tr,
        np.float32(0),
    )
    # Diversity = number of distinct content types in the window.
    _wc_tr = _cat_cs[:, _hi_tr] - _cat_cs[:, _lo_tr]
    _div_tr = np.where(
        _vol_tr > 0,
        (_wc_tr > 0).sum(axis=0).astype(np.float32),
        np.float32(1.0),
    )

    # If there are no test rows, return the train results early.
    if not _n_te:
        del _cat_cs, _auto_cs
        return (
            _tr_pos, _vol_tr, _af_tr, _div_tr,
            _te_pos,
            np.array([], np.float32), np.array([], np.float32), np.array([], np.float32),
        )

    # Cast test timestamps for use in window lookups.
    _te_ts_ns = _gr_te["_ts"].values.astype(np.int64)

    if test_history == "train_only":
        # Test rows: look up against sorted train timestamps to prevent leakage.
        _hi_te = np.searchsorted(_tr_ts_ns, _te_ts_ns, side="left").astype(np.int64)
        _lo_te = np.searchsorted(_tr_ts_ns, _te_ts_ns - win_ns, side="left").astype(np.int64)
        _vol_te = (_hi_te - _lo_te).astype(np.float32)
        _af_te = np.where(
            _vol_te > 0,
            (_auto_cs[_hi_te] - _auto_cs[_lo_te]) / _vol_te,
            np.float32(0),
        )
        _wc_te = _cat_cs[:, _hi_te] - _cat_cs[:, _lo_te]
        _div_te = np.where(
            _vol_te > 0,
            (_wc_te > 0).sum(axis=0).astype(np.float32),
            np.float32(1.0),
        )
    else:
        # "all_prior": test rows see train + prior test rows (deployment simulation).
        _te_auto = _gr_te["_auto"].values.astype(np.float32)
        _tr_ctp = _gr_tr["_ctp"].values
        _te_ctp = _gr_te["_ctp"].values

        # Concatenate train and test into a single chronologically ordered array.
        _combo_ts = np.concatenate([_tr_ts_ns, _te_ts_ns])
        _combo_auto = np.concatenate([_tr_auto, _te_auto])
        _combo_ctp = np.concatenate([_tr_ctp, _te_ctp])

        # Stable sort preserves original order for ties (train before test).
        _sort = np.argsort(_combo_ts, kind="stable")
        _combo_ts = _combo_ts[_sort]
        _combo_auto = _combo_auto[_sort]
        _combo_ctp = _combo_ctp[_sort]
        _combo_n = len(_combo_ts)

        # Re-encode content-type codes on the combined array.
        _, _combo_ctp_codes = np.unique(_combo_ctp, return_inverse=True)
        _combo_n_cats = int(_combo_ctp_codes.max()) + 1
        _combo_onehot = (
            _combo_ctp_codes[np.newaxis, :]
            == np.arange(_combo_n_cats, dtype=np.int32)[:, np.newaxis]
        ).astype(np.int32)
        _combo_cat_cs = np.hstack([
            np.zeros((_combo_n_cats, 1), np.int32),
            np.cumsum(_combo_onehot, axis=1),
        ])
        del _combo_onehot

        # Cumulative automation on the combined timeline.
        _combo_auto_cs = np.empty(_combo_n + 1, np.float32)
        _combo_auto_cs[0] = 0.0
        np.cumsum(_combo_auto, out=_combo_auto_cs[1:])

        # True sorted positions of test rows (original indices _n_tr .. _n_tr+_n_te-1).
        # np.searchsorted(..., side="left") is wrong for duplicate timestamps because
        # the stable sort places train rows before test rows for ties.
        _inv_sort = np.empty(_combo_n, dtype=np.int64)
        _inv_sort[_sort] = np.arange(_combo_n, dtype=np.int64)
        _hi_te = _inv_sort[_n_tr + np.arange(_n_te, dtype=np.int64)]
        _lo_te = np.searchsorted(_combo_ts, _te_ts_ns - win_ns, side="left").astype(np.int64)
        _vol_te = (_hi_te - _lo_te).astype(np.float32)
        _af_te = np.where(
            _vol_te > 0,
            (_combo_auto_cs[_hi_te] - _combo_auto_cs[_lo_te]) / _vol_te,
            np.float32(0),
        )
        _wc_te = _combo_cat_cs[:, _hi_te] - _combo_cat_cs[:, _lo_te]
        _div_te = np.where(
            _vol_te > 0,
            (_wc_te > 0).sum(axis=0).astype(np.float32),
            np.float32(1.0),
        )

        del _combo_ts, _combo_auto, _combo_ctp, _combo_auto_cs, _combo_cat_cs

    del _cat_cs, _auto_cs
    return (_tr_pos, _vol_tr, _af_tr, _div_tr, _te_pos, _vol_te, _af_te, _div_te)


def read_data_file(args, do_sample: bool = False, stride: int = 1):
    """
    Read a single parquet or CSV file, optionally stride-sampling.

    Parameters
    ----------
    args : tuple (index, file_path_str)
        Index used for ordering results and the path to the file.
    do_sample : bool, optional
        Whether to apply stride-based sub-sampling after loading.
    stride : int, optional
        Take every ``stride``-th row when ``do_sample`` is True.

    Returns
    -------
    tuple
        (index, pa.Table) preserving the input index for sort-merge.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.csv as pacsv

    # Unpack the work item.
    idx, fpath = args
    fp = Path(fpath)

    # Dispatch to the correct PyArrow reader based on file extension.
    if fp.suffix.lower() == ".parquet":
        tbl = pq.read_table(fp)
    elif str(fp).lower().endswith(".csv.gz"):
        import gzip
        with gzip.open(fp, "rt") as gz_f:
            tbl = pacsv.read_csv(gz_f)
    else:
        tbl = pacsv.read_csv(fp)

    # Optionally down-sample by taking every stride-th row.
    if do_sample and stride > 1:
        tbl = tbl.take(pa.array(range(0, len(tbl), stride)))

    return idx, tbl


# ── TwoStageClassifier ─────────────────────────────────────────────────────────

class TwoStageClassifier(BaseEstimator, ClassifierMixin):
    """
    Stage 1: binary (0=OTHER, 1=NOT-OTHER).
    Stage 2: multiclass on NOT-OTHER rows only.
    Probability stitching preserves the sklearn predict_proba contract.
    """

    def __init__(self, stage1, stage2, other_label):
        self.stage1 = stage1
        self.stage2 = stage2
        self.other_label = other_label

    def fit(self, X, y):
        """
        Fit the two-stage pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self
        """
        y = np.asarray(y)

        # Stage 1 learns to distinguish OTHER vs. everything else.
        self.stage1.fit(X, (y != self.other_label).astype(int))

        # Stage 2 is trained only on rows that are NOT-OTHER.
        mask = y != self.other_label
        Xm = X[mask] if not sp.issparse(X) else X[mask]
        self.stage2.fit(Xm, y[mask])

        # Store the full label space for probability alignment later.
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class label for each sample.
        """
        s1 = self.stage1.predict(X)
        n = X.shape[0]

        # Default every row to OTHER; overwrite where stage1 says NOT-OTHER.
        out = np.full(n, self.other_label, dtype=int)
        nm = s1 == 1
        if nm.any():
            out[nm] = self.stage2.predict(X[nm])
        return out

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Combines stage-1 probabilities (OTHER vs. NOT-OTHER) with stage-2
        class probabilities inside the NOT-OTHER branch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Probability estimates aligned to ``self.classes_``.
        """
        n = X.shape[0]
        nc = len(self.classes_)
        proba = np.zeros((n, nc))

        # Stage-1 probabilities: column 0 = OTHER, column 1 = NOT-OTHER.
        s1p = self.stage1.predict_proba(X)
        p_other = s1p[:, 0]
        p_not = s1p[:, 1]

        # Place the OTHER probability at the correct column index.
        other_pos = int(np.searchsorted(self.classes_, self.other_label))
        proba[:, other_pos] = p_other

        # Stage-2 probabilities are weighted by the NOT-OTHER probability mass.
        s2p = self.stage2.predict_proba(X)
        for j, cls in enumerate(self.stage2.classes_):
            proba[:, int(np.searchsorted(self.classes_, cls))] = p_not * s2p[:, j]

        return proba


# ── Training helpers ───────────────────────────────────────────────────────────

def evaluate(name, y_true, y_pred, tier, approach, results: dict | None = None,
             cv_scores=None):
    """
    Compute macro F1 and balanced accuracy, record in *results*.

    Parameters
    ----------
    name : str
        Model / experiment identifier used as the key in *results*.
    y_true, y_pred : array-like
        Ground-truth and predicted labels.
    tier : int or str
        Pipeline tier label for logging.
    approach : str
        Approach name for logging.
    results : dict or None
        Mutable dict to update in-place (keyed by model name).
    cv_scores : array-like or None
        Optional cross-validation F1 scores to include in the printed summary.

    Returns
    -------
    tuple
        (macro_f1, bal_acc)
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score

    # Macro F1 averages per-class F1, treating all classes equally.
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    # Balanced accuracy accounts for class imbalance.
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # Build the results entry.
    entry = {"macro_f1": macro_f1, "bal_acc": bal_acc, "tier": tier, "approach": approach}
    cv_str = ""
    if cv_scores is not None:
        entry["cv_f1_mean"] = float(cv_scores.mean())
        entry["cv_f1_std"] = float(cv_scores.std())
        cv_str = f"  cv={cv_scores.mean():.4f}±{cv_scores.std():.4f}"

    # Persist and print the metrics.
    if results is not None:
        results[name] = entry
    print(f"  {name:<45s}  macro F1={macro_f1:.4f}  bal_acc={bal_acc:.4f}{cv_str}")
    return macro_f1, bal_acc


def predict_dispatch(pipe, name: str, tier: int | str, X, X_t2=None):
    """
    Route predict / predict_proba to the right feature matrix based on tier.

    Parameters
    ----------
    pipe : estimator
        Fitted sklearn pipeline or model.
    name : str
        Model identifier (unused but kept for API consistency).
    tier : int or str
        If "2", use ``X_t2``; otherwise use ``X``.
    X, X_t2 : array-like
        Feature matrices for tier 1 and tier 2 respectively.

    Returns
    -------
    tuple
        (predictions, probabilities)
    """
    # Tier 2 models expect a differently structured feature matrix.
    if str(tier) == "2":
        return pipe.predict(X_t2), pipe.predict_proba(X_t2)
    return pipe.predict(X), pipe.predict_proba(X)


# ── Inference predictor ────────────────────────────────────────────────────────

class CategoryPredictor:
    """
    Wraps a loaded model artifact and config to provide feature engineering
    and scoring.

    Parameters
    ----------
    model_artifact : dict
        Pickle dict saved by _t.ipynb (keys: name, pipeline/model, target_labels, …).
    cfg : dict
        TOML config loaded by ``tomllib.load``.
    """

    def __init__(self, model_artifact: dict, cfg: dict):
        # Keep references to the serialized model and the feature-pipeline config.
        self.ma = model_artifact
        self.cfg = cfg
        self._fp = cfg["feature_pipeline"]

        # Unpack metadata stored at training time.
        self.target_labels = model_artifact["target_labels"]
        self.label_map_inv = model_artifact["label_map_inv"]
        self.feature_cols = model_artifact["FEATURE_COLS"]
        self.cat_features = model_artifact["CATEGORICAL_FEATURES"]
        self.num_features = model_artifact["NUMERICAL_FEATURES"]
        self.text_col = model_artifact["TEXT_COL"]
        self.text_col_2 = model_artifact.get("TEXT_COL_2", "incompatible_content_explanation")
        self.text_col_3 = model_artifact.get("TEXT_COL_3")
        self.preprocessor_t1 = model_artifact["preprocessor_t1"]
        self.preprocessor_t2 = model_artifact.get("preprocessor_t2")
        self.model_name = model_artifact["name"]
        self.tier = model_artifact.get("tier", 1)

        # Training-era per-platform automation rate — applied during inference
        # so the feature distribution matches what the model was trained on.
        self.platform_automation_rates = model_artifact.get(
            "platform_automation_rates", {}
        )

    # ── Feature engineering ────────────────────────────────────────────────────

    def engineer_features(self, df_raw: pd.DataFrame,
                          history_df: pd.DataFrame | None = None,
                          inplace: bool = False):
        """
        Apply the same feature engineering used during training.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw input data with optional columns such as ``content_type``,
            ``decision_visibility``, ``application_date``, etc.
        history_df : DataFrame | None
            Optional historical data to seed rolling-window features.
            Without history, the first rows of each platform in *df_raw*
            are scored with cold-start rolling values (0, 0, 1).
        inplace : bool
            If True, mutate *df_raw* in-place instead of copying.  Use only
            when *df_raw* is a temporary batch that will be discarded.

        Returns
        -------
        tuple
            (X, texts_icg, texts_ice, texts_df) where *X* is a DataFrame of
            ``feature_cols`` and the three Series are text columns for NLP.
        """
        df = df_raw if inplace else df_raw.copy()

        # ── Parse multi-value string columns into first-item and count features ──
        if "content_type" in df.columns:
            df["content_type_primary"] = df["content_type"].apply(parse_first)
            df["content_type_count"] = df["content_type"].apply(parse_count).astype(np.int32)
        if "territorial_scope" in df.columns:
            df["territorial_scope_count"] = df["territorial_scope"].apply(parse_count).astype(np.int32)
            df["territorial_scope_primary"] = df["territorial_scope"].apply(parse_first)
        if "decision_visibility" in df.columns:
            df["decision_visibility_primary"] = df["decision_visibility"].apply(parse_first)
            df["visibility_decision_count"] = df["decision_visibility"].apply(parse_count).astype(np.int32)

        # ── Binary flags for the presence of each decision type ──
        _vis_dec = (
            df["decision_visibility"].notna().astype(np.int8)
            if "decision_visibility" in df.columns
            else pd.Series(0, index=df.index, dtype=np.int8)
        )
        df["has_monetary_decision"] = (
            df["decision_monetary"].notna().astype(np.int8)
            if "decision_monetary" in df.columns
            else pd.Series(0, index=df.index, dtype=np.int8)
        )
        df["has_provision_decision"] = (
            df["decision_provision"].notna().astype(np.int8)
            if "decision_provision" in df.columns
            else pd.Series(0, index=df.index, dtype=np.int8)
        )
        df["has_account_decision"] = (
            df["decision_account"].notna().astype(np.int8)
            if "decision_account" in df.columns
            else pd.Series(0, index=df.index, dtype=np.int8)
        )

        # Aggregate decision actions and flag multi-action rows.
        df["total_decision_actions"] = (
            _vis_dec + df["has_monetary_decision"]
            + df["has_provision_decision"] + df["has_account_decision"]
        ).astype(np.int8)
        df["is_multi_action_decision"] = (df["total_decision_actions"] > 1).astype(np.int8)

        # ── Date-derived features ──
        app_dt = (pd.to_datetime(df["application_date"], errors="coerce")
                  if "application_date" in df.columns
                  else pd.Series(pd.NaT, index=df.index))
        cnt_dt = (pd.to_datetime(df["content_date"], errors="coerce")
                  if "content_date" in df.columns
                  else pd.Series(pd.NaT, index=df.index))
        df["app_year"] = app_dt.dt.year.astype("Int16")
        df["app_month"] = app_dt.dt.month.astype("Int8")
        df["content_year"] = cnt_dt.dt.year.astype("Int16")
        df["content_application_lag_days"] = (app_dt - cnt_dt).dt.days.astype("Int32")
        df["decision_facts_len"] = (
            df.get("decision_facts", pd.Series("", index=df.index)).fillna("").str.len()
        ).astype(np.int32)

        # ── Rolling platform features (3-hour window) ──
        _ts_col = "created_at" if "created_at" in df.columns else "application_date"
        _ts_vals = pd.to_datetime(
            df.get(_ts_col, pd.Series(pd.NaT, index=df.index)), errors="coerce"
        )
        # Determine whether timestamps have sub-day granularity; if not,
        # rolling windows would collapse to zero volume and are skipped.
        _has_subday = (
            len(df) > 1
            and _ts_vals.notna().any()
            and (_ts_vals.dropna().dt.time != pd.Timestamp("00:00:00").time()).any()
        )

        _win_hours = int(self._fp.get("rolling_window_hours", 3))
        df["platform_volume_3h"] = np.float32(0.0)
        df["platform_auto_fraction_3h"] = np.float32(0.0)
        df["platform_content_type_div_3h"] = np.float32(1.0)

        if _has_subday and "platform_name" in df.columns:
            # Assemble the rolling frame with the minimal columns required.
            _ctp = df.get("content_type_primary",
                          pd.Series("UNKNOWN", index=df.index)).fillna("UNKNOWN")
            _roll = pd.DataFrame({
                "platform_name": df["platform_name"].values,
                "_ts": _ts_vals.values,
                "_auto": (df.get("automated_detection", pd.Series("", index=df.index))
                          .fillna("").str.lower() == "yes").astype(float).values,
                "_ctp": _ctp.values,
            }, index=df.index)
            # Drop NaT rows so they don't corrupt the rolling computation.
            _roll = _roll[_roll["_ts"].notna()]

            # Build optional history frame for warm-starting rolling windows.
            _hist_roll = None
            if history_df is not None and "platform_name" in history_df.columns:
                _hist_ts = pd.to_datetime(
                    history_df.get(_ts_col,
                                   pd.Series(pd.NaT, index=history_df.index)),
                    errors="coerce",
                )
                if "content_type_primary" in history_df.columns:
                    _hist_ctp = history_df["content_type_primary"].fillna("UNKNOWN")
                elif "content_type" in history_df.columns:
                    _hist_ctp = history_df["content_type"].apply(parse_first).fillna("UNKNOWN")
                else:
                    _hist_ctp = pd.Series("UNKNOWN", index=history_df.index)
                _hist_roll = pd.DataFrame({
                    "platform_name": history_df["platform_name"].values,
                    "_ts": _hist_ts.values,
                    "_auto": (history_df.get("automated_detection",
                                              pd.Series("", index=history_df.index))
                              .fillna("").str.lower() == "yes").astype(float).values,
                    "_ctp": _hist_ctp.values,
                    "_is_batch": False,
                })
                _hist_roll = _hist_roll[_hist_roll["_ts"].notna()]

            # Convert window hours to nanoseconds for integer arithmetic.
            _win_ns = np.int64(_win_hours * 3600 * 1_000_000_000)
            for _plat, _grp in _roll.groupby("platform_name", sort=False):
                _grp = _grp.copy()
                _grp["_is_batch"] = True
                _s = _grp.sort_values("_ts")

                # Prepend historical rows (if any) to warm-start cumulative stats.
                if _hist_roll is not None:
                    _h = _hist_roll[_hist_roll["platform_name"] == _plat].sort_values("_ts")
                    if len(_h) > 0:
                        _s = pd.concat([_h, _s])

                _ts_ns = _s["_ts"].values.astype(np.int64)
                _n = len(_ts_ns)

                # Empty group after filtering should be skipped silently.
                if _n == 0:
                    continue

                # Cumulative automation flags for fast interval sums.
                _auto_vals = _s["_auto"].values.astype(np.float32)
                _auto_cs = np.empty(_n + 1, np.float32)
                _auto_cs[0] = 0.0
                np.cumsum(_auto_vals, out=_auto_cs[1:])

                # Content-type one-hot cumulative matrix.
                _, _ctp_codes = np.unique(_s["_ctp"].values, return_inverse=True)
                _n_cats = int(_ctp_codes.max()) + 1
                _onehot = (
                    _ctp_codes[np.newaxis, :]
                    == np.arange(_n_cats, dtype=np.int32)[:, np.newaxis]
                ).astype(np.int32)
                _cat_cs = np.hstack([
                    np.zeros((_n_cats, 1), np.int32),
                    np.cumsum(_onehot, axis=1),
                ])
                del _onehot

                # Vectorised window bounds via binary search on sorted timestamps.
                _lo = np.searchsorted(_ts_ns, _ts_ns - _win_ns, side="left").astype(np.int64)
                _hi = np.arange(_n, dtype=np.int64)
                _vol = (_hi - _lo).astype(np.float32)
                _af = np.where(
                    _vol > 0,
                    (_auto_cs[_hi] - _auto_cs[_lo]) / _vol,
                    np.float32(0),
                )
                _wc = _cat_cs[:, _hi] - _cat_cs[:, _lo]
                _div = np.where(
                    _vol > 0,
                    (_wc > 0).sum(axis=0).astype(np.float32),
                    np.float32(1.0),
                )

                # Write only the batch rows back to the output DataFrame.
                _batch_mask = _s["_is_batch"].values
                df.loc[_s.index[_batch_mask], "platform_volume_3h"] = _vol[_batch_mask]
                df.loc[_s.index[_batch_mask], "platform_auto_fraction_3h"] = _af[_batch_mask]
                df.loc[_s.index[_batch_mask], "platform_content_type_div_3h"] = _div[_batch_mask]

                del _cat_cs, _auto_cs

        # Use the training-era per-platform automation rate so the
        # feature distribution matches what the model was trained on.
        # Falls back to 0.0 for platforms not seen during training.
        df["platform_automation_rate"] = np.float32(0.0)
        if "platform_name" in df.columns and self.platform_automation_rates:
            df["platform_automation_rate"] = (
                df["platform_name"].map(self.platform_automation_rates).fillna(np.float32(0.0))
            )

        # Extract raw text columns for the NLP pipeline.
        texts_icg = df.get(self.text_col, pd.Series("", index=df.index)).fillna("")
        texts_ice = df.get(self.text_col_2, pd.Series("", index=df.index)).fillna("")
        _col3 = self.text_col_3 or "decision_facts"
        texts_df = df.get(_col3, pd.Series("", index=df.index)).fillna("")

        # ── Missing-value handling aligned with training configuration ──
        missing_str = self._fp["missing_string"]
        missing_num = self._fp["missing_numeric"]

        # Warn about any expected columns that are completely absent.
        _missing_cols = [c for c in self.feature_cols if c not in df.columns]
        if _missing_cols:
            import warnings
            warnings.warn(
                f"Missing expected feature columns: {_missing_cols}. "
                f"They will be filled with '{missing_str}' (categorical) or "
                f"{missing_num} (numeric)."
            )

        # Build the structured feature matrix, keeping only columns that exist.
        X = df[[c for c in self.feature_cols if c in df.columns]].copy()
        # Impute categorical columns with the designated missing string.
        for c in [f for f in self.cat_features if f in X.columns]:
            X[c] = X[c].fillna(missing_str).astype(str)
        # Coerce and impute numerical columns with the designated missing number.
        for n in [f for f in self.num_features if f in X.columns]:
            X[n] = pd.to_numeric(X[n], errors="coerce").fillna(missing_num)
        # Append fully missing columns so the column order matches training.
        for c in self.feature_cols:
            if c not in X.columns:
                X[c] = missing_str if c in self.cat_features else missing_num

        # Hopsworks' Kafka online-store writer (Avro serialisation) does not support
        # pandas nullable extension integer dtypes (Int16, Int8, Int32, etc.).
        # Since all nulls were already filled above, cast any remaining extension
        # integer columns to standard numpy int64 for Hopsworks compatibility.
        for _col in X.columns:
            if pd.api.types.is_extension_array_dtype(X[_col]) and pd.api.types.is_integer_dtype(X[_col]):
                X[_col] = X[_col].astype("int64")

        return X[self.feature_cols], texts_icg, texts_ice, texts_df

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _score_features(self, X: pd.DataFrame, texts_icg: pd.Series,
                        texts_ice: pd.Series, texts_df: pd.Series,
                        original_index) -> pd.DataFrame:
        """
        Run the loaded model on pre-engineered feature arrays.

        Parameters
        ----------
        X : pd.DataFrame
            Structured features.
        texts_icg, texts_ice, texts_df : pd.Series
            Text columns used in tier-2 models.
        original_index : pd.Index
            Index from the raw input, used to align the output rows.

        Returns
        -------
        pd.DataFrame
            Predictions, confidence, and per-class probability columns.
        """
        ma = self.ma

        # Tier-2 models fuse text directly into the structured feature frame.
        if str(self.tier) == "2":
            _col3 = self.text_col_3 or "decision_facts"
            X_t2 = X.copy()
            X_t2[self.text_col] = texts_icg.values
            X_t2[self.text_col_2] = texts_ice.values
            X_t2[_col3] = texts_df.values
            proba = ma["pipeline"].predict_proba(X_t2)
        else:
            # Tier-1 models consume structured features only.
            proba = ma["pipeline"].predict_proba(X)

        # Decode the highest-probability class index back to a label.
        pred_idx = np.argmax(proba, axis=1)
        pred_labels = [self.target_labels[i] for i in pred_idx]

        # Assemble the core output: predicted label and top confidence.
        out = pd.DataFrame(
            {"predicted_category": pred_labels, "confidence": proba.max(axis=1)},
            index=original_index,
        )

        # Build per-class probability columns, deduplicating long label names.
        _prob_names = {}
        for i, lbl in enumerate(self.target_labels):
            _base = f"prob_{lbl[:24]}"
            _name = _base
            _suffix = 0
            while _name in _prob_names.values():
                _name = f"{_base}_{_suffix:04d}"
                _suffix += 1
            _prob_names[i] = _name
            out[_name] = proba[:, i]
        del proba  # Free the dense probability matrix after columns are copied

        return out

    def predict(self, df_raw: pd.DataFrame,
                history_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Score a raw DataFrame using local feature engineering (no Feature Store).

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw input records.
        history_df : DataFrame | None
            Optional historical data to seed rolling-window features. Without
            history, the first rows of each platform in *df_raw* are scored
            with cold-start rolling values (0, 0, 1), which degrades accuracy.

        Returns
        -------
        pd.DataFrame
            Scoring results aligned to ``df_raw.index``.
        """
        # Warn when no history is provided because rolling features will be cold.
        if history_df is None:
            import warnings
            warnings.warn(
                "No history_df provided: rolling features are computed with "
                "batch-only history. The first rows of each platform will have "
                "degraded accuracy. Pass historical data as `history_df` to "
                "warm-start the rolling windows.",
                UserWarning,
                stacklevel=2,
            )

        # Engineer features, then score.
        X_new, texts_icg, texts_ice, texts_df = self.engineer_features(
            df_raw, history_df=history_df
        )
        return self._score_features(X_new, texts_icg, texts_ice, texts_df, df_raw.index)


# ── Live stream helpers ────────────────────────────────────────────────────────

def find_latest_sor_date(base_url: str, subpath: str, max_lookback: int = 10) -> str:
    """
    Return the most recent date string (YYYY-MM-DD) that has a published SoR zip.

    Parameters
    ----------
    base_url : str
        Root URL for the data repository.
    subpath : str
        Path segment between ``base_url`` and the zip filename.
    max_lookback : int
        Maximum number of past days to probe.

    Returns
    -------
    str
        ISO date string (YYYY-MM-DD) of the most recently available zip.

    Raises
    ------
    RuntimeError
        If no parquet zip is found within the look-back window.
    """
    import datetime
    import requests

    today = datetime.date.today()
    # Walk backwards day by day until a HEAD request succeeds.
    for delta in range(1, max_lookback + 1):
        ds = (today - datetime.timedelta(days=delta)).strftime("%Y-%m-%d")
        try:
            r = requests.head(
                f"{base_url}{subpath}/sor-global-{ds}-full.parquet.zip", timeout=10,
            )
            if r.status_code == 200:
                return ds
        except Exception:
            # Network or DNS errors are ignored so we keep probing earlier dates.
            continue

    raise RuntimeError(f"No parquet zip found in last {max_lookback} days")


def iter_sor_chunks(date_str: str, base_url: str, subpath: str, tail: int | None = None):
    """
    Yield (chunk_name, df) for parquet chunks in the daily SoR zip,
    in lexicographic (chronological) order.  Uses HTTP range requests so
    only the requested chunks are downloaded.  Pass tail=N to fetch only
    the last N chunks.

    Parameters
    ----------
    date_str : str
        ISO date string (YYYY-MM-DD) to retrieve.
    base_url : str
        Root URL for the data repository.
    subpath : str
        Path segment between ``base_url`` and the zip filename.
    tail : int or None
        If given, only yield the last ``tail`` chunks.

    Yields
    ------
    tuple
        (chunk_name, pd.DataFrame) for each parquet file inside the zip.
    """
    import remotezip

    # Construct the full URL to the daily zip bundle.
    url = f"{base_url}{subpath}/sor-global-{date_str}-full.parquet.zip"
    prefix = f"global___full/daily_dumps_chunked/sor-global-{date_str}-full/"

    with remotezip.RemoteZip(url) as rz:
        # List and sort all parquet entries under the expected prefix.
        chunk_files = sorted(
            n for n in rz.namelist()
            if n.startswith(prefix) and n.endswith(".parquet")
        )
        if not chunk_files:
            raise RuntimeError(f"No chunks found in {url}")
        # Optionally restrict to the most recent N chunks.
        if tail is not None:
            chunk_files = chunk_files[-tail:]
        # Stream each chunk into memory and parse with pandas.
        for path in chunk_files:
            name = path.split("/")[-1]
            yield name, pd.read_parquet(io.BytesIO(rz.read(path)))


def safe_div(a, b) -> float:
    """
    Return a / b, or 0.0 if b is zero.

    Parameters
    ----------
    a, b : float or int
        Numerator and denominator.

    Returns
    -------
    float
        The quotient, or 0.0 when division by zero would occur.
    """
    return a / b if b > 0 else 0.0


def compute_macro_prf(tp, fp, fn):
    """
    Compute macro-averaged precision, recall, and F1 from per-class TP/FP/FN arrays.

    Parameters
    ----------
    tp, fp, fn : array-like
        True positives, false positives, and false negatives per class.

    Returns
    -------
    tuple
        (precision, recall, f1) — all floats.
    """
    n = len(tp)
    # Per-class precision, recall, and F1 using safe division.
    prec = np.array([safe_div(tp[c], tp[c] + fp[c]) for c in range(n)])
    rec = np.array([safe_div(tp[c], tp[c] + fn[c]) for c in range(n)])
    f1 = np.array([safe_div(2 * prec[c] * rec[c], prec[c] + rec[c]) for c in range(n)])

    # Only average over classes that actually appeared in the data.
    mask = (tp + fp + fn) > 0
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0
    return float(prec[mask].mean()), float(rec[mask].mean()), float(f1[mask].mean())


def draw_stream_dashboard(
    n_done: int,
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray,
    hist_n: list, hist_f1: list, hist_pr: list, hist_re: list,
    stream_date: str, chunk_name: str,
):
    """
    Render the live-inference metrics dashboard.

    Parameters
    ----------
    n_done : int
        Total number of rows scored so far.
    tp, fp, fn : np.ndarray
        Per-class true positives, false positives, and false negatives.
    hist_n, hist_f1, hist_pr, hist_re : list
        Historical metric trajectories indexed by cumulative rows scored.
    stream_date : str
        Date label for the dashboard title.
    chunk_name : str
        Current chunk label for the dashboard title.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled dashboard figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    # Aggregate per-class counts into overall macro metrics.
    mp, mr, mf = compute_macro_prf(tp, fp, fn)
    total_tp = int(tp.sum())
    total_fp = int(fp.sum())
    total_fn = int(fn.sum())

    # Create a wide figure with two side-by-side subplots.
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

    # ── Left subplot: metric trajectories over rows scored ──
    ax0 = fig.add_subplot(gs[0])
    if hist_n:
        ax0.plot(hist_n, hist_f1, color="#6c8fff", lw=2.0,
                 label=f"Macro F1   {mf:.3f}")
        ax0.plot(hist_n, hist_pr, color="#34d399", lw=1.5, ls="--",
                 label=f"Precision  {mp:.3f}")
        ax0.plot(hist_n, hist_re, color="#fbbf24", lw=1.5, ls=":",
                 label=f"Recall     {mr:.3f}")
        ax0.axhline(hist_f1[-1], color="#6c8fff", lw=0.6, alpha=0.35)
    ax0.set_xlim(left=0)
    ax0.set_ylim(0, 1.05)
    ax0.set_xlabel("Rows scored")
    ax0.set_ylabel("Score")
    ax0.set_title(f"Macro F1 / Precision / Recall  (n={n_done:,})")
    ax0.legend(fontsize=9, loc="lower right")
    ax0.grid(True, alpha=0.22)

    # ── Right subplot: aggregate TP / FP / FN bar chart ──
    ax1 = fig.add_subplot(gs[1])
    vals = [total_tp, total_fp, total_fn]
    colors = ["#34d399", "#f87171", "#fbbf24"]
    bars = ax1.bar(["TP", "FP", "FN"], vals, color=colors, width=0.5)
    top = max(vals) if vals else 1
    # Annotate each bar with its numeric value just above the bar.
    for bar, v in zip(bars, vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + top * 0.012,
            f"{v:,}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax1.set_title("Aggregate TP / FP / FN  (one-vs-rest; TN omitted)")
    ax1.set_ylabel("Count")
    ax1.grid(True, axis="y", alpha=0.22)

    # Overall figure title with stream metadata.
    plt.suptitle(
        f"Live Inference  ·  {stream_date}  ·  {chunk_name}",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    return fig
