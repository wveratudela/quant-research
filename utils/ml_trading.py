from collections.abc import Callable

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


DECISION_CLASSES = np.array(
    [-1, 0, 1],
    dtype=int,
)

MARKOV_TRANSITION_MATRIX = np.array([
    [0.70, 0.15, 0.15],
    [0.15, 0.70, 0.15],
    [0.15, 0.15, 0.70],
])

CLASS_TO_INDEX = {
    -1: 0,
    0: 1,
    1: 2,
}



# ==========================================================
# 1. RESHAPE MARKET DATA
# ==========================================================

def reshape_market_data(
    df: pd.DataFrame,
    lookback: int = 25,
    prediction_horizon: int = 1,
    classification_threshold: float = 0.0,
    asset_col: str | None = None,
    date_col: str = "Date",
    close_col: str = "Close",
    volume_col: str | None = "Volume",
    high_col: str | None = "High",
    low_col: str | None = "Low",
    drop_incomplete_rows: bool = True,
) -> pd.DataFrame:
    """
    Convert chronological OHLCV data into one supervised-learning row
    per prediction date.

    Automatically handles common yfinance output:
    - Date stored as the DataFrame index
    - MultiIndex columns such as ('Close', 'AAPL')

    Lag convention:
        ret_1  = newest available return
        ret_2  = one period older
        ...
        ret_N  = oldest return in the lookback window

    Target:
        future_return = close[t + prediction_horizon] / close[t] - 1

        target = -1 if future_return < -classification_threshold
                  0 if within the threshold band
                  1 if future_return > classification_threshold
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if lookback < 1:
        raise ValueError("lookback must be at least 1.")

    if prediction_horizon < 1:
        raise ValueError("prediction_horizon must be at least 1.")

    data = df.copy()

    # ======================================================
    # NORMALIZE YFINANCE / MULTIINDEX COLUMNS
    # ======================================================

    if isinstance(data.columns, pd.MultiIndex):

        # Typical single-ticker yfinance structure:
        # level 0: Close, High, Low, Open, Volume
        # level 1: AAPL
        if data.columns.get_level_values(0).is_unique:
            data.columns = data.columns.get_level_values(0)

        # Defensive fallback for unusual MultiIndex structures.
        else:
            flattened_columns = []

            for column in data.columns:
                parts = [
                    str(part)
                    for part in column
                    if str(part) not in {"", "None"}
                ]

                flattened_columns.append(
                    "_".join(parts)
                )

            data.columns = flattened_columns

    # ======================================================
    # MOVE DATE INDEX INTO A COLUMN
    # ======================================================

    if date_col not in data.columns:

        if isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index()

            # The reset index column may be named Date, index, or something else.
            reset_date_column = data.columns[0]

            if reset_date_column != date_col:
                data = data.rename(
                    columns={
                        reset_date_column: date_col
                    }
                )

        elif data.index.name == date_col:
            data = data.reset_index()

        else:
            raise KeyError(
                f"'{date_col}' was not found as either a column "
                "or a DatetimeIndex."
            )

    # ======================================================
    # VALIDATE REQUIRED COLUMNS
    # ======================================================

    required_columns = {
        date_col,
        close_col,
    }

    missing_required = (
        required_columns - set(data.columns)
    )

    if missing_required:
        raise KeyError(
            "Missing required columns after normalization: "
            f"{sorted(missing_required)}.\n"
            f"Available columns: {data.columns.tolist()}"
        )

    if asset_col is not None and asset_col not in data.columns:
        raise KeyError(
            f"Asset column '{asset_col}' was not found."
        )

    data[date_col] = pd.to_datetime(
        data[date_col],
        errors="raise",
    )

    group_columns = (
        [asset_col]
        if asset_col is not None
        else []
    )

    sort_columns = group_columns + [date_col]

    data = (
        data
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )

    # ======================================================
    # BASE RETURN AND FUTURE TARGET
    # ======================================================

    if asset_col is not None:

        grouped = data.groupby(
            asset_col,
            sort=False,
            group_keys=False,
        )

        data["_return"] = grouped[close_col].pct_change(
            fill_method=None
        )

        data["_future_close"] = grouped[close_col].shift(
            -prediction_horizon
        )

    else:

        data["_return"] = data[close_col].pct_change(
            fill_method=None
        )

        data["_future_close"] = data[close_col].shift(
            -prediction_horizon
        )

    data["future_return"] = (
        data["_future_close"]
        / data[close_col].replace(0, np.nan)
        - 1.0
    )

    # ======================================================
    # OPTIONAL SOURCE SERIES
    # ======================================================

    source_series = {
        "ret": data["_return"],
    }

    if (
        volume_col is not None
        and volume_col in data.columns
    ):

        if asset_col is not None:
            data["_volume_change"] = (
                data
                .groupby(asset_col, sort=False)[volume_col]
                .pct_change(fill_method=None)
            )
        else:
            data["_volume_change"] = (
                data[volume_col]
                .pct_change(fill_method=None)
            )

        # Volume can occasionally contain zeros.
        data["_volume_change"] = data[
            "_volume_change"
        ].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        source_series["volchg"] = data[
            "_volume_change"
        ]

    if (
        high_col is not None
        and low_col is not None
        and high_col in data.columns
        and low_col in data.columns
    ):

        close_denominator = data[close_col].replace(
            0,
            np.nan,
        )

        data["_range"] = (
            data[high_col] - data[low_col]
        ) / close_denominator

        source_series["range"] = data["_range"]

    # ======================================================
    # BUILD TABULAR LAG WINDOW
    # ======================================================

    output_columns = group_columns + [date_col]

    output = data[output_columns].copy()

    for source_name, source_values in source_series.items():

        if asset_col is not None:

            grouped_source = source_values.groupby(
                data[asset_col],
                sort=False,
            )

            for lag in range(1, lookback + 1):
                output[f"{source_name}_{lag}"] = (
                    grouped_source.shift(lag - 1)
                )

        else:

            for lag in range(1, lookback + 1):
                output[f"{source_name}_{lag}"] = (
                    source_values.shift(lag - 1)
                )

    # ======================================================
    # TARGET
    # ======================================================

    output["future_return"] = data["future_return"]

    threshold = float(
        classification_threshold
    )

    output["target"] = np.select(
        [
            output["future_return"] < -threshold,
            output["future_return"] > threshold,
        ],
        [
            -1,
            1,
        ],
        default=0,
    ).astype(int)

    # ======================================================
    # CLEAN FINAL DATASET
    # ======================================================

    output = output.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if drop_incomplete_rows:

        generated_feature_columns = [
            column
            for column in output.columns
            if column.startswith(
                (
                    "ret_",
                    "volchg_",
                    "range_",
                )
            )
        ]

        output = output.dropna(
            subset=(
                generated_feature_columns
                + ["future_return"]
            )
        )

    return (
        output
        .reset_index(drop=True)
    )



# ==========================================================
# 2. FEATURE FACTORY
# ==========================================================

def build_feature_factory(
    df: pd.DataFrame,
    return_prefix: str = "ret_",
    volume_prefix: str = "volchg_",
    range_prefix: str = "range_",
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Add reusable feature families to a reshaped market dataset.

    The input is expected to come from `reshape_market_data()`.

    Returns
    -------
    featured_df
        Original reshaped dataset plus engineered features.

    feature_families
        Dictionary mapping family names to feature-column lists.

        This can be consumed directly by feature-family optimizers.
    """

    data = df.copy()

    def ordered_columns(prefix: str) -> list[str]:
        columns = [
            column
            for column in data.columns
            if column.startswith(prefix)
            and column[len(prefix):].isdigit()
        ]

        return sorted(
            columns,
            key=lambda column: int(column[len(prefix):]),
        )

    return_columns = ordered_columns(return_prefix)
    volume_columns = ordered_columns(volume_prefix)
    range_columns = ordered_columns(range_prefix)

    if not return_columns:
        raise ValueError(
            f"No return columns matching '{return_prefix}N' were found."
        )

    returns = data[return_columns]

    n_periods = len(return_columns)

    # lag 1 is most recent; larger lag numbers are older.
    recent_3 = returns.iloc[:, :min(3, n_periods)]
    recent_5 = returns.iloc[:, :min(5, n_periods)]
    recent_10 = returns.iloc[:, :min(10, n_periods)]

    old_3 = returns.iloc[:, -min(3, n_periods):]
    old_5 = returns.iloc[:, -min(5, n_periods):]
    old_10 = returns.iloc[:, -min(10, n_periods):]

    midpoint = max(1, n_periods // 2)

    recent_half = returns.iloc[:, :midpoint]
    old_half = returns.iloc[:, midpoint:]

    feature_families: dict[str, list[str]] = {
        "base_returns": return_columns,
        "base_volume": volume_columns,
        "base_range": range_columns,
        "distribution": [],
        "rolling": [],
        "trend": [],
        "pattern": [],
        "volume": [],
        "range": [],
        "interaction": [],
    }

    # ======================================================
    # Distribution family
    # ======================================================

    data["ret_mean"] = returns.mean(axis=1)
    data["ret_std"] = returns.std(axis=1)
    data["ret_min"] = returns.min(axis=1)
    data["ret_max"] = returns.max(axis=1)
    data["ret_median"] = returns.median(axis=1)

    data["ret_q10"] = returns.quantile(
        0.10,
        axis=1,
    )

    data["ret_q25"] = returns.quantile(
        0.25,
        axis=1,
    )

    data["ret_q75"] = returns.quantile(
        0.75,
        axis=1,
    )

    data["ret_q90"] = returns.quantile(
        0.90,
        axis=1,
    )

    data["ret_range"] = (
        data["ret_max"] - data["ret_min"]
    )

    data["ret_iqr"] = (
        data["ret_q75"] - data["ret_q25"]
    )

    data["ret_tail_range"] = (
        data["ret_q90"] - data["ret_q10"]
    )

    data["ret_mean_abs"] = returns.abs().mean(axis=1)

    data["ret_rms"] = np.sqrt(
        returns.pow(2).mean(axis=1)
    )

    data["ret_skew"] = returns.skew(axis=1)
    data["ret_kurtosis"] = returns.kurt(axis=1)

    distribution_features = [
        "ret_mean",
        "ret_std",
        "ret_min",
        "ret_max",
        "ret_median",
        "ret_q10",
        "ret_q25",
        "ret_q75",
        "ret_q90",
        "ret_range",
        "ret_iqr",
        "ret_tail_range",
        "ret_mean_abs",
        "ret_rms",
        "ret_skew",
        "ret_kurtosis",
    ]

    feature_families["distribution"] = distribution_features

    # ======================================================
    # Rolling/local-window family
    # ======================================================

    window_map = {
        3: recent_3,
        5: recent_5,
        10: recent_10,
    }

    rolling_features = []

    for window, values in window_map.items():

        mean_name = f"ret_mean_{window}"
        std_name = f"ret_std_{window}"
        min_name = f"ret_min_{window}"
        max_name = f"ret_max_{window}"

        data[mean_name] = values.mean(axis=1)
        data[std_name] = values.std(axis=1)
        data[min_name] = values.min(axis=1)
        data[max_name] = values.max(axis=1)

        rolling_features.extend([
            mean_name,
            std_name,
            min_name,
            max_name,
        ])

    data["ret_mean_old_3"] = old_3.mean(axis=1)
    data["ret_mean_old_5"] = old_5.mean(axis=1)
    data["ret_mean_old_10"] = old_10.mean(axis=1)

    data["ret_std_old_3"] = old_3.std(axis=1)
    data["ret_std_old_5"] = old_5.std(axis=1)
    data["ret_std_old_10"] = old_10.std(axis=1)

    data["ret_mean_shift_3"] = (
        data["ret_mean_3"] - data["ret_mean_old_3"]
    )

    data["ret_mean_shift_5"] = (
        data["ret_mean_5"] - data["ret_mean_old_5"]
    )

    data["ret_mean_shift_10"] = (
        data["ret_mean_10"] - data["ret_mean_old_10"]
    )

    data["ret_vol_shift_3"] = (
        data["ret_std_3"] - data["ret_std_old_3"]
    )

    data["ret_vol_shift_5"] = (
        data["ret_std_5"] - data["ret_std_old_5"]
    )

    data["ret_vol_shift_10"] = (
        data["ret_std_10"] - data["ret_std_old_10"]
    )

    rolling_features.extend([
        "ret_mean_old_3",
        "ret_mean_old_5",
        "ret_mean_old_10",
        "ret_std_old_3",
        "ret_std_old_5",
        "ret_std_old_10",
        "ret_mean_shift_3",
        "ret_mean_shift_5",
        "ret_mean_shift_10",
        "ret_vol_shift_3",
        "ret_vol_shift_5",
        "ret_vol_shift_10",
    ])

    feature_families["rolling"] = rolling_features

    # ======================================================
    # Trend family
    # ======================================================

    return_matrix = returns.to_numpy(
        dtype=float
    )

    # Reverse into chronological order:
    # oldest -> newest
    chronological_matrix = return_matrix[:, ::-1]

    x = np.arange(
        n_periods,
        dtype=float,
    )

    x_centered = x - x.mean()

    valid_mask = np.isfinite(
        chronological_matrix
    )

    filled_returns = np.where(
        valid_mask,
        chronological_matrix,
        np.nan,
    )

    row_means = np.nanmean(
        filled_returns,
        axis=1,
        keepdims=True,
    )

    centered_returns = filled_returns - row_means

    numerator = np.nansum(
        centered_returns * x_centered,
        axis=1,
    )

    denominator = np.sum(
        valid_mask * x_centered**2,
        axis=1,
    )

    data["trend_slope"] = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0,
    )

    data["recent_vs_old_mean"] = (
        recent_half.mean(axis=1)
        - old_half.mean(axis=1)
    )

    data["recent_vs_old_vol"] = (
        recent_half.std(axis=1)
        - old_half.std(axis=1)
    )

    data["latest_vs_mean"] = (
        returns.iloc[:, 0] - data["ret_mean"]
    )

    safe_std = data["ret_std"].replace(
        0,
        np.nan,
    )

    data["latest_zscore"] = (
        data["latest_vs_mean"] / safe_std
    ).fillna(0)

    trend_features = [
        "trend_slope",
        "recent_vs_old_mean",
        "recent_vs_old_vol",
        "latest_vs_mean",
        "latest_zscore",
    ]

    feature_families["trend"] = trend_features

    # ======================================================
    # Pattern family
    # ======================================================

    signs = np.sign(
        returns
    )

    data["positive_ratio"] = (
        returns.gt(0).mean(axis=1)
    )

    data["negative_ratio"] = (
        returns.lt(0).mean(axis=1)
    )

    data["zero_ratio"] = (
        returns.eq(0).mean(axis=1)
    )

    adjacent_signs = signs.to_numpy()[:, :-1]
    previous_signs = signs.to_numpy()[:, 1:]

    comparable = (
        np.isfinite(adjacent_signs)
        & np.isfinite(previous_signs)
        & (adjacent_signs != 0)
        & (previous_signs != 0)
    )

    changed = (
        adjacent_signs != previous_signs
    ) & comparable

    comparable_count = comparable.sum(axis=1)

    data["sign_change_ratio"] = np.divide(
        changed.sum(axis=1),
        comparable_count,
        out=np.zeros(
            len(data),
            dtype=float,
        ),
        where=comparable_count != 0,
    )

    positive_returns = returns.where(
        returns > 0
    )

    negative_returns = returns.where(
        returns < 0
    )

    data["positive_mean"] = positive_returns.mean(axis=1)
    data["negative_mean"] = negative_returns.mean(axis=1)

    data["gain_loss_ratio"] = (
        data["positive_mean"]
        / data["negative_mean"]
        .abs()
        .replace(0, np.nan)
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0)

    data["largest_jump"] = (
        returns
        .diff(axis=1)
        .abs()
        .max(axis=1)
    )

    data["path_smoothness"] = (
        returns
        .diff(axis=1)
        .abs()
        .mean(axis=1)
    )

    pattern_features = [
        "positive_ratio",
        "negative_ratio",
        "zero_ratio",
        "sign_change_ratio",
        "positive_mean",
        "negative_mean",
        "gain_loss_ratio",
        "largest_jump",
        "path_smoothness",
    ]

    feature_families["pattern"] = pattern_features

    # ======================================================
    # Volume family
    # ======================================================

    volume_features = []

    if volume_columns:

        volume = data[volume_columns]

        data["volume_change_mean"] = volume.mean(axis=1)
        data["volume_change_std"] = volume.std(axis=1)

        volume_recent_5 = volume.iloc[
            :,
            :min(5, len(volume_columns))
        ]

        volume_old_5 = volume.iloc[
            :,
            -min(5, len(volume_columns)):
        ]

        data["volume_change_recent_5"] = (
            volume_recent_5.mean(axis=1)
        )

        data["volume_change_old_5"] = (
            volume_old_5.mean(axis=1)
        )

        data["volume_regime_shift"] = (
            data["volume_change_recent_5"]
            - data["volume_change_old_5"]
        )

        volume_features = [
            "volume_change_mean",
            "volume_change_std",
            "volume_change_recent_5",
            "volume_change_old_5",
            "volume_regime_shift",
        ]

    feature_families["volume"] = volume_features

    # ======================================================
    # Range / volatility family
    # ======================================================

    range_features = []

    if range_columns:

        ranges = data[range_columns]

        data["range_mean"] = ranges.mean(axis=1)
        data["range_std"] = ranges.std(axis=1)
        data["range_max"] = ranges.max(axis=1)

        range_recent_5 = ranges.iloc[
            :,
            :min(5, len(range_columns))
        ]

        range_old_5 = ranges.iloc[
            :,
            -min(5, len(range_columns)):
        ]

        data["range_recent_5"] = (
            range_recent_5.mean(axis=1)
        )

        data["range_old_5"] = (
            range_old_5.mean(axis=1)
        )

        data["range_regime_shift"] = (
            data["range_recent_5"]
            - data["range_old_5"]
        )

        range_features = [
            "range_mean",
            "range_std",
            "range_max",
            "range_recent_5",
            "range_old_5",
            "range_regime_shift",
        ]

    feature_families["range"] = range_features

    # ======================================================
    # Interaction family
    # ======================================================

    data["trend_volatility_interaction"] = (
        data["trend_slope"] * data["ret_std"]
    )

    data["momentum_volatility_ratio"] = (
        data["ret_mean_5"]
        / data["ret_std_5"].replace(0, np.nan)
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0)

    data["activity_direction"] = (
        data["ret_mean_abs"]
        * (
            data["positive_ratio"]
            - data["negative_ratio"]
        )
    )

    data["shock_volatility_ratio"] = (
        data["largest_jump"]
        / data["ret_std"].replace(0, np.nan)
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0)

    interaction_features = [
        "trend_volatility_interaction",
        "momentum_volatility_ratio",
        "activity_direction",
        "shock_volatility_ratio",
    ]

    feature_families["interaction"] = interaction_features

    # Replace residual infinities.
    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return data, feature_families



def select_feature_families(feature_families, families):
    """
    Combine selected feature families into a single feature list.
    """

    selected = []

    for family in families:
        selected.extend(feature_families[family])

    return selected



def walk_forward_predict(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "target",
    date_column: str = "Date",
    model_factory: Callable | None = None,
    initial_train_size: float = 0.50,
    validation_size: float = 0.10,
    purge_rows: int = 1,
    include_remainder_in_last_fold: bool = True,
    minimum_validation_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run expanding-window walk-forward validation.

    Each fold trains on all observations before the validation period.
    Predictions retain their original dates and future returns.

    Parameters
    ----------
    data
        Featured supervised dataset sorted or sortable by date.

    feature_columns
        Model input columns.

    target_column
        Classification target.

    date_column
        Observation anchor date.

    model_factory
        Callable returning a fresh unfitted classifier.

        Example:
            lambda: LGBMClassifier(...)

    initial_train_size
        Fraction of observations used before the first prediction.

    validation_size
        Fraction of observations predicted per fold.

    purge_rows
        Observations excluded between training and validation.

    include_remainder_in_last_fold
        Append a short remainder to the final full validation block.

    minimum_validation_rows
        Minimum size of a standalone validation block. When omitted,
        the regular validation-block size is used.

    Returns
    -------
    predictions_df
        One row per out-of-sample observation, containing dates,
        actual targets, predictions and class probabilities.

    metrics_df
        One row per validation fold.
    """

    if model_factory is None:
        raise ValueError("model_factory must be supplied.")

    if purge_rows < 1:
        raise ValueError(
            "purge_rows must be at least 1 for weekly targets."
        )

    if not 0 < initial_train_size < 1:
        raise ValueError("initial_train_size must be between 0 and 1.")

    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    required_columns = (
        [date_column, target_column]
        + feature_columns
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}"
        )

    frame = (
        data[required_columns]
        .copy()
        .sort_values(date_column)
        .reset_index(drop=True)
    )

    n_rows = len(frame)

    first_validation_start = int(
        n_rows * initial_train_size
    )

    validation_rows = max(
        1,
        int(n_rows * validation_size),
    )

    if minimum_validation_rows is None:
        minimum_validation_rows = validation_rows

    if minimum_validation_rows < 1:
        raise ValueError("minimum_validation_rows must be at least 1.")

    if first_validation_start < 1:
        raise ValueError(
            "initial_train_size creates an empty training set."
        )

    if first_validation_start - purge_rows < 1:
        raise ValueError(
            "Purging empties the initial training window. Reduce "
            "purge_rows or increase initial_train_size."
        )

    remaining_rows = n_rows - first_validation_start
    full_blocks, remainder = divmod(
        remaining_rows,
        validation_rows,
    )

    validation_blocks = []
    block_start = first_validation_start

    for _ in range(full_blocks):
        validation_blocks.append(
            [block_start, block_start + validation_rows]
        )
        block_start += validation_rows

    if remainder:
        if include_remainder_in_last_fold and validation_blocks:
            validation_blocks[-1][1] = n_rows
        elif remainder >= minimum_validation_rows:
            validation_blocks.append([block_start, n_rows])

    if not validation_blocks and remaining_rows >= minimum_validation_rows:
        validation_blocks.append([first_validation_start, n_rows])

    if not validation_blocks:
        raise ValueError(
            "No validation block satisfies minimum_validation_rows."
        )

    fold_metrics = []
    fold_predictions = []

    for fold_number, (validation_start, validation_end) in enumerate(
        validation_blocks,
        start=1,
    ):

        train_end = validation_start - purge_rows

        if train_end < 1:
            raise ValueError(
                f"Purging empties training data for fold {fold_number}."
            )

        train = frame.iloc[:train_end]

        validation = frame.iloc[
            validation_start:validation_end
        ]

        if validation.empty:
            continue

        X_train = train[feature_columns]
        y_train = train[target_column]

        X_validation = validation[feature_columns]
        y_validation = validation[target_column]

        model = model_factory()

        model.fit(
            X_train,
            y_train,
        )

        predictions = (
            np.asarray(
                model.predict(X_validation)
            )
            .reshape(-1)
        )

        classes = np.asarray(
            model.classes_
        ).reshape(-1)

        probabilities = model.predict_proba(
            X_validation
        )

        prediction_frame = pd.DataFrame({
            date_column: validation[date_column].to_numpy(),
            "fold": fold_number,
            "actual": y_validation.to_numpy(),
            "prediction": predictions,
        })

        for class_index, class_label in enumerate(classes):
            prediction_frame[
                f"prob_{class_label}"
            ] = probabilities[:, class_index]

        fold_predictions.append(
            prediction_frame
        )

        fold_metrics.append({
            "fold": fold_number,
            "train_start": train[date_column].iloc[0],
            "train_end": train[date_column].iloc[-1],
            "validation_start": validation[date_column].iloc[0],
            "validation_end": validation[date_column].iloc[-1],
            "train_rows": len(train),
            "validation_rows": len(validation),
            "purge_rows": purge_rows,
            "accuracy": accuracy_score(
                y_validation,
                predictions,
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y_validation,
                predictions,
            ),
            "macro_f1": f1_score(
                y_validation,
                predictions,
                average="macro",
            ),
        })

    predictions_df = pd.concat(
        fold_predictions,
        ignore_index=True,
    )

    metrics_df = pd.DataFrame(
        fold_metrics
    )

    return predictions_df, metrics_df



def predictions_to_position(
    signal: pd.Series,
    initial_position: int = 0,
) -> pd.Series:
    """Convert ternary decisions into a persistent long/cash state."""

    if not isinstance(signal, pd.Series):
        raise TypeError("signal must be a pandas Series.")

    if signal.index.has_duplicates:
        raise ValueError("Decision dates must be unique.")

    if initial_position not in {0, 1}:
        raise ValueError("initial_position must be 0 or 1.")

    decisions = signal.sort_index()

    if decisions.isna().any():
        raise ValueError("Decisions must not contain missing values.")

    invalid = ~decisions.isin(DECISION_CLASSES)

    if invalid.any():
        invalid_values = decisions.loc[invalid].unique().tolist()
        raise ValueError(
            "Decisions must belong to {-1, 0, 1}; found "
            f"{invalid_values}."
        )

    current_position = int(initial_position)
    position_values = []

    for decision in decisions.astype(int):
        if decision == 1:
            current_position = 1
        elif decision == -1:
            current_position = 0

        position_values.append(current_position)

    return pd.Series(
        position_values,
        index=decisions.index,
        dtype="int64",
        name="position",
    )




# ==========================================================
# WEEKLY SUPERVISED DATASET
# ==========================================================

def build_weekly_supervised_dataset(
    ohlcv: pd.DataFrame,
    featured_daily: pd.DataFrame,
    feature_columns: list[str],
    buy_threshold: float = 0.035,
    sell_threshold: float = -0.020,
    date_column: str = "Date",
    open_column: str = "Open",
) -> pd.DataFrame:
    """
    Create one supervised observation per calendar week.

    Timeline
    --------
    Feature information:
        Daily data available through the final trading session
        before the new week begins.

    Decision and execution:
        First available market open of the new calendar week.

    Target:
        Return from the current week's first available open to
        the following week's first available open.

    Classes
    -------
     1:
        Future weekly return is greater than buy_threshold.

     0:
        Future weekly return lies between the sell and buy
        thresholds. The trading engine keeps its previous state.

    -1:
        Future weekly return is below sell_threshold.

    Parameters
    ----------
    ohlcv
        Daily OHLCV dataframe containing Date and Open.

    featured_daily
        Daily feature dataframe produced by build_feature_factory().
        Features must be anchored to the Date column.

    feature_columns
        Columns to carry into the weekly model dataset.

    buy_threshold
        Minimum future weekly return required for class +1.

    sell_threshold
        Maximum future weekly return assigned to class -1.

    date_column
        Name of the daily date column.

    open_column
        Name of the opening-price column.

    Returns
    -------
    pd.DataFrame
        One row per complete calendar-week decision period.
    """

    if sell_threshold >= buy_threshold:
        raise ValueError(
            "sell_threshold must be strictly less than buy_threshold."
        )

    required_ohlcv_columns = {
        date_column,
        open_column,
    }

    missing_ohlcv_columns = (
        required_ohlcv_columns
        - set(ohlcv.columns)
    )

    if missing_ohlcv_columns:
        raise KeyError(
            "Missing OHLCV columns: "
            f"{sorted(missing_ohlcv_columns)}"
        )

    required_feature_columns = (
        [date_column]
        + list(feature_columns)
    )

    missing_feature_columns = [
        column
        for column in required_feature_columns
        if column not in featured_daily.columns
    ]

    if missing_feature_columns:
        raise KeyError(
            "Missing daily feature columns: "
            f"{missing_feature_columns}"
        )

    market = (
        ohlcv[
            [
                date_column,
                open_column,
            ]
        ]
        .copy()
    )

    market[date_column] = pd.to_datetime(
        market[date_column]
    )

    market = (
        market
        .sort_values(date_column)
        .drop_duplicates(
            subset=date_column,
            keep="last",
        )
        .reset_index(drop=True)
    )

    # Calendar weeks run Monday through Sunday.
    market["week_id"] = (
        market[date_column]
        .dt.to_period("W-SUN")
    )

    # First available trading observation in each calendar week.
    weekly_market = (
        market
        .groupby(
            "week_id",
            sort=True,
            as_index=False,
        )
        .first()
        .rename(
            columns={
                date_column: "decision_date",
                open_column: "entry_open",
            }
        )
    )

    # Following week's first available open.
    weekly_market["next_decision_date"] = (
        weekly_market["decision_date"]
        .shift(-1)
    )

    weekly_market["next_entry_open"] = (
        weekly_market["entry_open"]
        .shift(-1)
    )

    weekly_market["future_week_return"] = (
        weekly_market["next_entry_open"]
        / weekly_market["entry_open"]
        - 1.0
    )

    # ------------------------------------------------------
    # Identify the final market observation before each
    # weekly decision date.
    # ------------------------------------------------------

    market_dates = market[
        [date_column]
    ].copy()

    weekly_market = pd.merge_asof(
        weekly_market.sort_values(
            "decision_date"
        ),
        market_dates.rename(
            columns={
                date_column: "feature_date"
            }
        ).sort_values(
            "feature_date"
        ),
        left_on="decision_date",
        right_on="feature_date",
        direction="backward",
        allow_exact_matches=False,
    )

    # ------------------------------------------------------
    # Attach the feature snapshot from the prior session.
    # ------------------------------------------------------

    daily_features = (
        featured_daily[
            required_feature_columns
        ]
        .copy()
        .rename(
            columns={
                date_column: "feature_date"
            }
        )
    )

    daily_features["feature_date"] = pd.to_datetime(
        daily_features["feature_date"]
    )

    weekly = weekly_market.merge(
        daily_features,
        on="feature_date",
        how="left",
        validate="one_to_one",
    )

    # ------------------------------------------------------
    # Ternary target
    # ------------------------------------------------------

    weekly["target"] = np.select(
        [
            weekly["future_week_return"]
            < sell_threshold,

            weekly["future_week_return"]
            > buy_threshold,
        ],
        [
            -1,
            1,
        ],
        default=0,
    ).astype(int)

    # Remove:
    # - weeks without a following week
    # - early weeks without a complete feature history
    weekly = weekly.dropna(
        subset=(
            feature_columns
            + [
                "future_week_return",
                "next_decision_date",
            ]
        )
    )

    weekly = weekly.sort_values("decision_date").reset_index(drop=True)

    if weekly["decision_date"].duplicated().any():
        raise ValueError("Weekly decision dates must be unique.")

    if not weekly["decision_date"].is_monotonic_increasing:
        raise ValueError("Weekly decision dates must be sorted.")

    if not (
        weekly["decision_date"] > weekly["feature_date"]
    ).all():
        raise ValueError(
            "Every decision_date must be strictly later than feature_date."
        )

    if not (
        weekly["next_decision_date"] > weekly["decision_date"]
    ).all():
        raise ValueError(
            "Every next_decision_date must be strictly later than "
            "decision_date."
        )

    if not (
        (weekly["entry_open"] > 0)
        & (weekly["next_entry_open"] > 0)
    ).all():
        raise ValueError(
            "entry_open and next_entry_open must both be positive."
        )

    output_columns = (
        [
            "decision_date",
            "feature_date",
            "next_decision_date",
            "entry_open",
            "next_entry_open",
            "future_week_return",
            "target",
        ]
        + feature_columns
    )

    return (
        weekly[output_columns]
        .reset_index(drop=True)
    )




# ----------------------------------------------------------
# Performance helper
# ----------------------------------------------------------

def performance_summary(
    backtest: pd.DataFrame,
    name: str,
    starting_capital: float,
    periods_per_year: int = 52,
) -> dict:
    """Calculate performance statistics for periodic backtest data."""

    if backtest.empty:
        raise ValueError("backtest must contain at least one row.")

    if starting_capital <= 0:
        raise ValueError("starting_capital must be positive.")

    if periods_per_year < 1:
        raise ValueError("periods_per_year must be at least 1.")

    returns = backtest["strategy_return"]
    equity = backtest["equity"]
    position = backtest["position"]

    final_value = equity.iloc[-1]
    total_return = final_value / starting_capital - 1.0

    if isinstance(equity.index, pd.DatetimeIndex):
        elapsed_days = (equity.index[-1] - equity.index[0]).days
        elapsed_years = elapsed_days / 365.25
    else:
        elapsed_years = len(backtest) / periods_per_year

    if elapsed_years > 0:
        cagr = (
            final_value / starting_capital
        ) ** (1.0 / elapsed_years) - 1.0
    else:
        cagr = np.nan

    annualized_volatility = (
        returns.std(ddof=1)
        * np.sqrt(periods_per_year)
    )

    annualized_return = (
        returns.mean()
        * periods_per_year
    )

    if annualized_volatility > 0:
        sharpe = (
            annualized_return
            / annualized_volatility
        )
    else:
        sharpe = np.nan

    max_drawdown = (
        backtest["drawdown"].min()
    )

    trade_events = (
        backtest["turnover"] > 0
    ).sum()

    entries = (
        backtest["position"]
        .diff()
        .fillna(backtest["position"])
        .eq(1)
        .sum()
    )

    return {
        "strategy": name,
        "final_value": final_value,
        "total_return": total_return,
        "CAGR": cagr,
        "annualized_volatility": annualized_volatility,
        "Sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "exposure": position.mean(),
        "entries": int(entries),
        "trade_events": int(trade_events),
        # Cumulative modeled return fraction, not a cash-currency amount.
        "total_cost": backtest["trading_cost"].sum(),
    }

# ----------------------------------------------------------
# Uniform ternary random decisions
#
# Complete ignorance:
# P(-1) = P(0) = P(+1) = 1/3
# ----------------------------------------------------------

def generate_uniform_random_decisions(
    dates,
    rng: np.random.Generator,
) -> pd.Series:

    decisions = rng.choice(
        DECISION_CLASSES,
        size=len(dates),
        p=[
            1 / 3,
            1 / 3,
            1 / 3,
        ],
    )

    return pd.Series(
        decisions,
        index=pd.to_datetime(dates),
        name="uniform_random_decision",
        dtype=int,
    )


def run_random_benchmark_simulations(
    generator_function,
    rng: np.random.Generator,
    benchmark_name: str,
    n_simulations: int,
    decision_dates,
    weekly_returns: pd.Series,
    starting_capital: float,
    transaction_cost: float,
    periods_per_year: int = 52,
) -> pd.DataFrame:
    """
    Run repeated random benchmark strategies using the same
    decision dates, execution logic, costs and backtest period
    as the ML strategy.
    """

    simulation_results = []

    for simulation in range(
        n_simulations
    ):

        random_decisions = generator_function(
            dates=decision_dates,
            rng=rng,
        )

        random_position = predictions_to_position(
            signal=random_decisions,
            initial_position=0,
        )

        random_backtest = backtest_weekly_position(
            weekly_return=weekly_returns,
            position=random_position,
            starting_capital=starting_capital,
            transaction_cost=transaction_cost,
        )

        metrics = performance_summary(
            random_backtest,
            benchmark_name,
            starting_capital=starting_capital,
            periods_per_year=periods_per_year,
        )

        metrics["simulation"] = simulation + 1

        simulation_results.append(
            metrics
        )

    return pd.DataFrame(
        simulation_results
    )



# ----------------------------------------------------------
# Backtest helper
# ----------------------------------------------------------

def backtest_weekly_position(
    weekly_return: pd.Series,
    position: pd.Series,
    starting_capital: float,
    transaction_cost: float = 0.0,
) -> pd.DataFrame:
    """Backtest positions held over aligned weekly open-to-open returns."""

    if not isinstance(weekly_return, pd.Series):
        raise TypeError("weekly_return must be a pandas Series.")

    if not isinstance(position, pd.Series):
        raise TypeError("position must be a pandas Series.")

    if weekly_return.empty:
        raise ValueError("weekly_return must contain at least one row.")

    if weekly_return.index.has_duplicates or position.index.has_duplicates:
        raise ValueError("Return and position indices must be unique.")

    if weekly_return.isna().any():
        raise ValueError("weekly_return must not contain missing values.")

    if starting_capital <= 0:
        raise ValueError("starting_capital must be positive.")

    if transaction_cost < 0:
        raise ValueError("transaction_cost must be non-negative.")

    missing_positions = weekly_return.index.difference(position.index)

    if not missing_positions.empty:
        raise ValueError(
            "Positions are missing for weekly return dates: "
            f"{missing_positions.tolist()}."
        )

    aligned_position = position.reindex(weekly_return.index)

    if aligned_position.isna().any():
        raise ValueError("Aligned weekly positions must not be missing.")

    if not aligned_position.isin([0, 1]).all():
        raise ValueError("Weekly positions must contain only 0 or 1.")

    aligned_position = aligned_position.astype(int).rename("position")
    asset_return = weekly_return.astype(float).rename("asset_return")

    turnover = aligned_position.diff().abs().astype(float)
    turnover.iloc[0] = abs(aligned_position.iloc[0])
    turnover.name = "turnover"

    trading_cost = (turnover * transaction_cost).rename("trading_cost")
    gross_strategy_return = (
        aligned_position * asset_return
    ).rename("gross_strategy_return")
    strategy_return = (
        gross_strategy_return - trading_cost
    ).rename("strategy_return")
    equity = (
        starting_capital * (1.0 + strategy_return).cumprod()
    ).rename("equity")
    drawdown = (equity / equity.cummax() - 1.0).rename("drawdown")

    return pd.concat(
        [
            asset_return,
            aligned_position,
            turnover,
            trading_cost,
            gross_strategy_return,
            strategy_return,
            equity,
            drawdown,
        ],
        axis=1,
    )


def backtest_position(
    asset_return: pd.Series,
    position: pd.Series,
    starting_capital: float,
    transaction_cost: float = 0.0,
) -> pd.DataFrame:
    """
    Generic daily-period backtester retained for compatibility.

    position:
        1 = invested
        0 = cash

    The supplied position is assumed to have already been shifted
    so it contains no same-day execution leakage.
    """

    position = (
        position
        .reindex(asset_return.index)
        .fillna(0.0)
        .astype(float)
    )

    # Position changes:
    # 0 -> 1 = entry
    # 1 -> 0 = exit
    turnover = position.diff().abs().fillna(position.abs())

    trading_cost = turnover * transaction_cost

    gross_strategy_return = (
        position * asset_return
    )

    net_strategy_return = (
        gross_strategy_return - trading_cost
    )

    equity = (
        starting_capital
        * (1.0 + net_strategy_return).cumprod()
    )

    running_peak = equity.cummax()

    drawdown = (
        equity / running_peak - 1.0
    )

    return pd.DataFrame({
        "asset_return": asset_return,
        "position": position,
        "turnover": turnover,
        "trading_cost": trading_cost,
        "gross_strategy_return": gross_strategy_return,
        "strategy_return": net_strategy_return,
        "equity": equity,
        "drawdown": drawdown,
    })


def generate_markov_random_decisions(
    dates,
    rng: np.random.Generator,
) -> pd.Series:

    decisions = np.empty(
        len(dates),
        dtype=int,
    )

    # Initial decision is uniformly random.
    decisions[0] = rng.choice(
        DECISION_CLASSES
    )

    for i in range(
        1,
        len(dates),
    ):
        previous_decision = decisions[i - 1]

        transition_probabilities = (
            MARKOV_TRANSITION_MATRIX[
                CLASS_TO_INDEX[previous_decision]
            ]
        )

        decisions[i] = rng.choice(
            DECISION_CLASSES,
            p=transition_probabilities,
        )

    return pd.Series(
        decisions,
        index=pd.to_datetime(dates),
        name="markov_random_decision",
        dtype=int,
    )


# ----------------------------------------------------------
# Random benchmark summary
# ----------------------------------------------------------

RANDOM_METRICS = [
    "final_value",
    "total_return",
    "CAGR",
    "annualized_volatility",
    "Sharpe",
    "max_drawdown",
    "exposure",
    "entries",
    "trade_events",
    "total_cost",
]


def summarize_random_results(
    results: pd.DataFrame,
    benchmark_name: str,
) -> pd.DataFrame:

    summary_rows = []

    for metric in RANDOM_METRICS:

        values = results[metric].dropna()

        summary_rows.append({
            "benchmark": benchmark_name,
            "metric": metric,
            "p05": values.quantile(0.05),
            "median": values.median(),
            "mean": values.mean(),
            "p95": values.quantile(0.95),
        })

    return pd.DataFrame(
        summary_rows
    )


# Add median random strategies to the main table.
def median_random_row(
    results: pd.DataFrame,
    name: str,
    starting_capital: float,
) -> dict:

    row = {
        column: results[column].median()
        for column in RANDOM_METRICS
    }

    row["strategy"] = name
    row["total_return"] = row["final_value"] / starting_capital - 1.0

    return row

def model_factory():

    return LGBMClassifier(
        objective="multiclass",
        num_class=3,

        num_leaves=8,
        max_depth=3,
        min_data_in_leaf=25,

        learning_rate=0.03,
        n_estimators=150,

        feature_fraction=0.80,
        bagging_fraction=0.80,
        bagging_freq=1,

        lambda_l1=1.0,
        lambda_l2=2.0,

        random_state=42,
        feature_fraction_seed=42,
        bagging_seed=42,

        n_jobs=-1,
        verbosity=-1,
    )
