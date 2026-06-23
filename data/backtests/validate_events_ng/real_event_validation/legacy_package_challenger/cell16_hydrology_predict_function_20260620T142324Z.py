def predict_hydrology_proba_df(
    df: pd.DataFrame,
    pixel_col: str = "pixel_key",
    time_col: str = "yyyymm",
) -> pd.DataFrame:
    return predict_hydrology_proba_from_long_df(
        df=df,
        pixel_col=pixel_col,
        time_col=time_col,
    )
