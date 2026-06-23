def predict_hydrology_proba_from_long_df(
    df: pd.DataFrame,
    pixel_col: str = "pixel_key",
    time_col: str = "yyyymm",
) -> pd.DataFrame:
    """
    Build valid hydrology windows from a long dataframe and return probabilities
    for rows whose end-timestep has a valid sequence.

    Required columns:
      - pixel_col
      - time_col
      - hydrology feature columns from packaged init manifest
    """
    X_seq, valid_index = build_hydrology_sequences(
        df=df,
        pixel_col=pixel_col,
        time_col=time_col,
    )

    if len(valid_index) == 0:
        return pd.DataFrame(columns=["p0", "p1", "p2", "p3"], index=pd.Index([]))

    proba = predict_hydrology_proba(X_seq)
    proba.index = valid_index
    return proba
