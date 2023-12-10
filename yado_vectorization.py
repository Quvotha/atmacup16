import numpy as np
import pandas as pd


def get_occurance_rate_array(df: pd.DataFrame, num_yado: int = 13806) -> np.ndarray:
    """宿が全体の何割のセッションに登場したかを示すベクトルを取得する。

    `yad_no` が1, 2, ..., `num_yado`であることが前提

    Parameters
    ----------
    df : pd.DataFrame
        ログ情報。
    num_yado: int, default = 13806
        宿番号が1番から何番まであるか。

    Returns
    -------
    np.ndarray
        宿が全セッションの何割に登場したか。宿番号 - 1 がインデックスになる。
    """
    rate = pd.merge(pd.DataFrame({"yad_no": list(range(1, num_yado + 1))}),
                    df.groupby("yad_no")["session_id"].nunique().reset_index(),
                    how="left").fillna(0.0)
    rate["session_id"] = rate["session_id"] / df["session_id"].nunique()
    return rate["session_id"].values


def get_cooccurance_rate_array(df: pd.DataFrame, num_yado: int = 13806) -> np.ndarray:
    """宿番号の共起行列を取得する。

    Parameters
    ----------
    df : pd.DataFrame
        ログ情報。
    num_yado: int, default = 13806
        宿番号が1番から何番まであるか。

    Returns
    -------
    np.ndarray
        共起行列。
    """