from typing import Tuple, Generator

from joblib import delayed, Parallel
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


def get_cooccurance_rate_array(df: pd.DataFrame, num_yado: int = 13806, n_jobs: int = -1, default: float = 0.) -> np.ndarray:
    """宿番号の共起行列を取得する。

    宿番号 no1, no2 に対して no1 が登場するセッションに占める no2 が登場するセッションの割合を求める。
    no1 = no2 の場合 no1 が登場するセッションに占める no1 がもう1度以上登場するセッションの割合を占める。    

    Parameters
    ----------
    df : pd.DataFrame
        ログ情報。
    num_yado: int, default = 13806
        宿番号が1番から何番まであるか。
    n_jobs: int, default = -1
        `joblib.Parallel` の初期化に使う並行数。
    default : float, default = 0.
        no1 がどのセッションにも登場しない宿番号だった場合に対応する行列の要素の値。

    Returns
    -------
    np.ndarray
        共起行列。
    """


    def co_occurance_of(no1: int, no2: 2) -> Tuple[int, int, float]:
        """宿番号 `no1` が登場するセッション総数に占める `no2` が登場するセッション数の割合を計算する。

        Parameters
        ----------
        no1, no2: int
            宿番号。

        Returns
        -------
        (no1, no2, rate) : Tuple[int, int, float]
            宿番号と割合。
        """
        if no1 == no2:
            df_ = df.query(f"yad_no == {no1}").copy()
            if df_.shape[0] == 0:
                rate = default
            else:
                total_occurances = df_["session_id"].nunique()
                count_by_session = df_["session_id"].value_counts()
                multiple_occurances = len(count_by_session[count_by_session > 1])
                rate = 1.0 * multiple_occurances / total_occurances
            return (no1, no2, rate)
        else:
            df_ = df.query(f"yad_no == {no1}").copy()
            if df_.shape[0] == 0:
                rate = default
            else:
                total_occurances = df_["session_id"].nunique()
                co_occurances = df[df["session_id"].isin(df_["session_id"])].query(f"yad_no == {no2}")["session_id"].nunique()
                rate = co_occurances / total_occurances
            return (no1, no2, rate)


    def gen_pairs() -> Generator[Tuple[int, int], None, None]:
        for no1 in range(1, 1 + num_yado):
            for no2 in range(1, 1 + num_yado):
                yield no1, no2


    results = Parallel(n_jobs=n_jobs)(delayed(co_occurance_of)(pair[0], pair[1]) for pair in gen_pairs())
    co_occurance_matrix = np.zeros((num_yado, num_yado))
    for (no1, no2, rate) in results:
        co_occurance_matrix[no1 - 1][no2 - 1] = rate
    return co_occurance_matrix


def get_continuous_occurance_rate_array(df: pd.DataFrame, num_yado: int = 13806, n_jobs: int = -1, default: float = 0.) -> np.ndarray:
    """宿番号 no1 が閲覧された直後に閲覧される宿番号が no2 である割合を計算する。

    あるセッションにおいて最後に閲覧された宿番号が no1 の場合、割合計算の対象外とする。
    
    Parameters
    ----------
    `get_cooccurance_rate_array` と同じ。


    Returns
    -------
    np.ndarray
        宿 no1 が閲覧された次に閲覧された宿が no2 である割合。
        宿番号 - 1 が対応するインデックスになる。
    """
    # ログの各行に直後に見た宿の情報を付与、セッションの最後のログ名は次の宿はあり得ない番号
    no_next_log = -1
    df = df.sort_values(["session_id", "seq_no"])  # 入力されたデータフレームを変更しない
    df["yad_no_next"] = df.groupby("session_id")["yad_no"].shift(-1, fill_value=no_next_log)


    def continuous_occurance_of(no1: int, no2: int) -> Tuple[int, int, float]:
        """宿番号 `no1` の閲覧ログに占める、直後に `no2` が閲覧された回数の割合を計算する。

        `no1` が10回閲覧されたとしてその直後に閲覧された宿が `no2` であった回数が4なら0.4になる。        

        Parameters
        ----------
        no1, no2: int
            宿番号。

        Returns
        -------
        (no1, no2, rate) : Tuple[int, int, float]
            宿番号と割合。
        """
        if no1 == no2:
            return (no1, no2, 0.)
        else:
            # no1 がセッションの最後に閲覧された宿ならばそのログは計算から除外する
            mask = (df["yad_no"] == no1) & (df["yad_no_next"] == no_next_log)  # True: 除外する
            df_ = df.loc[~mask, :]
            total_occurances = (df_["yad_no"] == no1).sum()
            if total_occurances < 1:
                rate = default
            else:
                continuous_occurances = ((df_["yad_no"] == no1) & (df_["yad_no_next"] == no2)).sum()
                rate = 1.0 * continuous_occurances / total_occurances
            return (no1, no2, rate)


    def gen_pairs() -> Generator[Tuple[int, int], None, None]:
        for no1 in range(1, 1 + num_yado):
            for no2 in range(1, 1 + num_yado):
                yield no1, no2


    results = Parallel(n_jobs=n_jobs)(delayed(continuous_occurance_of)(pair[0], pair[1]) for pair in gen_pairs())
    continuous_occurance_matrix = np.zeros((num_yado, num_yado))
    for (no1, no2, rate) in results:
        continuous_occurance_matrix[no1 - 1][no2 - 1] = rate
    return continuous_occurance_matrix