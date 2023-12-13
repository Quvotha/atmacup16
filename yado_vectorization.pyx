# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: infer_types=True
import os
import sys
from typing import Tuple, Generator

from joblib import delayed, Parallel
import numpy as np
import pandas as pd

cimport numpy as cnp

DIR = os.path.join(os.path.expanduser("~"), "atmacup16")
if DIR not in sys.path: sys.path.append(DIR)


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


cpdef cnp.ndarray[cnp.float64_t, ndim=2] get_cooccurance_rate_array(
    df : pd.DataFrame, num_yado: int = 13806
):
    """宿番号の共起行列を取得する。

    宿番号 no1, no2 に対して no1 が登場するセッションに占める no2 が登場するセッションの割合を求める。
    no1 = no2 の場合 no1 が登場するセッションに占める no1 がもう1度以上登場するセッションの割合を占める。    

    Parameters
    ----------
    df : pd.DataFrame
        ログ。
    num_yado: int, default = 13806
        宿番号が1番から何番まであるか。

    Returns
    -------
    co_occurance_matrix : cnp.ndarray[cnp.float64_t, ndim=2]
        共起行列。宿 no1 が閲覧されたセッションの内宿 no2 も閲覧された割合が no1 - 1 行 no2 - 1 列名の要素。
        no1 = no2 ならば no1 が閲覧されたセッションの内 no1 が再び閲覧された割合。
    """
    cdef int i, j, k, no1, no2
    cdef Py_ssize_t total_occurances, co_occurances
    cdef cnp.ndarray[cnp.float64_t, ndim=2] co_occurance_matrix = np.zeros((num_yado, num_yado), dtype=np.float64)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] session_ids, yad_numbers, session_ids_
    cdef cnp.ndarray[cnp.int64_t, ndim=1] counts
    cdef cnp.ndarray[cnp.int32_t, ndim=1] unique_sessions, session_counts
    cdef cnp.ndarray[cnp.int32_t, ndim=1] unique_session_ids1, unique_session_ids2
    cdef list session_lists = [np.empty(0, dtype=np.int32) for _ in range(num_yado)]

    df = df.sort_values(["session_id", "seq_no"])
    mapping = {id_: no for no, id_ in enumerate(df["session_id"].unique())}
    df["session_id_"] = df["session_id"].map(mapping)
    session_ids = df["session_id_"].to_numpy().astype("int32")
    yad_numbers = df["yad_no"].to_numpy().astype("int32")

    # 宿番号毎に登場するセッションを取得する、no1 = no2 のケースは共起行列の要素を求めてしまう
    for i in range(num_yado):
        session_ids_, counts = np.unique(session_ids[yad_numbers == i + 1], return_counts=True)
        session_lists[i] = session_ids_
        if len(session_ids_) > 0:
            co_occurance_matrix[i][i] = np.count_nonzero(counts > 1) / len(session_ids_)

    for i in range(num_yado):
        unique_session_ids1 = session_lists[i]
        for j in range(num_yado):
            no1 = i + 1
            no2 = j + 1

            # no1が登場するセッション数
            total_occurances = len(session_lists[i])
            if total_occurances > 0:

                # no1 と no2 が異なる宿ならば no1 の登場セッションの内 no2 も登場するセッション数の割合を求める
                if no1 == no2:
                    # もう計算済
                    continue
                else:
                    # 両方が登場するセッション数を求める
                    unique_session_ids2 = session_lists[j]
                    co_occurances = np.count_nonzero(np.in1d(unique_session_ids2, unique_session_ids1))
                    co_occurance_matrix[i, j] = co_occurances / <double>total_occurances
                    # print(i, j, co_occurances, total_occurances, co_occurances / total_occurances)
                    # print(co_occurance_matrix)
    return co_occurance_matrix


def get_continuous_occurance_rate_array(df: pd.DataFrame, num_yado: int = 13806) -> np.ndarray:
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
    # ログ発生順にソート
    df = df.sort_values(["session_id", "seq_no"])

    # セッション毎の最終ログを識別できるようにする
    session_continues = np.zeros(shape=df.shape[0], dtype=bool)  # True: セッション内で最後ではない
    session_continues[:-1] = df["session_id"].to_numpy()[:-1] == df["session_id"].to_numpy()[1:]
    df["session_continues"] = session_continues

    # 次のログで表示される宿
    yad_no_next = np.zeros(shape=df.shape[0], dtype=np.int32)  # 最終行の次はあり得ない宿番号
    yad_no_next[:-1] = df["yad_no"].to_numpy()[1:]
    df["yad_no_next"] = yad_no_next

    # セッション毎の最後のログを除く各宿番号の登場回数
    total_occurances = [0] * num_yado
    count_by_yad_no = df[df["session_continues"]]["yad_no"].value_counts()
    for yad_no, count in zip(count_by_yad_no.index, count_by_yad_no.values):
        total_occurances[yad_no - 1] = count

    # no1 の次に閲覧された宿が no2 である回数
    count_by_continuous_occurance = df[df["session_continues"]].groupby(["yad_no", "yad_no_next"])["session_id"].count()
    continuous_occurance_rate_array = np.zeros(shape=(num_yado, num_yado))
    for (no1, no2), cnt in zip(count_by_continuous_occurance.keys(), count_by_continuous_occurance.values):
        continuous_occurance_rate_array[no1 - 1][no2 - 1] = cnt / total_occurances[no1 - 1]
    return continuous_occurance_rate_array
