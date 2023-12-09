import unittest

import pandas as pd

import yado_vectorization


def get_test_df() -> pd.DataFrame:

    # セッション数5, 宿の種類4
    session_1 = pd.DataFrame()
    session_1["yad_no"] = [1, 3, 1]
    session_1["seq_no"] = list(range(session_1.shape[0]))
    session_1["session_id"] = "session1session1session1session1"

    session_2 = pd.DataFrame()
    session_2["yad_no"] = [2, 1]
    session_2["seq_no"] = list(range(session_2.shape[0]))
    session_2["session_id"] = "session2session2session2session2"

    session_3 = pd.DataFrame()
    session_3["yad_no"] = [4]
    session_3["seq_no"] = list(range(session_3.shape[0]))
    session_3["session_id"] = "session3session3session3session3"

    session_4 = pd.DataFrame()
    session_4["yad_no"] = [4, 2, 3, 1]
    session_4["seq_no"] = list(range(session_4.shape[0]))
    session_4["session_id"] = "session4session4session4session4"

    session_5 = pd.DataFrame()
    session_5["yad_no"] = [1]
    session_5["seq_no"] = list(range(session_5.shape[0]))
    session_5["session_id"] = "session5session5session5session5"

    testdata = pd.concat([session_1, session_2, session_3, session_4, session_5], ignore_index=True) \
              .sort_values(["session_id", "seq_no"])[["session_id", "seq_no", "yad_no"]]
    return testdata


class TestGetOccuranceRateArray(unittest.TestCase):
    pass


class TestGetCoOccuranceRateArray(unittest.TestCase):
    pass


class TestGetContinuousOccuranceRateArray(unittest.TestCase):
    pass


if __name__ == "__main__":
    print(get_test_df())