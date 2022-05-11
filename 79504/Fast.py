import pandas as pd
from joblib import Parallel, delayed
import multiprocessing


def parallel_apply(grouped_df, func) -> pd.DataFrame:
    """
    A function that parallelizes(!) the apply function of pandas over a GROUPED dataframe.
    Just pass a GROUPED dataframe along with a function that you want to apply over each group

    :param grouped_df: a grouped dataframe
    :param func: the function that you want to apply to each group of the dataframe
    :return processed_df: a dataframe that is the result concatenation of processed groups. IMPORTANT: the result are
    sorted the way the groups where, e.g. if the groups are sorted by their `co_id`, the processed_df will be the same.
    """
    processed_group = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in grouped_df)
    return pd.concat(processed_group)
