import sqlite3 as sl
import numpy as np
import io
print('DB/SETUP')
def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sl.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sl.register_adapter(np.ndarray, adapt_array)
sl.register_converter("array", convert_array)