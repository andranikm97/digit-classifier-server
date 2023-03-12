import sqlite3 as sl
import numpy as np
from . import setup

connection = sl.connect('./db/training.db', detect_types=sl.PARSE_DECLTYPES)

EXAMPLES="EXAMPLES"
with connection:
  output = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", [EXAMPLES])
  if output.fetchone() == None:
    connection.execute(f"""
            CREATE TABLE {EXAMPLES} (
                id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                createdAt DATETIME DEFAULT CURRENT_TIMESTAMP,
                image ARRAY NOT NULL,
                label INT NOT NULL,
                trained BOOLEAN DEFAULT 0, 
                trainedAt DATETIME
            );
        """)

def store_new_example(image, label):
  with connection:
    connection.execute(f"INSERT INTO {EXAMPLES} (image, label) values (?, ?)", [image, label])
