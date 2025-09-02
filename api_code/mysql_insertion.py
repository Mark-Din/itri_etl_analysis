import pandas as pd
from sqlalchemy import create_engine, text
import urllib.parse, os, csv

user = "root"
password = "!QAZ2wsx"
host = "localhost"
port = 3306
db   = "itri_poc"

def mysql_insertion(df, tbl):

    encoded_pw = urllib.parse.quote_plus(password)
    # IMPORTANT: allow_local_infile=1 for mysqlconnector (or local_infile=1 for PyMySQL)
    db_url = f"mysql+mysqlconnector://{user}:{encoded_pw}@{host}:{port}/{db}?allow_local_infile=1"

    engine = create_engine(db_url, pool_pre_ping=True)

    # 1) Ensure table exists with the right schema.
    #    If you don't have it yet, create it once with to_sql but with zero rows
    #    (or write a CREATE TABLE manually for precise types).
    #    Example (create empty table once):
    df.head(0).to_sql(tbl, engine, if_exists="replace", index=False)

    # 2) Write a clean TSV to disk (much safer than CSV for commas/quotes)
    tmp_path = "/tmp/whole_corp_load.tsv"  # on Windows: r"C:\temp\whole_corp_load.tsv"
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)

    # Write TSV: represent NULLs as \N (MySQL understands \N as NULL if not quoted)
    df.to_csv(
        tmp_path,
        sep="\t",
        index=False,
        na_rep="\\N",
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )

    # 3) Bulk load
    with engine.begin() as conn:

        # optional but helps: disable checks while loading
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        conn.execute(text("SET UNIQUE_CHECKS=0"))
        # If you have heavy indexes on the table, consider dropping them before and recreating after.

        # LOAD DATA (LOCAL needs allow_local_infile=1)
        result = conn.execute(text(f"""
            LOAD DATA LOCAL INFILE :path
            INTO TABLE {tbl}
            FIELDS TERMINATED BY '\t'
            ESCAPED BY '\\\\'
            LINES TERMINATED BY '\n'
            IGNORE 1 LINES
        """), {"path": tmp_path})

        conn.execute(text("SET UNIQUE_CHECKS=1"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    return result.rowcount