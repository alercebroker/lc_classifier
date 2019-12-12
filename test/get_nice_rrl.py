import pandas as pd
import psycopg2
import json


credentials_file = "alerceuser.json"
with open(credentials_file) as jsonfile:
    params = json.load(jsonfile)['params']

conn = psycopg2.connect(
    dbname=params['dbname'],
    user=params['user'],
    host=params['host'],
    password=params['password']
)

query = """
select oid, ra, dec, fid, mjd, magpsf_corr, sigmapsf_corr
from detections
where oid='ZTF18aayhpyh'
"""

data = pd.read_sql_query(con=conn, sql=query)
data.set_index('oid', inplace=True)
print(data.head())
data.to_pickle('data/nice_rrl.pkl')
