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
select sub.oid, detections.ra, detections.dec, detections.fid,
detections.mjd, detections.magpsf_corr, detections.sigmapsf_corr

from (
select oid
from objects
where nobs > 8
limit 1000
) sub
left join detections
on sub.oid=detections.oid
"""

data = pd.read_sql_query(con=conn, sql=query)
data.set_index('oid', inplace=True)
print(data.head())
data.to_pickle('data/subset.pkl')

print('Long LCs query...')
query = """
select sub.oid, detections.ra, detections.dec, detections.fid,
detections.mjd, detections.magpsf_corr, detections.sigmapsf_corr

from (
select oid
from objects
where nobs > 80
limit 1000
) sub
left join detections
on sub.oid=detections.oid
"""

data = pd.read_sql_query(con=conn, sql=query)
data.set_index('oid', inplace=True)
print(data.head())
data.to_pickle('data/subset_long.pkl')
