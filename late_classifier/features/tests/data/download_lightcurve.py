from alerce.api import AlerceAPI


alerce_api = AlerceAPI()
oid = 'ZTF18aazsabq'

detections = alerce_api.get_detections(oid, 'pandas')
non_detections = alerce_api.get_non_detections(oid, 'pandas')

detections.reset_index(inplace=True)
non_detections.reset_index(inplace=True)

detections.set_index('oid', inplace=True)
non_detections.set_index('oid', inplace=True)

print(detections.head())
print(non_detections.head())

detections.to_csv(f'{oid}_det.csv')
non_detections.to_csv(f'{oid}_nondet.csv')
