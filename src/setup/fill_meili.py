import pandas as pd
import meilisearch

descriptions = pd.read_json("../data/diagnosis_descriptions.json")

icd_9 = descriptions[descriptions["icd_version"] == 9]
icd_10 = descriptions[descriptions["icd_version"] == 10]

icd_9_docs = [{
            "id": i,
            "icd_code": item["icd_code"],
            "description": item["long_title"],
        } for i,item in icd_9.iterrows()]

icd_10_docs = [{
            "id": i,
            "icd_code": item["icd_code"],
            "description": item["long_title"],
        } for i,item in icd_10.iterrows()]


client = meilisearch.Client('http://127.0.0.1:7700', 'masterKey')
index = client.index('icd_9_descriptions')
index.add_documents(icd_9_docs)



index = client.index("icd_10_descriptions")
index.add_documents(icd_10_docs)
