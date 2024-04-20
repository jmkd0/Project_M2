import os
import pandas as pd
import json
import time
from google.cloud import pubsub_v1 #pip install --upgrade google-cloud-pubsub
credentials_path = "/Users/komlan/Project_M2/GCP_Project/data_stream/doubleclick-364815-8e1a01a8d071.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
publisher = pubsub_v1.PublisherClient()
topic_path = "projects/doubleclick-364815/topics/topic_doubleclick"
keys = ["Landing_Page_URL_ID","Referrer_URL","Impression_ID","Asset_ID","Active_View_Eligible_Impressions",
        "Active_View_Measurable_Impressions","Active_View_Viewable_Impressions","Activity_ID","Event_Time",
        "Event_Type","Event_Sub_Type","User_ID","Advertiser_ID","Campaign_ID","Floodlight_Configuration",
        "Ad_ID","Rendering_ID","Creative_Version","Site_ID_DCM","Placement_ID","Country_Code","State_Region",
        "Browser_Plateform_ID"]
data = pd.read_csv("/Users/komlan/Project_M2/GCP_Project/doubleclick_dataset.csv").drop(['Unnamed: 0'], axis=1)

cpt = 0
stream_data = data[100:]
print(data.shape)
for i, line in stream_data.iterrows():
    row = {}
    for j, val in enumerate(line):
        row[keys[j]] = val
    
    row["Landing_Page_URL_ID"] = str(row["Landing_Page_URL_ID"]).replace(" : ","_")
    row["User_ID"] = str(row["User_ID"])
    row["Ad_ID"] = str(row["Ad_ID"])
    row["Impression_ID"] = row["Impression_ID"].replace(" : ","_")
    row["Country_Code"] = None if pd.isnull(row["Country_Code"]) else row["Country_Code"]
    row["Event_Sub_Type"] = None if pd.isnull(row["Event_Sub_Type"]) else row["Event_Sub_Type"]
    attributes = json.dumps(row).encode("utf-8")
    future = publisher.publish(topic_path, attributes)
    cpt += 1