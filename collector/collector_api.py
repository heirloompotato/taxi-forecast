import json, requests, logging, os
from datetime import datetime, timezone
from google.cloud import storage
from google.auth.transport.requests import Request
from google.oauth2 import id_token

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BUCKET = os.environ['RAW_JSON_BUCKET']
LTA_URL = os.environ['LTA_URL']
RAIN_URL = os.environ['RAIN_URL']
TEMP_URL = os.environ['TEMP_URL']
HUMI_URL = os.environ['HUMI_URL']
FORE_URL = os.environ['FORE_URL']
ETL_ENDPOINT = os.environ['ETL_ENDPOINT']

client = storage.Client()
bucket = client.bucket(BUCKET)

def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

def upload(js: dict, tag: str):
    ts = now()
    name = f"{tag}/{tag}_{ts}.json"
    try:
        bucket.blob(name).upload_from_string(
            json.dumps(js, indent=2), content_type="application/json")
        logging.info(f"Uploaded {name}")
    except Exception as e:
        logging.error(f"INCIDENT: Failed to upload {name}: {e}")

def fetch(url):
    try:
        return requests.get(url, timeout=10).json()
    except Exception as e:
        logging.error(f"INCIDENT: Failed to fetch {url}: {e}")
        return {}

def trigger_etl(endpoint):
    try:
        audience = endpoint
        token = id_token.fetch_id_token(Request(), audience)
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.post(endpoint, headers=headers)
        logging.info(f"Triggered ETL job, response: {resp.status_code}")
    except Exception as e:
        logging.error(f"INCIDENT: Failed to trigger ETL job: {e}")

def main():
    utcnow = datetime.utcnow()
    is_10or40min = utcnow.minute in [10, 40]
    upload(fetch(LTA_URL),  "taxi")
    upload(fetch(RAIN_URL), "rain")
    upload(fetch(TEMP_URL), "temp")
    upload(fetch(HUMI_URL), "humid")

    # grab forecast only on the :10 or :40 minute mark
    if is_10or40min:
        upload(fetch(FORE_URL), "forecast")

    # Call etl_run.py to process the data
    trigger_etl(ETL_ENDPOINT)

if __name__ == "__main__":
    main()
