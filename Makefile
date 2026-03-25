.PHONY: download-nta-geojson
.PHONY: download-nyc-311

download-nta-geojson:
	poetry run python src/datasets/nyc/ingestion/download_nta_geojson.py

download-nyc-311:
	poetry run python src/datasets/nyc/ingestion/download_nyc_311.py --start-date $(START_DATE) --query-file ./src/datasets/nyc/queries/nyc_311.soql
