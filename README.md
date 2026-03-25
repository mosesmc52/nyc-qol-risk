# municipal-stress-engine
A city-agnostic engine for modeling latent operational stress in municipal systems, with NYC as the first deployment.

## Makefile

Available helper targets:

- `make download-nta-geojson`
  Runs `poetry run python src/datasets/nyc/ingestion/download_nta_geojson.py`

- `make download-nyc-311 START_DATE=2023-01-01`
  Runs `poetry run python src/datasets/nyc/ingestion/download_nyc_311.py --start-date $(START_DATE) --query-file ./src/datasets/nyc/queries/nyc_311.soql`

Note: `download_nyc_311.py` currently requires `--end-date` as well, so the `download-nyc-311` target needs one more parameter before it can execute successfully.
