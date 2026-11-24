# Container image for the Tanguar Haor monitor (GeoLattice Notebook 08_10).
#   docker build -t haor-monitor:1.0.0 .
#   docker run --rm -v "$PWD/monitor_data:/data" haor-monitor:1.0.0 run --start 2024-01-01 --end 2024-12-31
FROM python:3.11.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HAOR_DATA_DIR=/data

# rasterio and odc-stac install from wheels that bundle GDAL: no system GDAL needed
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . && useradd --create-home --uid 10001 monitor && mkdir /data && chown monitor /data

USER monitor
VOLUME ["/data"]
HEALTHCHECK NONE
ENTRYPOINT ["python", "-m", "haor_monitor"]
CMD ["run"]
