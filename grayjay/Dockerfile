# Grayjay Desktop in server mode, built from the upstream "latest" zip.
#
# Moved into the repo 2026-10-16 (was /opt/docker-config/grayjay/Dockerfile, an
# untracked file with three hand-tuned variants beside it). Two reasons it had to
# move: a locally built image with its build context outside git is not
# reproducible after a loss, and nothing in the estate could update it — it is
# not in a registry, so docker-update.sh/compose pull skip it ("No image to be
# pulled"), and it predated the entrypoint fix in ./entrypoint.sh, which is what
# actually lets a rebuilt payload reach the running server.
#
# Upstream publishes NO version or checksum endpoint (2026-10-16:
# /Apps/Grayjay.Desktop/version.json, AppVersion.json and *.zip.sha256 all 404,
# directory listing 403), so the only change signal is the zip's HTTP ETag:
#
#   curl -sI https://updater.grayjay.app/Apps/Grayjay.Desktop/Grayjay.Desktop-linux-x64.zip \
#     | grep -i etag
#
# GRAYJAY_ZIP_ETAG is therefore both the cache-buster (its value changes when
# upstream rebuilds the zip, which invalidates the wget layer) and the value
# scripts/docker-update/grayjay-update.sh compares against the running image's
# label to decide whether anything needs doing. Pass it as a build arg:
#
#   docker compose build --build-arg GRAYJAY_ZIP_ETAG="$(new etag)" grayjay
#
# GRAYJAY_ZIP_SHA256 is the optional PIN. Empty (the default) means "accept
# whatever upstream serves and record its sha256 in the image", which is the
# automatic path. Set it to freeze the version: the build then fails unless the
# downloaded zip matches, so an upstream change cannot land unreviewed. The
# running server's sha256 is copied to /data/.grayjay-image-sha256 at startup and
# is printed by the update script, so the value to pin is always discoverable:
#
#   docker run --rm --entrypoint cat media-stack-grayjay:latest /opt/grayjay.zip.sha256
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime deps for the bundled CEF/Chromium payload. Kept identical to the
# original hand-tuned context (which worked); ca-certificates and curl are added
# because the update script and the app's own fetches need them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget unzip ca-certificates curl \
    libx11-6 libxext6 libxcb1 libxcomposite1 libxrandr2 \
    libxfixes3 libxdamage1 libxcb-xfixes0 libnss3 libatk1.0-0 libcups2 \
    libxss1 libasound2t64 libatk-bridge2.0-0 libgtk-3-0 libdrm2 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/grayjay

ARG GRAYJAY_ZIP_URL=https://updater.grayjay.app/Apps/Grayjay.Desktop/Grayjay.Desktop-linux-x64.zip
# Cache-buster + provenance. See the header: this is the zip's HTTP ETag.
ARG GRAYJAY_ZIP_ETAG=unset
# Optional pin. Empty = accept and record.
ARG GRAYJAY_ZIP_SHA256=

# The zip has to be fetched from a URL whose content can change, which is the
# whole reason this image needs a rebuild path at all. `unzip` is done in a
# temp dir so a partially extracted payload can never look complete.
RUN set -eux; \
    wget -q --tries=5 --timeout=30 --waitretry=10 --retry-connrefused \
         -O /tmp/grayjay.zip "$GRAYJAY_ZIP_URL"; \
    if [ -n "$GRAYJAY_ZIP_SHA256" ]; then \
      echo "$GRAYJAY_ZIP_SHA256  /tmp/grayjay.zip" | sha256sum -c -; \
    fi; \
    sha256sum /tmp/grayjay.zip | awk '{print $1}' > /opt/grayjay.zip.sha256; \
    mkdir -p /tmp/gj; \
    unzip -q /tmp/grayjay.zip -d /tmp/gj; \
    mv /tmp/gj/Grayjay.Desktop-linux-x64-*/* .; \
    rm -rf /tmp/grayjay.zip /tmp/gj; \
    echo "payload files: $(find . -type f | wc -l)"; \
    cat /opt/grayjay.zip.sha256

LABEL org.opencontainers.image.title="grayjay-server" \
      org.opencontainers.image.description="Grayjay Desktop (server mode) from the upstream latest zip" \
      org.opencontainers.image.source="https://grayjay.app/" \
      grayjay.zip.url="$GRAYJAY_ZIP_URL" \
      grayjay.zip.etag="$GRAYJAY_ZIP_ETAG"

# Web UI and Sync.
EXPOSE 11338 12315

ENV GRAYJAY_MODE=server

COPY entrypoint.sh /usr/local/bin/grayjay-entrypoint
RUN chmod +x /usr/local/bin/grayjay-entrypoint

ENTRYPOINT ["/usr/local/bin/grayjay-entrypoint"]
