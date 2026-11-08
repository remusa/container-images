#!/bin/sh
# Grayjay server entrypoint: apply the image's payload to the persistent volume,
# then run the app from there.
#
# WHY THIS EXISTS (the bug it fixes, found 2026-10-16):
# the previous entrypoint was
#
#   mkdir -p /data && cp -r --update=none /opt/grayjay/* /data && cd /data && ./Grayjay...
#
# `--update=none` means "copy only files that do not exist yet". /data keeps the
# first payload it ever received (2026-07-26 here), so the binary that actually
# ran was the volume copy, frozen at first start: rebuilding the image would
# have changed nothing at all. That is why grayjay had no working update path
# even in principle.
#
# WHAT IS PRESERVED: everything the app creates that the payload does not ship —
# database.db, id, launch, enabledClients, port, log.txt, temp_files,
# curl-ca-bundle.pem (downloaded at runtime), downloaded/, downloads/,
# downloads_ongoing/, downloads_playlists/. Copying never deletes them, so state
# survives an upgrade.
#
# WHAT IS REMOVED: files that the PREVIOUS payload shipped and this one does
# not. Tracked via /data/.grayjay-payload-manifest, written on every start.
# Without that, a stale .so or bundled helper from an older version would linger
# forever and could still be loaded.
#
# Not done here: pruning anything the app created. There is no reliable list of
# what that is, so the safe direction is "never delete a file we did not ship".
set -eu

APP_SRC=/opt/grayjay
DATA=${DATA_DIR:-/data}
MANIFEST="$DATA/.grayjay-payload-manifest"

[ -d "$APP_SRC" ] || { echo "FATAL: no payload at $APP_SRC" >&2; exit 1; }
mkdir -p "$DATA"

# Payload file list, relative, sorted, LC_ALL=C so `comm` and the manifest are
# stable across runs and locales.
(cd "$APP_SRC" && find . -type f | LC_ALL=C sort) > /tmp/payload.new

if [ -f "$MANIFEST" ]; then
  # Lines only in the old manifest = shipped by the previous version, absent from
  # this one. Delete those; anything else in /data is app state.
  LC_ALL=C comm -23 "$MANIFEST" /tmp/payload.new | while IFS= read -r f; do
    rm -f "${DATA}/${f#./}"
  done
fi

# Overwrite the payload in place. `cp -a` keeps modes and symlinks; ownership is
# fixed up below because the app is run as root here but the volume's files have
# historically been owned by uid 1001, and a future `user:` in compose must not
# find a root-owned binary it cannot rewrite.
cp -a "$APP_SRC/." "$DATA/"

cp /tmp/payload.new "$MANIFEST"
if [ -f "$APP_SRC.zip.sha256" ]; then
  cp "$APP_SRC.zip.sha256" "$DATA/.grayjay-image-sha256"
fi

# Cosmetic on a root container, but it keeps the volume consistent with its
# history and keeps the app writable if it is ever run as 1001.
chown -R 1001:1001 "$DATA" 2>/dev/null || true

echo "grayjay payload applied: $(wc -l < "$MANIFEST") files, zip sha256 $(cat "$APP_SRC.zip.sha256" 2>/dev/null || echo unknown)"

cd "$DATA"
# --ignore-security is carried over from the original entrypoint; the server mode
# binds a web UI to a LAN address and the app refuses to start without it.
exec ./Grayjay "--${GRAYJAY_MODE:-server}" --ignore-security
