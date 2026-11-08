# container-images

Locally-built images for the homelab, built on GitHub Actions instead of on the
guests that run them.

Two images live here. Both existed because upstream provides no usable registry
image, and both were previously built **on the host that runs them** — first the
arrstack VM, then LXC `arrstack-nix` (104) — which meant multi-GB base pulls and
build caches landing on a 40 GB LXC rootfs.

| image | why it exists | upstream |
| --- | --- | --- |
| `ghcr.io/remusa/grayjay` | Grayjay Desktop in server mode. Upstream ships an unversioned zip with no version or checksum endpoint; the zip's HTTP ETag is the only change signal. | <https://grayjay.app/> |
| `ghcr.io/remusa/manga-translator-shim` | A REST endpoint layered on `zyddnys/manga-image-translator` for the library viewer's translate button. Upstream's image only speaks an internal pickled-executor API. | <https://github.com/zyddnys/manga-image-translator> |

Both are **published public** on GHCR on purpose: the guests that pull them run no
sops and hold no registry credentials, so an anonymous pull is the whole point.

## How updates happen

Neither workflow builds on a schedule alone — each first asks whether anything
actually changed, and exits without building if not:

- **grayjay** hashes the live upstream zip. The image tag *is* the upstream ETag,
  and the zip's sha256 is passed as `GRAYJAY_ZIP_SHA256`, so the Dockerfile's
  optional pin is exercised on every build (the build fails if upstream serves
  different bytes than the ones the workflow just measured).
- **manga-translator-shim** resolves the base image's current manifest digest and
  tags by that digest. It also rewrites `FROM …:main` to `FROM …@sha256:…` before
  building, so the shipped shim is pinned to the exact translator it was built
  against — the Dockerfile's own comment claimed this and the tag did not.

Consumers pin by digest, so a rebuild of an image cannot change what a host runs
until someone edits the pin.

## Consuming

```nix
virtualisation.oci-containers.containers.grayjay = {
  image = "ghcr.io/remusa/grayjay@sha256:<digest>";  # not :latest, not :<etag>
  pull = "always";
};
```

The digest is printed in each run's job summary.
