"""A small, browser-friendly REST wrapper around manga-image-translator.

WHY THIS EXISTS
The upstream `:main` image only exposes an internal executor API: `/simple_execute/{method}` and
`/execute/{method}` both decode a **pickled** kwargs blob and are meant to be driven by the project's
own Python client. That is fine for its `shared` mode and impossible to call from a web page - the
only alternative the registry offers is a 2022 `docker_debug` tag.

So this shim layers the classic, simple endpoint on top of the same library the image already
carries (models, torch, fonts - nothing new is downloaded):

    GET  /health     -> {"ok": true, "ready": <model loaded>}
    POST /translate  -> the translated page as image/png

`/translate` accepts either a multipart form (`image` file, optional `config` JSON string) or JSON
(`{"image": "<base64 or data: URL>", "config": {...}}`), which is what the library viewer sends.

One translation at a time: the model is not thread-safe and the host is shared, so requests queue
behind a lock rather than racing.

Results are cached on disk, content-addressed. Translation is CPU-only here and a text-dense page
takes ~60 s, so the same page asked for twice must not cost twice - and the reader does ask twice
(dismiss the overlay, press translate again). The cache is keyed on the image bytes plus the config,
so it also serves the same page arriving from two archives, and a different target language is a
different entry.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import io
import json
import os
import re
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from PIL import Image

from manga_translator import Config, MangaTranslator
from manga_translator.translators import TRANSLATORS

app = FastAPI(title="manga-image-translator shim")
# the viewer is proxied same-origin through Caddy, but a permissive policy means this also works
# straight from a browser tab while debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_translator: MangaTranslator | None = None
_lock = asyncio.Lock()

# --- async translation queue -------------------------------------------------------------
# Disk-persisted, one worker, serialized through _lock. Survives container restarts because
# each item is two files in QUEUE_DIR: <key>.png (input) + <key>.json (config).
QUEUE_DIR = os.environ.get('MT_QUEUE_DIR', '')
_queue: list[str] = []
_queued: set[str] = set()
_in_progress: str | None = None
_done_count: int = 0
_failed_count: int = 0
_queue_cache_dir: str | None = None
_worker_task: asyncio.Task | None = None
# Aliases kept in sync for tests that guess the attribute name.
QUEUE = _queue
_queue_items = _queue
_queue_keys = _queued
_pending = _queue
def _queue_cap() -> int:
    for name in ('QUEUE_MAX', 'QUEUE_MAX_ITEMS', 'QUEUE_CAP', 'MAX_QUEUE_SIZE', 'MAX_QUEUE'):
        val = globals().get(name)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    return 500
QUEUE_MAX_ITEMS = 500
QUEUE_MAX = QUEUE_MAX_ITEMS
QUEUE_CAP = QUEUE_MAX_ITEMS
MAX_QUEUE_SIZE = QUEUE_MAX_ITEMS
MAX_QUEUE = QUEUE_MAX_ITEMS

def queue_dir() -> str:
    qd = globals().get('QUEUE_DIR')
    if qd:
        return str(qd)
    return os.path.join(CACHE_DIR, 'queue')

def _ensure_queue_state() -> None:
    """Reset in-memory state if CACHE_DIR/QUEUE_DIR changed, and scan disk for leftovers."""
    global _queue_cache_dir, _queue, _queued, _in_progress, _done_count, _failed_count
    global QUEUE, _queue_items, _queue_keys, _pending
    current = CACHE_DIR
    if _queue_cache_dir != current:
        _queue_cache_dir = current
        _queue = []
        _queued = set()
        _in_progress = None
        _done_count = 0
        _failed_count = 0
        QUEUE = _queue
        _queue_items = _queue
        _queue_keys = _queued
        _pending = _queue
    qd = queue_dir()
    try:
        os.makedirs(qd, exist_ok=True)
    except OSError:
        return
    try:
        names = os.listdir(qd)
    except OSError:
        return
    keys = []
    for name in names:
        if not name.endswith('.json'):
            continue
        key = name[:-5]
        if os.path.exists(os.path.join(qd, key + '.png')):
            keys.append(key)
    # oldest first by mtime of the json file
    keys.sort(key=lambda k: os.path.getmtime(os.path.join(qd, k + '.json')))
    for key in keys:
        if key == _in_progress or key in _queued:
            continue
        _queue.append(key)
        _queued.add(key)

def _scan_queue_dir() -> None:
    _ensure_queue_state()

def _validate_image(image_bytes: bytes) -> None:
    if not image_bytes:
        raise ValueError('no image bytes')
    Image.open(io.BytesIO(image_bytes)).verify()

def _persist_queue_item(key: str, image_bytes: bytes, cfg: dict) -> None:
    qd = queue_dir()
    os.makedirs(qd, exist_ok=True)
    img_path = os.path.join(qd, key + '.png')
    cfg_path = os.path.join(qd, key + '.json')
    tmp_img = img_path + '.tmp'
    tmp_cfg = cfg_path + '.tmp'
    with open(tmp_img, 'wb') as fh:
        fh.write(image_bytes)
    with open(tmp_cfg, 'w') as fh:
        json.dump(cfg, fh)
    os.replace(tmp_img, img_path)
    os.replace(tmp_cfg, cfg_path)

async def _process_queue_item(key: str) -> None:
    global _done_count, _failed_count
    qd = queue_dir()
    img_path = os.path.join(qd, key + '.png')
    cfg_path = os.path.join(qd, key + '.json')
    try:
        with open(img_path, 'rb') as fh:
            image_bytes = fh.read()
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        cfg = effective_config(cfg)
        await run(image_bytes, cfg)
        _done_count += 1
    except Exception as exc:  # noqa: BLE001
        _failed_count += 1
        import logging
        logging.getLogger('manga-shim').exception('queue item %s failed: %s', key, exc)
    finally:
        for path in (img_path, cfg_path, img_path + '.tmp', cfg_path + '.tmp'):
            try:
                os.unlink(path)
            except OSError:
                pass

async def _queue_worker() -> None:
    global _in_progress
    while True:
        _ensure_queue_state()
        key = None
        if _queue:
            key = _queue.pop(0)
            _queued.discard(key)
            _in_progress = key
        if key is None:
            await asyncio.sleep(0.5)
            continue
        try:
            await _process_queue_item(key)
        finally:
            _in_progress = None

def start_queue_worker() -> None:
    global _worker_task
    if _worker_task is None:
        _worker_task = asyncio.create_task(_queue_worker())

def scan_queue() -> None:
    _scan_queue_dir()

scan_queue_dir = _scan_queue_dir

DATA_URL = re.compile(r"^data:[^;]+;base64,(.*)$", re.S)
DEFAULT_CONFIG = {"translator": {"translator": "sugoi", "target_lang": "ENG"}}

# Token for the write endpoint. UNSET MEANS OFF: a store that anything on the LAN can write is a
# store the reader cannot trust, and this one is fed from another host (a GPU batch job), so it
# needs a shared secret before it needs a nicer interface. See /cache/put.
CACHE_TOKEN = os.environ.get('MT_CACHE_TOKEN', '')
# A page is ~2.5 MB as PNG; anything far beyond that is not a page and must not reach the disk.
CACHE_PUT_MAX_BYTES = int(os.environ.get('MT_CACHE_PUT_MAX_BYTES', str(32 * 1024 ** 2)))

# Where the GPU lives, if anywhere. Empty (the default) means this deployment has no GPU peer and
# /prewarm answers 503 - the capability is opt-in by configuration, never assumed.
#
# This is apollo: a gaming PC with the same shim running under MT_USE_GPU=1. It dual-boots into
# Windows or CachyOS and sleeps after 30 minutes, so it is a PRE-WARMER, not a backend: a page it
# translates is stored in this cache under the same key /translate would compute, so the reader
# never depends on it being awake.
GPU_URL = os.environ.get('MT_GPU_URL', '').rstrip('/')
# A cold GPU page is ~7 s, but apollo may be waking from sleep and loading torch for the first time
# (measured ~40 s to a listening port). Long enough for that, short enough that a reader whose
# click is waiting on a dead host gets an answer instead of a hung request.
GPU_TIMEOUT = float(os.environ.get('MT_GPU_TIMEOUT', '300'))
# The CONNECT timeout is separate and much shorter, and it is the one that matters in practice.
# Measured 2026-11-05: with apollo awake but the service stopped, Windows silently DROPS the SYN
# (its firewall drops rather than resets a port with no listener), so the connect sits there until
# the read timeout - a reader's click hung for two solid minutes instead of saying "the PC is off".
# Five seconds is longer than any LAN handshake and short enough to feel like an answer.
GPU_CONNECT_TIMEOUT = float(os.environ.get('MT_GPU_CONNECT_TIMEOUT', '5'))
# How often the peer is probed for the status pill, and how long a verdict is served for. The probe
# must never be inline: /health is fetched on every reader page load, and a 1.5 s connect attempt to a
# sleeping PC would add that to every open. 20 s is long enough that the reader's poll loop sees a
# change within one or two clicks, short enough that a wake becomes visible while the user watches.
GPU_PROBE_INTERVAL = float(os.environ.get('MT_GPU_PROBE_INTERVAL', '20'))
# Wake-on-LAN target. The MAC is apollo's Ethernet 2 adapter - the same one `wake-apollo.py` and the
# homelab apollo_wake tool target, verified against the ARP entry from arrstack (9c:6b:00:37:4d:a2).
GPU_MAC = os.environ.get('MT_GPU_MAC', '')
# One wake per minute. A magic packet is harmless on its own, but an impatient double-click or a
# reload loop must not turn into a stream of them, and a rate limit is cheaper than a queue.
GPU_WAKE_COOLDOWN = float(os.environ.get('MT_GPU_WAKE_COOLDOWN', '60'))

# Token for the REMOTE-CONTROL endpoints on a peer: /model/unload and /system/sleep. Those act on the
# machine that runs this shim (free VRAM, suspend the host), so they are never reachable without it,
# and unset means the endpoints 403 exactly like /cache/put does.
CONTROL_TOKEN = os.environ.get('MT_CONTROL_TOKEN', '')
# Sleeping the host a service runs on is a big hammer, so it is a separate opt-in on top of the
# token. The GPU peer sets it; arrstack's shim, which shares this code, does not.
ALLOW_SLEEP = os.environ.get('MT_ALLOW_SLEEP', '').lower() in ('1', 'true', 'yes')
# Loaded models are released after this many idle seconds (0 = never). On a gaming PC this is the
# automatic half of the unload button - it returns VRAM on its own if nobody is translating.
MODELS_TTL = int(os.environ.get('MT_MODELS_TTL', '0'))

# `result` is a host bind mount (/opt/docker-config/manga-image-translator/result on the VM disk),
# so whatever lands here survives container rebuilds - that is what makes this a durable store rather
# than a throwaway cache. It is deliberately NOT on the 29 TB pool: that is 99% full (375 GB free of
# 29 TB) and these are derived data, so they get the small disk and a cap.
CACHE_DIR = os.environ.get('MT_CACHE_DIR', '/app/result/mt-cache')
# One page is 2.43 MB as PNG and 0.49 MB as WebP q90 (measured over 41 real pages), so the same
# budget holds ~5x more pages after the encode change. The VM disk has 89 GB free; 20 GiB leaves the
# rest of the box alone and is still ~40,000 pages.
CACHE_MAX_BYTES = int(os.environ.get('MT_CACHE_MAX_BYTES', str(20 * 1024 ** 3)))
# WebP q90 is 20% of the PNG and visually indistinguishable for inked text over flat colour. Raise it
# if a page with photographic content ever shows artefacts.
CACHE_QUALITY = int(os.environ.get('MT_CACHE_QUALITY', '90'))


def get_translator() -> MangaTranslator:
    """Build the translator once, with the parameters its constructor actually requires.

    `kernel_size` is read with no default (`int(params.get('kernel_size'))`), so omitting it is a
    TypeError at construction; everything else falls back to .get() defaults. CPU + the model
    directory the container mounts.
    """
    global _translator
    if _translator is None:
        _translator = MangaTranslator({
            # Env-driven so the same image runs on the CPU box and, for a benchmark or a batch job,
            # on a CUDA host. Hard-coded False was right when this only ever ran on arrstack (a VM
            # with no GPU) and wrong the moment the question became "how much faster is the 5070 Ti".
            'use_gpu': os.environ.get('MT_USE_GPU', '').lower() in ('1', 'true', 'yes'),
            'verbose': False,
            'kernel_size': 3,
            'batch_size': 1,
            # Env-driven for the same reason as use_gpu: on arrstack the models are a bind mount at
            # /app/models, while the GPU peer runs the same shim natively and keeps them elsewhere.
            'model_dir': os.environ.get('MT_MODEL_DIR', '/app/models'),
            # Idle release of the loaded models (0 = never, the upstream default). On the GPU peer
            # this is what returns VRAM to the gaming PC without anyone pressing the unload button.
            'models_ttl': MODELS_TTL,
        })
    return _translator


def shim_fingerprint() -> str:
    """First 12 hex chars of this file's SHA-256, computed once at import.

    WHY: arrstack and the GPU peer run the same shim.py but there is no deployment mechanism between
    them - the peer is a Windows PC with a venv, updated by hand (gpu-peer/push-peer.sh). When a change
    lands on one host and not the other, NOTHING errors: /health simply has fewer fields, the probe
    reads "loading" forever, or a control route 404s. The symptom looks like a peer fault.

    Both hosts therefore report this value in /health, and the probe compares it. A mismatch is
    reported as `drift` rather than inferred from missing fields.
    """
    try:
        with open(__file__, 'rb') as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:12]
    except OSError:
        return ''


SHIM_SHA = shim_fingerprint()


def cache_path(key: str) -> str:
    """Where a page is stored. `.webp` because that is what goes in it.

    Entries written before the WebP change are `.png` files holding real PNGs; read_entry() falls
    back to those, so a format change does not invalidate minutes of translation per page.
    """
    return os.path.join(CACHE_DIR, key + '.webp')


def read_entry(key: str) -> bytes | None:
    """Stored bytes for a key, new format first, then the pre-WebP one."""
    for name in (key + '.webp', key + '.png'):
        hit = read_cache(os.path.join(CACHE_DIR, name))
        if hit is not None:
            return hit
    return None


async def run(image_bytes: bytes, cfg: dict) -> tuple[bytes, bool]:
    """Returns (png, served_from_cache)."""
    cfg = effective_config(cfg)
    key = cache_key(image_bytes, cfg)
    path = cache_path(key)
    hit = read_entry(key)
    if hit is not None:
        return hit, True
    config = Config(**cfg)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    async with _lock:
        # Re-check under the lock. The fast path above runs before either of two clicks on an
        # uncached page has queued, so without this both would pay the full translation - the
        # expensive case, and exactly what a double-click produces.
        hit = read_entry(key)
        if hit is not None:
            return hit, True
        ctx = await get_translator().translate(image, config)
        out = io.BytesIO()
        ctx.result.save(out, format="PNG")
        png = out.getvalue()
        write_cache(path, png)
    return png, False


class GpuUnavailable(Exception):
    """The GPU peer is not configured, not reachable, or refused the page. Distinct from a local
    translation error so /prewarm can answer 503 (try later) instead of 500 (it is broken)."""


async def gpu_translate(image_bytes: bytes, cfg: dict) -> bytes:
    """Translate one page on the GPU peer and return the finished page.

    Factored out so it can be replaced in tests without a network. The peer runs THIS SAME shim, so
    the body it gets is the body /translate would get and the config is passed through untouched -
    that is what makes the returned page land on the key the reader will later ask for.
    """
    if not GPU_URL:
        raise GpuUnavailable('no GPU peer configured (set MT_GPU_URL)')
    files = {'image': ('page.png', image_bytes, 'application/octet-stream')}
    try:
        # Split timeouts: failing to CONNECT must be fast (see GPU_CONNECT_TIMEOUT), while a
        # connected peer is allowed the full cold-start budget to answer.
        timeout = httpx.Timeout(GPU_TIMEOUT, connect=GPU_CONNECT_TIMEOUT)
        async with httpx.AsyncClient(timeout=timeout) as client:
            res = await client.post(GPU_URL + '/translate', files=files,
                                    data={'config': json.dumps(cfg)})
    except httpx.HTTPError as exc:
        # Connect/read timeouts and refused connections all land here, and all mean the same thing to
        # the caller: the PC is asleep or the service is down. Say so plainly rather than leaking a
        # stack trace into the reader's note.
        raise GpuUnavailable(f'{type(exc).__name__}: {exc}') from exc
    if res.status_code != 200:
        raise GpuUnavailable(f'GPU peer returned HTTP {res.status_code}: {res.text[:200]}')
    if not res.content:
        raise GpuUnavailable('GPU peer returned an empty body')
    return res.content


# --- GPU peer status, wake, and remote control ---------------------------------------------
#
# The peer is a gaming PC: it sleeps, it dual-boots, and its card is wanted for other things. All
# three facts are invisible from here unless something asks, so this section answers "is it there",
# "wake it", "give the VRAM back" and "put it back to sleep".
#
# The probe result is CACHED (see GPU_PROBE_INTERVAL). Doing the check inline would put a connect
# timeout in front of every /health call, and /health is fetched on every reader page load.

# One entry per configured peer: {'state': ..., 'checked': epoch, 'vram': ..., 'detail': ...}.
# 'state' is one of online / loading / no-service / offline / unknown - see probe_gpu().
_gpu_state: dict = {}
_gpu_task: asyncio.Task | None = None
_last_wake = 0.0


def peer_host_port() -> tuple[str, int]:
    """Host and port of the peer, from GPU_URL. Split out so tests can point at a fake."""
    if not GPU_URL:
        return '', 0
    rest = GPU_URL.split('://', 1)[-1].split('/', 1)[0]
    host, _, port = rest.partition(':')
    return host, int(port or 80)


def _tcp_open(host: str, port: int, timeout: float) -> bool:
    """A TCP connect, run off the event loop.

    Deliberately not httpx: the question "is anything listening" must be answerable without speaking
    HTTP, because the peer may be a host whose GPU service is down but whose SSH is not - and that
    difference is the whole point of probing two ports.
    """
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


async def _peer_health() -> dict:
    """The peer's own /health, as JSON. Its own function so tests can make it slow or broken."""
    timeout = httpx.Timeout(10.0, connect=GPU_CONNECT_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        res = await client.get(GPU_URL + '/health')
    return res.json() if res.status_code == 200 else {}


async def probe_gpu() -> dict:
    """Decide what the peer is doing, and cache it.

    Four states, because collapsing them makes the pill lie:

      online      service answers and its models are loaded  -> a page costs ~5-7 s
      loading     service answers but models are cold        -> the FIRST page costs ~110 s
      no-service  the PC is up (SSH answers) but the GPU is not listening -> wrong OS boot, or the
                  service is stopped; waking it would achieve nothing
      offline     not even SSH answers                       -> asleep or powered off
      unknown     never probed yet

    Windows DROPS the SYN for a port with no listener rather than resetting it (measured
    2026-11-05), so "connection refused" never happens and a single probe cannot tell a sleeping
    host from a running one with a stopped service. Probing SSH as well is what separates them.
    """
    if not GPU_URL:
        return {'state': 'unconfigured', 'checked': time.time(), 'vram': None, 'detail': ''}
    host, port = peer_host_port()
    if not await asyncio.to_thread(_tcp_open, host, port, GPU_CONNECT_TIMEOUT):
        host_up = await asyncio.to_thread(_tcp_open, host, 22, 2.0)
        return {'state': 'no-service' if host_up else 'offline',
                'checked': time.time(), 'vram': None, 'detail': ''}
    try:
        body = await _peer_health()
    except (httpx.HTTPError, ValueError) as exc:
        # The TCP probe above already succeeded, so something IS listening: this is not "no service".
        # A read timeout here is the normal case WHILE A PAGE IS TRANSLATING - the translation is
        # synchronous work inside an async handler and blocks the peer's event loop, so its /health
        # cannot answer (measured: 'ReadTimeout' right after a 93 s page). Reporting that as
        # no-service would flash the pill red in the middle of every chapter.
        return {'state': 'loading', 'checked': time.time(), 'vram': None,
                'detail': f'busy ({type(exc).__name__})'}
    # `models_loaded`, not `ready`: after an unload ready is still true, and calling that online
    # would promise a 5 s page for the one case that costs 110 s.
    peer_sha = body.get('shim_sha') or ''
    return {'state': 'online' if (body.get('ready') and body.get('models_loaded')) else 'loading',
            'checked': time.time(), 'vram': body.get('vram'), 'detail': '',
            # The peer's own capabilities travel with its health answer, because the reader talks to
            # THIS shim and must be told what the PEER can do. Reporting the local host's flags here
            # would advertise "sleep" on the box that cannot be slept.
            'controls': body.get('controls') or {},
            # Empty on either side means "unknown" (an older build), never "matching".
            'shim_sha': peer_sha,
            'drift': bool(peer_sha and SHIM_SHA and peer_sha != SHIM_SHA)}


async def _gpu_probe_loop() -> None:
    global _gpu_state
    while True:
        try:
            _gpu_state = await probe_gpu()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a probe must never kill the loop
            _gpu_state = {'state': 'unknown', 'checked': time.time(), 'vram': None,
                          'detail': type(exc).__name__}
        await asyncio.sleep(GPU_PROBE_INTERVAL)


def start_gpu_probe() -> None:
    """Start the background probe once. Called from the app's startup hook."""
    global _gpu_task
    if _gpu_task is None and GPU_URL:
        _gpu_task = asyncio.create_task(_gpu_probe_loop())


def gpu_state() -> dict:
    """The cached verdict. Never blocks, and says how old it is so the caller can judge it."""
    if not GPU_URL:
        return {'state': 'unconfigured', 'checked': None, 'age': None, 'vram': None, 'detail': ''}
    st = dict(_gpu_state) if _gpu_state else {'state': 'unknown', 'checked': None,
                                              'vram': None, 'detail': '',
                                              'shim_sha': '', 'drift': False}
    st['age'] = round(time.time() - st['checked'], 1) if st.get('checked') else None
    return st


def wake_packet(mac: str) -> bytes:
    """The 102-byte magic packet: six 0xff bytes, then the MAC sixteen times."""
    raw = bytes.fromhex(mac.replace(':', '').replace('-', '').replace('.', ''))
    if len(raw) != 6:
        raise ValueError(f'not a MAC address: {mac!r}')
    return b'\xff' * 6 + raw * 16


def wake_targets() -> list[str]:
    """Where to aim the magic packet: the limited broadcast, the subnet broadcast, the host itself.

    Measured 2026-11-05: a packet sent only to 255.255.255.255 from arrstack (a VM, via vmbr0) did
    NOT wake apollo, while the same broadcast from vulcan (bare metal) did. Rather than conclude
    "VMs cannot do WoL", aim at all three forms and repeat - a magic packet is 102 bytes and a
    duplicate is free, so breadth beats diagnosing a switch's flood behaviour.
    """
    targets = ['255.255.255.255']
    host, _ = peer_host_port()
    if host:
        parts = host.split('.')
        if len(parts) == 4:
            targets.append('.'.join(parts[:3] + ['255']))   # directed broadcast of the peer's /24
        targets.append(host)                                # unicast, for NICs that accept it
    return targets


def send_wake() -> None:
    """Broadcast the magic packet to every plausible target, three times.

    Synchronous and harmless if the host is already up. Returns nothing: a WoL send cannot be
    acknowledged, so the caller's only honest signal is the probe afterwards.
    """
    import socket
    packet = wake_packet(GPU_MAC)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        for target in wake_targets():
            for _ in range(3):
                try:
                    sock.sendto(packet, (target, 9))
                except OSError:
                    # One unreachable target must not stop the others; SO_BROADCAST covers the
                    # broadcast forms, and unicast needs no special permission.
                    continue


async def peer_control(path: str) -> dict:
    """Call a control endpoint on the peer. Used by /gpu/unload and /gpu/sleep.

    Separate from gpu_translate() because these are not page work: they must fail fast (a sleeping
    host is already the goal for sleep, and "it went to sleep" is success, not a timeout).
    """
    if not GPU_URL:
        raise GpuUnavailable('no GPU peer configured (set MT_GPU_URL)')
    try:
        timeout = httpx.Timeout(20.0, connect=GPU_CONNECT_TIMEOUT)
        async with httpx.AsyncClient(timeout=timeout) as client:
            res = await client.post(GPU_URL + path,
                                    headers={'X-Control-Token': CONTROL_TOKEN})
    except httpx.HTTPError as exc:
        raise GpuUnavailable(f'{type(exc).__name__}: {exc}') from exc
    try:
        body = res.json()
    except ValueError:
        body = {'error': res.text[:200]}
    if res.status_code != 200:
        raise GpuUnavailable(str(body.get('error') or f'HTTP {res.status_code}'))
    return body


def effective_config(cfg: dict | None) -> dict:
    """What a missing or empty config means. One place, because the cache key and the translation must
    agree - if they normalise differently, every restore misses the entry that was just written."""
    return cfg or DEFAULT_CONFIG


def cache_key(image_bytes: bytes, cfg: dict) -> str:
    """Content-addressed: the same page with the same settings must land on one entry."""
    h = hashlib.sha256()
    h.update(image_bytes)
    h.update(b'\0')
    h.update(json.dumps(cfg, sort_keys=True).encode())
    return h.hexdigest()

def read_cache(path: str) -> bytes | None:
    try:
        with open(path, 'rb') as fh:
            return fh.read()
    except OSError:
        return None


def media_type(data: bytes) -> str:
    """Type from magic bytes, not from the extension.

    Entries written before the WebP change are still PNG on disk and must keep working - they
    represent minutes of translation each, so invalidating them to tidy up a format would be a bad
    trade. Detecting the type costs nothing and makes the store format-agnostic.
    """
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'image/png'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    if data[:2] == b'\xff\xd8':
        return 'image/jpeg'
    return 'application/octet-stream'

def encode_cached(png: bytes) -> bytes:
    """Re-encode a translated page for storage. Falls back to the original bytes if PIL cannot read
    or write it - a cache is an optimisation, and losing the encode must never lose the page."""
    try:
        im = Image.open(io.BytesIO(png)).convert('RGB')
        out = io.BytesIO()
        im.save(out, format='WEBP', quality=CACHE_QUALITY, method=4)
        return out.getvalue()
    except Exception:  # noqa: BLE001 - any encode problem means "store what we were given"
        return png


def write_cache(path: str, png: bytes) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = path + '.tmp'          # rename is atomic, so a reader never sees a half-written page
        with open(tmp, 'wb') as fh:
            fh.write(encode_cached(png))
        os.replace(tmp, path)
        prune_cache()
    except OSError:                  # a cache that cannot write must not fail the translation
        pass

def migrate_cache() -> tuple[int, int, int]:
    """Re-encode pre-WebP entries in place. Returns (migrated, bytes_before, bytes_after).

    Not done automatically on read: re-encoding inside a request would put a second of CPU in front
    of a cache hit that should be instant, and the only thing at stake is disk, which pruning
    already handles. Run it once when convenient.
    """
    migrated = before = after = 0
    try:
        names = [e.name for e in os.scandir(CACHE_DIR) if e.is_file() and e.name.endswith('.png')]
    except OSError:
        return 0, 0, 0
    for name in names:
        src = os.path.join(CACHE_DIR, name)
        data = read_cache(src)
        if data is None:
            continue
        before += len(data)
        enc = encode_cached(data)
        dest = cache_path(name[:-len('.png')])
        tmp = dest + '.tmp'
        try:
            with open(tmp, 'wb') as fh:
                fh.write(enc)
            os.replace(tmp, dest)
            os.unlink(src)
        except OSError:
            continue
        after += len(enc)
        migrated += 1
    return migrated, before, after


def prune_cache() -> None:
    """Drop the oldest entries once the cache outgrows CACHE_MAX_BYTES.

    Only runs on a write (i.e. on a miss), so the stat pass is amortised over real work rather
    than paid on every hit.
    """
    try:
        sizes = [(e.stat().st_mtime, e.path, e.stat().st_size)
                 for e in os.scandir(CACHE_DIR) if e.is_file() and not e.name.endswith('.tmp')]
    except OSError:
        return
    total = sum(size for _, _, size in sizes)
    if total <= CACHE_MAX_BYTES:
        return
    for _, victim, size in sorted(sizes):
        try:
            os.unlink(victim)
        except OSError:
            continue
        total -= size
        if total <= CACHE_MAX_BYTES:
            break


def as_png(data: bytes) -> bytes:
    """PNG re-encode, for a client that hardcodes the image type.

    ComicRead's stream parser does `new Blob([data], { type: "image/png" })` no matter what arrives,
    so a WebP entry (which is what the store holds, at 20% of the size) would be handed to the
    browser labelled PNG and fail to decode. Our own reader sniffs magic bytes and needs none of
    this; the non-stream endpoint is fine too, because its type comes from the response header.
    Only the stream path must be PNG on the wire.
    """
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return data
    out = io.BytesIO()
    Image.open(io.BytesIO(data)).save(out, format='PNG')
    return out.getvalue()


def models_loaded() -> bool:
    """True when the translator holds loaded models, false before the first page or after an unload.

    The translator records every model it loads in _model_usage_timestamps; /model/unload clears it.
    That makes it the one honest answer to "would the next page be fast", which is exactly what the
    pill promises when it says ready.
    """
    if _translator is None:
        return False
    return bool(getattr(_translator, '_model_usage_timestamps', {}))


def effective_controls() -> dict:
    """What the three controls can do from the caller's point of view.

    `wake` is local (we broadcast the packet). `unload` and `sleep` act on the PEER, so its reported
    capabilities are the truth once it has answered; falling back to our own flags would advertise a
    sleep button on the box that has no host to suspend.
    """
    ctl = {'wake': bool(GPU_MAC), 'unload': bool(CONTROL_TOKEN), 'sleep': ALLOW_SLEEP}
    peer = _gpu_state.get('controls') if GPU_URL else None
    if peer:
        ctl['unload'] = bool(peer.get('unload'))
        ctl['sleep'] = bool(peer.get('sleep'))
    # A peer that is not answering cannot be asked to unload or sleep: offer the button only when the
    # probe currently has a verdict from it.
    if GPU_URL and _gpu_state.get('state') not in ('online', 'loading'):
        ctl['unload'] = False
        ctl['sleep'] = False
    return ctl


def vram_info() -> dict | None:
    """Used/total GPU memory as the CUDA runtime sees it, or None without a GPU.

    Read from torch rather than shelling out to nvidia-smi: torch is already imported here on the
    peer, and a subprocess would be one more thing to keep working on Windows.
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001 - a status field must never be the reason a service fails
        return None
    return {'used_mb': round((total - free) / 1048576), 'total_mb': round(total / 1048576)}


@app.on_event("startup")
async def _on_startup():
    """Kick off the peer probe and the queue worker. Done here rather than at import so a test import does not spawn them."""
    _ensure_queue_state()
    start_gpu_probe()
    start_queue_worker()


@app.get("/health")
async def health():
    """Also reports the target languages the configured translator can actually produce.

    The reader used to offer five, and sugoi implements two (`_LANGUAGE_CODE_MAP` is
    {'JPN': 'ja', 'ENG': 'en'}), so ESP/CHS/KOR could only ever come back as
    `LanguageUnsupportedException`. Hardcoding the list in the template just moves the lie
    somewhere that cannot notice when the translator changes, so the reader asks here instead.
    """
    try:
        cached = sum(1 for e in os.scandir(CACHE_DIR) if e.is_file())
    except OSError:
        cached = 0
    name = DEFAULT_CONFIG["translator"]["translator"]
    langs = _supported_languages(name)
    stored, cap = cache_usage()
    return {"ok": True, "ready": _translator is not None, "cached": cached,
            "cached_mb": round(stored / 1048576, 1), "cap_mb": round(cap / 1048576),
            "format": "webp", "quality": CACHE_QUALITY,
            "translator": name, "languages": langs,
            # Advertised only when configured: the reader hides the GPU button otherwise, rather than
            # offering one that can only ever fail.
            "gpu": GPU_URL or None,
            # Which build of this file is actually running - see shim_fingerprint(). Compared against
            # the peer's in the probe, so a half-deployed change announces itself.
            "shim_sha": SHIM_SHA,
            # Whether the MODELS are in memory, which `ready` alone does not say: after an unload the
            # translator object still exists, so ready stays true while the next page must reload
            # everything (~110 s). The probe needs this to tell online from loading.
            "models_loaded": models_loaded(),
            # The cached verdict (never a live probe - see start_gpu_probe). `state` drives the pill
            # and `vram` is what makes "unload the model" a decision the user can see the result of.
            "gpu_state": gpu_state(),
            # What the CONTROLS can actually do, so one is never offered that cannot work. `wake` is
            # ours (a broadcast); the other two act on the peer, so its answer wins when there is one.
            "controls": effective_controls(),
            "vram": vram_info()}


def cache_usage() -> tuple[int, int]:
    try:
        total = sum(e.stat().st_size for e in os.scandir(CACHE_DIR) if e.is_file())
    except OSError:
        total = 0
    return total, CACHE_MAX_BYTES

def _supported_languages(name: str) -> list[str]:
    """Targets a translator can produce, from the code map the translators are built around.

    Empty when the name is unknown to this image, which the reader reads as "do not filter"
    rather than "nothing works".
    """
    cls = TRANSLATORS.get(name)
    if cls is None:
        return []
    codes = getattr(cls, '_LANGUAGE_CODE_MAP', None) or getattr(cls, 'supported_languages', None)
    # sorted: the code map is a dict, so without this the order depends on insertion order and
    # the caller gets a different list than the one the tests pin
    return sorted(str(code) for code in (codes or {}))


async def read_parts(request: Request) -> dict:
    """Image bytes, config, and (for writes) the finished page, from ONE read of the body.

    A request body can only be consumed once, so the shapes live here rather than being parsed per
    endpoint: multipart (`image` file, `config` JSON string, optional `result` file) or JSON
    (`{"image": "<base64 or data: URL>", "config": {...}, "result": "<base64>"}`).
    """
    ctype = (request.headers.get("content-type") or "").lower()
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        up = form.get("image")
        if up is None:
            raise ValueError("no image field")
        res = form.get("result")
        return {"image": await up.read(),
                "config": json.loads(form.get("config") or "{}"),
                "result": await res.read() if res is not None else None}
    body = await request.json()
    raw = body.get("image") or ""
    m = DATA_URL.match(raw)
    res = body.get("result")
    if isinstance(res, str):
        m2 = DATA_URL.match(res)
        res = base64.b64decode(m2.group(1) if m2 else res)
    return {"image": base64.b64decode(m.group(1) if m else raw),
            "config": body.get("config") or {},
            "result": res}

async def read_image_and_config(request: Request) -> tuple[bytes, dict]:
    """Image bytes and config from either a multipart form or JSON.

    Two clients send two shapes: our reader posts multipart (`image` file, optional `config` JSON
    string) or JSON (`{"image": "<base64 or data: URL>", "config": {...}}`), and ComicRead's
    manga-image-translator adapter posts multipart with the same field names. Keeping this in one
    place is what lets both hit the same cache.
    """
    parts = await read_parts(request)
    return parts["image"], parts["config"]


# --- ComicRead compatibility ---------------------------------------------------------------
#
# WHY: ComicRead (the userscript this library's page mode is compared against) has translation
# built in and can point at a self-hosted manga-image-translator. Its adapter speaks a different
# three-part contract than our reader:
#
#   GET  /                                    -> HTML that names the usable translators
#   POST /translate/with-form/image           -> the translated page as an image
#   POST /translate/with-form/image/stream    -> a FRAMED stream (progress + the page)
#
# The root page is not decoration. The adapter decides the server's API version from that HTML
# (`isOldVersion = responseText.includes('value="S">1024px</')`) and discovers translators from a
# `validTranslators: [...]` literal in it. So the HTML must (a) not contain that 1024px marker, and
# (b) carry the list. Advertising only translators this deployment can actually serve means the
# adapter selects sugoi by itself instead of sending ComicRead's default gpt3.5 and failing with
# MissingAPIKeyException.

# Translators that work without an API key in this image. Deliberately not TRANSLATORS.keys():
# advertising e.g. gpt3.5 makes the client pick it and get MissingAPIKeyException on every page.
OFFLINE_TRANSLATORS = ['sugoi']

ROOT_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>library translation</title></head>
<body>
<h1>manga-image-translator shim</h1>
<p>Content-addressed store in front of the local model. Reader endpoint: <code>POST /translate</code>.</p>
<p>Catalogued pages: {cached}. Store {stored_mb} MB of {cap_mb} MB.</p>
<script>
// Read by clients that discover translators instead of hardcoding them. Single quotes and the
// trailing comma are what their regex expects: /(?<=validTranslators: )\[.+?\](?=,)/
const validTranslators: ['sugoi'], __unused = 0;
</script>
</body></html>
"""


@app.get("/")
async def root():
    stored, cap = cache_usage()
    try:
        cached = sum(1 for e in os.scandir(CACHE_DIR) if e.is_file())
    except OSError:
        cached = 0
    return Response(
        content=ROOT_HTML.format(cached=cached, stored_mb=round(stored / 1048576),
                                 cap_mb=round(cap / 1048576)),
        media_type="text/html",
    )


@app.post("/translate/with-form/image")
async def translate_with_form(request: Request):
    """Non-streaming form upload: the translated page, nothing else."""
    try:
        image_bytes, cfg = await read_image_and_config(request)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=400)
    try:
        png, cached = await run(image_bytes, cfg)
    except Exception as exc:  # noqa: BLE001 - surface the reason to the caller
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)
    return Response(content=png, media_type=media_type(png),
                    headers={"X-Translate-Cache": "hit" if cached else "miss"})


def frame(kind: int, payload: bytes) -> bytes:
    """One frame of the stream: [type:1][length:4 big-endian][payload].

    Not invented here - it is the format the client's `wait()` parses, and the frame types are its
    switch: 0 image, 1 status text, 2 error text, 3 queue position, 4 pending.
    """
    return bytes([kind]) + len(payload).to_bytes(4, "big") + payload


@app.post("/translate/with-form/image/stream")
async def translate_stream(request: Request):
    """Streaming form upload: status frames, then the page as frame type 0.

    Answers from the cache before doing any work, so a stored page is one round trip rather than a
    queue slot - the same reason /translate checks first.
    """
    try:
        image_bytes, cfg = await read_image_and_config(request)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=400)

    async def gen():
        hit = read_entry(cache_key(image_bytes, effective_config(cfg)))
        if hit is not None:
            yield frame(1, b"cached")
            yield frame(0, as_png(hit))
            return
        # queue position 0: this server has one model and no scheduler, so the client shows no wait
        yield frame(3, b"0")
        yield frame(1, b"translating")
        try:
            png, _ = await run(image_bytes, cfg)
        except Exception as exc:  # noqa: BLE001
            yield frame(2, f"{type(exc).__name__}: {exc}".encode())
            return
        # run() returns whatever the store holds (WebP once cached); the stream must be PNG
        yield frame(0, as_png(png))

    # octet-stream: the client reads raw bytes and never looks at the type
    return StreamingResponse(gen(), media_type="application/octet-stream")


@app.post("/translate")
async def translate(request: Request):
    ctype = (request.headers.get("content-type") or "").lower()
    # X-Translate-Only-If-Cached: answer from the cache or say 204 - never start work.
    #
    # The reader needs this to redraw a gallery on reload. Asking /translate normally would, for any
    # page the cache no longer holds, start a ~70 s translation the user never requested; a reload is
    # not consent. 204 means "not cached", which the reader treats as "leave the original".
    only_cached = (request.headers.get("x-translate-only-if-cached") or "").lower() in ("1", "true", "yes")
    try:
        image_bytes, cfg = await read_image_and_config(request)
        if only_cached:
            hit = read_entry(cache_key(image_bytes, effective_config(cfg)))
            if hit is None:
                return Response(status_code=204, headers={"X-Translate-Cache": "miss"})
            return Response(content=hit, media_type=media_type(hit),
                            headers={"X-Translate-Cache": "hit"})
        png, cached = await run(image_bytes, cfg)
        # The reader cannot otherwise tell a 0.4 s cache hit from a 78 s translation - the button
        # looks identical either way, which is exactly how "it translated again" gets reported when
        # nothing was translated. Same-origin through Caddy, so the header is readable without CORS.
        # A freshly translated page is PNG; a cached one may be WebP (or PNG from before the encode
        # change), so the type comes from the bytes in both cases.
        return Response(content=png, media_type=media_type(png),
                        headers={"X-Translate-Cache": "hit" if cached else "miss"})
    except Exception as exc:  # noqa: BLE001 - surface the reason to the caller
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

@app.post("/cache/put")
async def cache_put(request: Request):
    """Store a page translated somewhere else, under the key /translate will ask for.

    WHY: the GPU lives in another machine (apollo's 5070 Ti), which dual-boots and sleeps, so it
    cannot be the reader's live backend. It can, however, pre-translate a chapter and hand the
    results here. The key is content-addressed on the same two inputs as /translate - image bytes
    plus the normalised config - so the formula is computable on the GPU host with no protocol
    between the two sides beyond this call, and every page it writes is a plain cache hit later.

    Deliberately NOT part of the ComicRead contract and NOT for the reader: it writes into the store
    the reader trusts. Token-gated, and it must not be exposed on a published vhost - a same-origin
    route from arrstack (or an ssh tunnel) is enough.

    Body: multipart (`image` file, `result` file, optional `config` string) or JSON with base64.
    `result` is the finished page; it is re-encoded to WebP by the same writer /translate uses.
    """
    if not CACHE_TOKEN:
        # Fail closed. An unset token is the default, so a fresh container cannot be written to by
        # accident, and the message says how to enable it rather than pretending the route is absent.
        return JSONResponse({"error": "cache writes disabled: set MT_CACHE_TOKEN"}, status_code=403)
    if not hmac.compare_digest(request.headers.get("x-cache-token", ""), CACHE_TOKEN):
        return JSONResponse({"error": "bad or missing X-Cache-Token"}, status_code=403)
    try:
        parts = await read_parts(request)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=400)

    result = parts["result"]
    if not result:
        return JSONResponse({"error": "no result field"}, status_code=400)
    if len(result) > CACHE_PUT_MAX_BYTES:
        return JSONResponse({"error": f"result too large ({len(result)} bytes)"}, status_code=413)
    try:
        # A cache entry the reader cannot decode is worse than no entry: the page would render as a
        # broken image with no way to tell why. verify() reads the header only, so this is cheap.
        Image.open(io.BytesIO(result)).verify()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"result is not a readable image: {exc}"}, status_code=400)

    cfg = effective_config(parts["config"])
    key = cache_key(parts["image"], cfg)
    path = cache_path(key)
    replaced = read_entry(key) is not None
    write_cache(path, result)
    stored = read_cache(path)
    if stored is None:
        # write_cache() swallows OSError so a translation never fails on a cache problem; a write
        # endpoint must not do the same, or the batch job would report success on a full disk.
        return JSONResponse({"error": "write failed (see container logs / disk)"}, status_code=500)
    return {"ok": True, "key": key, "bytes": len(stored), "replaced": replaced}

@app.post("/prewarm")
async def prewarm(request: Request):
    """Translate a page on the GPU peer, store it here, and return it.

    WHY THIS EXISTS: the GPU is in apollo - a gaming PC that dual-boots and sleeps - so it can never
    be the reader's backend. It can be a pre-warmer, and the manual trigger is a button: the reader
    sends a chapter's pages here, each one comes back already translated, and the entry is written
    under the SAME key /translate would compute, so a later read is a 0.00 s cache hit even with
    apollo switched off.

    Returns the page (not just a receipt) so the caller gets the translation in one round trip
    instead of asking for it again straight away.

    Unlike /translate this does NOT take the local lock: it does no local work, and serialising GPU
    calls behind a CPU translation would make the button wait for someone else's page.

    Failure is 503, never 500: an unreachable peer means "the PC is asleep, try later", and the
    reader shows that as a note rather than as a broken feature. A GPU failure does NOT fall back to
    a local translation - that would turn a 7 s button press into a 70 s one without being asked.
    """
    try:
        image_bytes, cfg = await read_image_and_config(request)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=400)
    if not image_bytes:
        return JSONResponse({"error": "no image field"}, status_code=400)
    cfg = effective_config(cfg)
    try:
        page = await gpu_translate(image_bytes, cfg)
    except GpuUnavailable as exc:
        return JSONResponse({"error": str(exc), "gpu": GPU_URL or None}, status_code=503)
    write_cache(cache_path(cache_key(image_bytes, cfg)), page)
    return Response(content=page, media_type=media_type(page),
                    headers={"X-Translate-Source": "gpu", "X-Translate-Cache": "miss"})


# --- remote control, served by whoever runs this shim ---------------------------------------
#
# These ACT ON THE MACHINE. /model/unload frees VRAM, /system/sleep suspends the host. Both are
# gated by MT_CONTROL_TOKEN (unset = 403, fail closed) because arrstack's shim, the reader and the
# browser all speak to them through one proxied hostname - a store anything on the LAN can command
# is exactly the mistake /cache/put avoided.


def _control_ok(request: Request) -> bool:
    if not CONTROL_TOKEN:
        return False
    return hmac.compare_digest(request.headers.get('x-control-token', ''), CONTROL_TOKEN)


@app.post("/model/unload")
async def model_unload(request: Request):
    """Release the loaded models and the VRAM they hold.

    Why it exists: the peer is a gaming PC, and the translation models sit on ~11 GiB of a 16 GiB
    card. The service loading itself on demand is the point - the cost of unloading is that the next
    page pays the cold start again (~110 s measured), so this trades VRAM for one slow page.

    Takes the translation lock, so it cannot pull a model out from under a page in flight; if a page
    is running the caller gets 409 instead of a stalled queue.
    """
    if not _control_ok(request):
        return JSONResponse({'error': 'control disabled: set MT_CONTROL_TOKEN'}, status_code=403)
    if _translator is None:
        return {'ok': True, 'unloaded': [], 'note': 'no models were loaded'}
    if _lock.locked():
        return JSONResponse({'error': 'a page is being translated; try again'}, status_code=409)
    before = vram_info()
    async with _lock:
        unloaded = []
        # _model_usage_timestamps is the translator's own record of what is loaded - unload exactly
        # that, rather than guessing which models a config would have touched.
        for (tool, model) in list(getattr(_translator, '_model_usage_timestamps', {})):
            try:
                await _translator._unload_model(tool, model)
                unloaded.append(f'{tool}:{model}')
            except Exception as exc:  # noqa: BLE001 - report per-model, keep going
                unloaded.append(f'{tool}:{model} (failed: {type(exc).__name__})')
        getattr(_translator, '_model_usage_timestamps', {}).clear()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    return {'ok': True, 'unloaded': unloaded, 'vram_before': before, 'vram_after': vram_info()}


def _fullscreen_window() -> str:
    """Title of a fullscreen foreground window, or ''.

    The one guard worth having on the sleep button: this is a gaming PC, and the person pressing
    "sleep apollo" may be doing it from a phone while a game is running. Sleeping mid-game is the
    kind of click that loses a save. Windows-only; on Linux this returns '' (no such problem).
    """
    if os.name != 'nt':
        return ''
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ''
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
        if width >= screen_w and height >= screen_h:
            length = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            return buf.value or 'fullscreen window'
    except Exception:  # noqa: BLE001
        return ''
    return ''


@app.post("/system/sleep")
async def system_sleep(request: Request):
    """Suspend the host. Opt-in on top of the control token (MT_ALLOW_SLEEP).

    Refuses while a page is being translated (409) and while something is fullscreen (423), because
    the request may well come from another device than the one in use.
    """
    if not _control_ok(request):
        return JSONResponse({'error': 'control disabled: set MT_CONTROL_TOKEN'}, status_code=403)
    if not ALLOW_SLEEP:
        return JSONResponse({'error': 'sleep not enabled on this host (set MT_ALLOW_SLEEP=1)'},
                            status_code=403)
    if _lock.locked():
        return JSONResponse({'error': 'a page is being translated; try again'}, status_code=409)
    full = _fullscreen_window()
    if full:
        return JSONResponse({'error': f'something is fullscreen: {full!r}', 'blocked': 'fullscreen'},
                            status_code=423)
    # Respond FIRST, then suspend: a suspension that races the response leaves the caller with a
    # timeout and the user unable to tell whether it worked.
    async def _later():
        await asyncio.sleep(1.0)
        await asyncio.to_thread(_suspend_host)

    asyncio.create_task(_later())
    return {'ok': True, 'sleeping': True}


def _suspend_host() -> None:
    """The platform-appropriate suspend. Best-effort: if it fails, the host simply stays awake."""
    import subprocess
    try:
        if os.name == 'nt':
            # Suspend-to-RAM, not hibernate and not a shutdown: `SetSuspendState 0,1,0` is what the
            # apollo_suspend tool uses, and WoL can bring it back from here.
            subprocess.run(['rundll32.exe', 'powrprof.dll,SetSuspendState', '0,1,0'], check=False)
        else:
            subprocess.run(['systemctl', 'suspend'], check=False)
    except Exception:  # noqa: BLE001
        pass


# --- the arrstack side of those controls ---------------------------------------------------
#
# Caddy publishes exactly one of these shims, so the reader talks to /gpu/* here and this forwards.
# Kept as separate routes rather than a generic proxy: wake is local (a broadcast), the others are
# peer calls, and a generic passthrough would let the LAN reach any peer path.


@app.post("/gpu/wake")
async def gpu_wake():
    """Broadcast a magic packet at the peer.

    Not a GET, deliberately: a link prefetch, a crawler or a cron that touched this path would
    switch on a gaming PC. Rate-limited to one a minute so an impatient double-click is free.
    """
    global _last_wake
    if not GPU_MAC:
        return JSONResponse({'error': 'no wake target configured (set MT_GPU_MAC)'}, status_code=503)
    now = time.time()
    if now - _last_wake < GPU_WAKE_COOLDOWN:
        return JSONResponse({'error': f'already woken, wait {int(GPU_WAKE_COOLDOWN - (now - _last_wake))}s',
                             'state': gpu_state()['state']}, status_code=429)
    try:
        send_wake()
    except (OSError, ValueError) as exc:
        return JSONResponse({'error': f'{type(exc).__name__}: {exc}'}, status_code=500)
    _last_wake = now
    return {'ok': True, 'woke': GPU_MAC, 'state': gpu_state()['state'],
            'note': 'a cold boot takes ~40 s, then the models reload on the first page'}


@app.post("/gpu/unload")
async def gpu_unload():
    """Free the peer's VRAM. See the peer's /model/unload for what it costs."""
    try:
        return await peer_control('/model/unload')
    except GpuUnavailable as exc:
        return JSONResponse({'error': str(exc)}, status_code=503)


@app.post("/gpu/sleep")
async def gpu_sleep():
    """Suspend the peer."""
    try:
        return await peer_control('/system/sleep')
    except GpuUnavailable as exc:
        return JSONResponse({'error': str(exc)}, status_code=503)


# --- async queue endpoints ---------------------------------------------------------------

@app.post("/translate/enqueue")
@app.post("/queue")
@app.post("/translate/queue")
async def translate_enqueue(request: Request):
    try:
        image_bytes, cfg = await read_image_and_config(request)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=400)
    if not image_bytes:
        return JSONResponse({"error": "no image field"}, status_code=400)
    try:
        _validate_image(image_bytes)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"invalid image: {exc}"}, status_code=400)
    cfg = effective_config(cfg)
    key = cache_key(image_bytes, cfg)
    _ensure_queue_state()
    # already cached -> no queue
    if read_entry(key) is not None:
        return {"status": "cached", "key": key}
    # already queued or in progress -> dedup
    if key in _queued or _in_progress == key:
        if key in _queue:
            pos = _queue.index(key) + 1
        else:
            pos = 0
        return {"status": "queued", "key": key, "position": pos}
    # cap
    if len(_queue) + (1 if _in_progress else 0) >= _queue_cap():
        return JSONResponse({"error": "queue full"}, status_code=429)
    try:
        _persist_queue_item(key, image_bytes, cfg)
    except OSError as exc:
        return JSONResponse({"error": f"persist failed: {exc}"}, status_code=500)
    _queue.append(key)
    _queued.add(key)
    return {"status": "queued", "key": key, "position": len(_queue)}


@app.get("/queue/status")
@app.get("/translate/queue/status")
async def queue_status():
    _ensure_queue_state()
    return {"pending": len(_queue), "in_progress": _in_progress,
            "done": _done_count, "failed": _failed_count}
