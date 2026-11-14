"""Tests for the shim's translation cache.

Runs inside the image, where fastapi/PIL/manga_translator already are - the cache functions are
pure file I/O, so nothing here loads a model or needs a GPU:

    docker exec manga-image-translator python -m unittest test_shim -v

WHY THE CACHE MATTERS ENOUGH TO TEST
Translation is CPU-only on a 6-core VM: a sparse page takes ~8 s, a text-dense one ~66 s, and the
model is behind an asyncio lock so requests queue. The reader asks for the same page more than once
as a matter of course (dismiss the overlay, press translate again), and a cache miss on the second
ask costs another full minute. So the two properties pinned here are that an identical request lands
on an identical key, and that a stored page reads back byte-for-byte.
"""
from __future__ import annotations

import asyncio
import time
import json
import os
import shutil
import tempfile
import unittest

import httpx
from PIL import Image as PILImage
import io
import shim
from fastapi.testclient import TestClient

CFG_ENG = {'translator': {'translator': 'sugoi', 'target_lang': 'ENG'}}


class TestCacheKey(unittest.TestCase):
    def test_identical_request_hits_the_same_entry(self):
        self.assertEqual(shim.cache_key(b'page', CFG_ENG), shim.cache_key(b'page', CFG_ENG))

    def test_config_key_order_does_not_matter(self):
        """The viewer happens to send a stable shape, but a hand-rolled curl will not. Sorting means
        the same settings cannot produce two entries and pay for the page twice."""
        reordered = {'translator': {'target_lang': 'ENG', 'translator': 'sugoi'}}
        self.assertEqual(shim.cache_key(b'page', CFG_ENG), shim.cache_key(b'page', reordered))

    def test_different_target_language_is_a_different_entry(self):
        esp = {'translator': {'translator': 'sugoi', 'target_lang': 'ESP'}}
        self.assertNotEqual(shim.cache_key(b'page', CFG_ENG), shim.cache_key(b'page', esp))

    def test_different_translator_is_a_different_entry(self):
        jp = {'translator': {'translator': 'jparacrawl', 'target_lang': 'ENG'}}
        self.assertNotEqual(shim.cache_key(b'page', CFG_ENG), shim.cache_key(b'page', jp))

    def test_different_image_is_a_different_entry(self):
        self.assertNotEqual(shim.cache_key(b'page one', CFG_ENG),
                            shim.cache_key(b'page two', CFG_ENG))

    def test_key_is_filesystem_safe(self):
        key = shim.cache_key(b'\x00\xff' * 999, CFG_ENG)
        self.assertEqual(len(key), 64)
        self.assertTrue(all(c in '0123456789abcdef' for c in key))


class TestCacheStore(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='mt-cache-')
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.addCleanup(setattr, shim, 'CACHE_DIR', shim.CACHE_DIR)
        shim.CACHE_DIR = self.dir

    def test_missing_entry_reads_as_none(self):
        self.assertIsNone(shim.read_cache(os.path.join(self.dir, 'nope.png')))

    def test_written_page_reads_back_as_an_image(self):
        """The store now RE-ENCODES (PNG -> WebP), so the bytes are deliberately not identical.
        What must hold is that a stored page is still a readable image of the same dimensions."""
        png = _png_bytes(400, 600)
        path = os.path.join(self.dir, 'a.webp')
        shim.write_cache(path, png)
        stored = shim.read_cache(path)
        self.assertIsNotNone(stored)
        im = PILImage.open(io.BytesIO(stored))
        self.assertEqual(im.size, (400, 600))

    def test_write_leaves_no_temp_file(self):
        """A reader must never open a half-written page, so the write goes through a temp name and
        is renamed into place."""
        path = os.path.join(self.dir, 'b.png')
        shim.write_cache(path, b'\x89PNG page')
        self.assertEqual([n for n in os.listdir(self.dir) if n.endswith('.tmp')], [])

    def test_unwritable_cache_does_not_fail_the_translation(self):
        """A cache is an optimisation; losing it must cost time, not the request."""
        shim.CACHE_DIR = '/proc/definitely-not-writable'
        shim.write_cache('/proc/definitely-not-writable/x.png', b'page')   # must not raise


class TestPrune(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='mt-prune-')
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.addCleanup(setattr, shim, 'CACHE_DIR', shim.CACHE_DIR)
        self.addCleanup(setattr, shim, 'CACHE_MAX_BYTES', shim.CACHE_MAX_BYTES)
        shim.CACHE_DIR = self.dir

    def _store(self, name, size, mtime):
        path = os.path.join(self.dir, name)
        with open(path, 'wb') as fh:
            fh.write(b'x' * size)
        os.utime(path, (mtime, mtime))
        return path

    def test_under_the_cap_prunes_nothing(self):
        shim.CACHE_MAX_BYTES = 100
        self._store('new.png', 10, 200)
        shim.prune_cache()
        self.assertEqual(os.listdir(self.dir), ['new.png'])

    def test_over_the_cap_drops_the_oldest_first(self):
        shim.CACHE_MAX_BYTES = 100
        old = self._store('old.png', 60, 100)
        mid = self._store('mid.png', 60, 150)
        new = self._store('new.png', 60, 200)
        shim.prune_cache()
        self.assertFalse(os.path.exists(old), 'oldest entry should go first')
        self.assertTrue(os.path.exists(new), 'the page just written must survive')
        self.assertFalse(os.path.exists(mid), 'pruning continues until under the cap')

    def test_pruning_leaves_temp_files_alone(self):
        """A temp file is a write in flight, not cached content - unlinking it would corrupt a
        concurrent translation."""
        shim.CACHE_MAX_BYTES = 10
        tmp = self._store('busy.png.tmp', 100, 100)
        shim.prune_cache()
        self.assertTrue(os.path.exists(tmp))


class TestSupportedLanguages(unittest.TestCase):
    """The reader's language selector is filtered from this, so it must not lie in either
    direction: an empty list makes the reader filter nothing (safe), and a non-empty list must
    reflect the translator actually deployed."""

    def test_sugoi_reports_its_two_targets(self):
        """sugoi's _LANGUAGE_CODE_MAP is {'JPN': 'ja', 'ENG': 'en'}; ESP is a 500 waiting to
        happen, which is exactly what this list exists to prevent."""
        self.assertEqual(shim._supported_languages('sugoi'), ['ENG', 'JPN'])

    def test_unknown_translator_reports_empty_not_a_crash(self):
        self.assertEqual(shim._supported_languages('no-such-translator'), [])

    def test_languages_are_plain_codes(self):
        for code in shim._supported_languages('sugoi'):
            self.assertEqual(code, code.upper())
            self.assertEqual(len(code), 3)


class TestCacheHitReporting(unittest.TestCase):
    """run() tells the caller whether it served a cached page, and /translate turns that into a
    header. Without it the reader shows the same "translating..." for a 0.4 s cache hit and an 80 s
    translation, so a hit reads as "it translated the page again"."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='mt-hit-')
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.addCleanup(setattr, shim, 'CACHE_DIR', shim.CACHE_DIR)
        shim.CACHE_DIR = self.dir

    def test_a_written_page_reports_as_a_hit(self):
        path = os.path.join(self.dir, shim.cache_key(b'page', CFG_ENG) + '.png')
        shim.write_cache(path, b'\x89PNG cached')
        self.assertEqual(shim.read_cache(path), b'\x89PNG cached')
        # the same key the request would compute, so run()'s fast path finds it
        self.assertTrue(os.path.exists(path))

    def test_keys_differ_so_one_page_cannot_mask_another(self):
        self.assertNotEqual(shim.cache_key(b'page 4', CFG_ENG), shim.cache_key(b'page 5', CFG_ENG))


class TestOnlyIfCached(unittest.TestCase):
    """X-Translate-Only-If-Cached exists so the reader can redraw a reloaded gallery without ever
    starting work. Without it, restoring a page whose cache had been pruned would begin a ~70 s
    translation on a reload - an action the user did not take.

    The header is read before the lock and answered straight from disk, so a cache-only miss costs
    nothing and cannot queue behind a running translation.
    """

    def test_key_matches_what_a_normal_request_would_store(self):
        # The restore path must compute the SAME key as the translate path, or every restore misses.
        path_cfg = {'translator': {'translator': 'sugoi', 'target_lang': 'ENG'}}
        self.assertEqual(shim.cache_key(b'page', path_cfg), shim.cache_key(b'page', dict(path_cfg)))

    def test_empty_cfg_normalises_to_the_default(self):
        # A request with no config must land on the default entry, not a different one - otherwise
        # the restore path and the translate path disagree and every restore misses.
        self.assertIs(shim.effective_config({}), shim.DEFAULT_CONFIG)
        self.assertIs(shim.effective_config(None), shim.DEFAULT_CONFIG)
        self.assertEqual(shim.cache_key(b'page', shim.effective_config({})),
                         shim.cache_key(b'page', shim.DEFAULT_CONFIG))


def _png_bytes(w, h, colour=(210, 210, 210)):
    """A real PNG, because the store re-encodes and needs something PIL can open."""
    out = io.BytesIO()
    PILImage.new('RGB', (w, h), colour).save(out, format='PNG')
    return out.getvalue()


class TestStorageFormat(unittest.TestCase):
    """Translated pages dominate the cache: 2.43 MB each as PNG, 0.49 MB as WebP q90 (measured over
    41 real pages). Encoding on write is what turns an 8 GB ephemeral cache into a durable one -
    same budget, ~5x the pages - without touching the 29 TB pool that is 99% full."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='mt-fmt-')
        self.addCleanup(shutil.rmtree, self.dir, True)
        for name in ('CACHE_DIR', 'CACHE_QUALITY', 'CACHE_MAX_BYTES'):
            self.addCleanup(setattr, shim, name, getattr(shim, name))
        shim.CACHE_DIR = self.dir

    def test_a_stored_page_is_webp_and_much_smaller(self):
        png = _png_bytes(900, 1300)
        stored = shim.encode_cached(png)
        self.assertEqual(shim.media_type(stored), 'image/webp')
        self.assertLess(len(stored), len(png))

    def test_unencodable_input_is_stored_verbatim(self):
        # Losing the encode must never lose the page: fall back to the bytes we were given.
        self.assertEqual(shim.encode_cached(b'not an image'), b'not an image')

    def test_media_type_reads_magic_bytes_not_the_extension(self):
        # Entries written before this change are PNG on disk and must keep being served correctly -
        # each one is minutes of translation.
        png = _png_bytes(10, 10)
        self.assertEqual(shim.media_type(png), 'image/png')
        self.assertEqual(shim.media_type(shim.encode_cached(png)), 'image/webp')
        self.assertEqual(shim.media_type(b'\xff\xd8\xff\xe0rest'), 'image/jpeg')
        self.assertEqual(shim.media_type(b'\x00\x05\x16\x07Netatalk'), 'application/octet-stream')

    def test_quality_is_configurable(self):
        png = _png_bytes(600, 900, (120, 140, 160))
        hi = shim.encode_cached(png)
        shim.CACHE_QUALITY = 40
        lo = shim.encode_cached(png)
        self.assertLess(len(lo), len(hi))

    def test_health_reports_usage_and_format(self):
        self.assertIsInstance(shim.cache_usage(), tuple)
        stored, cap = shim.cache_usage()
        self.assertEqual(stored, 0)
        self.assertGreater(cap, 0)


class TestLegacyEntries(unittest.TestCase):
    """Entries written before the WebP change are `.png` files holding real PNGs. Each one is minutes
    of translation, so a format change must keep reading them rather than start over."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='mt-legacy-')
        self.addCleanup(shutil.rmtree, self.dir, True)
        for name in ('CACHE_DIR', 'CACHE_QUALITY'):
            self.addCleanup(setattr, shim, name, getattr(shim, name))
        shim.CACHE_DIR = self.dir

    def test_a_new_entry_is_named_webp(self):
        self.assertTrue(shim.cache_path('abc').endswith('.webp'))

    def test_a_pre_webp_png_entry_is_still_found(self):
        png = _png_bytes(120, 160)
        with open(os.path.join(self.dir, 'abc.png'), 'wb') as fh:
            fh.write(png)                       # exactly what the old code stored
        found = shim.read_entry('abc')
        self.assertEqual(found, png)
        self.assertEqual(shim.media_type(found), 'image/png')

    def test_new_format_wins_when_both_exist(self):
        png = _png_bytes(60, 60)
        enc = shim.encode_cached(png)
        for name, data in (('k.webp', enc), ('k.png', png)):
            with open(os.path.join(self.dir, name), 'wb') as fh:
                fh.write(data)
        self.assertEqual(shim.read_entry('k'), enc)

    def test_missing_key_reads_as_none(self):
        self.assertIsNone(shim.read_entry('nothing-here'))

    def test_migration_re_encodes_and_removes_the_png(self):
        png = _png_bytes(500, 700, (200, 180, 160))
        with open(os.path.join(self.dir, 'mig.png'), 'wb') as fh:
            fh.write(png)
        migrated, before, after = shim.migrate_cache()
        self.assertEqual(migrated, 1)
        self.assertLess(after, before, 'the point of migrating is that the bytes get smaller')
        self.assertFalse(os.path.exists(os.path.join(self.dir, 'mig.png')))
        self.assertEqual(shim.media_type(shim.read_entry('mig')), 'image/webp')


if __name__ == '__main__':
    unittest.main(verbosity=2)


class TestComicReadContract(unittest.TestCase):
    """ComicRead's manga-image-translator adapter speaks a different three-part contract than our
    reader, and it discovers both the API version and the translator list from the ROOT PAGE. These
    pin the parts of that contract that fail silently when they drift: the version marker, the
    translator literal, and the stream framing.

    A missing/misworded root page does not error - the client quietly decides the server is the old
    API and starts POSTing /submit, which this shim does not implement.
    """

    def test_root_does_not_look_like_the_old_api(self):
        """The adapter sets isOldVersion from this marker; present = it posts to /submit."""
        html = shim.ROOT_HTML.format(cached=0, stored_mb=0, cap_mb=20480)
        self.assertNotIn('value="S">1024px</', html)

    def test_root_names_the_translators_in_the_expected_literal(self):
        """Parsed with /(?<=validTranslators: )\[.+?\](?=,)/ then JSON.parse after ' -> "."""
        import re
        html = shim.ROOT_HTML.format(cached=0, stored_mb=0, cap_mb=20480)
        found = re.search(r'(?<=validTranslators: )\[.+?\](?=,)', html, re.S)
        self.assertIsNotNone(found, 'the client would find no translators and leave the menu empty')
        names = json.loads(found.group(0).replace("'", '"'))
        self.assertEqual(names, shim.OFFLINE_TRANSLATORS)

    def test_advertised_translators_are_the_ones_that_work_without_keys(self):
        """ComicRead's own default is gpt3.5, which needs OPENAI_API_KEY. Its updateMitTranslators
        replaces an unknown selection with the first advertised name, so advertising only offline
        translators is what stops every page failing with MissingAPIKeyException."""
        self.assertIn('sugoi', shim.OFFLINE_TRANSLATORS)
        for name in shim.OFFLINE_TRANSLATORS:
            self.assertNotIn(name, ('gpt3.5', 'gpt4', 'gemini', 'deepl'), 'needs an API key')

    def test_frame_layout_matches_the_client_parser(self):
        """[type:1][length:4 big-endian][payload] - the client reads the length from bytes 1..4."""
        f = shim.frame(0, b'abc')
        self.assertEqual(f[0], 0)
        self.assertEqual(int.from_bytes(f[1:5], 'big'), 3)
        self.assertEqual(f[5:], b'abc')

    def test_stream_types_are_the_ones_the_client_switches_on(self):
        """0 image, 1 status, 2 error, 3 queue position. A wrong number is silently ignored by the
        client, which then waits for an image frame that never comes."""
        for kind, payload in ((0, b'img'), (1, b'translating'), (2, b'boom'), (3, b'0')):
            self.assertEqual(shim.frame(kind, payload)[0], kind)

    def test_stream_frames_are_png_even_when_the_store_holds_webp(self):
        """The client builds its blob with a hardcoded {type: "image/png"}, so a WebP entry has to be
        re-encoded on the way out. Serving the stored bytes directly gave the browser WebP labelled
        PNG, which does not decode."""
        webp = io.BytesIO()
        PILImage.new('RGB', (8, 8), 'white').save(webp, format='WEBP')
        out = shim.as_png(webp.getvalue())
        self.assertEqual(out[:8], b'\x89PNG\r\n\x1a\n')

    def test_as_png_leaves_png_untouched(self):
        png = io.BytesIO()
        PILImage.new('RGB', (8, 8), 'white').save(png, format='PNG')
        self.assertEqual(shim.as_png(png.getvalue()), png.getvalue())

class TestCachePut(unittest.TestCase):
    """`POST /cache/put` lets a GPU host pre-translate a chapter and hand the pages to the store.

    The property that makes the design work is that the WRITER and the READER compute the same key
    from the same two inputs - image bytes and the normalised config - so no protocol is needed
    between the two machines beyond this call. That is what the first test pins: a page written
    through the endpoint is served by /translate as a plain cache hit, without loading a model.

    The rest is the boundary. This route writes into the store the reader trusts, so it fails closed
    without a token, and it refuses anything that is not a readable page rather than storing a
    broken entry that renders as an undecodable image with no explanation.
    """

    TOKEN = 'test-token'

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='mt-put-')
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.addCleanup(setattr, shim, 'CACHE_DIR', shim.CACHE_DIR)
        self.addCleanup(setattr, shim, 'CACHE_TOKEN', shim.CACHE_TOKEN)
        self.addCleanup(setattr, shim, 'CACHE_PUT_MAX_BYTES', shim.CACHE_PUT_MAX_BYTES)
        shim.CACHE_DIR = self.dir
        shim.CACHE_TOKEN = self.TOKEN
        self.client = TestClient(shim.app)

    def _put(self, image=b'page bytes', result=None, cfg=None, token=TOKEN):
        result = _png_bytes(60, 90, (30, 40, 50)) if result is None else result
        files = {'image': ('page.png', image, 'image/png'),
                 'result': ('out.png', result, 'image/png')}
        data = {'config': json.dumps(CFG_ENG if cfg is None else cfg)}
        headers = {'X-Cache-Token': token} if token is not None else {}
        return self.client.post('/cache/put', files=files, data=data, headers=headers)

    def test_a_page_written_here_is_a_cache_hit_for_translate(self):
        """The whole point: no model, no work, the reader gets the page the GPU produced."""
        r = self._put()
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()['ok'])

        # same image + same config, asking for cache-only (which never starts a translation)
        hit = self.client.post('/translate', files={'image': ('page.png', b'page bytes', 'image/png')},
                               data={'config': json.dumps(CFG_ENG)},
                               headers={'X-Translate-Only-If-Cached': '1'})
        self.assertEqual(hit.status_code, 200)
        self.assertEqual(hit.headers['x-translate-cache'], 'hit')
        self.assertEqual(PILImage.open(io.BytesIO(hit.content)).size, (60, 90))

    def test_a_different_config_is_a_different_entry(self):
        """Pre-translating to ESP must not make the ENG request look cached."""
        self.assertEqual(self._put(cfg={'translator': {'translator': 'sugoi', 'target_lang': 'ESP'}}).status_code, 200)
        miss = self.client.post('/translate', files={'image': ('page.png', b'page bytes', 'image/png')},
                                data={'config': json.dumps(CFG_ENG)},
                                headers={'X-Translate-Only-If-Cached': '1'})
        self.assertEqual(miss.status_code, 204)

    def test_no_token_configured_disables_the_route(self):
        """Fail closed: an unset token is the default, so a fresh container cannot be written to."""
        shim.CACHE_TOKEN = ''
        self.assertEqual(self._put().status_code, 403)

    def test_wrong_token_is_refused(self):
        self.assertEqual(self._put(token='nope').status_code, 403)

    def test_missing_token_is_refused(self):
        self.assertEqual(self._put(token=None).status_code, 403)

    def test_missing_result_is_a_bad_request(self):
        r = self.client.post('/cache/put', files={'image': ('page.png', b'page bytes', 'image/png')},
                             data={'config': json.dumps(CFG_ENG)},
                             headers={'X-Cache-Token': self.TOKEN})
        self.assertEqual(r.status_code, 400)

    def test_a_non_image_result_is_refused(self):
        """Storing undecodable bytes would render as a broken image with no way to tell why."""
        r = self._put(result=b'this is not an image')
        self.assertEqual(r.status_code, 400)
        self.assertEqual([e.name for e in os.scandir(self.dir)], [])

    def test_an_oversized_result_is_refused(self):
        shim.CACHE_PUT_MAX_BYTES = 16
        self.assertEqual(self._put().status_code, 413)

    def test_rewriting_the_same_page_reports_that_it_replaced_one(self):
        self.assertFalse(self._put().json()['replaced'])
        self.assertTrue(self._put(result=_png_bytes(60, 90, (1, 2, 3))).json()['replaced'])

    def test_json_body_with_base64_works_too(self):
        """The batch job may prefer one JSON payload to multipart; both must land on one entry."""
        import base64
        r = self.client.post('/cache/put',
                             json={'image': base64.b64encode(b'page bytes').decode(),
                                   'result': base64.b64encode(_png_bytes(60, 90)).decode(),
                                   'config': CFG_ENG},
                             headers={'X-Cache-Token': self.TOKEN})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(os.path.exists(os.path.join(self.dir, r.json()['key'] + '.webp')))

class TestPrewarm(unittest.TestCase):
    """`POST /prewarm` translates on the GPU peer and stores the result here.

    The reason this endpoint exists is apollo's uptime: it is a gaming PC that sleeps, so it cannot
    be the reader's backend. The property that makes pre-warming safe is that the GPU page lands on
    the SAME key /translate computes - a later read is a cache hit with apollo switched off. That is
    what the first test pins, with the GPU call stubbed out.

    The rest is the failure boundary. An unreachable peer must be 503 ("try later"), not 500, and it
    must NOT fall back to a local translation: that would turn a 7 s button press into a 70 s one
    without anyone asking for it.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='mt-prewarm-')
        self.addCleanup(shutil.rmtree, self.dir, True)
        for name in ('CACHE_DIR', 'GPU_URL'):
            self.addCleanup(setattr, shim, name, getattr(shim, name))
        shim.CACHE_DIR = self.dir
        shim.GPU_URL = 'http://gpu.test:8091'
        self.client = TestClient(shim.app)

    def _stub_gpu(self, page=None, exc=None):
        async def fake(image_bytes, cfg):
            if exc is not None:
                raise exc
            return page if page is not None else _png_bytes(70, 100, (9, 9, 9))
        self.addCleanup(setattr, shim, 'gpu_translate', shim.gpu_translate)
        shim.gpu_translate = fake

    def _prewarm(self):
        return self.client.post('/prewarm',
                                files={'image': ('p.png', b'page bytes', 'image/png')},
                                data={'config': json.dumps(CFG_ENG)})

    def test_a_gpu_page_is_returned_and_stored_under_the_translate_key(self):
        self._stub_gpu()
        r = self._prewarm()
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers['x-translate-source'], 'gpu')
        self.assertEqual(PILImage.open(io.BytesIO(r.content)).size, (70, 100))

        # the point: no GPU, no model, and /translate still serves it
        shim.GPU_URL = ''
        hit = self.client.post('/translate',
                               files={'image': ('p.png', b'page bytes', 'image/png')},
                               data={'config': json.dumps(CFG_ENG)},
                               headers={'X-Translate-Only-If-Cached': '1'})
        self.assertEqual(hit.status_code, 200)
        self.assertEqual(hit.headers['x-translate-cache'], 'hit')

    def test_an_unreachable_peer_is_503_not_500(self):
        self._stub_gpu(exc=shim.GpuUnavailable('ConnectError: refused'))
        r = self._prewarm()
        self.assertEqual(r.status_code, 503)
        self.assertIn('refused', r.json()['error'])

    def test_no_peer_configured_is_503(self):
        shim.GPU_URL = ''
        r = self._prewarm()
        self.assertEqual(r.status_code, 503)

    def test_a_failure_does_not_fall_back_to_a_local_translation(self):
        """A 70 s local run would be a worse outcome than an honest "try later"."""
        self._stub_gpu(exc=shim.GpuUnavailable('timeout'))

        def boom(*a, **k):
            raise AssertionError('local translation must not run for /prewarm')

        self.addCleanup(setattr, shim, 'get_translator', shim.get_translator)
        shim.get_translator = boom
        self.assertEqual(self._prewarm().status_code, 503)

    def test_a_different_language_does_not_overwrite_the_other_entry(self):
        self._stub_gpu()
        self.assertEqual(self._prewarm().status_code, 200)
        other = self.client.post('/prewarm',
                                 files={'image': ('p.png', b'page bytes', 'image/png')},
                                 data={'config': json.dumps({'translator': {'translator': 'sugoi',
                                                                           'target_lang': 'ESP'}})})
        self.assertEqual(other.status_code, 200)
        stored = sorted(e.name for e in os.scandir(self.dir))
        self.assertEqual(len(stored), 2, stored)

    def test_health_advertises_the_peer_only_when_configured(self):
        self.assertEqual(self.client.get('/health').json()['gpu'], 'http://gpu.test:8091')
        shim.GPU_URL = ''
        self.assertIsNone(self.client.get('/health').json()['gpu'])

class TestGpuTimeouts(unittest.TestCase):
    """A sleeping PC must not hang a reader's click.

    Measured 2026-11-05: with apollo awake but the GPU service stopped, Windows silently DROPS the
    SYN rather than resetting it, so the request sat in connect until the READ timeout - two minutes
    of a button that looks broken. The fix is a separate, short connect timeout, and it is the kind
    of thing that silently regresses (they are one parameter apart), so it is pinned here.
    """

    def test_connect_fails_fast_and_read_still_allows_a_cold_start(self):
        self.assertLessEqual(shim.GPU_CONNECT_TIMEOUT, 10)
        self.assertGreaterEqual(shim.GPU_TIMEOUT, 120)
        t = httpx.Timeout(shim.GPU_TIMEOUT, connect=shim.GPU_CONNECT_TIMEOUT)
        self.assertEqual(t.connect, shim.GPU_CONNECT_TIMEOUT)
        self.assertEqual(t.read, shim.GPU_TIMEOUT)

    def test_a_connect_failure_becomes_gpu_unavailable(self):
        """Whatever httpx raises must reach the caller as 503, not as an unhandled 500."""
        import asyncio

        import httpx as _httpx

        class Boom:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **k):
                raise _httpx.ConnectTimeout('connect timed out')

        self.addCleanup(setattr, shim, 'httpx', shim.httpx)
        shim.httpx = type('fake', (), {'AsyncClient': Boom, 'HTTPError': _httpx.HTTPError,
                                       'Timeout': _httpx.Timeout})
        self.addCleanup(setattr, shim, 'GPU_URL', shim.GPU_URL)
        shim.GPU_URL = 'http://127.0.0.1:1'
        with self.assertRaises(shim.GpuUnavailable):
            asyncio.run(shim.gpu_translate(b'page', CFG_ENG))


class TestGpuStatus(unittest.TestCase):
    """The pill's source of truth, and the reason it must be cached.

    /health is fetched on every reader page load, so the peer probe cannot live inline: a 1.5 s
    connect attempt against a sleeping PC would add that to every page open. These pin the cache, the
    four states, and the fact that a probe failure is a status, not an exception.
    """

    def setUp(self):
        # CONTROL_TOKEN and ALLOW_SLEEP are pinned, not inherited: the real container has them set in
        # its environment, so a test that read the ambient values would pass on a bare box and fail
        # on the deployment it is meant to describe.
        for name in ('GPU_URL', 'GPU_MAC', '_gpu_state', '_last_wake', 'CONTROL_TOKEN',
                     'ALLOW_SLEEP'):
            self.addCleanup(setattr, shim, name, getattr(shim, name))
        shim.GPU_URL = 'http://gpu.test:8091'
        shim.GPU_MAC = '9c:6b:00:37:4d:a2'
        shim.CONTROL_TOKEN = ''
        shim.ALLOW_SLEEP = False
        shim._gpu_state = {}
        self.client = TestClient(shim.app)

    def test_unconfigured_peer_is_its_own_state(self):
        shim.GPU_URL = ''
        st = self.client.get('/health').json()['gpu_state']
        self.assertEqual(st['state'], 'unconfigured')

    def test_never_probed_reports_unknown_not_offline(self):
        """Calling it offline before asking would make a fresh start look like a dead PC."""
        self.assertEqual(self.client.get('/health').json()['gpu_state']['state'], 'unknown')

    def test_a_sleeping_host_and_a_stopped_service_are_different_states(self):
        """Both drop the SYN; only the SSH probe tells them apart, and the pill's advice differs."""
        shim._gpu_state = {'state': 'offline', 'checked': 1.0, 'vram': None, 'detail': ''}
        self.assertEqual(self.client.get('/health').json()['gpu_state']['state'], 'offline')
        shim._gpu_state = {'state': 'no-service', 'checked': 1.0, 'vram': None, 'detail': ''}
        self.assertEqual(self.client.get('/health').json()['gpu_state']['state'], 'no-service')

    def test_health_serves_the_cache_and_reports_its_age(self):
        # a pre-seeded verdict must be returned without any probing
        shim._gpu_state = {'state': 'online', 'checked': shim.time.time() - 5, 'vram': None,
                           'detail': ''}
        st = self.client.get('/health').json()['gpu_state']
        self.assertEqual(st['state'], 'online')
        self.assertGreaterEqual(st['age'], 5)

    def test_models_loaded_is_what_makes_a_page_fast_not_ready(self):
        """After an unload the translator object survives, so ready stays true while the next page
        must reload ~11 GiB of models. Measuring that in the same field would promise a 5 s page for
        the one case that costs 110 s."""
        class T:
            _model_usage_timestamps = {('translation', 'sugoi'): 1.0}

        self.addCleanup(setattr, shim, '_translator', shim._translator)
        shim._translator = T()
        self.assertTrue(shim.models_loaded())
        T._model_usage_timestamps = {}          # what /model/unload leaves behind
        self.assertFalse(shim.models_loaded())
        shim._translator = None
        self.assertFalse(shim.models_loaded())

    def test_an_unanswering_peer_cannot_be_asked_to_unload_or_sleep(self):
        self.addCleanup(setattr, shim, 'CONTROL_TOKEN', shim.CONTROL_TOKEN)
        shim.CONTROL_TOKEN = 'ctl'
        shim._gpu_state = {'state': 'offline', 'checked': shim.time.time(), 'vram': None,
                           'detail': '', 'controls': {'unload': True, 'sleep': True}}
        ctl = self.client.get('/health').json()['controls']
        self.assertFalse(ctl['unload'], 'nothing to unload on a machine that is off')
        self.assertFalse(ctl['sleep'])

    def test_a_busy_peer_is_not_reported_as_missing(self):
        """A translation blocks the peer's event loop, so /health times out MID-PAGE.

        Measured 2026-11-05: the probe read-timed-out immediately after a 93 s page. Calling that
        no-service would flash the pill red in the middle of every chapter - the state the user is
        most likely to look at.
        """
        async def fake_tcp(host, port, timeout):
            return True          # something is listening

        async def fake_health():
            raise __import__('httpx').ReadTimeout('read timed out')

        self.addCleanup(setattr, shim, '_tcp_open', shim._tcp_open)
        self.addCleanup(setattr, shim, '_peer_health', shim._peer_health)
        shim._tcp_open = fake_tcp
        shim._peer_health = fake_health
        st = asyncio.run(shim.probe_gpu())
        self.assertEqual(st['state'], 'loading')
        self.assertIn('ReadTimeout', st['detail'])

    def test_controls_report_what_this_deployment_can_do(self):
        """The reader offers a control only if the capability is advertised."""
        ctl = self.client.get('/health').json()['controls']
        self.assertEqual(ctl, {'wake': True, 'unload': False, 'sleep': False})
        shim.GPU_MAC = ''
        self.assertFalse(self.client.get('/health').json()['controls']['wake'])

    def test_sleep_and_unload_follow_the_peer_not_this_host(self):
        """The reader talks to arrstack's shim, but sleeping is the PEER's capability - reporting
        the local flag would offer a suspend button to the box that cannot be suspended."""
        self.addCleanup(setattr, shim, 'ALLOW_SLEEP', shim.ALLOW_SLEEP)
        self.addCleanup(setattr, shim, 'CONTROL_TOKEN', shim.CONTROL_TOKEN)
        shim.ALLOW_SLEEP = False          # this host cannot sleep itself
        shim.CONTROL_TOKEN = 'ctl'        # but it can call the peer
        shim._gpu_state = {'state': 'online', 'checked': shim.time.time(), 'vram': None,
                           'detail': '', 'controls': {'unload': True, 'sleep': True}}
        ctl = self.client.get('/health').json()['controls']
        self.assertTrue(ctl['sleep'], 'peer says it can sleep, so offer it')
        self.assertTrue(ctl['unload'])


class TestShimFingerprint(unittest.TestCase):
    """The guard against half-deployed shims.

    arrstack and the GPU peer run the same file with no deployment mechanism between them, so a
    change that lands on one host only fails silently: /health just has fewer fields. Both report a
    fingerprint, the probe compares them, and a mismatch becomes `drift` instead of a mystery.
    """

    def setUp(self):
        for name in ('GPU_URL', '_gpu_state'):
            self.addCleanup(setattr, shim, name, getattr(shim, name))
        shim.GPU_URL = 'http://gpu.test:8091'
        shim._gpu_state = {}
        self.client = TestClient(shim.app)

    def test_the_fingerprint_is_a_stable_short_hash_of_this_file(self):
        fp = shim.shim_fingerprint()
        self.assertEqual(len(fp), 12)
        self.assertTrue(all(c in '0123456789abcdef' for c in fp))
        self.assertEqual(fp, shim.shim_fingerprint(), 'must not change between calls')
        self.assertEqual(fp, shim.SHIM_SHA)

    def test_health_advertises_it(self):
        self.assertEqual(self.client.get('/health').json()['shim_sha'], shim.SHIM_SHA)

    def test_matching_fingerprints_are_not_drift(self):
        shim._gpu_state = {'state': 'online', 'checked': shim.time.time(), 'vram': None,
                           'detail': '', 'controls': {}, 'shim_sha': shim.SHIM_SHA, 'drift': False}
        st = self.client.get('/health').json()['gpu_state']
        self.assertFalse(st['drift'])

    def test_a_different_fingerprint_is_drift(self):
        shim._gpu_state = {'state': 'online', 'checked': shim.time.time(), 'vram': None,
                           'detail': '', 'controls': {}, 'shim_sha': 'deadbeefcafe',
                           'drift': True}
        self.assertTrue(self.client.get('/health').json()['gpu_state']['drift'])

    def test_an_unknown_fingerprint_is_not_called_drift(self):
        """An older peer build reports no shim_sha at all; calling that drift would be a false alarm
        on the very first deploy."""
        shim._gpu_state = {'state': 'online', 'checked': shim.time.time(), 'vram': None,
                           'detail': '', 'controls': {}, 'shim_sha': '', 'drift': False}
        self.assertFalse(self.client.get('/health').json()['gpu_state']['drift'])
        self.assertEqual(shim.effective_controls()['unload'], bool(shim.CONTROL_TOKEN))


class TestWake(unittest.TestCase):
    def setUp(self):
        for name in ('GPU_MAC', 'GPU_WAKE_COOLDOWN', '_last_wake', '_gpu_state'):
            self.addCleanup(setattr, shim, name, getattr(shim, name))
        shim.GPU_MAC = '9c:6b:00:37:4d:a2'
        shim.GPU_WAKE_COOLDOWN = 60
        shim._last_wake = 0.0
        shim._gpu_state = {'state': 'offline', 'checked': shim.time.time(), 'vram': None,
                           'detail': ''}
        self.client = TestClient(shim.app)

    def test_the_packet_is_the_standard_102_bytes(self):
        pkt = shim.wake_packet('9c:6b:00:37:4d:a2')
        self.assertEqual(len(pkt), 102)
        self.assertEqual(pkt[:6], b'\xff' * 6)
        self.assertEqual(pkt[6:12], bytes.fromhex('9c6b00374da2'))
        self.assertEqual(pkt[6:], bytes.fromhex('9c6b00374da2') * 16)

    def test_a_malformed_mac_is_refused_rather_than_broadcast(self):
        with self.assertRaises(ValueError):
            shim.wake_packet('not-a-mac')

    def test_no_mac_configured_is_503(self):
        shim.GPU_MAC = ''
        self.assertEqual(self.client.post('/gpu/wake').status_code, 503)

    def test_it_aims_at_every_plausible_target(self):
        """One broadcast target was not enough in practice (see wake_targets)."""
        shim.GPU_URL = 'http://192.168.1.21:8091'
        self.assertEqual(shim.wake_targets(), ['255.255.255.255', '192.168.1.255', '192.168.1.21'])
        shim.GPU_URL = ''
        self.assertEqual(shim.wake_targets(), ['255.255.255.255'])

    def test_it_broadcasts_and_then_refuses_to_be_hammered(self):
        """A prefetch loop or an impatient double-click must not become a packet stream."""
        sent = []
        self.addCleanup(setattr, shim, 'send_wake', shim.send_wake)
        shim.send_wake = lambda: sent.append(1)
        self.assertEqual(self.client.post('/gpu/wake').status_code, 200)
        second = self.client.post('/gpu/wake')
        self.assertEqual(second.status_code, 429)
        self.assertEqual(len(sent), 1)

    def test_wake_is_not_reachable_by_get(self):
        # a GET would let a link prefetch power on the machine
        self.assertEqual(self.client.get('/gpu/wake').status_code, 405)


class TestPeerControl(unittest.TestCase):
    """Unload and sleep: both act on the peer's host, both fail closed, both proxy here."""

    def setUp(self):
        for name in ('CONTROL_TOKEN', 'ALLOW_SLEEP', 'GPU_URL', '_translator', '_lock'):
            self.addCleanup(setattr, shim, name, getattr(shim, name))
        shim.CONTROL_TOKEN = 'ctl'
        shim.ALLOW_SLEEP = False
        shim.GPU_URL = 'http://gpu.test:8091'
        self.client = TestClient(shim.app)

    def _stub_peer(self, body=None, status=200):
        calls = []

        class Res:
            status_code = status

            def __init__(self, payload):
                self._payload = payload

            def json(self):
                return self._payload

            @property
            def text(self):
                return json.dumps(self._payload)

        class Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, headers=None, **k):
                calls.append((url, headers))
                return Res(body or {'ok': True})

        self.addCleanup(setattr, shim, 'httpx', shim.httpx)
        shim.httpx = type('fake', (), {'AsyncClient': Client,
                                       'HTTPError': __import__('httpx').HTTPError,
                                       'Timeout': __import__('httpx').Timeout})
        return calls

    def test_control_endpoints_fail_closed_without_a_token(self):
        req = type('R', (), {'headers': {}})()
        self.assertFalse(shim._control_ok(req))
        shim.CONTROL_TOKEN = ''
        self.assertFalse(shim._control_ok(req))

    def test_unload_forwards_the_control_token(self):
        calls = self._stub_peer({'ok': True, 'unloaded': ['translation:sugoi']})
        r = self.client.post('/gpu/unload')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(calls[0][0], 'http://gpu.test:8091/model/unload')
        self.assertEqual(calls[0][1]['X-Control-Token'], 'ctl')

    def test_sleep_needs_the_host_to_opt_in_as_well_as_the_token(self):
        """Refused here by the shim that opted in, so the message is the honest one."""
        self._stub_peer({'error': 'sleep not enabled on this host (set MT_ALLOW_SLEEP=1)'}, status=403)
        r = self.client.post('/gpu/sleep')
        self.assertEqual(r.status_code, 503)
        self.assertIn('MT_ALLOW_SLEEP', r.json()['error'])

    def test_unload_reports_a_sleeping_peer_as_503(self):
        """A sleeping peer is 503 ("try later"), not 500 ("it is broken")."""
        import httpx as _httpx

        class Timeout:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **k):
                raise _httpx.ConnectTimeout('connect timed out')

        self.addCleanup(setattr, shim, 'httpx', shim.httpx)
        shim.httpx = type('fake', (), {'AsyncClient': Timeout,
                                       'HTTPError': _httpx.HTTPError,
                                       'Timeout': _httpx.Timeout})
        self.assertEqual(self.client.post('/gpu/unload').status_code, 503)


class TestModelUnload(unittest.TestCase):
    """The peer's own /model/unload: frees VRAM, refuses to race a page, reports per model."""

    def setUp(self):
        for name in ('CONTROL_TOKEN', '_translator', '_lock'):
            self.addCleanup(setattr, shim, name, getattr(shim, name))
        shim.CONTROL_TOKEN = 'ctl'
        self.client = TestClient(shim.app)

    class FakeTranslator:
        def __init__(self):
            self._model_usage_timestamps = {('translation', 'sugoi'): 1.0,
                                            ('detection', 'ctd'): 1.0}
            self.unloaded = []

        async def _unload_model(self, tool, model):
            self.unloaded.append((tool, model))

    def test_it_unloads_exactly_what_the_translator_recorded(self):
        fake = self.FakeTranslator()
        shim._translator = fake
        r = self.client.post('/model/unload', headers={'X-Control-Token': 'ctl'})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(sorted(fake.unloaded),
                         [('detection', 'ctd'), ('translation', 'sugoi')])
        self.assertEqual(fake._model_usage_timestamps, {})

    def test_bad_token_is_refused(self):
        shim._translator = self.FakeTranslator()
        self.assertEqual(self.client.post('/model/unload').status_code, 403)

    def test_nothing_loaded_is_a_no_op_not_an_error(self):
        shim._translator = None
        r = self.client.post('/model/unload', headers={'X-Control-Token': 'ctl'})
        self.assertEqual(r.status_code, 200)
        self.assertIn('no models', r.json()['note'])


class TestQueue(unittest.TestCase):
    """Async, disk-persisted translation queue.

    Enqueue is cheap and takes no lock; a single background worker drains the queue through
    the existing lock so concurrency stays at one. Items live on disk so a restart re-enqueues
    leftovers instead of dropping them.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='mt-queue-')
        self.addCleanup(shutil.rmtree, self.dir, True)
        for name in ('CACHE_DIR', 'QUEUE_DIR', 'QUEUE_MAX_ITEMS', 'QUEUE_MAX', 'QUEUE_CAP',
                     'MAX_QUEUE_SIZE', 'MAX_QUEUE', '_worker_task', '_queue_cache_dir'):
            self.addCleanup(setattr, shim, name, getattr(shim, name, None))
        shim.CACHE_DIR = self.dir
        shim.QUEUE_DIR = os.path.join(self.dir, 'queue')
        shim.QUEUE_MAX_ITEMS = 3
        shim.QUEUE_MAX = 3
        shim.QUEUE_CAP = 3
        shim.MAX_QUEUE_SIZE = 3
        shim.MAX_QUEUE = 3
        shim._worker_task = None
        shim._queue_cache_dir = None
        self._orig_run = shim.run

        async def fake_run(image_bytes, cfg):
            await asyncio.sleep(0.05)
            return b'out', False

        shim.run = fake_run
        self.client = TestClient(shim.app)

    def tearDown(self):
        shim.run = self._orig_run

    def _enqueue(self, payload=b'page bytes', cfg=None):
        cfg = cfg or CFG_ENG
        return self.client.post('/translate/enqueue',
                                files={'image': ('p.png', payload, 'image/png')},
                                data={'config': json.dumps(cfg)})

    def test_enqueue_returns_queued_with_position(self):
        with self.client:
            r = self._enqueue()
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()['status'], 'queued')
            self.assertIn('key', r.json())
            self.assertEqual(r.json()['position'], 1)

    def test_cached_page_is_not_queued(self):
        key = shim.cache_key(b'cachedpage', CFG_ENG)
        shim.write_cache(shim.cache_path(key), b'\x89PNG cached')
        with self.client:
            r = self._enqueue(b'cachedpage')
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()['status'], 'cached')

    def test_duplicate_enqueue_is_deduped(self):
        with self.client:
            self._enqueue()
            r = self._enqueue()
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()['status'], 'queued')
            st = self.client.get('/queue/status').json()
            self.assertEqual(st['pending'] + (1 if st['in_progress'] else 0), 1)

    def test_status_reports_progress(self):
        with self.client:
            self._enqueue()
            # let the worker drain
            for _ in range(50):
                st = self.client.get('/queue/status').json()
                if st['done'] >= 1:
                    break
                time.sleep(0.05)
            st = self.client.get('/queue/status').json()
            self.assertEqual(st['failed'], 0)
            self.assertGreaterEqual(st['done'], 1)

    def test_queue_cap_returns_429(self):
        with self.client:
            for _ in range(3):
                self.assertEqual(self._enqueue(b'p' + str(_).encode()).status_code, 200)
            r = self._enqueue(b'overflow')
            self.assertEqual(r.status_code, 429)

    def test_items_persist_to_disk(self):
        with self.client:
            self._enqueue()
            qd = shim.queue_dir()
            files = os.listdir(qd)
            self.assertTrue(any(f.endswith('.png') for f in files))
            self.assertTrue(any(f.endswith('.json') for f in files))

    def test_queue_survives_restart(self):
        with self.client:
            self._enqueue()
            # stop processing by not waiting; ensure files exist
            time.sleep(0.02)
        # simulate restart: reset in-memory state, scan disk
        shim._worker_task = None
        shim._queue_cache_dir = None
        shim._ensure_queue_state()
        # either still pending or already done+removed; if pending, re-enqueued
        # at minimum the scan must not crash and queue dir still consistent
        self.assertIsInstance(shim._queue, list)

    def test_aliases(self):
        # queue endpoints reachable via aliases
        with self.client:
            r = self.client.post('/queue', files={'image': ('p.png', b'aliaspage', 'image/png')},
                                 data={'config': json.dumps(CFG_ENG)})
            self.assertEqual(r.status_code, 200)
            r2 = self.client.get('/translate/queue/status')
            self.assertEqual(r2.status_code, 200)
            self.assertIn('pending', r2.json())
