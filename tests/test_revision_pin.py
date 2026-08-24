"""The commit pin has to be the same in three places, or the bake is useless.

The image caches the snapshot for one revision. If the filter (or the vendored
image path) asks for a different one, resolution goes to the hub — which is the
thing baking the weights at build time exists to avoid, and which fails outright
on a cluster with no egress. Nothing enforces the agreement at runtime, so it is
enforced here.
"""
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _grep_constant(path: Path, name: str) -> str:
    pattern = re.compile(rf'^{name}\s*=\s*"([^"]+)"', re.M)
    match = pattern.search(path.read_text())
    assert match, f"{name} not found in {path}"
    return match.group(1)


class TestRevisionPin(unittest.TestCase):
    def test_filter_pin_is_a_full_commit_sha(self):
        """A branch or tag would reintroduce the moving reference the pin removes."""
        pin = _grep_constant(REPO_ROOT / "filter_sam3_detector" / "filter.py", "SAM3_REVISION")
        self.assertRegex(pin, SHA_RE)

    def test_vendored_image_path_uses_the_same_pin(self):
        """The vendored builder downloads its own checkpoint, so it carries its
        own constant; a re-sync from upstream would silently drop the pin."""
        filter_pin = _grep_constant(
            REPO_ROOT / "filter_sam3_detector" / "filter.py", "SAM3_REVISION"
        )
        vendored_pin = _grep_constant(
            REPO_ROOT / "sam3" / "sam3" / "model_builder.py", "SAM3_REVISION"
        )
        self.assertEqual(filter_pin, vendored_pin)

    def test_dockerfile_bakes_the_same_pin(self):
        filter_pin = _grep_constant(
            REPO_ROOT / "filter_sam3_detector" / "filter.py", "SAM3_REVISION"
        )
        dockerfile = (REPO_ROOT / "Dockerfile").read_text()
        self.assertIn(f'revision="{filter_pin}"', dockerfile)

    def test_vendored_downloads_pass_the_revision(self):
        """Both hf_hub_download calls, not just the checkpoint: resolving the
        config off `main` would still need the network."""
        source = (REPO_ROOT / "sam3" / "sam3" / "model_builder.py").read_text()
        downloads = [
            line for line in source.splitlines() if "hf_hub_download(" in line
        ]
        self.assertTrue(downloads, "no hf_hub_download call found")
        body = source[source.index("def download_ckpt_from_hf"):]
        body = body[: body.index("\ndef ", 1)]
        self.assertEqual(body.count("revision=SAM3_REVISION"), 2, body)


class TestResolveRevision(unittest.TestCase):
    """`revision` resolution, without the rest of setup().

    Extracted to a helper precisely so this can be asserted: the alternative is
    a test that drives model loading on a GPU host.
    """

    def setUp(self):
        try:
            from filter_sam3_detector.filter import (  # noqa: F401
                DEFAULT_MODEL_ID,
                SAM3_REVISION,
                resolve_revision,
            )
        except Exception as exc:  # pragma: no cover - import guard
            self.skipTest(f"filter module not importable: {exc}")

    def test_default_model_uses_the_shipped_pin(self):
        from filter_sam3_detector.filter import (
            DEFAULT_MODEL_ID,
            SAM3_REVISION,
            resolve_revision,
        )

        self.assertEqual(resolve_revision(DEFAULT_MODEL_ID, None), SAM3_REVISION)
        self.assertEqual(resolve_revision(DEFAULT_MODEL_ID, "   "), SAM3_REVISION)

    def test_explicit_revision_wins_for_the_default_model(self):
        from filter_sam3_detector.filter import DEFAULT_MODEL_ID, resolve_revision

        self.assertEqual(resolve_revision(DEFAULT_MODEL_ID, " abc123 "), "abc123")

    def test_overridden_model_requires_a_revision(self):
        """Falling back to the shipped pin would load a commit of a different
        repository; falling back to `main` would restore the moving reference."""
        from filter_sam3_detector.filter import resolve_revision

        with self.assertRaises(ValueError) as ctx:
            resolve_revision("org/other-sam", None)
        self.assertIn("revision", str(ctx.exception))
        self.assertIn("org/other-sam", str(ctx.exception))

    def test_overridden_model_accepts_its_own_revision(self):
        from filter_sam3_detector.filter import resolve_revision

        self.assertEqual(resolve_revision("org/other-sam", "deadbeef"), "deadbeef")
