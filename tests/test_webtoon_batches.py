import numpy as np
import cv2

from app.controllers.projects import ProjectController


def _page(h, w=50, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8), "G")


def _webp_size(arr):
    ok, buf = cv2.imencode(".webp", arr, [cv2.IMWRITE_WEBP_QUALITY, 85])
    assert ok
    return len(buf)


def test_batches_respect_byte_cap():
    pages = [_page(60, seed=i) for i in range(50)]
    # Tiny byte cap -> each page should land in its own batch (single pages are
    # never split even if they exceed the cap).
    out = ProjectController._build_batched_pages(pages, max_bytes=300)
    assert all(_webp_size(img) <= 300 or img.shape[0] == 60 for img, _ in out)
    # Total content preserved.
    total = sum(img.shape[0] for img, _ in out)
    assert total == 60 * 50


def test_batches_respect_height_cap():
    # 60 pages of height 500 (30000px total) with a 16000px cap -> a full
    # batch of 32 pages, then the remaining 28 pages fit under the cap.
    pages = [_page(500, seed=i) for i in range(60)]
    out = ProjectController._build_batched_pages(pages, max_bytes=10**9, max_height=16000)
    assert all(img.shape[0] <= 16000 for img, _ in out)
    assert sum(img.shape[0] for img, _ in out) == 500 * 60
    # Small pages pack up to the cap without exceeding it.
    heights = [img.shape[0] for img, _ in out]
    assert heights == [16000, 14000]


def test_batches_dont_cross_chapters():
    rng = np.random.default_rng(0)
    pages = [
        (rng.integers(0, 256, size=(15, 50, 3), dtype=np.uint8), g)
        for g in (["A"] * 5 + ["B"] * 5)
    ]
    out = ProjectController._build_batched_pages(pages, max_bytes=10**9, max_height=100)
    for img, grp in out:
        assert img.shape[0] <= 100
        assert grp in ("A", "B")
    # Chapter B never appears before chapter A finishes.
    seen = [grp for _, grp in out]
    assert seen.index("A") + seen.count("A") == seen.index("B")


def test_page_order_preserved_within_batch():
    # Distinct solid-color pages; concatenation order must match input order.
    colors = [(i, i, i) for i in range(0, 200, 40)]
    pages = [(np.full((10, 5, 3), c, dtype=np.uint8), "G") for c in colors]
    out = ProjectController._build_batched_pages(pages, max_bytes=10**9, max_height=100)
    combined = np.concatenate([img for img, _ in out], axis=0)
    assert combined.shape[0] == 10 * len(colors)
    for idx, c in enumerate(colors):
        assert tuple(combined[idx * 10, 0]) == c


def test_single_huge_page_not_split():
    big = (np.zeros((5000, 50, 3), dtype=np.uint8), "G")
    out = ProjectController._build_batched_pages([big], max_bytes=1, max_height=100)
    assert len(out) == 1
    assert out[0][0].shape[0] == 5000


def test_oversized_combined_batch_drops_last_page():
    # Two pages that each fit the cap alone but whose combination does not:
    # the post-check splits the last page into its own batch.
    page = _page(2000, seed=1)[0]
    size = _webp_size(page)
    out = ProjectController._build_batched_pages(
        [(page, "G"), (page, "G")], max_bytes=int(size * 1.5)
    )
    assert [img.shape[0] for img, _ in out] == [2000, 2000]
    assert len(out) == 2
