"""Regression tests for the 1.0 drop-in compatibility gaps (PyO3 layer).

Each test here pins a fix from the 1.0 preflight review: derived datasets
carrying the full API, custom-key round-tripping, pycocotools-format annToRLE,
pycocotools stats semantics, real Python warnings, and error types. None of
them need the gitignored data/ directory.
"""

import warnings

import hotcoco
import numpy as np
import pytest
from hotcoco import COCO, COCOeval, LVISeval, LVISResults, mask


def tiny_dataset():
    return {
        "images": [
            {"id": 1, "width": 100, "height": 100, "file_name": "a.jpg"},
            {"id": 2, "width": 100, "height": 100, "file_name": "b.jpg"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [50, 50, 20, 20], "area": 400, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}],
    }


def tiny_dt(gt):
    return gt.load_res([{"image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "score": 0.9}])


# ---------------------------------------------------------------------------
# Derived datasets carry the full API (subclass-loss fix)
# ---------------------------------------------------------------------------


class TestDerivedDatasetsKeepFullApi:
    def test_split_results_have_browse(self):
        coco = COCO(tiny_dataset())
        train, val = coco.split(val_frac=0.5)
        assert hasattr(type(train), "browse")
        assert callable(type(val).browse)

    def test_all_derivation_paths_have_browse(self):
        coco = COCO(tiny_dataset())
        derived = [coco.filter(cat_ids=[1]), coco.sample(n=1), tiny_dt(coco), COCO.merge([coco])]
        for d in derived:
            assert hasattr(type(d), "browse"), f"{type(d)} lost browse()"

    def test_browse_without_deps_or_dir_raises_cleanly(self):
        """browse() reaches the Python implementation (not AttributeError)."""
        coco = COCO(tiny_dataset())
        with pytest.raises((ValueError, ImportError)):
            coco.browse()


# ---------------------------------------------------------------------------
# Custom keys survive the Rust round-trip
# ---------------------------------------------------------------------------


class TestCustomKeysRoundTrip:
    def test_annotation_image_category_extras_survive(self):
        ds = tiny_dataset()
        ds["annotations"][0]["confidence_source"] = "human"
        ds["annotations"][0]["tags"] = ["hard", {"nested": 1}]
        ds["images"][0]["weather"] = "rainy"
        ds["categories"][0]["taxonomy_id"] = 42

        coco = COCO(ds)
        out = coco.dataset
        ann = next(a for a in out["annotations"] if a["id"] == 1)
        assert ann["confidence_source"] == "human"
        assert ann["tags"] == ["hard", {"nested": 1}]
        assert next(i for i in out["images"] if i["id"] == 1)["weather"] == "rainy"
        assert next(c for c in out["categories"] if c["id"] == 1)["taxonomy_id"] == 42

    def test_extras_survive_filter(self):
        ds = tiny_dataset()
        ds["annotations"][0]["custom"] = {"a": 1}
        coco = COCO(ds).filter(cat_ids=[1])
        anns = coco.dataset["annotations"]
        assert anns and anns[0]["custom"] == {"a": 1}

    def test_extras_visible_in_load_anns_and_anns(self):
        ds = tiny_dataset()
        ds["annotations"][1]["reviewer"] = "alice"
        coco = COCO(ds)
        assert coco.load_anns(2)[0]["reviewer"] == "alice"
        assert coco.anns[2]["reviewer"] == "alice"

    def test_non_serializable_extra_raises(self):
        ds = tiny_dataset()
        ds["annotations"][0]["bad"] = object()
        with pytest.raises(TypeError):
            COCO(ds)

    def test_slice_by_callable_sees_custom_image_keys(self):
        ds = tiny_dataset()
        ds["images"][0]["weather"] = "rainy"
        ds["images"][1]["weather"] = "sunny"
        gt = COCO(ds)
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        ev.evaluate()
        result = ev.slice_by(lambda img: img.get("weather"))
        assert "rainy" in result and "sunny" in result
        assert result["rainy"]["num_images"] == 1


# ---------------------------------------------------------------------------
# annToRLE returns pycocotools format
# ---------------------------------------------------------------------------


class TestAnnToRle:
    def test_ann_to_rle_pycocotools_format(self):
        ds = tiny_dataset()
        ds["annotations"][0]["segmentation"] = [[10.0, 10.0, 40.0, 10.0, 40.0, 40.0, 10.0, 40.0]]
        coco = COCO(ds)
        rle = coco.ann_to_rle(coco.anns[1])
        assert set(rle) == {"size", "counts"}
        assert rle["size"] == [100, 100]
        assert isinstance(rle["counts"], bytes)
        # Feeds straight back into the mask module, like pycocotools
        assert mask.decode(rle).sum() == mask.area(rle)
        assert coco.annToRLE(coco.anns[1]) == rle


# ---------------------------------------------------------------------------
# ev.stats: [] before summarize, ndarray after — and params copy docs hold
# ---------------------------------------------------------------------------


class TestStatsSemantics:
    def test_stats_empty_list_before_summarize(self):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        assert ev.stats == []

    def test_stats_ndarray_after_summarize(self):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        ev.run()
        stats = ev.stats
        assert isinstance(stats, np.ndarray)
        assert stats.dtype == np.float64
        assert stats.shape == (12,)
        # The pycocotools idiom this shape exists for:
        assert stats[0] >= 0.0

    def test_params_max_dets_is_a_list(self):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        assert ev.params.maxDets == [1, 10, 100]
        assert isinstance(ev.params.maxDets, list)
        # Whole-attribute assignment is the documented mutation path.
        ev.params.maxDets = [1, 10, 100, 200]
        assert ev.params.maxDets == [1, 10, 100, 200]


# ---------------------------------------------------------------------------
# Guards emit real Python warnings, not fd-2 writes
# ---------------------------------------------------------------------------


class TestGuardsWarn:
    def test_accumulate_before_evaluate_warns(self):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        with pytest.warns(UserWarning, match="accumulate"):
            ev.accumulate()

    def test_summarize_before_accumulate_warns(self):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        with pytest.warns(UserWarning, match="summarize"):
            ev.summarize()

    def test_f_scores_before_accumulate_warns(self):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        with pytest.warns(UserWarning, match="f_scores"):
            assert ev.f_scores() == {}

    def test_no_warning_on_correct_order(self):
        gt = COCO(tiny_dataset())
        ev = COCOeval(gt, tiny_dt(gt), "bbox")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ev.run()


# ---------------------------------------------------------------------------
# Error types: converters route through to_pyerr
# ---------------------------------------------------------------------------


class TestErrorTypes:
    def test_from_voc_missing_dir_is_oserror(self):
        with pytest.raises(OSError):
            COCO.from_voc("/nonexistent/voc-dir")

    def test_from_yolo_missing_data_yaml_is_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="data.yaml"):
            COCO.from_yolo(str(tmp_path))

    def test_from_cvat_malformed_xml_is_value_error(self, tmp_path):
        bad = tmp_path / "annotations.xml"
        bad.write_text("<annotations><image name='a.jpg'")
        with pytest.raises(ValueError):
            COCO.from_cvat(str(bad))

    def test_compare_mismatched_params_is_value_error(self):
        gt = COCO(tiny_dataset())
        ev_a = COCOeval(gt, tiny_dt(gt), "bbox")
        ev_a.evaluate()
        ev_b = COCOeval(gt, tiny_dt(gt), "bbox")
        ev_b.params.iouThrs = [0.5]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # deliberate non-reference config
            ev_b.evaluate()
        with pytest.raises(ValueError):
            hotcoco.compare(ev_a, ev_b)


# ---------------------------------------------------------------------------
# mask module round-trip and dict-in/dict-out
# ---------------------------------------------------------------------------


class TestMaskSurface:
    def test_encode_decode_round_trip(self):
        m = np.zeros((10, 12), dtype=np.uint8, order="F")
        m[2:5, 3:7] = 1
        rle = mask.encode(m)
        assert isinstance(rle["counts"], bytes)
        assert (mask.decode(rle) == m).all()

    def test_encode_3d_round_trip(self):
        m = np.zeros((10, 12, 2), dtype=np.uint8, order="F")
        m[2:5, 3:7, 0] = 1
        m[6:9, 1:4, 1] = 1
        rles = mask.encode(m)
        assert isinstance(rles, list) and len(rles) == 2
        assert (mask.decode(rles) == m).all()

    def test_fr_py_objects_dict_in_dict_out(self):
        uncompressed = {"size": [10, 10], "counts": [50, 10, 40]}
        out = mask.frPyObjects(uncompressed, 10, 10)
        assert isinstance(out, dict), "dict input must give dict output (pycocotools parity)"
        outs = mask.frPyObjects([uncompressed], 10, 10)
        assert isinstance(outs, list) and len(outs) == 1


# ---------------------------------------------------------------------------
# LVIS drop-in objects construct the documented types
# ---------------------------------------------------------------------------


class TestLvisDropIn:
    def test_lviseval_returns_cocoeval(self):
        gt = COCO(tiny_dataset())
        ev = LVISeval(gt, tiny_dt(gt), "bbox")
        assert isinstance(ev, COCOeval)

    def test_lvisresults_returns_coco(self):
        gt = COCO(tiny_dataset())
        res = LVISResults(gt, [{"image_id": 1, "category_id": 1, "bbox": [1, 1, 5, 5], "score": 0.5}])
        assert isinstance(res, COCO)


# ---------------------------------------------------------------------------
# load_warnings
# ---------------------------------------------------------------------------


class TestLoadWarnings:
    def test_clean_load_has_no_warnings(self):
        assert COCO(tiny_dataset()).load_warnings == []

    def test_duplicate_ann_ids_are_reported(self):
        ds = tiny_dataset()
        ds["annotations"][1]["id"] = 1  # collide with the first
        coco = COCO(ds)
        assert any("annotation" in w for w in coco.load_warnings)
