#!/usr/bin/env python3
import json
import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app"))

from disaster_core import (
    DisasterHallucinationDetector,
    EarthquakeUSGSTool,
    EvidenceToolRegistry,
    Geocoder,
    LLMClaimExtractor,
    _env_float,
)


DATASETS = (
    ("earthquakes_true.json", "earthquakes_true_result.json", "earthquakes_true_result_v2.json"),
    (
        "false_domestic_earthquake_batch_50.json",
        "false_domestic_result.json",
        "false_domestic_result_v2.json",
    ),
)


class ReplayClaimExtractor:
    def __init__(self, previous_result_name: str):
        payload = json.loads((ROOT / previous_result_name).read_text(encoding="utf-8"))
        self._by_text = {item["input_text"]: item.get("extraction", {}) for item in payload["items"]}
        self._completer = LLMClaimExtractor.__new__(LLMClaimExtractor)

    def extract_claims(self, text: str):
        extraction = copy.deepcopy(self._by_text.get(text, {"claims": []}))
        meta = dict(extraction.get("meta") or {})
        meta["replayed_extraction"] = True
        extraction["meta"] = meta
        return self._completer._complete_extraction(extraction, text)


def create_replay_detector(previous_result_name: str):
    geocoder = Geocoder()
    registry = EvidenceToolRegistry()
    registry.register(
        EarthquakeUSGSTool(
            geocoder=geocoder,
            radius_km=_env_float("DISASTER_USGS_RADIUS_KM", 100.0, 1.0, 1000.0),
            magnitude_tolerance=_env_float("DISASTER_MAGNITUDE_TOLERANCE", 0.5, 0.0, 2.0),
            reliable_min_magnitude=_env_float(
                "DISASTER_USGS_RELIABLE_MIN_MAGNITUDE", 4.5, 0.0, 10.0
            ),
        )
    )
    return DisasterHallucinationDetector(
        extractor=ReplayClaimExtractor(previous_result_name),
        registry=registry,
        geocoder=geocoder,
    )


def run_dataset(source_name: str, previous_result_name: str, output_name: str) -> None:
    detector = create_replay_detector(previous_result_name)
    source_path = ROOT / source_name
    output_path = ROOT / output_name
    texts = json.loads(source_path.read_text(encoding="utf-8"))["texts"]
    items = []
    for index, text in enumerate(texts, 1):
        items.append({"index": index, **detector.run(text)})
        print(f"{source_name}: {index}/{len(texts)}", flush=True)
    output_path.write_text(
        json.dumps({"total": len(texts), "items": items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"written: {output_path}", flush=True)


def main() -> None:
    for source_name, previous_result_name, output_name in DATASETS:
        run_dataset(source_name, previous_result_name, output_name)


if __name__ == "__main__":
    main()
