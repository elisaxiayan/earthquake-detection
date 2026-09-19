import json
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import requests
from openai import OpenAI


from key import OpenAI_KEY2, QWEN_KEY, AMAP_KEY


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso_z(dt: datetime) -> str:
    return _to_utc(dt).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso(value: str) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return _to_utc(datetime.fromisoformat(normalized))
    except ValueError:
        return None


def get_current_utc_time() -> datetime:
    """Single source of truth for current UTC time."""
    return datetime.now(timezone.utc)


def _is_chinese_context(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _default_timezone_for_text(text: str):
    return ZoneInfo("Asia/Shanghai") if _is_chinese_context(text) else timezone.utc


def _detect_operator(text: str) -> str:
    if re.search(r"(超过|大于|高于|多于)", text):
        return "gt"
    if re.search(r"(至少|不小于|不少于|不低于|以上|>=)", text):
        return "gte"
    if re.search(r"(小于|低于|少于|不到)", text):
        return "lt"
    if re.search(r"(至多|不大于|不超过|以下|<=)", text):
        return "lte"
    return "eq"


def _compare_numeric(actual: float, operator: str, expected: float, expected_max: float | None = None) -> bool:
    if operator == "eq":
        return actual == expected
    if operator == "gt":
        return actual > expected
    if operator == "gte":
        return actual >= expected
    if operator == "lt":
        return actual < expected
    if operator == "lte":
        return actual <= expected
    if operator == "between":
        if expected_max is None:
            return False
        return expected <= actual <= expected_max
    return False


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if minimum <= value <= maximum else default


def _extract_explicit_earthquake_facts(text: str) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    coords = re.search(r"北纬\s*([\d.]+)\s*度[，,]\s*东经\s*([\d.]+)\s*度", text)
    if coords:
        facts["lat"] = float(coords.group(1))
        facts["lon"] = float(coords.group(2))

    location = re.search(
        r"\d{1,2}\s*时\s*\d{1,2}\s*分\s*(?:在)?\s*(.+?)\s*[（(]\s*北纬",
        text,
    )
    if location:
        value = location.group(1).strip()
        if value and _is_chinese_context(value):
            facts["location"] = value

    magnitude = re.search(r"发生\s*([\d.]+)\s*级地震", text)
    if magnitude:
        facts["magnitude"] = float(magnitude.group(1))

    if re.search(r"(没有|未发生|没有发生)\s*.{0,8}地震", text):
        facts["polarity"] = "not_occurred"
    elif "地震" in text:
        facts["polarity"] = "occurred"
    return facts


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return earth_radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _normalize_admin_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = _clean_chinese_place(value)
    return re.sub(r"(省|市|县|区|自治州|自治县|地区|盟|旗|特别行政区)$", "", normalized)


_PLACE_SIMPLIFIED_TRANSLATION = str.maketrans(
    {
        "臺": "台",
        "灣": "湾",
        "縣": "县",
        "區": "区",
        "鄉": "乡",
        "鎮": "镇",
        "東": "东",
        "蘭": "兰",
        "蓮": "莲",
        "門": "门",
        "馬": "马",
        "義": "义",
        "連": "连",
    }
)


def _clean_chinese_place(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    simplified = value.translate(_PLACE_SIMPLIFIED_TRANSLATION)
    return "".join(re.findall(r"[\u4e00-\u9fff]+", simplified))


def _compare_admin_areas(source: dict[str, Any] | None, target: dict[str, Any] | None) -> tuple[str, str]:
    if not source or not target:
        return "无法解析", "输入地点或USGS事件地点无法解析为中文行政区"

    source_code = str(source.get("adcode") or "")
    target_code = str(target.get("adcode") or "")
    source_district = _normalize_admin_name(source.get("district"))
    target_district = _normalize_admin_name(target.get("district"))
    source_city = _normalize_admin_name(source.get("city"))
    target_city = _normalize_admin_name(target.get("city"))
    source_province = _normalize_admin_name(source.get("province"))
    target_province = _normalize_admin_name(target.get("province"))
    source_standard = _normalize_admin_name(source.get("standard_name"))
    target_standard = _normalize_admin_name(target.get("standard_name"))

    if source_code and target_code and source_code == target_code:
        return "同区县", ""
    if source_standard and target_standard and source_standard == target_standard:
        return "同区县", ""
    if source_district and target_district and source_district == target_district:
        return "同区县", ""
    if source_city and target_city and source_city == target_city:
        return "同城市", ""
    if source_province and target_province and source_province == target_province:
        return "同省附近", ""
    return "地名不一致", "输入地点与USGS事件行政区不一致，请结合距离人工复核"


class TimeNormalizer:
    """Normalize time expressions into absolute UTC interval."""

    @staticmethod
    def interval_from_text(text: str, now: datetime | None = None) -> dict[str, str]:
        tz = _default_timezone_for_text(text)
        now_utc = _to_utc(now or get_current_utc_time())
        now_local = now_utc.astimezone(tz)
        start_local = now_local - timedelta(hours=1)
        end_local = now_local

        explicit_match = re.search(
            r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日(?:\s*(\d{1,2})(?:\s*[:：时点]\s*(\d{1,2}))?\s*分?)?",
            text,
        )
        if explicit_match:
            year = int(explicit_match.group(1))
            month = int(explicit_match.group(2))
            day = int(explicit_match.group(3))
            hour_group = explicit_match.group(4)
            minute_group = explicit_match.group(5)
            if hour_group is not None:
                hour = min(int(hour_group), 23)
                minute = min(int(minute_group or 0), 59)
                point_local = datetime(year, month, day, hour, minute, tzinfo=tz)
                start_local = point_local - timedelta(minutes=3)
                end_local = point_local + timedelta(minutes=3)
            else:
                start_local = datetime(year, month, day, 0, 0, tzinfo=tz)
                end_local = start_local + timedelta(days=1)
            return {"start_time": _iso_z(start_local), "end_time": _iso_z(end_local)}

        future_hours = re.search(r"未来\s*(\d+)\s*小时", text)
        future_days = re.search(r"未来\s*(\d+)\s*天", text)
        past_hours = re.search(r"过去\s*(\d+)\s*小时", text)
        past_days = re.search(r"过去\s*(\d+)\s*天", text)
        past_minutes = re.search(r"过去\s*(\d+)\s*分钟", text)
        future_minutes = re.search(r"未来\s*(\d+)\s*分钟", text)
        point_match = re.search(r"(\d{1,2})点(?:\s*(\d{1,2})分?)?", text)

        if future_hours:
            start_local = now_local
            end_local = now_local + timedelta(hours=int(future_hours.group(1)))
        elif future_days:
            start_local = now_local
            end_local = now_local + timedelta(days=int(future_days.group(1)))
        elif future_minutes:
            start_local = now_local
            end_local = now_local + timedelta(minutes=int(future_minutes.group(1)))
        elif past_hours:
            start_local = now_local - timedelta(hours=int(past_hours.group(1)))
            end_local = now_local
        elif past_days:
            start_local = now_local - timedelta(days=int(past_days.group(1)))
            end_local = now_local
        elif past_minutes:
            start_local = now_local - timedelta(minutes=int(past_minutes.group(1)))
            end_local = now_local
        else:
            day_anchor = now_local
            if "明天" in text:
                day_anchor = now_local + timedelta(days=1)
            elif "昨天" in text:
                day_anchor = now_local - timedelta(days=1)

            if "现在" in text:
                start_local = now_local - timedelta(hours=1)
                end_local = now_local
            elif "今天" in text or "明天" in text or "昨天" in text:
                start_local = day_anchor.replace(hour=0, minute=0, second=0, microsecond=0)
                end_local = start_local + timedelta(days=1)

            if point_match:
                hour = min(int(point_match.group(1)), 23)
                minute = min(int(point_match.group(2) or 0), 59)
                point = day_anchor.replace(hour=hour, minute=minute, second=0, microsecond=0)
                # Specific time points are checked within ±3 minutes.
                start_local = point - timedelta(minutes=3)
                end_local = point + timedelta(minutes=3)

        if end_local <= start_local:
            end_local = start_local + timedelta(minutes=1)

        return {"start_time": _iso_z(start_local), "end_time": _iso_z(end_local)}

    @staticmethod
    def normalize_claim_interval(claim: dict[str, Any], text: str) -> dict[str, Any]:
        normalized = dict(claim)
        interval = normalized.get("time_interval", {})
        now = get_current_utc_time()
        start = _parse_iso(interval.get("start_time"))
        end = _parse_iso(interval.get("end_time"))
        if start is None or end is None:
            normalized["time_interval"] = TimeNormalizer.interval_from_text(text, now)
            return normalized

        has_explicit_cn_datetime = bool(re.search(r"\d{4}年\s*\d{1,2}月\s*\d{1,2}日", text))
        if _is_chinese_context(text) and has_explicit_cn_datetime:
            normalized["time_interval"] = TimeNormalizer.interval_from_text(text, now)
            return normalized

        if end <= start:
            end = start + timedelta(minutes=1)

        # Post-correct LLM time drift using text semantics.
        has_future_hint = bool(re.search(r"(未来|稍后|接下来)", text))
        has_past_hint = bool(re.search(r"(过去|此前|之前|刚刚)", text))
        has_now_hint = "现在" in text

        if has_future_hint and end <= now:
            normalized["time_interval"] = TimeNormalizer.interval_from_text(text, now)
            return normalized
        if has_past_hint and start >= now:
            normalized["time_interval"] = TimeNormalizer.interval_from_text(text, now)
            return normalized
        if has_now_hint and not (start <= now <= end):
            normalized["time_interval"] = TimeNormalizer.interval_from_text(text, now)
            return normalized

        normalized["time_interval"] = {"start_time": _iso_z(start), "end_time": _iso_z(end)}
        return normalized


class LLMClaimExtractor:
    """Extract disaster claims into normalized structured objects."""

    def __init__(self, api_key: str, base_url: str | None = None, model: str = "gpt-4o-mini"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=30.0,
            max_retries=1,
        )
        self.model = model

    def extract_claims(self, text: str) -> dict[str, Any]:
        current_utc = _iso_z(get_current_utc_time())
        system_prompt = """
You are an information extraction system for disaster fact-checking.

Return STRICT JSON:
{
  "claims": [
    {
      "hazard_type": "earthquake",
      "location": "string",
      "lat": float | null,
      "lon": float | null,
      "fact_type": "occurrence | magnitude | count | fatality | injury",
      "polarity": "occurred | not_occurred | null",
      "operator": "eq | gt | gte | lt | lte | between | null",
      "value": float | null,
      "value_max": float | null,
      "unit": "Mw | events | people | null",
      "time_interval": {
        "start_time": "ISO8601 UTC, e.g. 2026-04-27T00:00:00Z",
        "end_time": "ISO8601 UTC, e.g. 2026-04-27T02:00:00Z"
      }
    }
  ]
}

Rules:
- Extract only disaster claims that can be checked.
- For this system, set hazard_type to earthquake.
- Preserve the location in the language and wording used by the user. Do not translate Chinese locations.
- If a statement says an earthquake occurred and includes a magnitude, output BOTH an occurrence claim and a magnitude claim with the same location, coordinates, and time.
- Convert relative time (past/future/minutes/hours/days/now/specific time) into absolute interval.
- Use CURRENT_UTC_TIME as the anchor for all relative time expressions.
- "没有地震/未发生地震" => fact_type=occurrence, polarity=not_occurred.
- "发生地震/有地震" => fact_type=occurrence, polarity=occurred.
- "X级地震" => fact_type=magnitude, value=X, operator defaults to eq unless wording implies comparison.
- "发生N次地震" => fact_type=count, value=N.
- "伤亡/死亡/受伤" => fact_type=fatality or injury.
- If location missing, use Taiwan with lat=23.6978 lon=120.9605.
- JSON only.
""".strip()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "system", "content": f"CURRENT_UTC_TIME={current_utc}"},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content)
            return self._complete_extraction(payload, text)
        except Exception as exc:
            payload = {
                "claims": [self._fallback_extract(text)],
                "meta": {"fallback": True, "error_type": type(exc).__name__, "error": str(exc)},
            }
            return self._complete_extraction(payload, text)

    def _complete_extraction(self, payload: dict[str, Any], text: str) -> dict[str, Any]:
        claims = [dict(claim) for claim in payload.get("claims", []) if isinstance(claim, dict)]
        facts = _extract_explicit_earthquake_facts(text)
        completed: list[str] = []

        for claim in claims:
            claim["hazard_type"] = claim.get("hazard_type") or "earthquake"
            if facts.get("location"):
                claim["location"] = facts["location"]
                claim["location_source"] = "input_text"
            if isinstance(facts.get("lat"), float) and isinstance(facts.get("lon"), float):
                claim["lat"] = facts["lat"]
                claim["lon"] = facts["lon"]

        by_type = {claim.get("fact_type"): claim for claim in claims}
        base = next(iter(claims), {"hazard_type": "earthquake"})

        if facts.get("polarity") and "occurrence" not in by_type:
            occurrence = dict(base)
            occurrence.update(
                {
                    "hazard_type": "earthquake",
                    "fact_type": "occurrence",
                    "polarity": facts["polarity"],
                    "operator": "eq",
                    "value": None,
                    "value_max": None,
                    "unit": None,
                    "time_interval": TimeNormalizer.interval_from_text(text),
                }
            )
            claims.append(occurrence)
            by_type["occurrence"] = occurrence
            completed.append("occurrence")

        if isinstance(facts.get("magnitude"), float) and "magnitude" not in by_type:
            magnitude = dict(base)
            magnitude.update(
                {
                    "hazard_type": "earthquake",
                    "fact_type": "magnitude",
                    "polarity": None,
                    "operator": _detect_operator(text),
                    "value": facts["magnitude"],
                    "value_max": None,
                    "unit": "Mw",
                    "time_interval": TimeNormalizer.interval_from_text(text),
                }
            )
            claims.append(magnitude)
            by_type["magnitude"] = magnitude
            completed.append("magnitude")

        if isinstance(facts.get("magnitude"), float) and "magnitude" in by_type:
            by_type["magnitude"]["value"] = facts["magnitude"]
            by_type["magnitude"]["unit"] = by_type["magnitude"].get("unit") or "Mw"
        if facts.get("polarity") and "occurrence" in by_type:
            by_type["occurrence"]["polarity"] = facts["polarity"]

        magnitude_value = by_type.get("magnitude", {}).get("value")
        if isinstance(magnitude_value, (int, float)) and "occurrence" in by_type:
            by_type["occurrence"]["reported_magnitude"] = float(magnitude_value)

        ordered_claims = []
        for fact_type in ("occurrence", "magnitude"):
            if fact_type in by_type:
                ordered_claims.append(by_type[fact_type])
        ordered_claims.extend(
            claim for claim in claims if claim.get("fact_type") not in {"occurrence", "magnitude"}
        )

        meta = dict(payload.get("meta") or {})
        if completed:
            meta["completed_claims"] = completed
        if facts.get("location"):
            meta["location_source"] = "input_text"
        payload["claims"] = ordered_claims
        if meta:
            payload["meta"] = meta
        return payload

    def _fallback_extract(self, text: str) -> dict[str, Any]:
        polarity = "not_occurred" if re.search(r"(没有|未发生|无).{0,2}地震", text) else "occurred"
        operator = _detect_operator(text)
        magnitude_match = re.search(r"(\d+(?:\.\d+)?)\s*级", text)
        count_match = re.search(r"(\d+)\s*(次|起)", text)
        people_match = re.search(r"(\d+)\s*(人)", text)

        fact_type = "occurrence"
        value: float | None = None
        value_max: float | None = None
        unit: str | None = None
        if re.search(r"(死亡|遇难|伤亡)", text):
            fact_type = "fatality"
            value = float(people_match.group(1)) if people_match else None
            unit = "people"
        elif re.search(r"(受伤|伤者)", text):
            fact_type = "injury"
            value = float(people_match.group(1)) if people_match else None
            unit = "people"
        elif magnitude_match:
            fact_type = "magnitude"
            value = float(magnitude_match.group(1))
            unit = "Mw"
        elif count_match:
            fact_type = "count"
            value = float(count_match.group(1))
            unit = "events"

        location = "Taiwan"
        lat = 23.6978
        lon = 120.9605

        loc_match = re.search(r"(在|于)([^，。,\s]{1,20})(发生|有|出现|过去|未来|现在)", text)
        if loc_match:
            location = loc_match.group(2)
            lat = None
            lon = None

        return {
            "hazard_type": "earthquake",
            "location": location,
            "lat": lat,
            "lon": lon,
            "polarity": polarity,
            "fact_type": fact_type,
            "operator": operator,
            "value": value,
            "value_max": value_max,
            "unit": unit,
            "time_interval": TimeNormalizer.interval_from_text(text),
        }


class Geocoder:
    AMAP_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"
    AMAP_REVERSE_URL = "https://restapi.amap.com/v3/geocode/regeo"
    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
    NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
    OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

    def __init__(self):
        self._cache: dict[str, tuple[float, float]] = {}
        self._reverse_cache: dict[tuple[float, float], dict[str, Any] | None] = {}
        self._last_nominatim_request_at = 0.0

    def _wait_for_nominatim(self) -> None:
        elapsed = time.monotonic() - self._last_nominatim_request_at
        if elapsed < 1.05:
            time.sleep(1.05 - elapsed)
        self._last_nominatim_request_at = time.monotonic()

    def resolve(self, location: str) -> tuple[float, float] | None:
        if not location:
            return None
        if location in self._cache:
            return self._cache[location]

        # Prefer AMap for Chinese addresses; fallback to Nominatim/Open-Meteo.
        point = self._resolve_amap(location)
        if point is None:
            point = self._resolve_nominatim(location)
        if point is None:
            point = self._resolve_open_meteo(location)

        if point is not None:
            self._cache[location] = point
        return point

    def reverse_resolve(self, lat: float, lon: float) -> dict[str, Any] | None:
        cache_key = (round(float(lat), 4), round(float(lon), 4))
        if cache_key in self._reverse_cache:
            return self._reverse_cache[cache_key]

        area = self._reverse_amap(float(lat), float(lon))
        if area is None:
            area = self._reverse_nominatim(float(lat), float(lon))
        self._reverse_cache[cache_key] = area
        return area

    @staticmethod
    def _standard_admin_name(province: str, city: str, district: str, fallback: str = "") -> str:
        parts: list[str] = []
        for value in (province, city, district):
            if value and value not in parts:
                parts.append(value)
        return "".join(parts) or fallback

    def _reverse_amap(self, lat: float, lon: float) -> dict[str, Any] | None:
        if not AMAP_KEY:
            return None
        try:
            params = {
                "key": AMAP_KEY,
                "location": f"{lon},{lat}",
                "output": "JSON",
                "extensions": "base",
                "radius": 1000,
            }
            response = requests.get(self.AMAP_REVERSE_URL, params=params, timeout=8)
            response.raise_for_status()
            payload = response.json()
            if str(payload.get("status")) != "1":
                return None
            regeocode = payload.get("regeocode") or {}
            component = regeocode.get("addressComponent") or {}

            def scalar(value: Any) -> str:
                return value if isinstance(value, str) else ""

            province = scalar(component.get("province"))
            city = scalar(component.get("city")) or scalar(component.get("province"))
            district = scalar(component.get("district"))
            formatted = scalar(regeocode.get("formatted_address"))
            standard_name = self._standard_admin_name(province, city, district, formatted)
            if not standard_name:
                return None
            return {
                "province": province,
                "city": city,
                "district": district,
                "adcode": scalar(component.get("adcode")),
                "standard_name": standard_name,
                "source": "amap",
            }
        except Exception:
            return None

    def _reverse_nominatim(self, lat: float, lon: float) -> dict[str, Any] | None:
        try:
            self._wait_for_nominatim()
            params = {
                "lat": lat,
                "lon": lon,
                "format": "jsonv2",
                "zoom": 10,
                "addressdetails": 1,
                "accept-language": "zh-CN",
            }
            headers = {"User-Agent": "disaster-verifier/1.0"}
            response = requests.get(self.NOMINATIM_REVERSE_URL, params=params, headers=headers, timeout=8)
            response.raise_for_status()
            payload = response.json()
            address = payload.get("address") or {}
            province = _clean_chinese_place(address.get("state") or address.get("province") or "")
            city = _clean_chinese_place(
                address.get("city")
                or address.get("municipality")
                or address.get("county")
                or address.get("town")
                or ""
            )
            district = _clean_chinese_place(
                address.get("district")
                or address.get("city_district")
                or address.get("county")
                or ""
            )
            display_name = _clean_chinese_place(payload.get("display_name") or "")
            standard_name = self._standard_admin_name(province, city, district, display_name)
            if not standard_name:
                return None
            return {
                "province": province,
                "city": city,
                "district": district,
                "adcode": "",
                "standard_name": standard_name,
                "source": "nominatim",
            }
        except Exception:
            return None

    def _resolve_amap(self, location: str) -> tuple[float, float] | None:
        amap_key = AMAP_KEY
        if not amap_key:
            return None

        try:
            params = {
                "key": amap_key,
                "address": location,
                "output": "JSON",
            }
            response = requests.get(self.AMAP_GEOCODE_URL, params=params, timeout=8)
            response.raise_for_status()
            payload = response.json()
            if str(payload.get("status")) != "1":
                return None
            if int(payload.get("count", "0")) < 1:
                return None

            geocodes = payload.get("geocodes") or []
            if not geocodes:
                return None
            location_str = geocodes[0].get("location", "")
            if "," not in location_str:
                return None
            lon_str, lat_str = location_str.split(",", 1)
            return (float(lat_str), float(lon_str))
        except Exception:
            return None

    def _resolve_nominatim(self, location: str) -> tuple[float, float] | None:
        try:
            self._wait_for_nominatim()
            params = {"q": location, "format": "jsonv2", "limit": 1}
            headers = {"User-Agent": "disaster-verifier/1.0"}
            response = requests.get(self.NOMINATIM_URL, params=params, headers=headers, timeout=8)
            response.raise_for_status()
            payload = response.json()
            if not payload:
                return None
            return (float(payload[0]["lat"]), float(payload[0]["lon"]))
        except Exception:
            return None

    def _resolve_open_meteo(self, location: str) -> tuple[float, float] | None:
        try:
            params = {"name": location, "count": 1, "language": "zh", "format": "json"}
            response = requests.get(self.OPEN_METEO_GEOCODE_URL, params=params, timeout=8)
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results") or []
            if not results:
                return None
            first = results[0]
            return (float(first["latitude"]), float(first["longitude"]))
        except Exception:
            return None


class EvidenceTool(Protocol):
    hazard_type: str

    def fetch_evidence(self, claim: dict[str, Any]) -> dict[str, Any]:
        ...

    def verify(self, claim: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
        ...


class EarthquakeUSGSTool:
    """Earthquake fact tool backed by USGS earthquake feed."""

    hazard_type = "earthquake"
    BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    def __init__(
        self,
        geocoder: Geocoder,
        radius_km: float = 100.0,
        magnitude_tolerance: float = 0.5,
        reliable_min_magnitude: float = 4.5,
    ):
        self.geocoder = geocoder
        self.radius_km = radius_km
        self.magnitude_tolerance = magnitude_tolerance
        self.reliable_min_magnitude = reliable_min_magnitude
        self._evidence_cache: dict[tuple[tuple[str, Any], ...], dict[str, Any]] = {}

    def fetch_evidence(self, claim: dict[str, Any]) -> dict[str, Any]:
        interval = claim["time_interval"]
        start = _parse_iso(interval["start_time"])
        end = _parse_iso(interval["end_time"])
        if start is None or end is None:
            raise ValueError("time_interval must use ISO8601.")

        now = datetime.now(timezone.utc)
        # If any portion of the interval is in the future, treat it as non-observable.
        if end > now:
            return {"features": [], "meta": {"future_only": True}}

        params = {
            "format": "geojson",
            "starttime": _iso_z(start),
            "endtime": _iso_z(end),
            "orderby": "time-asc",
            "limit": 200,
        }

        # Optional optimization for magnitude checks.
        if claim.get("fact_type") == "magnitude":
            op = claim.get("operator")
            value = claim.get("value")
            if isinstance(value, (int, float)) and op in ("gt", "gte"):
                params["minmagnitude"] = float(value)

        lat = claim.get("lat")
        lon = claim.get("lon")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            params["latitude"] = float(lat)
            params["longitude"] = float(lon)
            params["maxradiuskm"] = float(claim.get("radius_km", self.radius_km))

        cache_key = tuple(sorted(params.items()))
        if cache_key in self._evidence_cache:
            return self._evidence_cache[cache_key]

        response = requests.get(self.BASE_URL, params=params, timeout=12)
        response.raise_for_status()
        payload = response.json()
        self._evidence_cache[cache_key] = payload
        return payload

    def _rank_candidates(
        self,
        claim: dict[str, Any],
        features: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        lat = claim.get("lat")
        lon = claim.get("lon")
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            return []

        interval = claim.get("time_interval") or {}
        start = _parse_iso(interval.get("start_time"))
        end = _parse_iso(interval.get("end_time"))
        midpoint_ms = None
        if start is not None and end is not None:
            midpoint_ms = ((start.timestamp() + end.timestamp()) / 2) * 1000

        radius_km = float(claim.get("radius_km", self.radius_km))
        source_admin = claim.get("admin") if isinstance(claim.get("admin"), dict) else None
        if source_admin is None and features:
            source_admin = self.geocoder.reverse_resolve(float(lat), float(lon))
            if source_admin is not None:
                claim["admin"] = source_admin
                claim["standard_location"] = source_admin.get("standard_name")
        match_rank = {"同区县": 0, "同城市": 1, "同省附近": 2, "无法解析": 3, "地名不一致": 4}
        candidates: list[dict[str, Any]] = []

        for feature in features:
            geometry = feature.get("geometry") or {}
            coordinates = geometry.get("coordinates") or []
            if len(coordinates) < 2:
                continue
            try:
                event_lon = float(coordinates[0])
                event_lat = float(coordinates[1])
            except (TypeError, ValueError):
                continue

            distance_km = _haversine_km(float(lat), float(lon), event_lat, event_lon)
            if distance_km > radius_km:
                continue

            event_admin = self.geocoder.reverse_resolve(event_lat, event_lon)
            place_match, place_warning = _compare_admin_areas(source_admin, event_admin)
            props = feature.get("properties") or {}
            event_time_ms = props.get("time")
            time_diff_seconds = 0.0
            if midpoint_ms is not None and isinstance(event_time_ms, (int, float)):
                time_diff_seconds = abs(float(event_time_ms) - midpoint_ms) / 1000

            candidates.append(
                {
                    "feature": feature,
                    "distance_km": distance_km,
                    "event_admin": event_admin,
                    "place_match": place_match,
                    "place_warning": place_warning,
                    "time_diff_seconds": time_diff_seconds,
                    "sort_key": (
                        match_rank.get(place_match, 5),
                        distance_km,
                        time_diff_seconds,
                    ),
                }
            )

        candidates.sort(key=lambda candidate: candidate["sort_key"])
        return candidates

    @staticmethod
    def _sample_event(candidate: dict[str, Any]) -> dict[str, Any]:
        feature = candidate["feature"]
        props = feature.get("properties") or {}
        event_time_ms = props.get("time")
        event_time = None
        if isinstance(event_time_ms, (int, float)):
            event_time = _iso_z(datetime.fromtimestamp(event_time_ms / 1000, tz=timezone.utc))
        event_admin = candidate.get("event_admin") or {}
        return {
            "time": event_time,
            "magnitude": props.get("mag"),
            "place_raw": props.get("place"),
            "place_zh": event_admin.get("standard_name") or "无法解析",
            "distance_km": round(float(candidate["distance_km"]), 1),
            "place_match": candidate["place_match"],
            "place_warning": candidate["place_warning"],
        }

    def verify(self, claim: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
        if evidence.get("meta", {}).get("future_only"):
            return {"claim": claim, "status": "unknown", "reason": "future_interval_not_observable"}

        features = evidence.get("features", [])
        candidates = self._rank_candidates(claim, features)
        event_count = len(candidates)
        fact_type = claim.get("fact_type") or "occurrence"
        operator = claim.get("operator") or "eq"
        value = claim.get("value")
        value_max = claim.get("value_max")

        top_events = [self._sample_event(candidate) for candidate in candidates[:3]]
        warnings = [event["place_warning"] for event in top_events[:1] if event.get("place_warning")]

        if fact_type == "occurrence":
            polarity = claim.get("polarity")
            if polarity == "occurred":
                if event_count > 0:
                    status = "supported"
                    reason = None
                else:
                    reported_magnitude = claim.get("reported_magnitude")
                    if (
                        isinstance(reported_magnitude, (int, float))
                        and float(reported_magnitude) < self.reliable_min_magnitude
                    ):
                        status = "unknown"
                        reason = "below_usgs_reliable_coverage"
                    else:
                        status = "contradicted"
                        reason = None
            elif polarity == "not_occurred":
                status = "supported" if event_count == 0 else "contradicted"
                reason = None
            else:
                return {"claim": claim, "status": "unknown", "reason": "unsupported_polarity"}
            result = {
                "claim": claim,
                "fact_type": fact_type,
                "event_count": event_count,
                "sample_events": top_events,
                "status": status,
                "warnings": warnings,
            }
            if reason:
                result["reason"] = reason
                result["reliable_min_magnitude"] = self.reliable_min_magnitude
            return result

        if fact_type == "count":
            if not isinstance(value, (int, float)):
                return {"claim": claim, "status": "unknown", "reason": "missing_count_value"}
            actual_count = float(event_count)
            supported = _compare_numeric(
                actual_count,
                operator,
                float(value),
                float(value_max) if isinstance(value_max, (int, float)) else None,
            )
            return {
                "claim": claim,
                "fact_type": fact_type,
                "actual_value": actual_count,
                "operator": operator,
                "expected_value": float(value),
                "expected_value_max": value_max,
                "sample_events": top_events,
                "status": "supported" if supported else "contradicted",
            }

        if fact_type == "magnitude":
            magnitude_candidates = [
                candidate
                for candidate in candidates
                if isinstance(candidate["feature"].get("properties", {}).get("mag"), (int, float))
            ]
            if not magnitude_candidates:
                return {"claim": claim, "status": "unknown", "reason": "no_magnitude_data_in_interval"}
            if not isinstance(value, (int, float)):
                return {"claim": claim, "status": "unknown", "reason": "missing_magnitude_value"}
            best_candidate = magnitude_candidates[0]
            actual_mag = float(best_candidate["feature"]["properties"]["mag"])
            expected_mag = float(value)
            if operator == "eq":
                supported = abs(actual_mag - expected_mag) <= self.magnitude_tolerance + 1e-9
            else:
                supported = _compare_numeric(
                    actual_mag,
                    operator,
                    expected_mag,
                    float(value_max) if isinstance(value_max, (int, float)) else None,
                )
            return {
                "claim": claim,
                "fact_type": fact_type,
                "actual_value": actual_mag,
                "operator": operator,
                "expected_value": expected_mag,
                "expected_value_max": value_max,
                "tolerance": self.magnitude_tolerance if operator == "eq" else None,
                "sample_events": top_events,
                "warnings": warnings,
                "status": "supported" if supported else "contradicted",
            }

        if fact_type in ("fatality", "injury"):
            return {
                "claim": claim,
                "fact_type": fact_type,
                "status": "unknown",
                "reason": "insufficient_evidence_source_for_casualty_metrics",
            }

        return {"claim": claim, "status": "unknown", "reason": "unsupported_fact_type"}


class ClaimSchemaNormalizer:
    @staticmethod
    def normalize(claim: dict[str, Any], text: str) -> dict[str, Any]:
        normalized = dict(claim)
        fact_type = normalized.get("fact_type")
        if not fact_type:
            if re.search(r"(死亡|遇难|伤亡)", text):
                fact_type = "fatality"
            elif re.search(r"(受伤|伤者)", text):
                fact_type = "injury"
            elif re.search(r"\d+\s*级", text):
                fact_type = "magnitude"
            elif re.search(r"\d+\s*(次|起)", text):
                fact_type = "count"
            else:
                fact_type = "occurrence"
        normalized["fact_type"] = fact_type

        if normalized.get("operator") is None:
            normalized["operator"] = _detect_operator(text)
        if "value_max" not in normalized:
            normalized["value_max"] = None

        if fact_type == "magnitude" and normalized.get("value") is None:
            match = re.search(r"(\d+(?:\.\d+)?)\s*级", text)
            if match:
                normalized["value"] = float(match.group(1))
                normalized["unit"] = normalized.get("unit") or "Mw"
        if fact_type == "count" and normalized.get("value") is None:
            match = re.search(r"(\d+)\s*(次|起)", text)
            if match:
                normalized["value"] = float(match.group(1))
                normalized["unit"] = normalized.get("unit") or "events"
        if fact_type in ("fatality", "injury") and normalized.get("unit") is None:
            normalized["unit"] = "people"

        if fact_type == "occurrence" and normalized.get("polarity") is None:
            normalized["polarity"] = (
                "not_occurred" if re.search(r"(没有|未发生|无).{0,2}地震", text) else "occurred"
            )
        return normalized


class EvidenceToolRegistry:
    def __init__(self):
        self._tools: dict[str, EvidenceTool] = {}

    def register(self, tool: EvidenceTool) -> None:
        self._tools[tool.hazard_type] = tool

    def get(self, hazard_type: str) -> EvidenceTool | None:
        return self._tools.get(hazard_type)


class DisasterHallucinationDetector:
    """Pipeline orchestrator for multi-tool disaster hallucination checks."""

    def __init__(self, extractor: LLMClaimExtractor, registry: EvidenceToolRegistry, geocoder: Geocoder):
        self.extractor = extractor
        self.registry = registry
        self.geocoder = geocoder

    def _resolve_coordinates(self, claim: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(claim)
        lat = normalized.get("lat")
        lon = normalized.get("lon")
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            location = normalized.get("location", "")
            resolved = self.geocoder.resolve(location)
            if resolved is None:
                return normalized
            normalized["lat"], normalized["lon"] = resolved
            lat, lon = resolved

        return normalized

    def run(self, text: str) -> dict[str, Any]:
        extraction = self.extractor.extract_claims(text)
        claims = extraction.get("claims", [])
        results: list[dict[str, Any]] = []

        for raw_claim in claims:
            claim = TimeNormalizer.normalize_claim_interval(raw_claim, text)
            claim = ClaimSchemaNormalizer.normalize(claim, text)
            claim = self._resolve_coordinates(claim)
            hazard_type = claim.get("hazard_type", "")
            tool = self.registry.get(hazard_type)

            if tool is None:
                results.append({"claim": claim, "status": "unknown", "reason": "no_tool_for_hazard"})
                continue

            try:
                evidence = tool.fetch_evidence(claim)
                results.append(tool.verify(claim, evidence))
            except Exception as exc:
                results.append(
                    {
                        "claim": claim,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

        return {"input_text": text, "results": results, "extraction": extraction}

    def run_batch(self, texts: list[str]) -> dict[str, Any]:
        return {"total": len(texts), "items": [{"index": i + 1, **self.run(t)} for i, t in enumerate(texts)]}


def create_detector_from_env() -> DisasterHallucinationDetector:
    provider = os.getenv("DISASTER_LLM_PROVIDER", "qwen").lower()
    model = os.getenv("DISASTER_LLM_MODEL", "qwen-plus")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY") or OpenAI_KEY2
        base_url = os.getenv("DISASTER_LLM_BASE_URL")
        model = os.getenv("DISASTER_LLM_MODEL", "gpt-4o-mini")
    else:
        api_key = os.getenv("QWEN_KEY") or QWEN_KEY
        base_url = os.getenv(
            "DISASTER_LLM_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    if not api_key:
        raise ValueError("Missing API key. Set QWEN_KEY or OPENAI_API_KEY.")

    geocoder = Geocoder()
    extractor = LLMClaimExtractor(api_key=api_key, base_url=base_url, model=model)
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
    return DisasterHallucinationDetector(extractor=extractor, registry=registry, geocoder=geocoder)
