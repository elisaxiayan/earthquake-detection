import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol, Literal
from zoneinfo import ZoneInfo

import requests
from openai import OpenAI

from bs4 import BeautifulSoup
import random

from key import OpenAI_KEY2, QWEN_KEY, AMAP_KEY
from pypinyin import lazy_pinyin, Style


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



def has_valid_location(claim: dict[str, Any]) -> bool:
    lat = claim.get("lat")
    lon = claim.get("lon")
    location = claim.get("location")

    if location in (None, "", "unknown"):
        return False

    return isinstance(lat, (int, float)) and isinstance(lon, (int, float))

def iso_z_to_beijing_datetime(value: str) -> datetime:
    BJ_TZ = ZoneInfo("Asia/Shanghai")
    dt = _parse_iso(value)

    if dt is None:
        raise ValueError(f"Invalid ISO datetime: {value}")

    return dt.astimezone(BJ_TZ)


def claim_interval_to_beijing_dates(time_interval: dict[str, str]) -> tuple[str, str]:
    start = _parse_iso(time_interval["start_time"])
    end = _parse_iso(time_interval["end_time"])

    BJ_TZ = ZoneInfo("Asia/Shanghai")

    if start is None or end is None:
        raise ValueError("time_interval must use ISO8601.")

    if end <= start:
        end = start + timedelta(seconds=1)

    begtime = start.astimezone(BJ_TZ).date().isoformat()

    # end_time 是右边界，所以减 1 秒再取日期，避免多查一天
    endtime = (end - timedelta(seconds=1)).astimezone(BJ_TZ).date().isoformat()

    return begtime, endtime

def assistant_tool_for_fetch_earthquake_cn_events(page_id : str, claim : dict[str : Any]) -> list[dict[str : Any]]:

    time_interval = claim["time_interval"]
    st_time, ed_time = claim_interval_to_beijing_dates(time_interval)

    minlon = float(claim["lon"]) - 10
    maxlon = float(claim["lon"]) + 10
    minlat = float(claim["lat"]) - 10
    maxlat = float(claim["lat"]) + 10

    return fetch_earthquake_cn_events(page_id=page_id, begtime=st_time, endtime=ed_time, min_lon=minlon, max_lon=maxlon, min_lat=minlat, max_lat=maxlat)

#pageid有earthquake_subao, earthquake_zhengshi, earthquake_csn, earthquake_sdqdz, 这里使用subao，不然会报错
def fetch_earthquake_cn_events(
    page_id: str,
    begtime: str,
    endtime: str,
    min_m: float = 3,
    max_m: float = 10,
    min_lon: float = -180.0,
    max_lon: float = 180.0,
    min_lat: float = -90.0,
    max_lat: float = 90.0,
    min_depth: float = 0,
    max_depth: float = 1000,
    locationselect: str = "world",
):
    session = requests.Session()

    referer = f"https://data.earthquake.cn/datashare/report.shtml?PAGEID={page_id}"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/149.0.0.0 Safari/537.36 Edg/149.0.0.0"
        ),
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
        "Origin": "https://data.earthquake.cn",
        "Referer": referer,
    }

    # 先访问对应页面拿 cookie
    session.get(referer, headers=headers, timeout=15)

    url = "https://data.earthquake.cn/datashare/report.shtml"

    params = {
        "random": str(random.random())
    }

    component_guid = f"{page_id}_guid_catalog"
    table_id = f"{page_id}_guid_catalog_data"

    data = {
        "DISPLAY_TYPE": "1",
        "PAGEID": page_id,
        "refreshComponentGuid": component_guid,

        "begtime": begtime,
        "endtime": endtime,

        "minM": str(min_m),
        "maxM": str(max_m),

        "minLon": str(min_lon),
        "maxLon": str(max_lon),
        "minLat": str(min_lat),
        "maxLat": str(max_lat),

        "minDepths": str(min_depth),
        "maxDepths": str(max_depth),

        "locationselect": locationselect,

        "SEARCHREPORT_ID": "catalog",
        "WX_ISAJAXLOAD": "true",
    }

    response = session.post(
        url,
        params=params,
        data=data,
        headers=headers,
        timeout=20,
    )

    response.raise_for_status()
    response.encoding = response.apparent_encoding

    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table", id=table_id)
    if table is None:
        return []

    events = []

    for tr in table.find_all("tr"):
        cells = [
            div.get_text(strip=True)
            for div in tr.find_all("div", class_="cls-data-content-list")
        ]

        if len(cells) >= 8 and cells[0].isdigit():
            events.append({
                "index": int(cells[0]),
                "time_bj": cells[1],
                "lon": float(cells[2]),
                "lat": float(cells[3]),
                "depth_km": float(cells[4]),
                "magnitude": float(cells[5]),
                "place": cells[6],
                "event_type": cells[7],
            })

    return events

#把中文地名转成类似 Guangdong / Guangzhou 的形式。
def chinese_place_to_pinyin(name: str) -> str:
    if not name:
        return ""

    # 去掉常见行政区后缀
    suffixes = ["特别行政区", "壮族自治区", "回族自治区", "维吾尔自治区", "自治区", "省", "市", "地区", "盟", "自治州", "州", "县", "区", "旗"]
    
    clean = name.strip()

    for suffix in suffixes:
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)]
            break
    
    if (clean == "吉林"):
        return "Jilin"
    if (clean == "陕西"):
        return "Shaanxi"
    
    syllables = lazy_pinyin(clean, style=Style.NORMAL)

    # 广东 -> guang dong -> Guangdong
    return "".join(part.capitalize() for part in syllables)

#通过经纬度long,lat获取对应的province和city
def reverse_geocode_amap(lat: float, lon: float) -> dict[str, Any] | None:
    amap_key = AMAP_KEY
    if not amap_key:
        return None

    url = "https://restapi.amap.com/v3/geocode/regeo"

    params = {
        "key": amap_key,
        "location": f"{lon},{lat}",  # 高德这里是 lon,lat
        "extensions": "base",
        "radius": 1000,
        "output": "JSON",
    }

    try:
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()

        if str(data.get("status")) != "1":
            return None

        regeocode = data.get("regeocode") or {}
        component = regeocode.get("addressComponent") or {}

        return {
            "formatted_address": regeocode.get("formatted_address"),
            "province": component.get("province"),
            "city": component.get("city"),
            "district": component.get("district"),
            "township": component.get("township"),
            "adcode": component.get("adcode"),
            "citycode": component.get("citycode"),
        }

    except Exception:
        return None

def functionDealWithLocation(claim : str, earthquake : dict) -> bool | None:
    lati = earthquake["geometry"]["coordinates"][1]
    long = earthquake["geometry"]["coordinates"][0]
    temp_pc = reverse_geocode_amap(lat = lati, lon = long)
    if temp_pc is None:
        return None
    usgs_province = temp_pc["province"]
    usgs_city = temp_pc["city"]
    usgs_district = temp_pc["district"]
    temp_pc = reverse_geocode_amap(lat = claim["lat"], lon = claim["lon"])
    if temp_pc is None:
        return None
    claim_province = temp_pc["province"]
    claim_city = temp_pc["city"]
    claim_district = temp_pc["district"]
    claim_level = claim["location_level"]
    if (claim_level == "province"):
        return claim_province == usgs_province
    elif (claim_level == "city"):
        return (claim_province == usgs_province) and (claim_city == usgs_city)
    elif (claim_level == "district"):
        return (claim_province == usgs_province) and (claim_city == usgs_city) and (claim_district == usgs_district)
    else :
        return None

def functionDealWithLocationCN(claim : str, cn_earthquake : dict) -> bool | None:
    lati = cn_earthquake["lat"]
    long = cn_earthquake["lon"]
    temp_pc = reverse_geocode_amap(lat = lati, lon = long)
    if temp_pc is None:
        return None
    subao_province = temp_pc["province"]
    subao_city = temp_pc["city"]
    subao_district = temp_pc["district"]
    temp_pc = reverse_geocode_amap(lat = claim["lat"], lon = claim["lon"])
    if temp_pc is None:
        return None
    claim_province = temp_pc["province"]
    claim_city = temp_pc["city"]
    claim_district = temp_pc["district"]
    claim_level = claim["location_level"]
    if (claim_level == "province"):
        return claim_province == subao_province
    elif (claim_level == "city"):
        return (claim_province == subao_province) and (claim_city == subao_city)
    elif (claim_level == "district"):
        return (claim_province == subao_province) and (claim_city == subao_city) and (claim_district == subao_district)
    else :
        return None

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
        has_future_hint = bool(re.search(r"(未来|稍后|接下来|之后|以后)", text))
        has_past_hint = bool(re.search(r"(过去|此前|之前|刚刚|以前)", text))
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
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    #在这里添加一个新的function，用于提前判断灾害类型
    def detect_hazard_type(self, text) -> str | None:
        system_prompt = """
        You are a disaster system for disaster detection, please detect the hazard type in the text.
        Here are 4 different hazard types: earthquake, flood, wildfire, others.
        others is seen as one type, which means the text is not about earthquake, flood or wildfire.

        After detection, only return one of these exact words(comma seen as seperator for those words):
        earthquake,
        flood,
        wildfire,
        others.

        Do not return explanation.
        """.strip()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0,
            )
            hazard_type = response.choices[0].message.content.strip().lower()
            return hazard_type
        except Exception:
            return None
    #这里为止

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
      "location_level": "province | city | district | null",
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
- Convert relative time (past/future/minutes/hours/days/now/specific time) into absolute interval.
- Use CURRENT_UTC_TIME as the anchor for all relative time expressions.
- "没有地震/未发生地震" => fact_type=occurrence, polarity=not_occurred.
- "发生地震/有地震" => fact_type=occurrence, polarity=occurred.
- "X级地震" => fact_type=magnitude, value=X, operator defaults to eq unless wording implies comparison.
- "发生N次地震" => fact_type=count, value=N.
- "伤亡/死亡/受伤" => fact_type=fatality or injury.
- Claim decomposition rules:
  - Decompose the input text into as many independently checkable claims as possible.
  - For every earthquake-related statement, always create one base occurrence claim.
  - The occurrence claim is mandatory because whether an earthquake occurred is the fundamental check for all other earthquake attributes.
  - If the text states that an earthquake occurred, create an occurrence claim with fact_type="occurrence" and polarity="occurred".
  - If the text states that no earthquake occurred, create an occurrence claim with fact_type="occurrence" and polarity="not_occurred".
  - If the same statement also includes magnitude, count, casualty, injury, or other checkable attributes, create additional separate claims for those attributes.
  - Do not merge occurrence, magnitude, count, fatality, or injury into one claim.
  - Claims extracted from the same earthquake statement should reuse the same location, location_level, lat, lon, and time_interval unless the text gives different values.
  - If the text says "X级地震", create:
    1. one occurrence claim with polarity="occurred";
    2. one magnitude claim with fact_type="magnitude", value=X.
  - If the text says "发生N次地震", create:
    1. one occurrence claim with polarity="occurred";
    2. one count claim with fact_type="count", value=N.
  - If the text says there were casualties or injuries, create separate fatality or injury claims in addition to the occurrence claim.
  - Return multiple claims whenever one sentence contains multiple independently verifiable facts.

- The "location" field must use the official Chinese administrative name, not pinyin or English.
- Normalize informal Chinese place names to official Chinese names, but do not infer a more specific location than the user stated.
  Examples:
  "北京" => "北京市"
  "四川" / "四川省" => "四川省"
  "成都" / "成都市" => "四川省成都市"
  "广州" / "广州市" => "广东省广州市"
  "洛隆县" => "西藏自治区昌都市洛隆县"

- The "location_level" field describes the administrative level of the most specific location explicitly mentioned by the user.
- Allowed values for "location_level" are:
  "province": province-level administrative regions, including provinces, autonomous regions, municipalities directly under the central government, special administrative regions, and Taiwan.
  "city": prefecture-level administrative regions, including prefecture-level cities, autonomous prefectures, prefectures, and leagues.
  "district": county-level administrative regions, including districts, counties, county-level cities, banners, and autonomous counties.
  null: use only when location is missing or the administrative level cannot be determined.
- Do not decide location_level only by the suffix "市". Some "市" are province-level, some are prefecture-level, and some are county-level.
  Examples:
  "北京市" => location="北京市", location_level="province"
  "重庆市" => location="重庆市", location_level="province"
  "成都市" => location="四川省成都市", location_level="city"
  "格尔木市" => location="青海省海西蒙古族藏族自治州格尔木市", location_level="district"
- If the user only mentions a province-level place, return only the province-level name.
  Example: "四川省发生地震" => location="四川省", location_level="province", not location="四川省成都市".
- If the user mentions a city/prefecture-level place, return province + city/prefecture.
  Example: "成都市发生地震" => location="四川省成都市", location_level="city".
  Example: "海西州发生地震" => location="青海省海西蒙古族藏族自治州", location_level="city".
- If the user mentions a county/district-level place, return province + city/prefecture + county/district when known.
  Example: "洛隆县发生地震" => location="西藏自治区昌都市洛隆县", location_level="district".
- Do not add district/county/township/street information unless it is explicitly mentioned or necessary to disambiguate the named place.

- If location missing, use "unknown" with lat=null lon=null and location_level=null.
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
            return json.loads(response.choices[0].message.content)
        except Exception as exc:
            return {
                "claims": [self._fallback_extract(text)],
                "meta": {"fallback": True, "error_type": type(exc).__name__, "error": str(exc)},
            }

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

        location = "unknown"
        lat = None
        lon = None
        location_level = None

        loc_match = re.search(r"(在|于)([^，。,\s]{1,20})(发生|有|出现|过去|未来|现在)", text)
        if loc_match:
            location = loc_match.group(2)
            lat = None
            lon = None
            

        return {
            "hazard_type": "earthquake",
            "location": location,
            "location_level" : location_level,
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
    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
    OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

    def __init__(self):
        self._cache: dict[str, tuple[float, float]] = {}

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
            params["maxradiuskm"] = float(claim.get("radius_km", 550))

        response = requests.get(self.BASE_URL, params=params, timeout=12)
        response.raise_for_status()
        return response.json()

    def verify(self, claim: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
        if evidence.get("meta", {}).get("future_only"):
            return {"claim": claim, "status": "unknown", "reason": "future_interval_not_observable"}

        cn_events = assistant_tool_for_fetch_earthquake_cn_events(page_id="earthquake_subao", claim=claim) 
        features = evidence.get("features", [])


        for feature in features[:]:
            if (functionDealWithLocation(claim=claim, earthquake=feature) is False):
                features.remove(feature)
        for cn_event in cn_events[:]:
            if (functionDealWithLocationCN(claim=claim, cn_earthquake=cn_event) is False):
                cn_events.remove(cn_event)

        
        event_count = len(features) + len(cn_events)
        fact_type = claim.get("fact_type") or "occurrence"
        operator = claim.get("operator") or "eq"
        value = claim.get("value")
        value_max = claim.get("value_max")


        if fact_type == "occurrence":
            polarity = claim.get("polarity")
            if polarity == "occurred":
                supported = event_count > 0
            elif polarity == "not_occurred":
                supported = event_count == 0
            else:
                return {"claim": claim, "status": "unknown", "reason": "unsupported_polarity"}
            return {
                "claim": claim,
                "fact_type": fact_type,
                "event_count": event_count,
                "status": "supported" if supported else "contradicted",
            }

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
                "status": "supported" if supported else "contradicted",
            }

        if fact_type == "magnitude":
            mags = [
                float(feature.get("properties", {}).get("mag"))
                for feature in features
                if isinstance(feature.get("properties", {}).get("mag"), (int, float))
            ]

            #加入subao检索内容
            for i in cn_events:
                if i["magnitude"]:
                    mags.append(i["magnitude"])

            if not mags:
                return {"claim": claim, "status": "unknown", "reason": "no_magnitude_data_in_interval"}
            if not isinstance(value, (int, float)):
                return {"claim": claim, "status": "unknown", "reason": "missing_magnitude_value"}
            

            expected_mag = float(value)
            supported = False
            actual_mag = None


            if operator == "eq":
                for temp in mags:
                    if (abs(temp - expected_mag) <= 0.4):
                        supported = True
                        actual_mag = temp
                        break
            else:
                for temp in mags:
                    supported = _compare_numeric(
                        temp,
                        operator,
                        expected_mag,
                        float(value_max) if isinstance(value_max, (int, float)) else None,
                    )
                    if (supported == True):
                        actual_mag = temp
                        break

            return {
                "claim": claim,
                "fact_type": fact_type,
                "actual_value": actual_mag,
                "operator": operator,
                "expected_value": expected_mag,
                "expected_value_max": value_max,
                "tolerance": 0.4 if operator == "eq" else None, 
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
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            return normalized

        location = normalized.get("location", "")
        resolved = self.geocoder.resolve(location)
        if resolved is None:
            return normalized

        normalized["lat"], normalized["lon"] = resolved
        return normalized

    def run(self, text: str) -> dict[str, Any]:
        extraction = self.extractor.extract_claims(text)
        claims = extraction.get("claims", [])
        results: list[dict[str, Any]] = []

        for raw_claim in claims:
            claim = TimeNormalizer.normalize_claim_interval(raw_claim, text)
            claim = ClaimSchemaNormalizer.normalize(claim, text)
            claim = self._resolve_coordinates(claim)

            if not has_valid_location(claim):
                results.append({
                    "claim": claim,
                    "status": "unknown",
                    "reason": "missing_or_unresolved_location",
                })
                continue

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

    extractor = LLMClaimExtractor(api_key=api_key, base_url=base_url, model=model)
    registry = EvidenceToolRegistry()
    registry.register(EarthquakeUSGSTool())
    geocoder = Geocoder()
    return DisasterHallucinationDetector(extractor=extractor, registry=registry, geocoder=geocoder)
