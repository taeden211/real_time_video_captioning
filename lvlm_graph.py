import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any

from dotenv import load_dotenv

try:
    import openai
except ImportError as exc:
    raise ImportError("pip install openai") from exc


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = os.getenv("OPENAI_SCENE_MODEL", "gpt-5.4-mini")
IMAGE_DIR = os.getenv("IMAGE_DIR", "data")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output1")
MAX_IMAGES = None
CANDIDATES_PER_IMAGE = 2
TEMPERATURE = 0.0
MAX_WORKERS = 6
MAX_RETRIES = 3
RETRY_DELAY = 5
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


# ---------------------------------------------------------------------------
# Ontology
# ---------------------------------------------------------------------------

WORKER_LABEL = "근로자"

TIER1_LABELS = {
    "근로자",
    "굴착기",
    "타워크레인",
    "이동식크레인",
    "덤프트럭",
    "지게차",
    "로드롤러",
    "콘크리트펌프카",
    "레미콘",
    "파일드라이버",
    "고소작업차",
    "불도저",
    "천공기",
    "로더",
    "그레이더",
    "페이버",
    "소형트럭",
    "승용차",
    "살수차",
}

TIER2_LABELS = {
    "철근",
    "합판",
    "파이프",
    "H빔",
    "토사",
    "거푸집",
    "목재",
    "자재",
    "개구부",
}

TIER3_LABELS = {
    "난간",
    "안전망",
    "방호울타리",
    "가설울타리",
    "동바리",
    "비계",
    "작업발판",
}

PREFERRED_LABELS = sorted(
    TIER1_LABELS
    | TIER2_LABELS
    | TIER3_LABELS
    | {
        "콘크리트구조물",
        "교각",
        "거더",
        "옹벽",
        "슬래브",
        "기초",
        "경고표지",
        "신호수",
        "사다리",
        "수역",
        "안전모",
        "안전대",
        "가스배관",
    }
)

FORBIDDEN_LABELS = {"장비", "차량", "기계", "구조물", "시설", "물체", "물건"}

CATEGORY_PREDICATES = {
    "functional": {"operating", "loading", "carrying", "working_on", "walking_on"},
    "structural": {"on", "inside", "attached_to", "supported_by", "connected_to"},
    "spatial": {"next_to", "above", "below", "behind", "in_front_of"},
    "safety": {"too_close_to", "approaching", "blocking"},
}

RELATION_CATEGORY_PRIORITY = {
    "spatial": 1,
    "structural": 2,
    "functional": 3,
    "safety": 4,
}

VALID_PREDICATES = sorted({p for preds in CATEGORY_PREDICATES.values() for p in preds})
VALID_CATEGORIES = sorted(CATEGORY_PREDICATES)
VALID_HAZARDS = ["추락", "낙하물", "충돌", "협착", "전도", "감전", "익수"]

EQUIPMENT_LABELS = TIER1_LABELS - {WORKER_LABEL}
WATER_LABEL = "수역"
SLOPE_CONTEXT_LABELS = {"토사", "콘크리트구조물", "슬래브", "비계", "작업발판"}

ID_PREFIXES = {
    "근로자": "worker",
    "비계": "scaffold",
    "동바리": "prop",
    "작업발판": "platform",
    "난간": "guardrail",
    "안전망": "safety_net",
    "방호울타리": "barrier",
    "가설울타리": "temp_fence",
    "굴착기": "excavator",
    "타워크레인": "tower_crane",
    "이동식크레인": "mobile_crane",
    "덤프트럭": "dump_truck",
    "지게차": "forklift",
    "로드롤러": "roller",
    "콘크리트펌프카": "pump_car",
    "레미콘": "mixer_truck",
    "파일드라이버": "pile_driver",
    "고소작업차": "aerial_lift",
    "불도저": "bulldozer",
    "천공기": "drill",
    "로더": "loader",
    "그레이더": "grader",
    "페이버": "paver",
    "소형트럭": "small_truck",
    "승용차": "car",
    "살수차": "sprinkler",
    "철근": "rebar",
    "합판": "plywood",
    "파이프": "pipe",
    "H빔": "hbeam",
    "토사": "soil",
    "거푸집": "formwork",
    "목재": "wood",
    "자재": "material",
    "개구부": "opening",
    "콘크리트구조물": "structure",
    "교각": "pier",
    "거더": "girder",
    "옹벽": "retaining_wall",
    "슬래브": "slab",
    "기초": "foundation",
    "경고표지": "sign",
    "신호수": "signal_worker",
    "사다리": "ladder",
    "수역": "water",
    "안전모": "helmet",
    "안전대": "safety_belt",
    "가스배관": "gas_pipe",
}

SYNONYM_MAP = {
    "백호": "굴착기",
    "포크레인": "굴착기",
    "굴삭기": "굴착기",
    "크레인": "이동식크레인",
    "기중기": "이동식크레인",
    "롤러": "로드롤러",
    "머캐덤롤러": "로드롤러",
    "진동롤러": "로드롤러",
    "믹서트럭": "레미콘",
    "콘크리트믹서트럭": "레미콘",
    "펌프카": "콘크리트펌프카",
    "콘크리트펌프": "콘크리트펌프카",
    "고소차": "고소작업차",
    "스카이차": "고소작업차",
    "표지판": "경고표지",
    "안전표지": "경고표지",
    "임시펜스": "가설울타리",
    "가설펜스": "가설울타리",
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

OBJECT_SYSTEM_PROMPT = f"""You are a vision-language model for annotating construction site safety scenes.

Goals:
- Extract only objects that are actually visible in the image.
- Produce outputs that will be used for text and scene-graph embedding similarity comparisons.
- Prioritize consistency so that similar visual scenes use the same labels, id prefixes, and relationship directions.

Output rules:
- Follow the JSON schema strictly.
- Do not invent non-visible objects, hidden PPE, or speculative hazards.
- Exclude ambiguous objects or use more conservative labels and lower confidence.
    "- Write `scene_description` as 2–4 concise, factual sentences in Korean.
    - `scene_description` should include scene type, main equipment/workers, observed actions, and surrounding structure/soil/water context (in Korean).
    - Do not assert hazards in `scene_description`; record hazards only in the `hazards` field when clearly visible.
- Bboxes are normalized coordinates [x1, y1, x2, y2] relative to the full image; values must be between 0.0 and 1.0.

Object grouping:
- Tier 1: Workers and all heavy equipment/vehicles must always be individual objects; never group them.
- Tier 2: Rebar, plywood, pipes, H-beams, soil, formwork, wood, material, and openings may be grouped only when three or more of the same type appear in the same area.
- Tier 2 exception: If a specific material is directly interacting with a worker or equipment, separate it as an individual object.
- Tier 3: Guardrails, safety nets, barriers, temporary fences, props, scaffolds, and work platforms may be grouped when they form continuous or repetitive installations.
- `count` is for internal validation: individuals = 1, groups = approximate visible count.

Label rules:
- Prefer these labels: {", ".join(PREFERRED_LABELS)}
- Do not use forbidden generic labels: {", ".join(sorted(FORBIDDEN_LABELS))}
- Use `material` only as a last resort. Prefer specific labels such as rebar, plywood, pipe, wood, formwork, H-beam, soil.
- Record PPE only in a worker's `attributes.ppe`; do not create separate objects for worn helmets or safety belts.

ID rules:
- IDs must use the standard label prefixes.
- Examples: worker_1, excavator_1, tower_crane_1, mobile_crane_1, dump_truck_1, roller_1, mixer_truck_1, barrier_group_1, guardrail_group_1, rebar_group_1.
- Number objects left-to-right.

Heavy equipment distinctions:
- Excavator: visible boom-arm-bucket structure.
- Forklift: two front forks are visible.
- Tower crane: fixed tower and horizontal jib are visible.
- Mobile crane: boom mounted on a truck/crawler is visible.
- Dump truck: a large truck with an open bed is visible.
- Roller: a cylindrical compaction drum is visible.
- Concrete pump: a vehicle with a folding boom for pumping concrete.
- Mixer truck: a rotating drum mixer is visible.

Scan strategy:
- Inspect the image in four quadrants (upper-left, upper-right, lower-left, lower-right) and merge the results.
- Do not miss small workers, water edges, excavation faces, guardrails, or temporary barriers that are important for safety context.
"""

OBJECT_USER_PROMPT = """Extract visible construction site objects from the image.

Procedure:
1. First provide a short scene summary.
2. Identify workers, heavy equipment/vehicles, structures, temporary installations, safety elements, materials, and water/soil contexts important to the scene.
3. For each object, include `id`, `label`, `count`, `bbox`, `location`, `attributes`, `confidence`, and `evidence`.
4. Exclude objects that are not visible or are speculative.
5. Standardize id prefixes and number same-type objects left-to-right.

Return all textual fields (`scene_description`, `evidence`, etc.) in Korean.
"""

RELATION_SYSTEM_PROMPT = f"""You are a model that, given a list of construction site objects and the image, generates scene-graph `relationships` and `hazards`.

Key constraints:
- Do not create new objects.
- `relationships` and `hazards` must reference only the provided object ids.
- Generate only directly visible relationships.
- Keep relations concise and meaningful; avoid creating many unnecessary spatial relations that add noise for embedding comparisons.
- For any (sub_id, obj_id) pair produce only the most important relation.
- Do not duplicate equivalent bidirectional spatial relations.
- Priority order: safety > functional > structural > spatial.
- If a functional relation is clear for a pair, do not also add a spatial relation for the same pair.

Relation categories and predicates:
- functional: operating, loading, carrying, working_on, walking_on
- structural: on, inside, attached_to, supported_by, connected_to
- spatial: next_to, above, below, behind, in_front_of
- safety: too_close_to, approaching, blocking

Direction rules:
- functional: actor or moving/holding object is `sub_id`, target is `obj_id`.
- structural on/inside: the object that is on/inside is `sub_id`, the supporting/containing object is `obj_id`.
- attached_to/supported_by/connected_to: the smaller or dependent object is `sub_id`.
- safety: the worker/equipment/object exposed to risk is `sub_id`, the hazard source or obstacle is `obj_id`.
- spatial: prefer the more important object as `sub_id`; if equal importance, choose the left-most object as `sub_id`.

Hazard generation principles:
- Allowed hazards: {", ".join(VALID_HAZARDS)}
- Record hazards only when a clear hazardous condition is visible in the image.
- If evidence is ambiguous, return an empty `hazards` list.
- Fall: generate only when a worker is at a drop edge or excavation face and the fall direction and missing protection are clearly visible.
- Falling object: generate only when suspended loads, high-level materials, or clear upper-lower contexts are visible.
- Collision: generate only when operating/moving equipment and a worker/equipment are very close.
- Entrapment: generate only when a worker is in a visible pinch point between equipment and fixed structure or materials.
- Overturn: generate only when ladders, equipment, or loads appear visually unstable.
- Electrocution: generate only when exposed wires, electrical equipment, or power lines adjacent to work are clearly visible.
- Drowning: generate only when a worker is at a water edge or in water and insufficient protection is clearly visible.
- Do not generate drowning when only water is present or equipment is near shallow water without clear worker exposure.
- Equipment collaboration is a functional relation; do not create a collision hazard for coordinated equipment work alone.
- Do not create a collision hazard merely because cars and equipment appear in the same scene.
- Do not create fall/overturn hazards solely because excavation faces or soil slopes are visible.
"""

RELATION_USER_PROMPT = """Given the object list and the image, generate `relationships` and `hazards`.

Notes:
- Do not re-output the `objects` array. Return only `relationships` and `hazards`.
- Reference only the provided object ids; do not create new ids or objects.
- Add relations and hazards only when there is visual evidence.
- Write short factual `evidence` and `reason` sentences in Korean.
- If there are no relationships, return an empty list for `relationships`.
- If there are no hazards, return an empty list for `hazards`.
"""


# ---------------------------------------------------------------------------
# JSON schemas
# ---------------------------------------------------------------------------

OBJECT_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "label": {"type": "string"},
        "count": {"type": "integer", "minimum": 1},
        "bbox": {
            "type": "array",
            "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "minItems": 4,
            "maxItems": 4,
        },
        "location": {"type": "string"},
        "attributes": {
            "type": "object",
            "properties": {
                "ppe": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "state": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["ppe", "state"],
            "additionalProperties": False,
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence": {"type": "string"},
    },
    "required": [
        "id",
        "label",
        "count",
        "bbox",
        "location",
        "attributes",
        "confidence",
        "evidence",
    ],
    "additionalProperties": False,
}

OBJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scene_description": {"type": "string"},
        "objects": {
            "type": "array",
            "items": OBJECT_ITEM_SCHEMA,
        },
    },
    "required": ["scene_description", "objects"],
    "additionalProperties": False,
}

RELATION_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sub_id": {"type": "string"},
        "predicate": {"type": "string", "enum": VALID_PREDICATES},
        "obj_id": {"type": "string"},
        "category": {"type": "string", "enum": VALID_CATEGORIES},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence": {"type": "string"},
    },
    "required": ["sub_id", "predicate", "obj_id", "category", "score", "evidence"],
    "additionalProperties": False,
}

HAZARD_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "related_object_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "hazard": {"type": "string", "enum": VALID_HAZARDS},
        "reason": {"type": "string"},
    },
    "required": ["related_object_ids", "hazard", "reason"],
    "additionalProperties": False,
}

SCENE_GRAPH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scene_description": {"type": "string"},
        "objects": {
            "type": "array",
            "items": OBJECT_ITEM_SCHEMA,
        },
        "relationships": {
            "type": "array",
            "items": RELATION_ITEM_SCHEMA,
        },
        "hazards": {
            "type": "array",
            "items": HAZARD_ITEM_SCHEMA,
        },
    },
    "required": ["scene_description", "objects", "relationships", "hazards"],
    "additionalProperties": False,
}

RELATION_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relationships": {
            "type": "array",
            "items": RELATION_ITEM_SCHEMA,
        },
        "hazards": {
            "type": "array",
            "items": HAZARD_ITEM_SCHEMA,
        },
    },
    "required": ["relationships", "hazards"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def image_to_data_uri(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime_map = {
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }
    mime = mime_map.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def find_images() -> list[str]:
    images = []
    for root, dirs, files in os.walk(IMAGE_DIR):
        dirs.sort()
        for filename in sorted(files):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                images.append(os.path.join(root, filename))
    return images


def output_stem_for_image(img_path: str) -> str:
    rel_path = os.path.relpath(img_path, IMAGE_DIR)
    stem = os.path.splitext(rel_path)[0]
    return stem.replace(os.sep, "__").replace("/", "__")


def output_paths_for_image(img_path: str) -> list[str]:
    stem = output_stem_for_image(img_path)
    return [
        os.path.join(OUTPUT_DIR, f"{stem}_{idx}.json")
        for idx in range(1, CANDIDATES_PER_IMAGE + 1)
    ]


def slug_prefix(label: str) -> str:
    prefix = ID_PREFIXES.get(label)
    if prefix:
        return prefix
    slug = re.sub(r"[^a-zA-Z0-9가-힣]+", "_", label).strip("_").lower()
    return slug or "object"


def clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, number))


def normalize_bbox(bbox: Any) -> list[float]:
    if not isinstance(bbox, list):
        return [0.0, 0.0, 1.0, 1.0]
    values = [clamp_float(v, 0.0, 1.0, 0.0) for v in bbox[:4]]
    while len(values) < 4:
        values.append(0.0)
    x1, y1, x2, y2 = values
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def normalize_object(
    obj: dict[str, Any], index: int
) -> tuple[dict[str, Any], list[str]]:
    warnings = []
    label = str(obj.get("label", "")).strip() or "자재"
    original_label = label
    label = SYNONYM_MAP.get(label, label)
    if label != original_label:
        warnings.append(f"Normalized label '{original_label}' -> '{label}'")

    obj["label"] = label
    obj["count"] = max(1, int(obj.get("count") or 1))
    obj["bbox"] = normalize_bbox(obj.get("bbox"))
    obj["location"] = str(obj.get("location", "")).strip() or "이미지 내부"
    obj["confidence"] = clamp_float(obj.get("confidence"), 0.0, 1.0, 0.5)
    obj["evidence"] = (
        str(obj.get("evidence", "")).strip() or "이미지에서 해당 객체가 보인다."
    )

    attrs = obj.get("attributes")
    if not isinstance(attrs, dict):
        attrs = {}
    ppe = attrs.get("ppe") if isinstance(attrs.get("ppe"), list) else []
    state = attrs.get("state") if isinstance(attrs.get("state"), list) else []
    obj["attributes"] = {
        "ppe": [str(v) for v in ppe if str(v).strip()],
        "state": [str(v) for v in state if str(v).strip()],
    }
    if label != WORKER_LABEL:
        obj["attributes"]["ppe"] = []

    id_value = str(obj.get("id", "")).strip()
    if not id_value:
        id_value = f"{slug_prefix(label)}_{index}"
    if (
        obj["count"] > 1
        and label in (TIER2_LABELS | TIER3_LABELS)
        and "group_" not in id_value
    ):
        id_value = f"{slug_prefix(label)}_group_{index}"
    obj["id"] = re.sub(r"[^a-zA-Z0-9_]+", "_", id_value).strip("_") or f"object_{index}"

    if label in FORBIDDEN_LABELS:
        warnings.append(f"Forbidden generic label '{label}' on {obj['id']}")
    if label in TIER1_LABELS and obj["count"] > 1:
        warnings.append(
            f"Tier 1 object '{obj['id']}' should be individual; count reset to 1"
        )
        obj["count"] = 1
    return obj, warnings


def canonicalize_object_ids(
    objects: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    counters: dict[str, int] = {}
    id_map: dict[str, str] = {}
    canonical_objects = []

    for obj in objects:
        old_id = obj["id"]
        label = obj["label"]
        prefix = slug_prefix(label)
        is_group = obj.get("count", 1) > 1 and label in (TIER2_LABELS | TIER3_LABELS)
        counter_key = f"{prefix}_group" if is_group else prefix
        counters[counter_key] = counters.get(counter_key, 0) + 1
        new_id = (
            f"{prefix}_group_{counters[counter_key]}"
            if is_group
            else f"{prefix}_{counters[counter_key]}"
        )

        obj = deepcopy(obj)
        obj["id"] = new_id
        id_map[old_id] = new_id
        canonical_objects.append(obj)

    return canonical_objects, id_map


def normalize_object_result(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(data, dict):
        raise ValueError("Top-level output must be an object")
    if (
        not isinstance(data.get("scene_description"), str)
        or not data["scene_description"].strip()
    ):
        raise ValueError("scene_description must be a non-empty string")
    if not isinstance(data.get("objects"), list):
        raise ValueError("objects must be a list")

    warnings = []
    clean_objects = []
    seen_ids = set()
    for index, obj in enumerate(data["objects"], 1):
        if not isinstance(obj, dict):
            warnings.append("Skipped non-object item in objects")
            continue
        clean_obj, obj_warnings = normalize_object(obj, index)
        warnings.extend(obj_warnings)
        if clean_obj["label"] in FORBIDDEN_LABELS:
            warnings.append(
                f"Skipped object with forbidden generic label: {clean_obj['id']} "
                f"({clean_obj['label']})"
            )
            continue
        base_id = clean_obj["id"]
        suffix = 2
        while clean_obj["id"] in seen_ids:
            clean_obj["id"] = f"{base_id}_{suffix}"
            suffix += 1
        seen_ids.add(clean_obj["id"])
        clean_objects.append(clean_obj)

    clean_objects, id_map = canonicalize_object_ids(clean_objects)
    for old_id, new_id in id_map.items():
        if old_id != new_id:
            warnings.append(f"Canonicalized object id '{old_id}' -> '{new_id}'")

    return {
        "scene_description": data["scene_description"].strip(),
        "objects": clean_objects,
    }, warnings


def validate_scene_graph(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    object_result, warnings = normalize_object_result(
        {
            "scene_description": data.get("scene_description", ""),
            "objects": data.get("objects", []),
        }
    )
    valid_ids = {obj["id"] for obj in object_result["objects"]}
    labels_by_id = {obj["id"]: obj["label"] for obj in object_result["objects"]}

    rel_candidates = []
    for rel in data.get("relationships", []):
        if not isinstance(rel, dict):
            warnings.append("Skipped non-object relationship")
            continue
        sid = str(rel.get("sub_id", "")).strip()
        oid = str(rel.get("obj_id", "")).strip()
        pred = str(rel.get("predicate", "")).strip()
        cat = str(rel.get("category", "")).strip()
        if sid not in valid_ids or oid not in valid_ids:
            warnings.append(f"Skipped relationship with unknown id: {sid} -> {oid}")
            continue
        if sid == oid:
            warnings.append(f"Skipped self relationship: {sid}")
            continue
        if pred not in VALID_PREDICATES:
            warnings.append(
                f"Skipped invalid predicate '{pred}' between {sid} and {oid}"
            )
            continue
        if cat not in CATEGORY_PREDICATES or pred not in CATEGORY_PREDICATES[cat]:
            warnings.append(f"Skipped predicate/category mismatch: {pred}/{cat}")
            continue
        rel_candidates.append(
            {
                "sub_id": sid,
                "predicate": pred,
                "obj_id": oid,
                "category": cat,
                "score": clamp_float(rel.get("score"), 0.0, 1.0, 0.5),
                "evidence": str(rel.get("evidence", "")).strip(),
            }
        )

    best_rels_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for rel in rel_candidates:
        pair = (rel["sub_id"], rel["obj_id"])
        current = best_rels_by_pair.get(pair)
        if current is None:
            best_rels_by_pair[pair] = rel
            continue

        rel_rank = (
            RELATION_CATEGORY_PRIORITY.get(rel["category"], 0),
            rel["score"],
        )
        current_rank = (
            RELATION_CATEGORY_PRIORITY.get(current["category"], 0),
            current["score"],
        )
        if rel_rank > current_rank:
            warnings.append(
                f"Replaced duplicate relationship {pair[0]} -> {pair[1]} "
                f"with higher-priority {rel['category']}/{rel['predicate']}"
            )
            best_rels_by_pair[pair] = rel
        else:
            warnings.append(
                f"Skipped lower-priority duplicate relationship {pair[0]} -> {pair[1]} "
                f"{rel['category']}/{rel['predicate']}"
            )

    clean_rels = list(best_rels_by_pair.values())

    functional_pairs = {
        frozenset((rel["sub_id"], rel["obj_id"]))
        for rel in clean_rels
        if rel["category"] == "functional"
    }

    clean_hazards = []
    for haz in data.get("hazards", []):
        if not isinstance(haz, dict):
            warnings.append("Skipped non-object hazard")
            continue
        ref_ids = haz.get("related_object_ids", [])
        if not isinstance(ref_ids, list):
            warnings.append("Skipped hazard with invalid related_object_ids")
            continue
        ref_ids = [str(rid).strip() for rid in ref_ids if str(rid).strip()]
        if not ref_ids or any(rid not in valid_ids for rid in ref_ids):
            warnings.append(f"Skipped hazard with unknown ids: {ref_ids}")
            continue
        hazard = str(haz.get("hazard", "")).strip()
        if hazard not in VALID_HAZARDS:
            warnings.append(f"Skipped invalid hazard '{hazard}'")
            continue
        ref_labels = [labels_by_id.get(rid, "") for rid in ref_ids]
        reason = str(haz.get("reason", "")).strip()
        ref_label_set = set(ref_labels)
        if hazard in {"익수", "추락", "협착"} and WORKER_LABEL not in ref_label_set:
            warnings.append(
                f"Skipped unsupported {hazard} hazard without worker: {ref_ids}"
            )
            continue
        if hazard == "익수":
            if WATER_LABEL not in ref_label_set:
                warnings.append(
                    f"Skipped unsupported 익수 hazard without water: {ref_ids}"
                )
                continue
            has_water_edge = any(
                keyword in reason for keyword in ("가장자리", "물가", "수역", "개방")
            )
            has_missing_protection = any(
                keyword in reason for keyword in ("방호", "난간", "울타리", "안전")
            )
            if not (has_water_edge and has_missing_protection):
                warnings.append(
                    f"Skipped weak 익수 hazard without edge and protection evidence: {ref_ids}"
                )
                continue
        if hazard == "추락":
            if not (ref_label_set & SLOPE_CONTEXT_LABELS):
                warnings.append(
                    f"Skipped unsupported 추락 hazard without fall context: {ref_ids}"
                )
                continue
            has_fall_edge = any(
                keyword in reason
                for keyword in ("가장자리", "고저차", "굴착면", "높", "낙하")
            )
            has_missing_protection = any(
                keyword in reason for keyword in ("방호", "난간", "울타리", "안전망")
            )
            if not (has_fall_edge and has_missing_protection):
                warnings.append(
                    f"Skipped weak 추락 hazard without fall edge and protection evidence: {ref_ids}"
                )
                continue
        if hazard == "협착":
            if not (
                ref_label_set
                & (EQUIPMENT_LABELS | {"콘크리트구조물", "자재", "파이프"})
            ):
                warnings.append(
                    f"Skipped unsupported 협착 hazard without pinch source: {ref_ids}"
                )
                continue
            if not any(
                keyword in reason
                for keyword in ("사이", "끼", "협착", "좁", "장비", "고정")
            ):
                warnings.append(
                    f"Skipped weak 협착 hazard without explicit pinch evidence: {ref_ids}"
                )
                continue
        if hazard == "낙하물" and not any(
            keyword in reason for keyword in ("매달", "상부", "고소", "낙하", "하부")
        ):
            warnings.append(
                f"Skipped weak 낙하물 hazard without overhead evidence: {ref_ids}"
            )
            continue
        if hazard == "전도" and not any(
            keyword in reason for keyword in ("기울", "불안정", "경사", "전도", "넘어")
        ):
            warnings.append(
                f"Skipped weak 전도 hazard without instability evidence: {ref_ids}"
            )
            continue
        if hazard == "충돌":
            has_worker = WORKER_LABEL in ref_label_set
            has_functional_pair = any(
                frozenset(pair) in functional_pairs
                for pair in (
                    (ref_ids[i], ref_ids[j])
                    for i in range(len(ref_ids))
                    for j in range(i + 1, len(ref_ids))
                )
            )
            if has_functional_pair and not has_worker:
                warnings.append(
                    f"Skipped collision hazard for equipment collaboration: {ref_ids}"
                )
                continue
            has_safety_relation = any(
                rel["category"] == "safety"
                and rel["predicate"] in {"too_close_to", "approaching"}
                and rel["sub_id"] in ref_ids
                and rel["obj_id"] in ref_ids
                for rel in clean_rels
            )
            if not has_safety_relation:
                warnings.append(
                    f"Skipped weak collision hazard without safety relation: {ref_ids}"
                )
                continue
            if not any(
                keyword in reason
                for keyword in ("가까", "접근", "경로", "이동", "작동", "충돌")
            ):
                warnings.append(
                    f"Skipped weak collision hazard without proximity/motion evidence: {ref_ids}"
                )
                continue
        clean_hazards.append(
            {
                "related_object_ids": ref_ids,
                "hazard": hazard,
                "reason": reason,
            }
        )

    return {
        "scene_description": object_result["scene_description"],
        "objects": object_result["objects"],
        "relationships": clean_rels,
        "hazards": clean_hazards,
    }, warnings


def to_legacy_output_format(graph: dict[str, Any]) -> dict[str, Any]:
    legacy_objects = []
    for obj in graph.get("objects", []):
        attrs = obj.get("attributes") if isinstance(obj.get("attributes"), dict) else {}
        state = [str(v) for v in attrs.get("state", []) if str(v).strip()]
        legacy_attrs = {"state": state}
        if obj.get("label") == WORKER_LABEL:
            ppe = [str(v) for v in attrs.get("ppe", []) if str(v).strip()]
            legacy_attrs = {"ppe": ppe, "state": state}

        legacy_objects.append(
            {
                "id": obj.get("id", ""),
                "label": obj.get("label", ""),
                "attributes": legacy_attrs,
                "location": obj.get("location", ""),
            }
        )

    return {
        "scene_description": graph.get("scene_description", ""),
        "objects": legacy_objects,
        "relationships": graph.get("relationships", []),
        "hazards": graph.get("hazards", []),
    }


def extract_json_from_response(resp: Any) -> str:
    output_text = getattr(resp, "output_text", None)
    if output_text:
        return output_text
    if getattr(resp, "choices", None):
        return resp.choices[0].message.content
    if getattr(resp, "output", None):
        chunks = []
        for item in resp.output:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)
        if chunks:
            return "".join(chunks)
    raise ValueError("Could not extract text from API response")


def call_json_api(
    client: openai.OpenAI,
    system_prompt: str,
    user_text: str,
    data_uri: str,
    schema_name: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": schema,
        },
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            try:
                resp = client.responses.create(
                    model=MODEL,
                    input=[
                        {
                            "role": "system",
                            "content": [{"type": "input_text", "text": system_prompt}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": user_text},
                                {
                                    "type": "input_image",
                                    "image_url": data_uri,
                                    "detail": "high",
                                },
                            ],
                        },
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": schema_name,
                            "strict": True,
                            "schema": schema,
                        }
                    },
                    temperature=TEMPERATURE,
                )
            except (AttributeError, TypeError):
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_text},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": data_uri, "detail": "high"},
                                },
                            ],
                        },
                    ],
                    response_format=response_format,
                    temperature=TEMPERATURE,
                )
            return json.loads(extract_json_from_response(resp))
        except (openai.RateLimitError, openai.APITimeoutError) as exc:
            print(
                f"  API retryable error ({attempt}/{MAX_RETRIES}): {exc}",
                file=sys.stderr,
            )
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY * attempt)
        except (json.JSONDecodeError, ValueError) as exc:
            print(
                f"  Invalid model output ({attempt}/{MAX_RETRIES}): {exc}",
                file=sys.stderr,
            )
            if attempt == MAX_RETRIES:
                raise

    raise RuntimeError("unreachable")


def call_scene_graph_api(
    client: openai.OpenAI, data_uri: str
) -> tuple[dict[str, Any], list[str]]:
    raw_objects = call_json_api(
        client=client,
        system_prompt=OBJECT_SYSTEM_PROMPT,
        user_text=OBJECT_USER_PROMPT,
        data_uri=data_uri,
        schema_name="construction_objects",
        schema=OBJECT_SCHEMA,
    )
    object_result, warnings = normalize_object_result(raw_objects)

    relation_user_text = (
        RELATION_USER_PROMPT
        + "\n\n입력 objects JSON:\n"
        + json.dumps(object_result, ensure_ascii=False, indent=2)
    )
    raw_relations = call_json_api(
        client=client,
        system_prompt=RELATION_SYSTEM_PROMPT,
        user_text=relation_user_text,
        data_uri=data_uri,
        schema_name="construction_scene_relations",
        schema=RELATION_RESULT_SCHEMA,
    )

    raw_graph = {
        "scene_description": object_result["scene_description"],
        "objects": deepcopy(object_result["objects"]),
        "relationships": raw_relations.get("relationships", []),
        "hazards": raw_relations.get("hazards", []),
    }

    graph, graph_warnings = validate_scene_graph(raw_graph)
    warnings.extend(graph_warnings)
    return graph, warnings


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_image(
    client: openai.OpenAI, img_path: str, index: int, total: int
) -> tuple[bool, list[str], str]:
    name = os.path.basename(img_path)
    out_paths = output_paths_for_image(img_path)

    if all(os.path.exists(out_path) for out_path in out_paths):
        return True, [], f"[{index}/{total}] {name} ... already exists (skipped)"

    data_uri = image_to_data_uri(img_path)
    messages = []
    warnings_all = []

    for candidate_idx, out_path in enumerate(out_paths, 1):
        result, warnings = call_scene_graph_api(client, data_uri)
        warnings_all.extend(f"c{candidate_idx}: {w}" for w in warnings)
        output_result = to_legacy_output_format(result)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_result, f, ensure_ascii=False, indent=2)

        obj_count = len(output_result["objects"])
        group_count = sum(1 for obj in result["objects"] if obj.get("count", 1) > 1)
        rel_count = len(output_result["relationships"])
        haz_count = len(output_result["hazards"])
        messages.append(
            f"c{candidate_idx}: objects={obj_count} groups={group_count} "
            f"rels={rel_count} hazards={haz_count}"
        )

    return (
        True,
        warnings_all,
        f"[{index}/{total}] {name} ... OK " + " | ".join(messages),
    )


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = find_images()
    if MAX_IMAGES is not None:
        images = images[:MAX_IMAGES]

    if not images:
        print(f"ERROR: No images in {IMAGE_DIR}/", file=sys.stderr)
        sys.exit(1)

    print(
        f"Processing {len(images)} images with {MODEL} "
        f"({CANDIDATES_PER_IMAGE} candidates/image, {MAX_WORKERS} workers)...\n"
    )

    success = 0
    fail = 0
    all_warnings = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_image, client, img_path, i, len(images)): img_path
            for i, img_path in enumerate(images, 1)
        }

        for future in as_completed(futures):
            img_path = futures[future]
            try:
                ok, warnings, message = future.result()
                print(message)
                for warning in warnings:
                    print(f"    WARN: {warning}", file=sys.stderr)
                all_warnings.extend(warnings)
                success += int(ok)
                fail += int(not ok)
            except Exception as exc:
                print(f"{os.path.basename(img_path)} ... FAIL {exc}", file=sys.stderr)
                fail += 1

    if all_warnings:
        report_path = os.path.join(OUTPUT_DIR, "_warnings.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(all_warnings, f, ensure_ascii=False, indent=2)
        print(f"\nSaved warnings to {report_path}")

    print(f"\nDone. success={success} fail={fail}")


if __name__ == "__main__":
    main()
