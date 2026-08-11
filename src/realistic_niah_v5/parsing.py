from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from realistic_niah.parsing import parse_total, split_reasoning_and_final
from realistic_niah_v3.city_list_termination import (
    CityListTerminationCut,
    find_first_terminated_gold_city_list,
)
from realistic_niah_v3.first_list_cutoff import (
    align_text_exact_token_prefix,
    exact_token_prefix_length,
)


PARSER_UPSTREAM_REPOSITORY = "https://github.com/TheWayLost/niah-parser"
PARSER_UPSTREAM_COMMIT = "8ebf6b7af4770d8c91e6540d474505e23ad57c8c"
PARSER_IMPLEMENTATION = "realistic_niah_v3.find_first_terminated_gold_city_list"
PARSER_FILE_SHA256 = {
    "city_list_termination.py": "bb2cd01275a4dbfd339a388a3a830c4c2aa762ec14ad89ca04fd98bbc1b64728",
    "first_list_cutoff.py": "72781f9060d21fd6c693da4c0b0c0ad58831a031d37bc50fed21ee860ded66b7",
    "gold_city_cutoff.py": "bc5c37f410b96008023724f3f88895f82fdd39d9a0a05163427f1d3e017c03a9",
}
PARSER_SCHEMA_VERSION = "realistic_niah_v5_oracle_trace_v1"
SITE_SCHEMA_VERSION = "realistic_niah_v5_trace_sites_v1"

_TOTAL_RE = re.compile(r"(?im)^\s*Total\s*:")
_TOTAL_ANYWHERE_RE = re.compile(r"(?i)(?<!\w)Total\s*:")
_INTEGER_AFTER_TOTAL_RE = re.compile(r"[ \t\r\n]*(?P<answer>[+-]?\d+)")
_INDEXED_MARKER_RE = re.compile(r"^[ \t]*\d+[.)][ \t]*")
_BULLET_MARKER_RE = re.compile(r"^[ \t]*(?:[-\u2022]|\*(?!\*))[ \t]*")
_ORDINAL_MARKER_RE = re.compile(
    r"^[ \t]*(?:\*{1,2}|`)?(?:first|second|third|fourth|fifth|sixth|"
    r"seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|"
    r"fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|"
    r"then|finally)(?:\*{1,2}|`)?(?:[.):,-](?:\*{1,2}|`)?[ \t]*|[ \t]+)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class TraceCharSite:
    site_id: str
    site_kind: str
    occurrence: int | None
    city: str | None
    marker: int | str | None
    boundary_kind: str | None
    char_start: int
    char_end: int
    primary: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraceTokenSite:
    char_site: TraceCharSite
    alignment_eligible: bool
    alignment_status: str
    alignment_strategy: str | None
    prefix_token_count: int | None
    shared_baseline_prefix_tokens: int
    retokenized_suffix_tokens: int
    literal_token_start: int | None
    literal_token_end: int | None
    prefix_token_ids_sha256: str | None
    prefix_token_ids: tuple[int, ...] = ()

    @property
    def endpoint_token(self) -> int | None:
        if self.prefix_token_count is None or self.prefix_token_count < 1:
            return None
        return self.prefix_token_count - 1

    def to_dict(self, *, include_token_ids: bool = False) -> dict[str, Any]:
        payload = {
            **self.char_site.to_dict(),
            "alignment_eligible": self.alignment_eligible,
            "alignment_status": self.alignment_status,
            "alignment_strategy": self.alignment_strategy,
            "prefix_token_count": self.prefix_token_count,
            "endpoint_token": self.endpoint_token,
            "shared_baseline_prefix_tokens": self.shared_baseline_prefix_tokens,
            "retokenized_suffix_tokens": self.retokenized_suffix_tokens,
            "literal_token_start": self.literal_token_start,
            "literal_token_end": self.literal_token_end,
            "prefix_token_ids_sha256": self.prefix_token_ids_sha256,
        }
        if include_token_ids:
            payload["prefix_token_ids"] = list(self.prefix_token_ids)
        return payload


def infer_model_family(row: Mapping[str, Any], override: str | None = None) -> str:
    if override:
        family = str(override).lower()
    elif row.get("model_family"):
        family = str(row["model_family"]).lower()
    else:
        label = " ".join(
            str(row.get(key, ""))
            for key in ("model_label", "model", "model_id")
        ).lower()
        if "qwen3" in label:
            family = "qwen3"
        elif "gemma" in label:
            family = "gemma4"
        else:
            raise ValueError("Cannot infer qwen3/gemma4 model family")
    aliases = {"qwen": "qwen3", "gemma": "gemma4", "gemma-4": "gemma4"}
    family = aliases.get(family, family)
    if family not in {"qwen3", "gemma4"}:
        raise ValueError(f"Unsupported V5 parser family: {family}")
    return family


def raw_output_text(row: Mapping[str, Any]) -> str:
    for key in ("raw_output_text", "completion_text_raw", "output_text"):
        if row.get(key) is not None:
            return str(row[key])
    baseline = row.get("baseline")
    if isinstance(baseline, Mapping) and baseline.get("raw_output_text") is not None:
        return str(baseline["raw_output_text"])
    raise ValueError("Record has no raw native-thinking output text")


def output_token_ids(row: Mapping[str, Any]) -> tuple[int, ...]:
    for key in ("output_token_ids", "generated_token_ids"):
        value = row.get(key)
        if value is not None:
            return tuple(int(token) for token in value)
    raise ValueError("Record has no output/generated token IDs")


def prompt_token_ids(row: Mapping[str, Any]) -> tuple[int, ...]:
    for key in ("input_ids", "prompt_token_ids"):
        value = row.get(key)
        if value is not None:
            return tuple(int(token) for token in value)
    raise ValueError("Record has no prompt input IDs")


def gold_records(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("gold_records", "gold_pairs", "relevant_records"):
        value = row.get(key)
        if value is None:
            continue
        records: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                if "city" not in item:
                    raise ValueError(f"Gold record lacks city: {item}")
                records.append(dict(item))
            else:
                records.append({"city": str(item)})
        return records
    raise ValueError("Record has no oracle city registry")


def _marker_end(text: str, start: int, end: int, marker_kind: str | None) -> int | None:
    item = text[start:end]
    patterns: tuple[re.Pattern[str], ...]
    if marker_kind == "indexed":
        patterns = (_INDEXED_MARKER_RE,)
    elif marker_kind == "bullet":
        patterns = (_BULLET_MARKER_RE,)
    elif marker_kind == "ordinal":
        patterns = (_ORDINAL_MARKER_RE,)
    else:
        patterns = (_INDEXED_MARKER_RE, _BULLET_MARKER_RE, _ORDINAL_MARKER_RE)
    for pattern in patterns:
        match = pattern.match(item)
        if match and match.end() > 0:
            return start + match.end()
    return None


def _city_span(text: str, start: int, end: int, city: str) -> tuple[int, int] | None:
    item = text[start:end]
    pattern = re.compile(r"(?<!\w)" + re.escape(city) + r"(?!\w)", re.IGNORECASE)
    matches = list(pattern.finditer(item))
    if not matches:
        return None
    match = matches[-1]
    return start + match.start(), start + match.end()


def _post_boundary_end(text: str, item_end: int) -> int:
    end = int(item_end)
    if text.startswith("\r\n", end):
        return end + 2
    if end < len(text) and text[end] in "\r\n":
        return end + 1
    return end


def _answer_query_span(text: str, reasoning_end: int | None) -> tuple[int, int] | None:
    start = int(reasoning_end or 0)
    matches = list(_TOTAL_RE.finditer(text, pos=start))
    if not matches:
        matches = list(_TOTAL_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    colon = text.find(":", match.start(), match.end())
    if colon < 0:
        return None
    return match.start(), colon + 1


def _answer_query_v2_span(
    text: str, reasoning_end: int | None
) -> tuple[int, int] | None:
    """Locate the literal ``Total:`` prefix before the numeric answer.

    Gemma native-thinking completions can place a decoded channel-control token
    immediately before ``Total:``, so the legacy line-anchored expression does
    not register an answer-query site even though the baseline token sequence
    contains a valid answer prefix.  V2 deliberately relaxes only that anchor;
    it still selects the last ``Total:`` after the reasoning boundary and ends
    at the colon.  The later token-alignment audit remains authoritative.
    """

    start = int(reasoning_end or 0)
    matches = list(_TOTAL_ANYWHERE_RE.finditer(text, pos=start))
    if not matches:
        matches = list(_TOTAL_ANYWHERE_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    colon = text.find(":", match.start(), match.end())
    if colon < 0:
        return None
    return match.start(), colon + 1


def _answer_query_v3_span(
    text: str, reasoning_end: int | None
) -> tuple[int, int] | None:
    """End at the literal baseline token immediately before the answer digit.

    V2 ends at the colon in ``Total: 3``. Both registered tokenizers encode
    the following space as its own baseline token, so V3 includes the native
    whitespace separator and ends immediately before the first integer.
    Token alignment remains authoritative for every individual row.
    """

    start = int(reasoning_end or 0)
    matches = list(_TOTAL_ANYWHERE_RE.finditer(text, pos=start))
    if not matches:
        matches = list(_TOTAL_ANYWHERE_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    colon = text.find(":", match.start(), match.end())
    if colon < 0:
        return None
    answer = _INTEGER_AFTER_TOTAL_RE.match(text, pos=colon + 1)
    if answer is None:
        return None
    answer_start = int(answer.start("answer"))
    if answer_start <= colon:
        return None
    return match.start(), answer_start


def trace_char_sites(raw_text: str, parser: CityListTerminationCut) -> list[TraceCharSite]:
    if not parser.detected:
        return []
    lengths = {
        len(parser.item_markers),
        len(parser.item_gold_cities),
        len(parser.item_start_chars),
        len(parser.item_end_chars),
        len(parser.item_boundary_kinds),
    }
    if len(lengths) != 1:
        raise RuntimeError("Parser item arrays are not index-aligned")
    sites: list[TraceCharSite] = []
    for offset, (marker, city, start, end, boundary) in enumerate(
        zip(
            parser.item_markers,
            parser.item_gold_cities,
            parser.item_start_chars,
            parser.item_end_chars,
            parser.item_boundary_kinds,
        ),
        start=1,
    ):
        if not 0 <= start < end <= len(raw_text):
            raise RuntimeError(f"Parser item span is invalid: [{start}, {end})")
        common = {
            "occurrence": offset,
            "city": str(city),
            "marker": marker,
            "boundary_kind": str(boundary),
        }
        marker_end = _marker_end(raw_text, start, end, parser.marker_kind)
        if marker_end is not None:
            sites.append(
                TraceCharSite(
                    site_id=f"marker_end:{offset}",
                    site_kind="marker_end",
                    char_start=start,
                    char_end=marker_end,
                    primary=False,
                    **common,
                )
            )
        city_span = _city_span(raw_text, start, end, str(city))
        if city_span is not None:
            sites.append(
                TraceCharSite(
                    site_id=f"city_end:{offset}",
                    site_kind="city_end",
                    char_start=city_span[0],
                    char_end=city_span[1],
                    primary=False,
                    **common,
                )
            )
        sites.append(
            TraceCharSite(
                site_id=f"item_end:{offset}",
                site_kind="item_end",
                char_start=start,
                char_end=end,
                primary=True,
                **common,
            )
        )
        post_end = _post_boundary_end(raw_text, end)
        sites.append(
            TraceCharSite(
                site_id=f"post_boundary:{offset}",
                site_kind="post_boundary",
                char_start=start,
                char_end=post_end,
                primary=False,
                **common,
            )
        )
    if parser.cut_char is not None and parser.list_start_char is not None:
        sites.append(
            TraceCharSite(
                site_id="list_cut",
                site_kind="list_cut",
                occurrence=parser.item_count,
                city=(parser.item_gold_cities[-1] if parser.item_gold_cities else None),
                marker=(parser.item_markers[-1] if parser.item_markers else None),
                boundary_kind=parser.boundary_kind,
                char_start=int(parser.list_start_char),
                char_end=int(parser.cut_char),
                primary=False,
            )
        )
    answer = _answer_query_span(raw_text, parser.reasoning_end_char)
    if answer is not None:
        sites.append(
            TraceCharSite(
                site_id="answer_query",
                site_kind="answer_query",
                occurrence=None,
                city=None,
                marker=None,
                boundary_kind="total_colon",
                char_start=answer[0],
                char_end=answer[1],
                primary=False,
            )
        )
    answer_v2 = _answer_query_v2_span(raw_text, parser.reasoning_end_char)
    if answer_v2 is not None:
        sites.append(
            TraceCharSite(
                site_id="answer_query_v2",
                site_kind="answer_query_v2",
                occurrence=None,
                city=None,
                marker=None,
                boundary_kind="total_colon_relaxed_anchor_v2",
                char_start=answer_v2[0],
                char_end=answer_v2[1],
                primary=False,
            )
        )
    answer_v3 = _answer_query_v3_span(raw_text, parser.reasoning_end_char)
    if answer_v3 is not None:
        sites.append(
            TraceCharSite(
                site_id="answer_query_v3",
                site_kind="answer_query_v3",
                occurrence=None,
                city=None,
                marker=None,
                boundary_kind="literal_token_before_numeric_answer_v3",
                char_start=answer_v3[0],
                char_end=answer_v3[1],
                primary=False,
            )
        )
    return sites


def parse_trace_record(
    row: Mapping[str, Any], *, model_family: str | None = None
) -> dict[str, Any]:
    family = infer_model_family(row, model_family)
    raw = raw_output_text(row)
    gold = gold_records(row)
    parser = find_first_terminated_gold_city_list(
        raw,
        model_family=family,
        gold_records=gold,
    )
    reasoning, final = split_reasoning_and_final(
        raw,
        prompt_mode="native_thinking",
        reasoning_expected=True,
    )
    parsed_count = parse_total(final)
    exact_count = parsed_count == len(gold) if parsed_count is not None else False
    return {
        "schema_version": PARSER_SCHEMA_VERSION,
        "parser_upstream_repository": PARSER_UPSTREAM_REPOSITORY,
        "parser_upstream_commit": PARSER_UPSTREAM_COMMIT,
        "parser_implementation": PARSER_IMPLEMENTATION,
        "parser_file_sha256": dict(PARSER_FILE_SHA256),
        "request_id": row.get("request_id", row.get("stimulus_id")),
        "stimulus_id": row.get("stimulus_id"),
        "model_label": row.get("model_label", row.get("model")),
        "model_family": family,
        "seed": row.get("seed"),
        "split": row.get("split"),
        "gold_count": len(gold),
        "parsed_count": parsed_count,
        "exact_count": exact_count,
        "reasoning_text": reasoning,
        "final_text": final,
        "parser": parser.to_dict(),
        "char_sites": [site.to_dict() for site in trace_char_sites(raw, parser)],
    }


def _token_ids_sha256(values: Sequence[int]) -> str:
    payload = ",".join(str(int(value)) for value in values).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def align_trace_sites(
    tokenizer: Any,
    *,
    raw_text: str,
    baseline_output_token_ids: Iterable[int],
    sites: Iterable[TraceCharSite],
) -> list[TraceTokenSite]:
    baseline = tuple(int(value) for value in baseline_output_token_ids)
    aligned: list[TraceTokenSite] = []
    for site in sites:
        start = exact_token_prefix_length(
            tokenizer,
            raw_text=raw_text,
            output_token_ids=baseline,
            cut_char=site.char_start,
        )
        end = exact_token_prefix_length(
            tokenizer,
            raw_text=raw_text,
            output_token_ids=baseline,
            cut_char=site.char_end,
        )
        prefix = align_text_exact_token_prefix(
            tokenizer,
            raw_text=raw_text,
            output_token_ids=baseline,
            cut_char=site.char_end,
        )
        token_ids = tuple(prefix.token_ids) if prefix.eligible else ()
        aligned.append(
            TraceTokenSite(
                char_site=site,
                alignment_eligible=bool(prefix.eligible),
                alignment_status=str(prefix.status),
                alignment_strategy=prefix.strategy,
                prefix_token_count=(len(token_ids) if prefix.eligible else None),
                shared_baseline_prefix_tokens=int(prefix.shared_baseline_prefix_tokens),
                retokenized_suffix_tokens=int(prefix.retokenized_suffix_tokens),
                literal_token_start=start,
                literal_token_end=end,
                prefix_token_ids_sha256=(
                    _token_ids_sha256(token_ids) if token_ids else None
                ),
                prefix_token_ids=token_ids,
            )
        )
    return aligned


def parse_and_align_record(
    row: Mapping[str, Any], tokenizer: Any, *, model_family: str | None = None
) -> dict[str, Any]:
    parsed = parse_trace_record(row, model_family=model_family)
    parser = CityListTerminationCut(**parsed["parser"])
    sites = trace_char_sites(raw_output_text(row), parser)
    aligned = align_trace_sites(
        tokenizer,
        raw_text=raw_output_text(row),
        baseline_output_token_ids=output_token_ids(row),
        sites=sites,
    )
    parsed["site_schema_version"] = SITE_SCHEMA_VERSION
    parsed["token_sites"] = [site.to_dict() for site in aligned]
    parsed["alignment_summary"] = {
        "sites": len(aligned),
        "eligible": sum(site.alignment_eligible for site in aligned),
        "literal_baseline": sum(
            site.alignment_strategy == "literal_baseline_token_prefix"
            for site in aligned
        ),
        "retokenized": sum(
            site.alignment_strategy == "text_exact_boundary_retokenization"
            for site in aligned
        ),
    }
    return parsed
