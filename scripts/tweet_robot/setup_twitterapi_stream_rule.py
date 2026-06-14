#!/usr/bin/env python
"""Create/update the TwitterApi.io WebSocket filter rule for one stream tweet.

Run this once per active source tweet before starting the robot. It configures a
source-conversation rule like ``conversation_id:<source_tweet_id>`` and activates
it. This script does not touch robot state, cameras, policies, or X posting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import requests

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from config import AppConfig  # noqa: E402

_BASE_URL = "https://api.twitterapi.io"
_GET_RULES_PATH = "/oapi/tweet_filter/get_rules"
_ADD_RULE_PATH = "/oapi/tweet_filter/add_rule"
_UPDATE_RULE_PATH = "/oapi/tweet_filter/update_rule"


def _rule_tag(cfg: AppConfig, source_tweet_id: str) -> str:
    prefix = cfg.twitterapi_stream_rule_tag_prefix.strip() or "tweet_robot"
    return f"{prefix}_{source_tweet_id}"


def _rule_value(source_tweet_id: str) -> str:
    return f"conversation_id:{source_tweet_id}"


def _post_json(path: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(
        _BASE_URL + path,
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _get_rules(api_key: str) -> list[dict[str, Any]]:
    resp = requests.get(_BASE_URL + _GET_RULES_PATH, headers={"X-API-Key": api_key}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    return list(payload.get("rules") or [])


def _find_rule(rules: list[dict[str, Any]], *, tag: str, value: str) -> dict[str, Any] | None:
    for rule in rules:
        if rule.get("tag") == tag:
            return rule
    for rule in rules:
        if rule.get("value") == value:
            return rule
    return None


def ensure_rule(cfg: AppConfig, source_tweet_id: str) -> dict[str, Any]:
    if not cfg.twitterapi_io_api_key:
        raise RuntimeError("TWITTERAPI_IO_API_KEY is required.")

    tag = _rule_tag(cfg, source_tweet_id)
    value = _rule_value(source_tweet_id)
    interval_s = max(0.1, float(cfg.twitterapi_stream_rule_interval_s))

    rules = _get_rules(cfg.twitterapi_io_api_key)
    existing = _find_rule(rules, tag=tag, value=value)
    if existing is None:
        add_payload = {"tag": tag, "value": value, "interval_seconds": interval_s}
        added = _post_json(_ADD_RULE_PATH, cfg.twitterapi_io_api_key, add_payload)
        rule_id = str(added.get("rule_id") or "")
        if not rule_id:
            raise RuntimeError(f"TwitterApi.io did not return rule_id: {added}")
        print(f"Created rule: id={rule_id} tag={tag} value={value}")
    else:
        rule_id = str(existing.get("rule_id") or "")
        if not rule_id:
            raise RuntimeError(f"Existing rule is missing rule_id: {existing}")
        print(f"Found existing rule: id={rule_id} tag={existing.get('tag')} value={existing.get('value')}")

    update_payload = {
        "rule_id": rule_id,
        "tag": tag,
        "value": value,
        "interval_seconds": interval_s,
        "is_effect": 1,
    }
    updated = _post_json(_UPDATE_RULE_PATH, cfg.twitterapi_io_api_key, update_payload)
    if updated.get("status") not in (None, "success"):
        raise RuntimeError(f"Rule activation did not report success: {updated}")

    return {
        "rule_id": rule_id,
        "tag": tag,
        "value": value,
        "interval_seconds": interval_s,
        "is_effect": 1,
        "update_response": updated,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tweet-id", required=True, help="Source tweet conversation ID to stream.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = AppConfig.from_env()
    rule = ensure_rule(cfg, str(args.source_tweet_id))
    print("Active TwitterApi.io stream rule:")
    for key in ("rule_id", "tag", "value", "interval_seconds", "is_effect"):
        print(f"  {key}: {rule[key]}")
    print("You can now start the tweet robot; the WebSocket should receive matching events.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
