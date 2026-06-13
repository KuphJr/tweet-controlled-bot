#!/usr/bin/env python
"""Read-only TwitterApi.io inspector for one tweet/reply ID.

This is intentionally separate from the live robot process. It does not write
state, post to X, open cameras, or touch the robot. It only reads TwitterApi.io
and prints enough metadata to explain whether the live reader would see/skip an
item.

Example:

    python scripts/tweet_robot/inspect_twitterapi_item.py \
        --source-tweet-id 2065449958984085509 \
        --target-tweet-id 2065455447033696643
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://api.twitterapi.io"
ENV_PATH = Path(__file__).resolve().parent / ".env"
DEFAULT_STATE_FILE = Path(__file__).resolve().parents[2] / "runtime" / "tweet_robot_state.json"


def _load_api_key() -> str:
    if os.getenv("TWITTERAPI_IO_API_KEY"):
        return os.environ["TWITTERAPI_IO_API_KEY"]
    if not ENV_PATH.exists():
        return ""
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("TWITTERAPI_IO_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _tweet_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("tweets", "replies", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and isinstance(value.get("tweets"), list):
            return value["tweets"]
    return []


def _parent_id(tweet: dict[str, Any]) -> str | None:
    for key in ("inReplyToId", "inReplyToTweetId", "inReplyToStatusId", "in_reply_to_status_id"):
        value = tweet.get(key)
        if value:
            return str(value)
    return None


def _summarize(tweet: dict[str, Any]) -> dict[str, Any]:
    author = tweet.get("author") or {}
    return {
        "id": str(tweet.get("id")),
        "text": tweet.get("text"),
        "createdAt": tweet.get("createdAt"),
        "isReply": tweet.get("isReply"),
        "inReplyToId": _parent_id(tweet),
        "conversationId": tweet.get("conversationId"),
        "inReplyToUsername": tweet.get("inReplyToUsername"),
        "author_userName": author.get("userName"),
        "author_name": author.get("name"),
        "author_followers": author.get("followers"),
        "author_isBlueVerified": author.get("isBlueVerified"),
    }


def _get(path: str, params: dict[str, Any], api_key: str) -> dict[str, Any] | None:
    resp = requests.get(
        BASE_URL + path,
        headers={"X-API-Key": api_key},
        params=params,
        timeout=20,
    )
    print(f"GET {path} params={params} -> {resp.status_code}")
    if not resp.ok:
        print(resp.text[:800])
        return None
    return resp.json()


def _scan_endpoint(
    *,
    path: str,
    source_tweet_id: str,
    target_tweet_id: str,
    api_key: str,
    max_pages: int,
) -> dict[str, Any] | None:
    cursor = ""
    total = 0
    for page in range(1, max_pages + 1):
        params: dict[str, Any] = {"tweetId": source_tweet_id}
        if cursor:
            params["cursor"] = cursor
        payload = _get(path, params, api_key)
        if payload is None:
            return None

        rows = _tweet_list(payload)
        print(
            f"  page={page} rows={len(rows)} has_next_page={payload.get('has_next_page')} "
            f"next_cursor={payload.get('next_cursor') or payload.get('nextCursor')}"
        )
        for raw in rows:
            total += 1
            if str(raw.get("id")) == target_tweet_id:
                print(f"  found target after scanning {total} row(s)")
                return raw

        cursor = payload.get("next_cursor") or payload.get("nextCursor") or ""
        if not payload.get("has_next_page") or not cursor:
            break
    print(f"  target not found after scanning {total} row(s)")
    return None


def _try_direct_lookup(target_tweet_id: str, api_key: str) -> dict[str, Any] | None:
    checks = [
        ("/twitter/tweet/by_ids", {"tweet_ids": target_tweet_id}),
        ("/twitter/tweet/by_ids", {"tweetIds": target_tweet_id}),
        ("/twitter/tweets", {"tweet_ids": target_tweet_id}),
        ("/twitter/tweet/detail", {"tweetId": target_tweet_id}),
        ("/twitter/tweet/thread_context", {"tweetId": target_tweet_id}),
    ]
    for path, params in checks:
        payload = _get(path, params, api_key)
        if payload is None:
            continue
        stack: list[Any] = [payload]
        seen: set[int] = set()
        while stack:
            obj = stack.pop()
            obj_id = id(obj)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            if isinstance(obj, dict):
                if str(obj.get("id")) == target_tweet_id:
                    return obj
                stack.extend(obj.values())
            elif isinstance(obj, list):
                stack.extend(obj)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tweet-id", required=True)
    parser.add_argument("--target-tweet-id", required=True)
    parser.add_argument("--max-pages", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = _load_api_key()
    if not api_key:
        raise SystemExit("TWITTERAPI_IO_API_KEY not found in env or scripts/tweet_robot/.env")

    print("Read-only inspection. No state/post/robot actions will be performed.")
    print(f"source={args.source_tweet_id} target={args.target_tweet_id}")

    target = _scan_endpoint(
        path="/twitter/tweet/replies",
        source_tweet_id=args.source_tweet_id,
        target_tweet_id=args.target_tweet_id,
        api_key=api_key,
        max_pages=args.max_pages,
    )
    if target is None:
        print("\nNot found in replies pages; trying direct lookup endpoints...")
        target = _try_direct_lookup(args.target_tweet_id, api_key)

    if target is None:
        print("\nTarget was not found by replies scan or direct lookup.")
        return 1

    summary = _summarize(target)
    print("\nTarget summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    parent = summary["inReplyToId"]
    conversation_id = summary["conversationId"]
    in_source_conversation = conversation_id == args.source_tweet_id
    would_accept_v2 = bool(parent and in_source_conversation)
    print("\nCurrent reader decision hints:")
    print(f"  direct_reply_to_source: {parent == args.source_tweet_id}")
    print(f"  in_source_conversation: {in_source_conversation}")
    print(f"  would_accept_from_replies_v2: {would_accept_v2}")
    print("  note: seen/processed checks are handled by the live state file, not this script")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
