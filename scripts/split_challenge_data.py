#!/usr/bin/env python3
"""Split prompts/challenge-data.json into markdown files under data/."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("data/challenge-data.json")
DEFAULT_OUTPUT_DIR = Path("data/extracted_data")


def slugify(value: str) -> str:
    """Return a filesystem-friendly slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "untitled"


def yaml_list(items: list[str]) -> str:
    return "\n".join(f"  - {json.dumps(item)}" for item in items)


def create_preamble(challenge: dict[str, Any], skills: list[str]) -> str:
    return (
        "---\n"
        f"id: {challenge['id']}\n"
        f"title: {json.dumps(challenge['title'])}\n"
        f"slug: {challenge['slug']}\n"
        f"url: {challenge['url']}\n"
        f"difficulty: {challenge['difficulty']}\n"
        "skills:\n"
        f"{yaml_list(skills)}\n"
        "---\n\n"
    )


def challenge_to_markdown(challenge: dict[str, Any]) -> str:
    skills = challenge.get("skills", [])
    skill_text = ", ".join(skills)
    preamble = create_preamble(challenge, skills)
    return (
        f"{preamble}"
        f"# {challenge['title']}\n\n"
        f"**Difficulty:** {challenge['difficulty']}\n\n"
        f"**URL:** {challenge['url']}\n\n"
        f"**Skills:** {skill_text}\n\n"
        "## Description\n\n"
        f"{challenge['description']}\n"
    )


def getting_started_to_markdown(data: dict[str, Any]) -> str:
    lines = ["# Getting Started", ""]
    lines.append("## Steps")
    lines.extend(
        f"{index}. {step}" for index, step in enumerate(data.get("steps", []), 1)
    )
    lines.append("")

    recommended = data.get("recommended_first_challenges", [])
    if recommended:
        lines.append("## Recommended First Challenges")
        lines.extend(f"- {challenge}" for challenge in recommended)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def troubleshooting_to_markdown(data: dict[str, Any]) -> str:
    lines = ["# Troubleshooting", ""]
    if data.get("general_advice"):
        lines.extend([data["general_advice"], ""])

    lines.append("## Steps")
    lines.extend(
        f"{index}. {step}" for index, step in enumerate(data.get("steps", []), 1)
    )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def faq_to_markdown(items: list[dict[str, str]]) -> str:
    lines = ["# FAQ", ""]
    for item in items:
        lines.extend([f"## {item['question']}", "", item["answer"], ""])
    return "\n".join(lines).rstrip() + "\n"


def index_to_markdown(challenges: list[dict[str, Any]]) -> str:
    lines = ["# Coding Challenges", ""]
    for challenge in challenges:
        filename = challenge_filename(challenge)
        lines.append(
            f"- [{challenge['title']}]({filename}) "
            f"({challenge['difficulty']}) — {challenge['description']}"
        )
    lines.append("")
    return "\n".join(lines)


def challenge_filename(challenge: dict[str, Any]) -> str:
    challenge_id = int(challenge["id"])
    slug = slugify(str(challenge.get("slug") or challenge.get("title") or challenge_id))
    return f"{challenge_id:03d}-{slug}.md"


def write_markdown_files(input_path: Path, output_dir: Path) -> int:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    challenges = payload.get("challenges", [])

    output_dir.mkdir(parents=True, exist_ok=True)

    for challenge in challenges:
        (output_dir / challenge_filename(challenge)).write_text(
            challenge_to_markdown(challenge), encoding="utf-8"
        )

    if "getting_started" in payload:
        (output_dir / "getting-started.md").write_text(
            getting_started_to_markdown(payload["getting_started"]), encoding="utf-8"
        )
    if "troubleshooting" in payload:
        (output_dir / "troubleshooting.md").write_text(
            troubleshooting_to_markdown(payload["troubleshooting"]), encoding="utf-8"
        )
    if "faq" in payload:
        (output_dir / "faq.md").write_text(
            faq_to_markdown(payload["faq"]), encoding="utf-8"
        )

    (output_dir / "index.md").write_text(
        index_to_markdown(challenges), encoding="utf-8"
    )
    return len(challenges)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split challenge-data.json into individual markdown files."
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="Input JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output markdown directory",
    )
    args = parser.parse_args()

    count = write_markdown_files(args.input, args.output_dir)
    print(f"Wrote {count} challenge markdown files to {args.output_dir}")


if __name__ == "__main__":
    main()
