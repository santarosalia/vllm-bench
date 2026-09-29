#!/usr/bin/env python3
"""vLLM serve 벤치를 실행하고 결과를 results/에 저장한다.

서버 주소와 모델은 .env 에서 읽는다. 이미 설정된 환경 변수가 있으면
파일보다 우선한다.

사용법:
  python bench.py run
  python bench.py run --num-prompts 20
  python bench.py show
  python bench.py show latest
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results"
ENV_FILE = ROOT / ".env"

DEFAULT_NUM_PROMPTS = 10
DEFAULT_INPUT_LEN = 1024
DEFAULT_OUTPUT_LEN = 256

SUMMARY_FIELDS = [
    ("completed", "성공 요청", 0),
    ("failed", "실패 요청", 0),
    ("duration", "소요 시간 (s)", 2),
    ("total_input_tokens", "입력 토큰", 0),
    ("total_output_tokens", "생성 토큰", 0),
    ("request_throughput", "요청 처리량 (req/s)", 2),
    ("output_throughput", "출력 토큰 처리량 (tok/s)", 2),
    ("total_token_throughput", "전체 토큰 처리량 (tok/s)", 2),
    ("max_output_tokens_per_s", "최대 출력 처리량 (tok/s)", 2),
    ("max_concurrent_requests", "최대 동시 요청", 2),
]

LATENCY_METRICS = [
    ("ttft", "TTFT", "첫 토큰까지"),
    ("tpot", "TPOT", "출력 토큰당 (첫 토큰 제외)"),
    ("itl", "ITL", "토큰 간 지연"),
    ("e2el", "E2EL", "요청 전체 지연"),
]


def parse_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_server_env() -> None:
    """`.env`를 읽는다. 이미 설정된 환경 변수는 유지한다."""
    for key, value in parse_env_file(ENV_FILE).items():
        os.environ.setdefault(key, value)


def server_setting(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"{name} 이 없습니다. .env 를 확인하세요.")
    return value


def find_vllm() -> str:
    override = os.environ.get("VLLM_BIN")
    if override:
        return override
    local = ROOT / ".venv" / "bin" / "vllm"
    if local.is_file():
        return str(local)
    found = shutil.which("vllm")
    if found:
        return found
    sys.exit(
        "vllm 실행 파일을 찾지 못했습니다. "
        "프로젝트 .venv에 vllm이 설치돼 있는지 확인하세요."
    )


def check_server(base_url: str, model: str) -> None:
    url = base_url.rstrip("/") + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        sys.exit(
            f"{url} 이 vLLM OpenAI API가 아닙니다 (HTTP {exc.code}).\n"
            f"GPU 머신의 vLLM 컨테이너가 이 주소로 떠 있는지 확인하세요."
        )
    except Exception as exc:
        sys.exit(
            f"서버에 연결하지 못했습니다: {url}\n{exc}\n"
            "vLLM 서버를 띄운 뒤 다시 실행하세요."
        )

    model_ids = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)]
    if model_ids and model not in model_ids:
        joined = ", ".join(str(item) for item in model_ids)
        print(f"안내: 요청 모델이 서버 목록에 없습니다. 서버 모델: {joined}")


def result_files() -> list[Path]:
    if not RESULT_DIR.is_dir():
        return []
    return sorted(RESULT_DIR.glob("*.json"), key=lambda path: path.stat().st_mtime)


def load_result(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        sys.exit(f"결과 파일이 비어 있습니다: {path}")
    # --append-result 로 여러 JSON이 이어진 경우 마지막 실행을 본다.
    decoder = json.JSONDecoder()
    obj, end = decoder.raw_decode(text)
    rest = text[end:].strip()
    while rest:
        obj, end = decoder.raw_decode(rest)
        rest = rest[end:].strip()
    if not isinstance(obj, dict):
        sys.exit(f"결과 JSON 형식이 아닙니다: {path}")
    return obj


def fmt(value, digits: int) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int) and digits == 0:
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def resolve_show_target(name: str | None) -> Path:
    files = result_files()
    if not files:
        sys.exit(f"저장된 결과가 없습니다: {RESULT_DIR}")
    if name is None or name == "latest":
        return files[-1]
    candidate = Path(name)
    if candidate.is_file():
        return candidate
    inside = RESULT_DIR / name
    if inside.is_file():
        return inside
    sys.exit(f"결과 파일을 찾지 못했습니다: {name}")


def print_summary_row(path: Path, data: dict) -> None:
    completed = data.get("completed", "-")
    failed = data.get("failed", "-")
    throughput = fmt(data.get("output_throughput"), 2)
    ttft = fmt(data.get("mean_ttft_ms"), 1)
    e2el = fmt(data.get("p99_e2el_ms") or data.get("mean_e2el_ms"), 1)
    print(
        f"{data.get('date', '-'):<17} "
        f"{completed!s:>4}/{failed!s:<4} "
        f"{throughput:>10} tok/s  "
        f"TTFT {ttft:>8} ms  "
        f"E2E {e2el:>8} ms  "
        f"{path.name}"
    )


def print_result(path: Path, data: dict) -> None:
    print(f"파일    {path}")
    print(f"날짜    {data.get('date', '-')}")
    print(f"모델    {data.get('model_id', '-')}")
    print(f"백엔드  {data.get('backend', '-')}")
    print(f"프롬프트 {data.get('num_prompts', '-')}")
    print(f"요청률  {data.get('request_rate', '-')}")
    for key in ("base_url", "endpoint", "input_len", "output_len"):
        if key in data:
            print(f"{key}: {data[key]}")

    print()
    print(f"{'항목':<28}{'값':>12}")
    print("-" * 40)
    for key, label, digits in SUMMARY_FIELDS:
        if key not in data:
            continue
        print(f"{label:<28}{fmt(data[key], digits):>12}")

    for key, short, title in LATENCY_METRICS:
        rows = []
        for stat, stat_label in (
            ("mean", "평균"),
            ("median", "중앙값"),
            ("p50", "P50"),
            ("p90", "P90"),
            ("p99", "P99"),
        ):
            field = f"{stat}_{key}_ms"
            if field in data:
                rows.append((stat_label, data[field]))
        if not rows:
            continue
        print()
        print(f"{short}  {title} (ms)")
        print("-" * 40)
        for stat_label, value in rows:
            print(f"{stat_label:<28}{fmt(value, 2):>12}")

    input_lens = data.get("input_lens") or []
    output_lens = data.get("output_lens") or []
    ttfts = data.get("ttfts") or []
    latencies = data.get("latencies") or []
    errors = data.get("errors") or []
    count = max(len(input_lens), len(output_lens), len(ttfts), len(latencies))
    if count:
        print()
        print("요청별")
        print(f"{'#':>4} {'입력':>8} {'출력':>8} {'TTFT ms':>12} {'E2E ms':>12}  오류")
        print("-" * 64)
        for index in range(count):
            ttft_ms = ttfts[index] * 1000 if index < len(ttfts) else None
            e2e_ms = latencies[index] * 1000 if index < len(latencies) else None
            error = errors[index] if index < len(errors) and errors[index] else ""
            in_len = input_lens[index] if index < len(input_lens) else "-"
            out_len = output_lens[index] if index < len(output_lens) else "-"
            print(
                f"{index + 1:>4} {in_len!s:>8} {out_len!s:>8} "
                f"{fmt(ttft_ms, 1):>12} {fmt(e2e_ms, 1):>12}  {error}"
            )


def cmd_show(args: argparse.Namespace) -> None:
    if args.name is None:
        files = result_files()
        if not files:
            sys.exit(f"저장된 결과가 없습니다: {RESULT_DIR}")
        print(f"{'날짜':<17} {'성공/실패':>9} {'출력 처리량':>16}  {'평균 TTFT':>14}  {'E2E':>12}  파일")
        for path in files:
            print_summary_row(path, load_result(path))
        print()
        print("자세히 보려면: python bench.py show latest")
        return
    path = resolve_show_target(args.name)
    print_result(path, load_result(path))


def cmd_run(args: argparse.Namespace) -> None:
    check_server(args.base_url, args.model)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for path in result_files()}

    command = [
        find_vllm(),
        "bench",
        "serve",
        "--backend",
        args.backend,
        "--base-url",
        args.base_url,
        "--endpoint",
        args.endpoint,
        "--model",
        args.model,
        "--num-prompts",
        str(args.num_prompts),
        "--random-input-len",
        str(args.random_input_len),
        "--random-output-len",
        str(args.random_output_len),
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(RESULT_DIR),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,99",
        "--metadata",
        f"base_url={args.base_url}",
        f"endpoint={args.endpoint}",
        f"input_len={args.random_input_len}",
        f"output_len={args.random_output_len}",
        *([item for item in args.extra if item != "--"]),
    ]

    print("실행:", " ".join(command))
    print(f"결과 디렉터리: {RESULT_DIR}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    log_path = RESULT_DIR / "latest.log"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log_file:
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
    code = process.wait()

    created = [path for path in result_files() if path.resolve() not in before]
    saved = created[-1] if created else (result_files()[-1] if result_files() else None)
    if saved is not None:
        saved_log = saved.with_suffix(".log")
        saved_log.write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
        print()
        print(f"저장됨: {saved}")
        print(f"로그:   {saved_log}")
        print("보기:   python bench.py show latest")
    if code != 0:
        sys.exit(code)


def build_parser() -> argparse.ArgumentParser:
    load_server_env()
    parser = argparse.ArgumentParser(description="vLLM serve 벤치 실행과 결과 조회")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="벤치를 실행하고 JSON으로 저장")
    run.add_argument("--base-url", default=server_setting("VLLM_BASE_URL"))
    run.add_argument("--endpoint", default=server_setting("VLLM_ENDPOINT"))
    run.add_argument("--backend", default=server_setting("VLLM_BACKEND"))
    run.add_argument("--model", default=server_setting("VLLM_MODEL"))
    run.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS)
    run.add_argument("--random-input-len", type=int, default=DEFAULT_INPUT_LEN)
    run.add_argument("--random-output-len", type=int, default=DEFAULT_OUTPUT_LEN)
    run.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="그대로 vllm bench serve에 넘길 추가 인자",
    )
    run.set_defaults(func=cmd_run)

    show = sub.add_parser("show", help="저장된 벤치 결과 조회")
    show.add_argument(
        "name",
        nargs="?",
        help="파일 경로, 파일 이름, 또는 latest. 생략하면 목록",
    )
    show.set_defaults(func=cmd_show)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
