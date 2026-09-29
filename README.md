# vllm-bench

이미 떠 있는 vLLM 서버에 `vllm bench serve`를 보내고, 결과를 `results/`에 저장해서 다시 본다.

이 저장소는 클라이언트다. 모델은 GPU 머신의 vLLM에서 서빙하고, 여기서는 HTTP로 벤치만 친다.

```
bench run
    │  HTTP
    ▼
vLLM  /v1/chat/completions
```

Apple Silicon, Python 3.12, `uv`가 필요하다. `vllm`과 `vllm-metal`은 macOS arm64 휠만 잠겨 있다.

## 준비

```bash
uv sync
cp .env.example .env
```

`.env`만 읽는다. 셸에 같은 변수가 있으면 파일보다 우선한다.

| 변수 | 의미 |
| --- | --- |
| `VLLM_BASE_URL` | 서버 주소 |
| `VLLM_ENDPOINT` | API 경로 |
| `VLLM_BACKEND` | `vllm bench serve` 백엔드 |
| `VLLM_MODEL` | 서버에 올라간 모델 이름 |

## 실행

```bash
uv run bench run
uv run bench show
uv run bench show latest
```

`run`은 실행 전에 `{VLLM_BASE_URL}/v1/models`를 확인한다. 기본 부하는 프롬프트 10개, 입력 1024토큰, 출력 256토큰이다.

```bash
uv run bench run --num-prompts 20 --random-input-len 512 --random-output-len 128
```

`--` 뒤에 붙인 인자는 `vllm bench serve`로 그대로 넘어간다.

```bash
uv run bench run -- --max-concurrency 4
```

## 결과

`results/`에 JSON과 같은 이름의 로그가 남는다.

- `show`는 실행 목록이다. 출력 처리량, 평균 TTFT, E2E를 보여 준다.
- `show latest` 또는 `show results/<파일>.json`은 지연 백분위(P50, P90, P99)와 요청별 TTFT, E2E를 보여 준다.

백분위는 TTFT, TPOT, ITL, E2EL의 50, 90, 99다.
