# Lab 20: Multi-Agent Research System

Hệ thống nghiên cứu multi-agent hoàn chỉnh: **Supervisor + Researcher + Analyst + Writer +
Critic**, chạy trên LangGraph, benchmark với single-agent baseline. Bắt đầu từ starter
skeleton của bài lab, mọi `TODO(student)` đã được triển khai đầy đủ và mở rộng thêm:

- **LLM provider:** OpenAI hoặc **DeepSeek** (OpenAI-compatible API), chọn qua `LLM_PROVIDER`.
- **Search:** Tavily thật, fallback mock nếu không có key.
- **Structured output:** Analyst/Critic trả JSON có schema (Pydantic), tự retry khi model
  trả sai định dạng.
- **Parallel research:** Researcher chia query thành 3 sub-query, search song song.
- **Streaming CLI:** `multi-agent` in tiến trình theo từng bước thay vì đợi xong mới in.
- **Checkpointing:** LangGraph `MemorySaver`, resume được nếu crash giữa chừng.
- **LLM-as-judge:** benchmark chấm điểm câu trả lời bằng một LLM call độc lập, song song
  với heuristic quality score.
- **HTML dashboard:** `make benchmark` sinh cả `benchmark_report.md` và
  `benchmark_report.html` (biểu đồ so sánh, theme-aware, tự chứa).

## Learning outcomes

Sau 2 giờ lab, học viên cần có thể:

1. Thiết kế role rõ ràng cho nhiều agent.
2. Xây dựng shared state đủ thông tin cho handoff.
3. Thêm guardrail tối thiểu: max iterations, timeout, retry/fallback, validation.
4. Trace được luồng chạy và giải thích agent nào làm gì.
5. Benchmark single-agent vs multi-agent theo quality, latency, cost.

## Architecture mục tiêu

```text
User Query
   |
   v
Supervisor / Router
   |------> Researcher Agent  -> research_notes
   |------> Analyst Agent     -> analysis_notes
   |------> Writer Agent      -> final_answer
   |
   v
Trace + Benchmark Report
```

## Cấu trúc repo

```text
.
├── src/multi_agent_research_lab/
│   ├── agents/              # Agent interfaces + skeletons
│   ├── core/                # Config, state, schemas, errors
│   ├── graph/               # LangGraph workflow skeleton
│   ├── services/            # LLM, search, storage clients
│   ├── evaluation/          # Benchmark/evaluation skeleton
│   ├── observability/       # Logging/tracing hooks
│   └── cli.py               # CLI entrypoint
├── configs/                 # YAML configs for lab variants
├── docs/                    # Lab guide, rubric, design notes
├── tests/                   # Unit tests for skeleton behavior
├── notebooks/               # Optional notebook entrypoint
├── scripts/                 # Helper scripts
├── .env.example             # Environment variables template
├── pyproject.toml           # Python project config
├── Dockerfile               # Containerized dev/runtime
└── Makefile                 # Common commands
```

## Quickstart

### 1. Tạo môi trường

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev,llm]"
cp .env.example .env
```

### 2. Cấu hình API keys

Mở `.env` và điền key cần thiết. `LLM_PROVIDER` chọn `openai` hoặc `deepseek`.

```bash
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=...
# hoặc
OPENAI_API_KEY=...

# optional
LANGSMITH_API_KEY=...
TAVILY_API_KEY=...
```

Không có key nào? Mọi client (`LLMClient`, `SearchClient`) tự fallback sang chế độ mock
deterministic — toàn bộ workflow vẫn chạy được end-to-end để test/CI.

### 3. Chạy smoke test

```bash
make test
python -m multi_agent_research_lab.cli --help
```

### 4. Chạy baseline

```bash
python -m multi_agent_research_lab.cli baseline \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

### 5. Chạy multi-agent workflow

```bash
python -m multi_agent_research_lab.cli multi-agent \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

In tiến trình theo từng bước (Supervisor → Researcher → Analyst → Writer → Critic), sau đó
in `ResearchState` đầy đủ dạng JSON.

### 6. Chạy benchmark

```bash
python -m multi_agent_research_lab.cli benchmark
# hoặc
make benchmark
```

Chạy baseline + multi-agent trên các query trong `configs/lab_default.yaml`, ghi
`reports/benchmark_report.md` và `reports/benchmark_report.html`.

## Milestones trong 2 giờ lab

| Thời lượng | Milestone | File gợi ý |
|---:|---|---|
| 0-15' | Setup, chạy baseline skeleton | `cli.py`, `services/llm_client.py` |
| 15-45' | Build Supervisor / router | `agents/supervisor.py`, `graph/workflow.py` |
| 45-75' | Thêm Researcher, Analyst, Writer | `agents/*.py`, `core/state.py` |
| 75-95' | Trace + benchmark single vs multi | `observability/tracing.py`, `evaluation/benchmark.py` |
| 95-115' | Peer review theo rubric | `docs/peer_review_rubric.md` |
| 115-120' | Exit ticket | `docs/lab_guide.md` |

## Quy ước production trong repo

- Tách rõ `agents`, `services`, `core`, `graph`, `evaluation`, `observability`.
- Không hard-code API key trong code.
- Tất cả input/output chính dùng Pydantic schema.
- Có type hints, linting, formatting, unit test tối thiểu.
- Có logging/tracing hook ngay từ đầu.
- Không để agent chạy vô hạn: dùng `max_iterations`, `timeout_seconds`.
- Có benchmark report thay vì chỉ demo output đẹp.

## Trạng thái implementation

Không còn `TODO(student)` nào trong `src/`. Xem `docs/design_template.md` để biết chi
tiết thiết kế (agent roles, shared state, routing policy, guardrails, benchmark plan) và
`docs/lab_guide.md` cho exit ticket đã trả lời dựa trên số liệu benchmark thật.

```bash
make lint       # ruff — sạch
make test       # pytest — 9/9 pass
mypy src        # strict — sạch
```

## Deliverables

1. GitHub repo (repo này).
2. Trace: `LANGSMITH_API_KEY` (tùy chọn) để có trace UI thật; luôn có `state.trace` cục bộ.
3. `reports/benchmark_report.md` + `reports/benchmark_report.html` — so sánh single vs
   multi-agent bằng latency/cost/quality heuristic/judge score/citation coverage.
4. Failure mode + cách fix: xem exit ticket trong `docs/lab_guide.md`.

## References

- Anthropic: Building effective agents — https://www.anthropic.com/engineering/building-effective-agents
- OpenAI Agents SDK orchestration/handoffs — https://developers.openai.com/api/docs/guides/agents/orchestration
- LangGraph concepts — https://langchain-ai.github.io/langgraph/concepts/
- LangSmith tracing — https://docs.smith.langchain.com/
- Langfuse tracing — https://langfuse.com/docs
