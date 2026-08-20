# Design Document: Multi-Agent Research System

## Problem

Xây dựng một research assistant nhận một câu hỏi nghiên cứu (research query), tìm thông
tin từ nhiều nguồn, phân tích độ tin cậy/mâu thuẫn giữa các nguồn, và viết một câu trả
lời cuối cùng có trích dẫn (citation) rõ ràng — thay vì chỉ trả lời từ kiến thức nội tại
của một LLM duy nhất.

## Why multi-agent?

Single-agent (một LLM call trực tiếp) nhanh và rẻ, nhưng:

- Không có bước tìm kiếm nguồn tường minh — câu trả lời dựa hoàn toàn vào tri thức đã
  huấn luyện của model, dễ lỗi thời hoặc bịa (hallucination) mà không có gì để kiểm chứng.
- Không tách biệt "thu thập dữ liệu" và "tổng hợp câu trả lời" nên khó debug: khi câu trả
  lời sai, không biết là do thiếu dữ liệu hay do suy luận sai.
- Không có bước tự phê bình (self-critique) độc lập.

Multi-agent (Supervisor + Researcher + Analyst + Writer + Critic) tách rõ trách nhiệm,
cho phép mỗi bước dùng system prompt chuyên biệt và để lại "hồ sơ" (shared state) có thể
kiểm tra sau. Đổi lại là chi phí và độ trễ cao hơn — đúng như kết quả benchmark đo được
trong `reports/benchmark_report.md`.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Quyết định route tiếp theo dựa trên state hiện có; enforce `max_iterations` | `ResearchState` | `route_history` mới | Loop vô hạn nếu thiếu stop condition → chặn bằng `max_iterations` + fallback sau 2 lần route giống nhau kèm lỗi |
| Researcher | Tìm nguồn (`SearchClient`) và tóm tắt thành `research_notes` | `request.query` | `sources`, `research_notes` | Search provider fail → bắt exception, ghi vào `state.errors`, supervisor route tiếp thay vì crash |
| Analyst | Trích xuất luận điểm chính, so sánh quan điểm, đánh giá bằng chứng yếu | `research_notes` | `analysis_notes` | Thiếu `research_notes` → raise `ValueError` bắt ở trong, ghi lỗi vào state |
| Writer | Tổng hợp `final_answer` có trích dẫn `[n]` trỏ về `sources` | `research_notes`, `analysis_notes`, `sources` | `final_answer` | Thiếu `analysis_notes` → tương tự Analyst |
| Critic | Fact-check câu trả lời cuối, đo citation coverage | `final_answer`, `sources` | `AgentResult` ghi vào `agent_results`, không sửa `final_answer` | Không có final_answer → bỏ qua, ghi lỗi |

## Shared state

`ResearchState` (xem `src/multi_agent_research_lab/core/state.py`) là nguồn dữ liệu duy
nhất truyền qua toàn bộ workflow:

- `request` — query gốc, không đổi trong suốt vòng đời.
- `iteration`, `route_history` — dùng để enforce guardrail và debug thứ tự agent chạy.
- `sources`, `research_notes` — output của Researcher; Analyst/Writer đọc lại.
- `analysis_notes` — output của Analyst; Writer đọc lại.
- `final_answer` — output của Writer; Critic đọc lại, benchmark dùng để tính quality/citation.
- `agent_results` — lịch sử đầy đủ (bao gồm token/cost usage) của mọi agent, dùng cho benchmark.
- `trace` — sự kiện chi tiết hơn `route_history` (vd. `researcher.error`), dùng để debug.
- `errors` — danh sách lỗi non-fatal, Supervisor đọc để quyết định fallback.

Mỗi field tồn tại vì có ít nhất một agent downstream cần đọc lại nó — không có field nào
chỉ để "cho đẹp".

## Routing policy

```text
supervisor:
  if iteration >= max_iterations: -> done (hard stop)
  if not sources or not research_notes: -> researcher
  elif not analysis_notes: -> analyst
  elif not final_answer: -> writer
  elif critic chưa chạy: -> critic
  else: -> done

  # fallback: nếu route hiện tại lặp lại đúng route trước đó VÀ có lỗi trong state
  # -> bỏ qua bước đó, đi tiếp theo thứ tự researcher -> analyst -> writer -> done
```

Đây là state-machine thuần (không tốn LLM call) vì quyết định hoàn toàn dựa trên field
nào đã có/thiếu trong state — rẻ, nhanh, và dễ test (xem `tests/test_agents_todo.py`).

## Guardrails

- **Max iterations:** `Settings.max_iterations` (mặc định 6, đọc từ `MAX_ITERATIONS` env).
  Supervisor kiểm tra ở đầu mỗi lần chạy, force `done` nếu vượt.
- **Timeout:** `Settings.timeout_seconds` truyền vào mỗi HTTP call tới LLM/search provider
  (OpenAI/DeepSeek SDK `timeout=`, Tavily `httpx` request timeout).
- **Retry:** `LLMClient.complete` dùng `tenacity` (3 lần, exponential backoff) cho lỗi
  provider tạm thời. `SearchClient` fallback từ Tavily sang mock search nếu request fail.
- **Fallback:**
  - LLM: không có API key hợp lệ (OpenAI/DeepSeek) → mock LLM client trả response
    deterministic thay vì crash toàn bộ pipeline.
  - Search: không có `TAVILY_API_KEY` hoặc Tavily lỗi → mock search trả 5 nguồn giả lập,
    đủ để test luồng end-to-end.
  - Supervisor: agent lặp lại thất bại 2 lần liên tiếp → route sang bước kế tiếp thay vì
    thử lại vô hạn.
- **Validation:** Toàn bộ input/output chính (`ResearchQuery`, `AgentResult`,
  `SourceDocument`, `BenchmarkMetrics`) đều là Pydantic model — sai kiểu dữ liệu bị chặn
  ngay ở boundary (CLI parse, agent output).

## Benchmark plan

Bộ query benchmark (`configs/lab_default.yaml → benchmark.queries`):

1. "Research GraphRAG state-of-the-art and write a 500-word summary"
2. "Compare single-agent and multi-agent workflows for customer support"
3. "Summarize production guardrails for LLM agents"

Metric đo (`evaluation/benchmark.py`):

| Metric | Cách đo |
|---|---|
| Latency | wall-clock giữa lúc gọi runner và lúc trả về state |
| Cost | tổng `cost_usd` ước tính từ token usage của mọi `AgentResult` |
| Quality | heuristic 0-10: độ dài câu trả lời + citation coverage + số lỗi non-fatal |
| Citation coverage | tỉ lệ câu trong `final_answer` có `[n]` trỏ về nguồn |
| Failure rate | 1.0 nếu runner crash hoặc không sinh được `final_answer`, ngược lại 0.0 |

Expected outcome: multi-agent chậm hơn và tốn hơn baseline (nhiều LLM call hơn), nhưng
citation coverage cao hơn hẳn (baseline không có bước search nên coverage luôn rỗng).
Quality score có thể ngang hoặc thấp hơn baseline khi search provider chỉ là mock — đây là
một failure mode thực tế cần ghi nhận: multi-agent chỉ thắng khi nguồn tìm được có chất
lượng, nếu không nó chỉ thêm chi phí mà không thêm giá trị.
