# Lab Guide: Multi-Agent Research System

## Scenario

Bạn cần xây dựng một research assistant có thể nhận câu hỏi dài, tìm thông tin, phân tích và viết câu trả lời cuối cùng. Lab yêu cầu so sánh hai cách làm:

1. **Single-agent baseline**: một agent làm toàn bộ.
2. **Multi-agent workflow**: Supervisor điều phối Researcher, Analyst, Writer.

## Quy tắc quan trọng

- Không thêm agent nếu không có lý do rõ ràng.
- Mỗi agent phải có responsibility riêng.
- Shared state phải đủ rõ để debug.
- Phải có trace hoặc log cho từng bước.
- Phải benchmark, không chỉ nhìn output bằng cảm tính.

## Milestone 1: Baseline

File gợi ý:

- `src/multi_agent_research_lab/cli.py`
- `src/multi_agent_research_lab/services/llm_client.py`

TODO(student): thay baseline placeholder bằng một call LLM thật.

## Milestone 2: Supervisor

File gợi ý:

- `src/multi_agent_research_lab/agents/supervisor.py`
- `src/multi_agent_research_lab/graph/workflow.py`

TODO(student): implement routing policy.

Gợi ý câu hỏi thiết kế:

- Khi nào gọi Researcher?
- Khi nào gọi Analyst?
- Khi nào gọi Writer?
- Khi nào stop?
- Nếu agent fail thì retry hay fallback?

## Milestone 3: Worker agents

File gợi ý:

- `src/multi_agent_research_lab/agents/researcher.py`
- `src/multi_agent_research_lab/agents/analyst.py`
- `src/multi_agent_research_lab/agents/writer.py`

TODO(student): implement từng worker.

## Milestone 4: Trace và benchmark

File gợi ý:

- `src/multi_agent_research_lab/observability/tracing.py`
- `src/multi_agent_research_lab/evaluation/benchmark.py`
- `src/multi_agent_research_lab/evaluation/report.py`

Benchmark tối thiểu:

| Metric | Cách đo gợi ý |
|---|---|
| Latency | wall-clock time |
| Cost | token usage hoặc provider usage |
| Quality | rubric 0-10 do peer review |
| Citation coverage | số claims có source / tổng claims chính |
| Failure rate | số query fail / tổng query |

## Troubleshooting

### macOS: lỗi SSL certificate khi gọi API qua HTTPS (Tavily, OpenAI, ...)

Triệu chứng: khi implement `SearchClient` (hoặc bất kỳ HTTPS call nào) trên macOS, bạn có thể gặp lỗi kiểu:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
unable to get local issuer certificate
```

Nguyên nhân: Python cài từ python.org trên macOS **không dùng** certificate store của hệ điều hành, nên không tìm thấy CA bundle hợp lệ. Đây là lỗi môi trường, **không phải** do API key sai.

Cách khắc phục (chọn 1 trong 3):

1. **Chạy script cài certificate đi kèm Python** (nhanh nhất):

   ```bash
   /Applications/Python\ 3.12/Install\ Certificates.command
   ```

   (thay `3.12` bằng version Python của bạn)

2. **Dùng `certifi` trong code** — thêm `certifi` vào dependencies, rồi tạo SSL context khi gọi HTTPS:

   ```python
   import certifi
   import ssl
   from urllib.request import urlopen

   ssl_context = ssl.create_default_context(cafile=certifi.where())
   urlopen(request, timeout=timeout, context=ssl_context)
   ```

3. **Set biến môi trường** trỏ tới CA bundle của certifi (không cần đổi code):

   ```bash
   export SSL_CERT_FILE=$(python -m certifi)
   ```

## Exit ticket

Mỗi nhóm trả lời 2 câu:

1. Case nào nên dùng multi-agent? Vì sao?

   Khi câu hỏi cần tổng hợp thông tin từ nhiều nguồn bên ngoài, có yêu cầu trích dẫn
   (citation) tường minh, và việc tách "tìm nguồn / phân tích / viết" giúp mỗi bước dùng
   prompt chuyên biệt và dễ debug hơn khi kết quả sai. Trong benchmark ở
   `reports/benchmark_report.md`, multi-agent luôn cho citation coverage > 0 trong khi
   baseline luôn là rỗng — vì baseline không có bước search nên không có gì để trích dẫn.
   Multi-agent cũng đáng dùng khi cần audit trail: `route_history` + `trace` +
   `agent_results` cho biết chính xác agent nào tạo ra phần nào của câu trả lời.

2. Case nào không nên dùng multi-agent? Vì sao?

   Khi câu hỏi đơn giản, có thể trả lời tốt từ kiến thức sẵn có của LLM, hoặc khi latency/
   cost là ưu tiên hàng đầu. Số liệu benchmark thực đo được (DeepSeek, cùng 3 query) cho
   thấy multi-agent chậm hơn baseline 13-51 giây và tốn gấp 3-5 lần chi phí token, trong
   khi quality score (heuristic) lại thấp hơn baseline 0.6-1.4 điểm — vì search provider
   trong môi trường lab chỉ là mock, nên nguồn thu thập được không có giá trị thông tin
   thực, kéo quality xuống dù citation coverage tăng. Đây là failure mode quan trọng: thêm
   agent không tự động cải thiện chất lượng nếu bước "tìm nguồn" không có dữ liệu thật —
   multi-agent chỉ đáng chi phí khi `SearchClient` được nối với một provider thật (Tavily/
   Bing) thay vì mock.
