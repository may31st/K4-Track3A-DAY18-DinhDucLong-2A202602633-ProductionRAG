# Individual Reflection: Lab 18 - Production RAG

Họ và tên: Đinh Đức Long  
Khóa: K4 - Track 3A  
Ngày hoàn thành: 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Đối chiếu các khái niệm trong bài giảng với phần triển khai thực tế trong lab:

| Lecture Concept | Module | Hàm cụ thể | Phân tích & Thực tế triển khai |
|----------------|--------|-------------|--------------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Tính cosine similarity giữa vector nhúng của các câu liên tiếp bằng `all-MiniLM-L6-v2`. Ngắt chunk khi độ tương đồng thấp hơn ngưỡng `SEMANTIC_THRESHOLD` (0.85) để giữ trọn ý của câu thay vì cắt cứng theo số ký tự. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Tách theo cấu trúc cha con: chunk con (256 ký tự) dùng để match vector và từ khóa cho chuẩn, còn chunk cha (2048 ký tự) trả về làm ngữ cảnh đầy đủ cho LLM sinh câu trả lời. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Cắt theo các đề mục Markdown (`#`, `##`, `###`) và lưu tên `section` vào metadata, nhờ đó giữ liền mạch bảng biểu và các điều khoản quy chế nội bộ. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | Kết hợp thứ hạng bằng thuật toán RRF giữa BM25 (dùng Underthesea tách từ tiếng Việt) và Dense Search (vector 1024 chiều từ BGE-M3), bù đắp điểm yếu bắt keyword của dense search và hiểu ngữ nghĩa của lexical search. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Dùng mô hình `BAAI/bge-reranker-v2-m3` chấm điểm lại top 20 candidate bằng full attention giữa câu hỏi và đoạn văn, sau đó chỉ lấy top 3 đoạn liên quan nhất đưa vào prompt. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Đánh giá qua 4 tiêu chí cốt lõi: Faithfulness (mức độ bám sát ngữ cảnh), Answer Relevancy (trả lời đúng trọng tâm câu hỏi), Context Precision (độ chuẩn của thứ tự chunk lấy về) và Context Recall (độ phủ thông tin). |
| Diagnostic Tree & Failure Analysis | M4 | `failure_analysis()` | Tự động phân loại lỗi (hallucination, thiếu chunk, chunk nhiễu, prompt chưa chuẩn) dựa vào chỉ số thấp nhất để khoanh vùng đúng khâu cần chỉnh sửa. |
| Contextual embeddings & HyQA | M5 | `_enrich_single_call()` / `enrich_chunks()` | Gom vào 1 lần gọi API để vừa tạo Contextual Prepend (tóm tắt vị trí của chunk trong tài liệu), vừa sinh câu hỏi giả định HyQA và gán metadata. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- Lỗi kỹ thuật gặp phải (Exact error message):
  1. `openai.RateLimitError: Error code: 429 - Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: ...`
  2. `AttributeError: 'ChatOpenAI' object has no attribute 'set_run_config'` khi chạy thư viện Ragas.
  3. `ModuleNotFoundError: No module named 'pypdf'` và lỗi encoding trên terminal Windows (`cp1258 codec can't encode character`).

- Nguyên nhân gốc rễ và cách debug:
  1. Rate Limit & Quota: Bản free tier của Google AI Studio chỉ cho 5 RPM và 20 request/ngày trên mỗi model. Khi chạy bộ 20 câu hỏi kèm enrichment cho 104 chunks, số lượng request vượt ngưỡng khiến API trả về mã 429 liên tục và pipeline bị nghẽn do retry lặp lại.
     Cách xử lý: Viết thêm cơ chế Circuit Breaker và fallback sang trích xuất câu chính cục bộ (Extractive Fallback) khi gặp lỗi quota hoặc timeout, giúp pipeline chạy tiếp mà không bị crash.
  2. Ragas Wrapper: Ragas bản mới yêu cầu các model LangChain phải bọc trong wrapper riêng.
     Cách xử lý: Bọc `ChatOpenAI` qua `LangchainLLMWrapper` và `HuggingFaceEmbeddings` qua `LangchainEmbeddingsWrapper`.
  3. Môi trường và encoding: Thiếu package trong môi trường ảo và terminal Windows mặc định dùng bảng mã cp1258.
     Cách xử lý: Cài bổ sung package vào `.venv` và thêm `sys.stdout.reconfigure(encoding="utf-8")` ở đầu các file chạy script.

- Kiến thức cần đào sâu thêm:
  - Cần tìm hiểu kỹ hơn cơ chế batching và async loop khi kết nối LangChain/Ragas với các provider ngoài OpenAI.
  - Tối ưu bộ nhớ khi load model Transformer: cần chuyển sang dạng singleton cho `CrossEncoder` và `SentenceTransformer` để tránh nạp lại trọng số mỗi lần gọi hàm, giảm tải RAM và thời gian khởi tạo.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống Trợ lý AI Hỏi đáp Quy chế & Hợp đồng Pháp lý Doanh nghiệp

#### 1. Hiện trạng
- Pipeline hiện tại: Dùng Naive RAG cơ bản (cắt chunk cố định 500 ký tự theo khoảng trắng, Dense Search đơn thuần với OpenAI `text-embedding-3-small`, lấy top 5 gửi thẳng vào LLM).
- Vấn đề gặp phải:
  - Bảng lương, biểu phí và điều khoản phạt hay bị cắt đôi giữa các chunk, khiến câu trả lời bị sai số liệu hoặc thiếu điều kiện ràng buộc.
  - Thường bỏ sót các thuật ngữ viết tắt hoặc số hiệu văn bản (ví dụ: "Nghị định 13", "MFA", "PVI").
  - LLM thỉnh thoảng tự suy diễn số ngày xử lý hoặc thẩm quyền ký duyệt khi context tìm về không đúng chỗ.

#### 2. Kế hoạch cải tiến
1. Chunking: Kết hợp Hierarchical Chunking và Structure-Aware Chunking. Giữ nguyên bảng biểu và cấu trúc điều khoản theo Markdown, chia chunk con 256 ký tự và lưu `parent_id` trỏ về toàn bộ điều luật cha.
2. Retrieval: Áp dụng Hybrid Search (BM25 + Dense Search) kết hợp thuật toán RRF ($k=60$). Dùng Underthesea tách từ tiếng Việt để BM25 bắt chính xác từng số hiệu văn bản và thuật ngữ pháp lý.
3. Reranking: Dùng `BAAI/bge-reranker-v2-m3` lọc từ top 30 xuống top 3 đến 5 đoạn sát nhất, hạn chế tình trạng lost in the middle và tiết kiệm token cho prompt.
4. Evaluation: Dựng pipeline CI/CD tự động chấm điểm với RAGAS trên bộ test 100 câu hỏi nghiệp vụ, kết hợp cây chẩn đoán để phân loại lỗi sau mỗi lần cập nhật kho dữ liệu.
5. Enrichment: Dùng Contextual Prepend để gắn tên chương/mục vào từng chunk nhỏ và sinh trước câu hỏi giả định HyQA cho các điều khoản tra cứu thường xuyên.

#### 3. Kế hoạch triển khai theo tuần
- Tuần 1: Chuẩn hóa lại khâu tiền xử lý dữ liệu: hoàn thiện OCR cho tài liệu scan và cấu hình Structure-Aware Chunking cho toàn bộ quy chế nội bộ.
- Tuần 2: Cấu hình Qdrant vector database, tạo chỉ mục BM25 tiếng Việt và ghép module Hybrid RRF Search.
- Tuần 3: Tích hợp Cross-Encoder Reranker và xây dựng bộ benchmark RAGAS đánh giá tự động trên môi trường Staging.
- Tuần 4: Đo kiểm độ trễ, tối ưu cache, đóng gói API bằng FastAPI và mở thử nghiệm nội bộ cho phòng Pháp chế và Nhân sự.
