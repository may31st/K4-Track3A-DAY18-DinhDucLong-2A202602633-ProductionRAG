# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Đinh Đức Long  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.9800 | 0.9800 | +0.0000 |
| Answer Relevancy | 0.8812 | 0.8591 | -0.0221 |
| Context Precision | 0.9000 | 0.9000 | +0.0000 |
| Context Recall | 0.7665 | 0.7017 | -0.0648 |

---

## Bottom-5 Failures

### #1
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Thẩm quyền phê duyệt thuộc Tổng Giám đốc hoặc Ban Giám đốc đối với các giao dịch mua sắm thiết bị trên 50 triệu VNĐ.
- **Got:** Context trích đoạn quy trình mua sắm chung nhưng thiếu bảng phân quyền hạn mức tài chính chi tiết.
- **Worst metric:** context_recall (0.4542)
- **Error Tree:** Output chưa chính xác → Context thiếu bảng phân quyền phê duyệt theo mức 50 triệu → Retrieval bỏ sót tài liệu hạn mức → Query "thiết bị 55 triệu" chưa match tốt với "thẩm quyền phê duyệt chi tiêu".
- **Root cause:** Khi cắt đoạn Hierarchical (child 256 ký tự), bảng hạn mức phê duyệt mua sắm bị tách nhỏ khiến điều kiện giá trị và người ký duyệt nằm ở hai chunk khác nhau.
- **Suggested fix:** Cải thiện chunking bằng Structure-Aware Chunking để giữ nguyên khối bảng hạn mức, đồng thời bổ sung từ khóa BM25 về hạn mức số tiền.

### #2
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Đơn xin nghỉ phép không lương trên 14 ngày phải được Giám đốc Khối hoặc Tổng Giám đốc phê duyệt sau khi Trưởng bộ phận đồng ý.
- **Got:** Context chỉ trích xuất được quy định chung: nhân viên được nghỉ không lương tối đa 30 ngày mỗi năm.
- **Worst metric:** context_recall (0.5722)
- **Error Tree:** Output thiếu cấp phê duyệt cho mốc 20 ngày → Context thiếu điều khoản phân cấp thẩm quyền nghỉ dài hạn → Retrieval chỉ lấy được chunk nói về số ngày tối đa.
- **Root cause:** Quy trình duyệt nghỉ dài ngày nằm ở phần thủ tục nhân sự thay vì mục định mức ngày nghỉ; embedding kéo về các chunk có cụm "nghỉ phép không lương" nhiều nhất.
- **Suggested fix:** Sử dụng Contextual Prepend để bổ sung bối cảnh tài liệu cha vào chunk và tăng top_k của Hybrid Search lên 30 trước khi đưa vào Cross-Encoder Reranker.

### #3
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Mức lương thử việc bằng tối thiểu 85% lương chính thức và dải lương trần cho vị trí Junior theo thang bảng lương công ty.
- **Got:** Context chỉ có quy định chung về thời gian thử việc và tỷ lệ 85%, thiếu dải lương số tuyệt đối cho cấp Junior.
- **Worst metric:** context_recall (0.5577)
- **Error Tree:** Output không có con số cụ thể → Context thiếu bảng lương chi tiết → Retrieval chưa liên kết được giữa vị trí "Junior" và khung lương.
- **Root cause:** Bảng lương nhiều cột bị vỡ định dạng khi cắt chunk theo số ký tự cố định, làm mất liên kết giữa vị trí công việc và dải lương tương ứng.
- **Suggested fix:** Áp dụng Structure-Aware Chunking giữ trọn bảng Markdown, kết hợp làm giàu metadata `level: junior` ở bước M5.

### #4
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quy định tạm ứng yêu cầu hoàn ứng trong 15 ngày; nếu quá hạn 20 ngày sẽ bị tính chế tài phạt lãi chậm nộp hoặc trừ vào lương kỳ kế tiếp.
- **Got:** Context chỉ trích được thời hạn tạm ứng tiêu chuẩn nhưng thiếu điều khoản về chế tài phạt quá hạn.
- **Worst metric:** context_recall (0.5577)
- **Error Tree:** Output thiếu điều khoản phạt → Context thiếu quy định xử lý quá hạn hoàn ứng → Lệch từ khóa giữa câu hỏi và văn bản quy chế.
- **Root cause:** Lệch từ vựng (lexical gap): Câu hỏi dùng từ thông thường "bị phạt", trong khi tài liệu quy chế dùng cụm "chế tài xử lý vi phạm thời hạn hoàn ứng".
- **Suggested fix:** Dùng kỹ thuật HyQA ở Module 5 để sinh trước các câu hỏi giả định có từ ngữ thông dụng như "bị phạt", "trễ hạn" vào metadata của chunk.

### #5
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 12 ngày cơ bản + 1 ngày thâm niên (mỗi 5 năm cộng 1 ngày) = 13 ngày phép; dải lương vị trí Senior theo thang bảng lương.
- **Got:** Context chỉ lấy được điều khoản về cộng ngày phép thâm niên, thiếu hẳn khung lương Senior.
- **Worst metric:** context_recall (0.5636)
- **Error Tree:** Output chỉ trả lời được 1 vế → Context thiếu vế thứ 2 (khung lương Senior) → Đây là câu hỏi ghép hai ý độc lập (multi-intent query).
- **Root cause:** Câu hỏi truy vấn hai chính sách nằm ở hai văn bản riêng biệt (Quy chế nghỉ phép và Quy chế lương thưởng); hệ thống truy vấn đơn (Single-hop) bị kéo lệch theo cụm từ "thâm niên phép năm".
- **Suggested fix:** Bổ sung bước Query Decomposition ở đầu pipeline để tách câu hỏi kép thành hai truy vấn con độc lập: số ngày phép cho 9 năm thâm niên và khung lương nhân viên Senior, sau đó gộp kết quả tìm kiếm từ cả hai.

---

## Case Study (cho presentation)

**Câu hỏi phân tích:**  
*"Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"*

**Error Tree walkthrough:**
1. **Output đúng?** Sai một nửa. Mô hình tính đúng số ngày phép năm (13 ngày) nhưng bỏ sót dải lương của Senior.
2. **Context đúng?** Sai. Context trả về gồm 3 đoạn từ `So_tay_nhan_vien.md` và `Quy_che_noi_bo.md` về chính sách nghỉ phép và thâm niên, không có đoạn nào từ tài liệu lương thưởng.
3. **Query rewrite:** Chưa tối ưu. Query nguyên bản gửi thẳng vào BM25 và Dense Search mang mật độ từ khóa cao về "thâm niên", "nghỉ phép" nên các chunk về ngày phép lấn át bảng lương.
4. **Khâu cần khắc phục:**
   - Tiền xử lý câu hỏi: Phân rã câu hỏi ghép (Query Decomposition) để tìm kiếm độc lập trên hai tài liệu.
   - Indexing: Structure-Aware Chunking để bảng lương không bị cắt rời.

**Nếu có thêm 1 giờ làm bài:**
- Triển khai Query Decomposition / Sub-query routing để tự động nhận diện câu hỏi đa ý và sinh các truy vấn con song song.
- Cấu hình Dynamic Top-K Retrieval: thay vì cố định top 20 candidates cho Reranker, dùng ngưỡng điểm số (score threshold) để giữ lại các tài liệu tiềm năng trước khi Cross-Encoder chấm điểm.
