# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Vũ Đình Đăng
**Mã sinh viên:** 2A202602946
**Khoá:** A20-K4
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Google Colab T4 · 16 GB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (Vietnamese) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 66% |
| DPO: β / tốc độ học (lr) / số epoch | 0,1 / 5e-6 / 1 |
| Giám khảo | Hội đồng `Skywork-Reward-V2-Qwen3-4B` + `Skywork-Reward-V2-Llama-3.2-3B`; sanity accuracy 100% |
| Chi phí | 0 đồng (Google Colab miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | Không được lưu trong artifact |
| VRAM cao nhất | Không được lưu trong artifact |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0,0981 |
| Độ chính xác reward trên held-out | 68% |
| Margin trên held-out | 0,0859 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 574,40 → 599,52 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Trên tập huấn luyện, reward của cả câu `chosen` và `rejected` đều tăng từ xấp xỉ 0. Ở cuối quá trình, reward `chosen` đạt khoảng
0,4035, còn `rejected` đạt khoảng 0,3054, tạo reward gap 0,0981. Vì `chosen` tăng và duy trì cao hơn `rejected`, đây không phải
trường hợp likelihood displacement, nơi reward của `chosen` giảm nhưng margin vẫn tăng do `rejected` giảm nhanh hơn. Đường
held-out cũng đi cùng hướng với train: reward `chosen` đạt 0,4163, reward `rejected` đạt 0,3304 và margin dương 0,0859. Khoảng
cách held-out nhỏ hơn train nhưng không đảo dấu; reward accuracy 68% cũng cao hơn mức ngẫu nhiên. Vì vậy chưa thấy dấu hiệu chỉ
ghi nhớ tập huấn luyện, dù chênh lệch không lớn và vẫn cần thận trọng khi khái quát. Chẩn đoán tự động `INTENDED` khớp với đồ thị:
hai reward cùng tăng, nhưng reward của câu được ưu tiên tăng nhanh hơn và tạo margin dương trên cả train lẫn held-out.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 4 | 8 | 38 | 46,0% ([39,0%; 53,0%]) | 46,74% | 50,0% |
| hữu ích — helpfulness (4) | 4 | 0 | 1 | 3 | 37,5% ([12,5%; 50,0%]) | 37,5% | 100,0% |
| an toàn — safety (4) | 4 | 1 | 0 | 3 | 62,5% ([50,0%; 87,5%]) | 66,67% | 0,0% |

Giám khảo: hội đồng Skywork Reward V2 Qwen3-4B và Llama-3.2-3B · sanity accuracy: 100% · `score_length_spearman`: 0,0066 (Qwen3) và 0,1080 (Llama)

Khoảng tin cậy 95% của held-out là [39%; 53%], có chứa 50%, nên kết quả chưa chứng minh DPO tốt hơn SFT. Sanity accuracy 100%
cho thấy hai reward model phân biệt đúng các cặp kiểm tra tiếng Việt, nhưng điều đó không loại bỏ hoàn toàn thiên lệch của giám khảo.
DPO dài hơn trung bình khoảng 25 ký tự, tuy nhiên chỉ 50% câu dài hơn thắng và tương quan điểm–độ dài của hai giám khảo rất thấp;
vì vậy chưa có bằng chứng rõ về length hacking. Qwen3 cho win rate 50%, còn Llama cho 42%; chênh lệch 8 điểm phần trăm đáng lưu ý
nhưng chưa đủ để khẳng định preference leakage. Mức đồng thuận giữa hai giám khảo là 86,21%.

Ở ví dụ hữu ích h3 (email xin nghỉ chăm con ốm), câu DPO ngắn gọn hơn và vẫn giữ đủ chủ đề, thời gian nghỉ, lời xin phép và cách kết
thư; đây là cải thiện hợp lý so với bản SFT hơi dài. Ngược lại, ví dụ h2 cho thấy cả hai câu đều gợi ý nguyên liệu không có sẵn, nên DPO
không bảo đảm cải thiện tính đúng đắn. Với an toàn s2, cả hai mô hình đều từ chối viết lời đe dọa, nhưng DPO bổ sung các phương án
giải quyết xung đột như nói chuyện trực tiếp hoặc nhờ người lớn đáng tin cậy. Đây là lý do hợp lý để ưu tiên DPO ở ví dụ an toàn đó.
Một hạn chế chung là đầu ra còn xuất hiện token `<tool_call>`/`</tool_call>`, cần được xử lý ở chat template hoặc bước hậu xử lý.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | | | | |
| 0.5 | | | | |

Tôi chưa chạy β-sweep. Tôi dự đoán β nhỏ sẽ cho phép mô hình dịch chuyển xa hơn khỏi mô hình tham chiếu, nên margin có thể lớn hơn
nhưng rủi ro làm giảm chất lượng hoặc thay đổi độ dài cũng cao hơn. Với β lớn, chính sách bị giữ gần mô hình tham chiếu hơn nên margin
có thể nhỏ hơn; cần nhiều seed để biết khác biệt về reward accuracy có vượt nhiễu hay không.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

Quyết định quan trọng nhất của tôi là dùng β = 0,1 với learning rate 5e-6 trên tier T4. Phương án thay thế là tăng β lên 0,5 để giữ
chính sách gần mô hình SFT hơn, hoặc giảm xuống 0,05 để tạo tín hiệu ưu tiên mạnh hơn; tôi cũng có thể dùng learning rate nhỏ hơn
nhằm giảm dao động. Tôi chọn cấu hình hiện tại vì đây là mức cân bằng thực dụng cho LoRA DPO: đủ mạnh để tạo margin trong số bước
huấn luyện hạn chế, nhưng không quá mạnh so với mô hình tham chiếu. Kết quả xác nhận một phần lựa chọn này. Reward gap train đạt
0,0981, margin held-out đạt 0,0859 và chẩn đoán là `INTENDED`; tức reward của `chosen` tăng nhanh hơn `rejected` trên cả hai tập.
Tuy nhiên, đánh giá đầu ra làm tôi thận trọng hơn: held-out win rate chỉ 46%, khoảng tin cậy [39%; 53%] chứa 50%, đồng thời có tới
38/50 cặp hòa. Như vậy tín hiệu reward nội tại tốt hơn chưa chuyển thành cải thiện đầu ra có ý nghĩa thống kê. Nếu làm lại, tôi sẽ
chạy β-sweep với ít nhất ba seed, giữ nguyên split và báo trung bình cùng độ lệch chuẩn. Tôi cũng sẽ sửa chat template để loại các
token `<tool_call>` thừa, vì lỗi định dạng có thể làm giảm điểm của cả SFT và DPO. Cuối cùng, tôi sẽ bổ sung một giám khảo khác họ
với Skywork và đánh giá thủ công một mẫu nhỏ để kiểm tra xem kết luận có phụ thuộc vào hội đồng reward model hiện tại hay không.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

_Trả lời ở đây._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

_Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?_

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?_

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Reward margin và accuracy trên held-out đều cải thiện theo đúng hướng, nhưng win rate đầu ra vẫn chỉ 46% và chưa khác 50% có ý nghĩa.
Điều này nhắc tôi rằng tối ưu preference loss không đồng nghĩa trực tiếp với chất lượng sinh tốt hơn theo giám khảo độc lập.
