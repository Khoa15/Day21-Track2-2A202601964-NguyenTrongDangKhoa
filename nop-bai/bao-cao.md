# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Nguyễn Trọng Đăng Khoa |
| MSSV | 2A202601964 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/Khoa15/Day21-Track2-2A202601964-NguyenTrongDangKhoa |
| Ngày nộp | 21/08/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 50 | 0.05 | 2 | 0.605 | 0.846 |
| 2 | 100 | 0.1 | 3 | 0.711 | 0.878 |
| 3 | 200 | 0.2 | 5 | 0.715 | 0.874 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.2`, `max_depth=5`.

**Lý do:** Lần chạy 3 có F1 score cao nhất (0.715). Lần chạy 2 có accuracy cao hơn (0.878) nhưng F1 thấp hơn (0.711), cho thấy accuracy bị ảnh hưởng bởi class imbalance. Giữa n_estimators và learning_rate có sự đánh đổi: với 200 estimators và learning rate 0.2, model vẫn hội tụ tốt và tránh được local minimum.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Bộ dữ liệu Adult có tỷ lệ class imbalance 75/25 (lớp ≤50K chiếm 75%). Một mô hình luôn trả lời "thu nhập thấp" đã đạt accuracy 0.75 mà không học được gì. F1 của lớp dương (thu nhập > 50K) đo khả năng cân bằng giữa precision và recall của lớp thiểu số — điều mà accuracy hoàn toàn bỏ qua. Không dùng average="weighted" hay "macro" vì chúng tính trung bình F1 của cả hai lớp, trong khi ta chỉ quan tâm đến khả năng phát hiện người có thu nhập cao.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| DVC pull lỗi 401 Invalid Credentials | `.dvc/config` có `credentialpath = ../sa-key.json` trỏ đến file không tồn tại trong CI | Copy sa-key.json vào đúng vị trí trong workflow trước khi dvc pull |
| Service income-api crash trên server | sklearn version mismatch: model train với 1.4.2 nhưng server chạy 1.7.2 | Thêm `pip install --force-reinstall` trong deploy script để đảm bảo đúng version |

---

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7032 | 0.8700 |
| Bước 3 (thêm `train_batch2`) | 0.7207 | 0.8760 |

**Nhận xét:** Thêm dữ liệu train_batch2 giúp cải thiện F1 (+0.0175) và accuracy (+0.006). Điều này cho thấy dữ liệu mới bổ sung thêm thông tin hữu ích cho model, giúp model học tốt hơn trên cả hai lớp.

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

- [ ] Bonus 1 - Tracking MLflow từ xa với DagsHub: ___
- [ ] Bonus 2 - Điều chỉnh ngưỡng quyết định: ___
- [ ] Bonus 3 - Báo cáo precision / recall tự động: ___
- [ ] Bonus 4 - Hoàn trả về phiên bản trước: ___
- [ ] Bonus 5 - Cảnh báo lệch lạc dữ liệu: ___
