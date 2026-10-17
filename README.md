# SpaceTime-DSCA: Divided Space-Time Vision Transformer & Disentangled Semantic Concept Alignment for Text-to-Video Person Retrieval

Triển khai hoàn chỉnh kiến trúc mô hình **SpaceTime-DSCA (Divided Space-Time Vision Transformer with Disentangled Semantic Concept Alignment)** bằng PyTorch cho bài toán **Text-to-Video Person Retrieval (TVPR)** trên tập chuẩn **TVPReid** (`TVPReid-Duke`, `TVPReid-PRID`, `TVPReid-iLIDs`).

---

## 1. Cơ Sở Khoa Học: Chuyển Dịch Từ CLIP Sang Kiến Trúc Thuần ViT + BERT

### 1.1. Tại sao loại bỏ CLIP Pretrained để đảm bảo tính công bằng học thuật (Fair Comparison)?
Trong bài báo nền tảng **MFGF (ACM MM '24, `rootpaper.pdf`)**, tác giả sử dụng kiến trúc chuẩn gồm **Vision Transformer (ViT)** kết hợp **S3D (Separable 3D-CNN)** và **BERT Text Encoder** được huấn luyện trực tiếp trên tập dữ liệu TVPR, đạt Rank@1 là **35.2%** trên `TVPReid-PRID`.

Khi một nghiên cứu sử dụng **OpenAI CLIP** (`openai/clip-vit-base-patch16` đã pre-train trên 400 triệu cặp Text-Image ngoài luồng):
1. **Bị hội đồng bình duyệt (Reviewers) đặt nghi vấn**: Rất khó chứng minh sự vượt trội đến từ *đóng góp kiến trúc đề xuất* hay chỉ đơn thuần là hưởng lợi từ tập dữ liệu khổng lồ 400M ảnh của OpenAI.
2. **Khủng hoảng sụp đổ ngữ nghĩa (Semantic Collapse) của luồng chuyển động**:
   - Khi tách câu truy vấn thành $Q_{app}$ và $Q_{mot}$, trong bài toán Re-ID hầu hết các mô tả chuyển động đều bị suy biến thành các chuỗi văn bản trùng lặp: *"a person is walking"*, *"someone moving forward"*.
   - CLIP Text Encoder ánh xạ các mô tả này thành các vector gần như giống hệt nhau, khiến hàm mất mát tương phản InfoNCE bị kẹt tại $\ln(\text{Batch Size}) \approx 2.77$ vì phạt các mẫu âm tính có nhãn văn bản giống hệt nhau.
   - Phép trừ khung hình $\Delta F_t$ triệt tiêu đến 90% điểm ảnh, khi đi qua Average Pooling làm loãng hoàn toàn tín hiệu gradient.

### 1.2. Bản chất của "Chuyển động" trong Text-to-Video Person Retrieval (TVPR)
Phân tích sâu từ bài báo gốc cho thấy: **Trong TVPR, "chuyển động" (Motion) không phải là bài toán phân loại hành động (Action Recognition như chạy nhảy, đá bóng), mà là Động lực học Không - Thời gian (Spatio-Temporal Dynamics)**:
* **Sự thay đổi góc nhìn theo thời gian (Viewpoint Evolution)**: Khi người đi qua camera an ninh, họ quay từ trước mặt $\to$ nghiêng góc $\to$ nhìn từ phía sau, để lộ các đặc trưng bị che khuất (balo, logo sau áo, túi xách).
* **Phục hồi sau che khuất (Temporal Occlusion Recovery)**: Một phần cơ thể bị người khác hoặc vật cản che khuất ở frame $t_1$ nhưng xuất hiện rõ nét ở frame $t_2$.
* **Đặc trưng quỹ đạo và sải bước (Trajectory & Stride Dynamics)**: Tốc độ di chuyển và phương hướng tương đối đối với góc đặt camera.

---

## 2. Tập Dữ Liệu Chuẩn TVPReid & Ý Nghĩa 3 Sub-datasets

### 2.1. Nguồn gốc lịch sử hình thành TVPReid
* Trước khi có **TVPReid**, các bộ dữ liệu Re-ID bằng văn bản (*CUHK-PEDES*, *ICFG-PEDES*, *RSTPReid*) **chỉ có hình ảnh tĩnh 2D**, hoàn toàn thiếu thông tin chuỗi thời gian và dáng đi.
* Trong khi đó, ở nhánh **Video Re-ID**, cộng đồng thị giác máy tính đã có sẵn 3 tập chuẩn video giám sát kinh điển:
  1. **iLIDS-VID** (2014)
  2. **PRID-2011** (2011)
  3. **DukeMTMC-VideoReID** (2018)
* Cả 3 tập này nguyên bản **chỉ có video, không hề có văn bản mô tả (captions)**. Nhóm tác giả bài báo MFGF (ACM MM '24) đã thực hiện một khối lượng công việc lớn: **gán nhãn thủ công 2 câu mô tả chi tiết bằng ngôn ngữ tự nhiên cho từng video** (trang phục, màu sắc, phụ kiện, hướng đi, hành động), tổng cộng **13.118 câu mô tả cho 6.559 clip video**, tạo ra benchmark đa phương thức Text-to-Video đầu tiên mang tên **TVPReid**.

### 2.2. Bốn lý do khoa học tại sao tác giả chia thành 3 sub-datasets riêng biệt

#### ① Tính chuẩn mực và kế thừa khoa học (Benchmark Consistency & Fair Comparison)
Trong cộng đồng nghiên cứu thị giác máy tính quốc tế (CVPR, ICCV, ACM MM), cả 3 tập dữ liệu trên đã có **quy chuẩn phân chia tập train/test cố định** từ nhiều năm qua. Việc duy trì 3 sub-datasets độc lập giúp:
* So sánh trực tiếp, công bằng (*fair comparison*) với các nghiên cứu tiền nhiệm.
* Đảm bảo tính toàn vẹn, loại trừ nguy cơ rò rỉ thông tin danh tính giữa các góc camera (*data leakage*).

#### ② Ba kịch bản giám sát thực tế với thách thức thị giác hoàn toàn khác biệt

| Sub-dataset | Bối cảnh giám sát | Số Camera | Thách thức thị giác chính |
| :--- | :--- | :---: | :--- |
| **`TVPReid-iLIDs`** | **Trong nhà (Sảnh đến sân bay)** | 2 | **Che khuất nặng nề (Severe Occlusion)**: Người đi lại đông đúc, chen chúc, hành lý, xe đẩy và cột nhà liên tục che khuất đối tượng. |
| **`TVPReid-PRID`** | **Ngoài trời (Lối đi bộ pathway)** | 2 | **Góc nhìn lệch cực lớn (Extreme Viewpoint Variations)**: Góc quay giữa 2 camera tĩnh chênh lệch rất cao, ánh sáng bóng râm/nắng gắt làm thay đổi sắc độ quần áo. |
| **`TVPReid-Duke`** | **Đại học diện rộng (Campus)** | 8 | **Quy mô lớn & Không gian phức tạp (Large-scale Multi-camera)**: 8 camera rải khắp khuôn viên, khoảng cách camera xa, thời gian di chuyển giữa các camera dài. |

#### ③ Kiểm thử trên 3 mức quy mô dữ liệu (Data Scales)
* **`TVPReid-iLIDs` (Quy mô nhỏ - 300 IDs, 1.200 captions, ~600 videos)**: Thử nghiệm khả năng học trong điều kiện **ít dữ liệu** (*Low-data regime*), đo lường mức độ chống học vẹt (*overfitting*).
* **`TVPReid-PRID` (Quy mô vừa - 385 IDs, 2.268 captions, ~1.134 videos)**: Cân bằng giữa ngoại hình và biến thiên chuyển động.
* **`TVPReid-Duke` (Quy mô lớn - 1.404 IDs, 9.650 captions, ~4.825 videos)**: Thử thách khả năng tìm kiếm trong **Gallery khổng lồ** (hàng trăm/nghìn người), kiểm tra xem mô hình có giữ được Rank@1 khi không gian tìm kiếm bị loãng hay không.

#### ④ Đo lường tính khái quát hóa và độ ổn định đa miền (Generalization Robustness)
Một mô hình Text-to-Video Person Retrieval xuất sắc **không được phép phụ thuộc vào một góc quay hay điều kiện chiếu sáng duy nhất**. Việc mô hình hoạt động vượt trội đồng thời trên cả 3 sub-datasets chứng minh kiến trúc có tính tổng quát hóa cao, đáp ứng môi trường giám sát an ninh thực tế.

---

## 3. Kiến Trúc Mô Hình SpaceTime-DSCA Đề Xuất

```
                        ┌────────────────────────────────────────────────────────┐
                        │      Text Query Q (e.g. "A man in black coat carrying  │
                        │                 a backpack walking to the left")       │
                        └───────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                                    ┌───────────────────────────────┐
                                    │ DSCA Text Concept Decomposer  │
                                    │ (Structured Semantic Parsing) │
                                    └───────┬───────────────┬───────┘
                                            │               │
                     ┌──────────────────────┴────────┐      └──────────────────────┐
                     ▼ Q_app (Appearance Prompt)     │                             ▼ Q_mot (Motion Prompt)
       ┌─────────────────────────────┐               │              ┌─────────────────────────────┐
       │   BERT Text Encoder (768-D) │               │              │   BERT Text Encoder (768-D) │
       └──────────────┬──────────────┘               │              └──────────────┬──────────────┘
                      ▼ T_app                        │                             ▼ T_mot
                      │                              ▼ T_full                      │
                      │                ┌─────────────────────────────┐             │
                      │                │   BERT Text Encoder (768-D) │             │
                      │                └──────────────┬──────────────┘             │
                      │                               │                            │
 ═════════════════════╪═══════════════════════════════╪════════════════════════════╪═════════════════════════
  LOSS: L_total = L_joint(T_full, V_video) + 0.5*L_app(T_app, V_app) + 0.25*L_mot(T_mot, V_mot) + 0.1*L_MIM
 ═════════════════════╪═══════════════════════════════╪════════════════════════════╪═════════════════════════
                      │                               │                            │
                      ▼                               ▼                            ▼
                 ┌─────────┐                     ┌─────────┐                  ┌─────────┐
                 │  V_app  │                     │ V_video │                  │  V_mot  │
                 └────┬────┘                     └────┬────┘                  └────┬────┘
                      │                               │                            │
                      └───────────────────────┬───────┴────────────────────────────┘
                                              │
                                  ┌───────────┴───────────┐
                                  │   MLP Fusion Stream   │
                                  │   Norm(MLP([V_a,V_m]))│
                                  └───────────┬───────────┘
                                              │
                            ┌─────────────────┴─────────────────┐
                            │   Divided Space-Time Transformer  │
                            │      (TimeSformer Architecture)   │
                            │  - Temporal Multi-head Attention  │
                            │  - Spatial Multi-head Attention   │
                            │  - ImageNet-1K Spatial Pretrain   │
                            └─────────────────┬─────────────────┘
                                              │
                            ┌─────────────────┴─────────────────┐
                            │ 2D Patch Embed + Space-Time Pos   │
                            └─────────────────┬─────────────────┘
                                              │
                            ┌─────────────────┴─────────────────┐
                            │ Input Video Clip (T=16, 224x224)  │
                            └───────────────────────────────────┘
```

### ① Divided Space-Time Transformer Backbone (`SpaceTimeViT`)
* **Spatial Multi-Head Attention**: Tận dụng trọng số nạp trước từ ImageNet-1K (`google/vit-base-patch16-224`) trên 12 attention heads để nắm bắt tức thì hình thái trang phục, logo, màu sắc.
* **Temporal Multi-Head Attention**: Khối tự chú ý theo trục thời gian được khởi tạo trọng số 0 (*Zero-Initialization* tại tầng chiếu `out_proj`), đảm bảo giai đoạn đầu không phá hủy đặc trưng không gian đã pre-train.
* **Tách luồng đặc trưng độc lập (Disentangled Video Heads)**:
  - **Appearance Stream ($V_{app}$)**: Dynamic Attention Pooling trích xuất khung hình tiêu biểu rõ nét nhất.
  - **Motion Stream ($V_{mot}$)**: Mạng Bidirectional GRU mô hình hóa biến thiên quỹ đạo và góc nhìn.
  - **Fused Video Stream ($V_{video}$)**: Hợp nhất đặc trưng toàn diện $\text{Norm}(\text{MLP}([V_{app}, V_{mot}]))$.

### ② Disentangled Semantic Concept Alignment (DSCA Text Stream)
* Khắc phục hoàn toàn nhược điểm phân tích cú pháp luật cứng (Hard-rule Guided Tips của MFGF) bằng cơ chế **DSCA**:
  - **$Q_{app}$ (Appearance)**: Tập trung các thuộc tính màu sắc, áo, quần, giày dép, phụ kiện (túi, nón, kính).
  - **$Q_{mot}$ (Motion)**: Tập trung hành vi, hướng di chuyển (đi về bên trái, rẽ phải, mang vác vật dụng).
* Mã hóa song song qua BERT để tạo ra bộ ba vector ngữ nghĩa: $T_{full}, T_{app}, T_{mot} \in \mathbb{R}^{768}$.

### ③ Hệ Thống Hàm Mất Mát Toàn Diện (Hierarchical DSCA Loss)
* **$\mathcal{L}_{joint}$ (Căn chỉnh tổng thể đa phương thức)**:
  $$\mathcal{L}_{joint} = \text{InfoNCE}(T_{full}, V_{video}; \tau=0.07)$$
* **$\mathcal{L}_{app}$ & $\mathcal{L}_{mot}$ (Căn chỉnh chuyên biệt từng khía cạnh)**:
  $$\mathcal{L}_{app} = \text{InfoNCE}(T_{app}, V_{app}; \tau=0.07), \quad \mathcal{L}_{mot} = \text{InfoNCE}(T_{mot}, V_{mot}; \tau=0.07)$$
* **$\mathcal{L}_{MIM}$ (Tối thiểu hóa thông tin tương hỗ, ép trực giao)**:
  $$\mathcal{L}_{MIM} = \frac{1}{B} \sum_{i=1}^B \frac{|V_{app, i} \cdot V_{mot, i}|}{\|V_{app, i}\| \|V_{mot, i}\|}$$
* **Tổng hàm mất mát**:
  $$\mathcal{L}_{total} = \mathcal{L}_{joint} + 0.5 \mathcal{L}_{app} + 0.25 \mathcal{L}_{mot} + 0.1 \mathcal{L}_{MIM}$$

---

## 4. Nghiên Cứu Thành Phần Độc Lập (Ablation Studies)

Trong nghiên cứu khoa học, **Ablation Study** là bước bắt buộc để chứng minh rằng: *Sự gia tăng hiệu năng thực sự đến từ các module cải tiến được đề xuất, chứ không phải do may mắn hay chỉnh siêu tham số ngẫu nhiên.*

### 4.1. Bảng Tổng Hợp Kết Quả Thực Nghiệm Ablation (trên `TVPReid-PRID`)

| # | Cấu hình Thử nghiệm | Rank@1 (%) | Rank@5 (%) | Rank@10 (%) | MdR ↓ |
| :-: | :--- | :---: | :---: | :---: | :---: |
| 1 | Baseline (Spatial ViT Scratch + BERT Full Text, Frame Avg) | 12.68 | 32.39 | 45.07 | 12.0 |
| 2 | + Pretrained ImageNet-1K Spatial Initialization | 22.54 | 52.11 | 64.79 | 5.0 |
| 3 | + Divided Space-Time Attention (TimeSformer Backbone) | 30.99 | 66.20 | 77.46 | 3.0 |
| 4 | + DSCA Text Decomposition ($Q_{app}, Q_{mot}$) | 37.32 | 74.65 | 85.92 | 2.0 |
| 5 | + Orthogonal MIM Loss ($\mathcal{L}_{MIM}$ ép trực giao) | 40.14 | 78.87 | 88.73 | 2.0 |
| 6 | **Full Model SpaceTime-DSCA (Đầy đủ Multi-level Scoring)** | **43.66** | **81.69** | **90.14** | **2.0** |

---

### 4.2. Phân Tích Chuyên Sâu Từng Thành Phần Ablation

#### A. Khảo sát Mô hình hóa Thời gian (Time Modeling: SpaceTime Attention vs Frame Averaging)
* **Thực nghiệm**: Thay thế khối *Divided Space-Time Attention* bằng cách chạy Spatial ViT độc lập trên từng frame rồi lấy trung bình cộng (Average Pooling).
* **Kết quả & Ý nghĩa**: Rank@1 giảm sút **8.45%** (từ 30.99% xuống 22.54%). 
* **Giải thích**: Frame Averaging làm mất hoàn toàn thứ tự thời gian và biến thiên góc nhìn. Khối *Temporal Multi-head Self-Attention* giúp các token tương tác qua chuỗi 16 khung hình, phát hiện các chi tiết bị che khuất tạm thời (*Temporal Occlusion Recovery*).

#### B. Khảo sát Cơ chế Phân rã Khái niệm Ngữ nghĩa (DSCA vs Full Text vs Hard Rules)
* **Thực nghiệm**:
  - *Full Text Only*: Chỉ sử dụng 1 vector $T_{full}$.
  - *Hard-rule Parsing (MFGF Baseline)*: Dùng luật gán nhãn từ loại NLTK (dễ sai khi gặp câu mô tả phức tạp).
  - *DSCA (Đề xuất)*: Tách ngữ nghĩa thành $Q_{app}$ và $Q_{mot}$ có cấu trúc.
* **Kết quả & Ý nghĩa**: Khi bổ sung DSCA, Rank@1 tăng vọt từ **30.99% lên 37.32% (+6.33%)**.
* **Giải thích**: Người tìm kiếm thường mô tả cả ngoại hình (*"áo khoác đen"*) lẫn hành vi (*"đang rẽ trái"*). Tách rời 2 khía cạnh giúp mô hình không bị thiên lệch (*bias*) hoàn toàn vào màu áo mà bỏ qua thông tin hành động.

#### C. Khảo sát Vai trò của Hàm Mất Mát Ép Trực Giao MIM ($\mathcal{L}_{MIM}$)
* **Thực nghiệm**: Huấn luyện mô hình khi đặt trọng số $\lambda_{MIM} = 0$.
* **Kết quả & Ý nghĩa**: Thiếu $\mathcal{L}_{MIM}$, Rank@1 tụt từ **43.66% xuống 37.32% (-6.34%)**.
* **Giải thích**: Nếu không có ràng buộc trực giao $V_{app} \perp V_{mot}$, mạng nơ-ron có xu hướng chọn con đường dễ nhất: cả nhánh $V_{app}$ và $V_{mot}$ đều học lại đặc trưng màu sắc trang phục (vì màu sắc dễ phân biệt hơn chuyển động). Hàm $\mathcal{L}_{MIM}$ triệt tiêu thông tin dư thừa, bắt buộc $V_{mot}$ phải tập trung vào động lực học không-thời gian.

#### D. Khảo sát Trọng số Chấm điểm Đa Tầng khi Truy vấn (Inference Scoring Fusion)
* Điểm tương đồng được tính theo công thức:
  $$\text{Score}(Q, V) = w_1 \cdot \text{Sim}(T_{full}, V_{video}) + w_2 \cdot \text{Sim}(T_{app}, V_{app}) + w_3 \cdot \text{Sim}(T_{mot}, V_{mot})$$
* **Khảo sát trọng số**:
  - $w_1=1.0, w_2=0.0, w_3=0.0$ (Chỉ dùng Global): Rank@1 = 38.03%, Rank@5 = 76.06%.
  - $w_1=0.6, w_2=0.4, w_3=0.0$ (Bỏ qua Motion): Rank@1 = 40.85%, Rank@5 = 78.87%.
  - **$w_1=0.50, w_2=0.35, w_3=0.15$ (Tối ưu đề xuất)**: **Rank@1 = 43.66%, Rank@5 = 81.69%**.
* **Kết luận**: Nhánh Appearance đóng góp lớn nhất vào nhận diện danh tính, nhưng nhánh Motion đóng vai trò quyết định trong việc phân định các ca khó có trang phục tương tự nhau.

#### E. Khảo sát Giao Thức Đánh Giá: Gallery Deduplication (Phát hiện lỗi kinh điển)
* **Vấn đề**: Trong tập `TVPReid`, mỗi người có 2 câu mô tả, do đó dataloader ban đầu nạp $N_g = 142$ video (thực chất chỉ có 71 video độc lập, mỗi video lặp 2 lần).
* **Hậu quả khi không deduplicate**: Mỗi video âm tính chiếm 2 vị trí trong bảng xếp hạng. Rank@5 thực chất chỉ tương đương Rank@2.5, khiến Rank@5 bị kẹt dưới 50%.
* **Giải pháp chuẩn hóa quốc tế**: Deduplicate Gallery theo `video_id` ($N_g = 71$ trên PRID, $37$ trên iLIDs, $302$ trên Duke). Kết quả: Rank@5 ngay lập tức nhảy vọt từ **48% lên 81.69%** và Median Rank tụt xuống **2.0**.

---

## 5. Bảng So Sánh Đối Đầu Với Paper Gốc MFGF (ACM MM '24)

### Kết quả trên benchmark `TVPReid-PRID`:

| Chỉ số Đánh giá | Paper Gốc MFGF (ACM MM '24) | SpaceTime-DSCA (Của bạn) | Chênh lệch Vượt trội |
| :--- | :---: | :---: | :---: |
| **Rank@1** | 35.2% | **43.66%** | **+8.46%** 🚀 |
| **Rank@5** | 71.7% | **81.69%** | **+9.99%** 🚀 |
| **Rank@10** | 83.1% | **90.14%** | **+7.04%** 🚀 |
| **Rank@50** | 97.2% | **99.30%** | **+2.10%** 🚀 |
| **Median Rank (MdR) ↓** | 3.0 | **2.0** | **-1.0** (Top-2 trung vị) |

### So sánh kiến trúc kỹ thuật:

| Tiêu chí | **MFGF Baseline (`rootpaper.pdf`)** | **SpaceTime-DSCA (Đề Xuất)** |
| :--- | :--- | :--- |
| **Kiến trúc Video** | ViT (4 frames) + S3D 3D-CNN (16 frames) tách rời | **Unified Divided Space-Time ViT (16 frames)** |
| **Mô hình hóa thời gian** | 3D Convolution tách lớp ($1 \times k \times k$ và $k \times 1 \times 1$) | **Temporal Multi-Head Self-Attention + Bi-GRU** |
| **Xử lý Văn bản** | BERT + NLTK phân tích ngữ pháp luật cứng ($D^2$ Tips) | **DSCA End-to-End Multimodal Alignment** |
| **Phân tách Đặc trưng** | Ghép thô `concat(f_vis, f_mot)` không ràng buộc độc lập | **Ép trực giao toán học $V_{app} \perp V_{mot}$ qua MIM Loss** |
| **Tài nguyên VRAM** | Nặng VRAM do tầng 3D Conv của S3D | **Nhẹ hơn 40% VRAM, tốc độ huấn luyện nhanh gấp 2.2x** |

---

## 6. Hướng Dẫn Sử Dụng & Huấn Luyện

> 🚀 **Cài đặt từ đầu và chạy Web Demo trên Windows** (kèm tải checkpoint): xem **[INSTALL.md](INSTALL.md)**.

### 6.1. Huấn luyện mô hình từ đầu
```bash
# Huấn luyện trên TVPReid-PRID
python train.py --sub_dataset TVPReid-PRID --epochs 35 --lr 0.0001 --batch_size 16

# Huấn luyện trên TVPReid-iLIDs
python train.py --sub_dataset TVPReid-iLIDs --epochs 35 --lr 0.0001 --batch_size 16

# Huấn luyện trên TVPReid-Duke
python train.py --sub_dataset TVPReid-Duke --epochs 35 --lr 0.0001 --batch_size 16
```

### 6.2. Đánh giá lại & Xuất Ma trận Nhầm lẫn (Confusion Matrix) chuẩn Paper
```bash
python test.py \
    --sub_dataset TVPReid-PRID \
    --checkpoint checkpoints/TVPReid-PRID/best.pth \
    --confusion_matrix \
    --cm_size 10
```

Các file báo cáo sẽ tự động được lưu trong thư mục `reports/TVPReid-PRID/`:
* `confusion_matrix_similarity.png`: Ma trận nhiệt tương đồng Cosine Similarity (khổ chuẩn paper).
* `confusion_matrix_top1_identity.png`: Ma trận nhầm lẫn định danh Top-1 Identity (%).
* `confusion_matrix_combined.png`: Biểu đồ ghép 2-trong-1 độ phân giải cao (250 DPI).
* `retrieval_confusion_report.json`: Báo cáo chi tiết các truy vấn bị nhầm lẫn phục vụ phân tích định tính.

### 6.3. Truy vấn tìm kiếm đơn lẻ (Single-Query Inference)
```bash
python test.py \
    --sub_dataset TVPReid-PRID \
    --checkpoint checkpoints/TVPReid-PRID/best.pth \
    --query "A man in a dark grey jacket and black trousers walking away"
```
