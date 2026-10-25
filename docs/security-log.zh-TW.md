# 安全性紀錄

[English](security-log.md) | 繁體中文

- 日期: 2026-10-25
- 分支: `fix/security-upgrades`
- 範圍: `requirements.txt`、`groundingdino/`

## 摘要

gradio 與 inference 升級到已修補且可安裝的版本，transformers 升到 5.x，並修正內含的 GroundingDINO 程式以相容。pillow、filelock、wheel 因為 gradio 與 inference 限制版本，仍有未修正的漏洞。

## 修改內容

| 項目 | 修改前 | 修改後 | 原因 |
|---|---|---|---|
| gradio | 4.19.2 | 6.20.0 | 任意檔案讀取、SSRF、路徑穿越漏洞（6.16 修正） |
| inference[yolo-world] | 0.9.13 | 1.8.2 | 0.9.13 已無法安裝：依賴的 inference-cli 已從 PyPI 撤下 |
| transformers | unpinned | >=5.5.0 | CVE-2026-5241 與更早的遠端執行程式碼、ReDoS 漏洞 |
| `groundingdino/models/GroundingDINO/bertwarper.py` | - | local fallbacks | transformers 5 從 `BertModel` 移除了 `get_extended_attention_mask`、`invert_attention_mask`、`get_head_mask` |

## 使用前須知

- gradio 固定在 6.20.0：更新的 6.x 版本與 inference 1.8.2 要求的 huggingface-hub 版本衝突。
- inference 1.8.2 比 0.9.13 新；原本的註解提到 0.9.13 在 `yolo_world/l` 稍好，因此偵測結果可能略有差異。

## 尚未解決的問題

### 被上游限制的套件版本

| 套件 | 版本 | 漏洞 | 無法修正的原因 | 影響與因應 |
|---|---|---|---|---|
| pillow | 12.3.0 | 41 筆（13.0.0 修正） | gradio 6.16 以上要求 `pillow<13`，目前沒有允許 Pillow 13 的 gradio 版本。 | 解碼惡意製作的圖片（例如 TIFF、PDF）可能讓程式當掉或越界讀取記憶體。請只在本機執行 Gradio 介面（不要開 `share=True`、不要對外開放連接埠），只開啟可信任的圖片；等 gradio 允許 Pillow 13 後再升級。 |
| filelock | 3.32.7 | 3 筆（4.2.1 修正） | inference 1.8.2 要求 `filelock<=4.0.0`。 | Windows 上的 `SoftFileLock` 與 SQLite 版 `ReadWriteLock` 的鎖定處理有缺陷。這裡只用於 Hugging Face 模型快取，單人本機使用的風險低。 |
| wheel | 0.45.1 | 2 筆（0.46.2 修正） | inference 1.8.2 要求 `wheel<=0.45.1`。 | 用 `wheel unpack` 解開惡意製作的 wheel 時可能路徑穿越或更改檔案權限。本程式執行時不會解開 wheel。 |

## 驗證方式

- 以 osv-scanner 掃描解析後的 `requirements.txt`，只剩上表的套件。
- CPU、torch 2.14.1、transformers 5.20.2：GroundingDINO 能依設定建立模型，`predict()`、`annotate()` 可正常執行。
- 把模型模組換成假模組後，Gradio 介面能正常建立並提供服務。

## 未驗證項目

- 用 GPU 與實際權重執行推論（測試環境沒有 GPU 與權重檔）。
