# So sánh các cấu hình training

## 📊 So sánh Timeline Training

### ❌ CẤU HÌNH CŨ (Trong file run.ps1 ban đầu)

```
Thông số:
  - epochs: 20
  - arch_after: 10
  - transformer_warmup_epochs: 5
```

**Timeline:**
```
Epoch 0-9:   ❌ Weight training only (NO architecture learning)
             → Warmup ends at epoch 5, but arch doesn't start until 10
             → WASTE 4 epochs (epoch 6-9) doing nothing!
             
Epoch 10-19: ✓ Weight + Arch training (only 10 epochs for learning)
             → Too few epochs to converge
```

**Vấn đề:**
- ❌ Lãng phí 4 epochs (6-9) sau warmup
- ❌ Chỉ 10 epochs để học architecture (quá ít!)
- ❌ Transformers chưa kịp hội tụ

---

### ⚠️ CẤU HÌNH NHANH (run_quick_test.ps1) - CHỈ ĐỂ TEST

```
Thông số:
  - epochs: 12
  - arch_after: 3
  - transformer_warmup_epochs: 3
```

**Timeline:**
```
Epoch 0-2:  Weight training only (warmup model)
Epoch 3-11: Weight + Arch training (9 epochs)
```

**Mục đích:**
- ✓ Verify transformer learning fix works
- ✓ Quick feedback (12 epochs ~1-2 hours)
- ⚠️ KHÔNG đủ cho final model (chỉ để test)

---

### ✅ CẤU HÌNH TỐT (run.ps1 - ĐÃ SỬA)

```
Thông số:
  - epochs: 50
  - arch_after: 5
  - transformer_warmup_epochs: 5
```

**Timeline:**
```
Epoch 0-4:   Weight training only (warmup model baseline)
Epoch 5-9:   Weight + Arch training (transformer warmup phase)
             → Transformers explore with warmup bonus
             
Epoch 10-49: Weight + Arch training (full learning - 40 epochs!)
             → Enough time for convergence
             → Transformers learn optimal configuration
```

**Ưu điểm:**
- ✅ Không lãng phí epoch nào
- ✅ 45 epochs cho architecture learning (đủ để hội tụ)
- ✅ Warmup smooth, transition tốt
- ✅ Transformers có đủ thời gian để học

---

## 📈 So sánh số lượng epochs học

| Cấu hình | Tổng epochs | Arch learning epochs | % thời gian học | Đánh giá |
|----------|-------------|---------------------|----------------|----------|
| Cũ (20 epochs) | 20 | 10 | 50% | ❌ Quá ít |
| Quick test | 12 | 9 | 75% | ⚠️ Chỉ để test |
| **Tốt (50 epochs)** | **50** | **45** | **90%** | ✅ **Khuyến nghị** |

---

## 🎯 Khuyến nghị sử dụng

### 1. Lần đầu sau fix → Chạy QUICK TEST
```powershell
.\run_quick_test.ps1
```
**Mục đích:** Verify fix works (TransON thay đổi, gradient flow)  
**Thời gian:** ~1-2 giờ  
**Kết quả mong đợi:**
```
Epoch 3:  TransON: ['0.646', '0.646', '0.646', '0.646'] TransGrad: 0.000312
Epoch 5:  TransON: ['0.649', '0.643', '0.651', '0.645'] TransGrad: 0.000287
Epoch 8:  TransON: ['0.653', '0.639', '0.657', '0.644'] TransGrad: 0.000301
Epoch 11: TransON: ['0.658', '0.632', '0.664', '0.641'] TransGrad: 0.000295
```
→ Nếu thấy TransON thay đổi → ✅ Fix thành công!

### 2. Sau khi verify → Chạy FULL TRAINING
```powershell
.\run.ps1
```
**Mục đích:** Train model hoàn chỉnh  
**Thời gian:** ~5-8 giờ  
**Kết quả mong đợi:**
- Architecture converges sau ~30-40 epochs
- Transformers quyết định ON/OFF rõ ràng (0.9+ hoặc 0.1-)
- Validation performance tốt

---

## ⚙️ Tùy chỉnh nâng cao

### Nếu muốn training NHANH hơn nhưng vẫn đủ:
```powershell
--epochs 30
--arch_after 5
--transformer_warmup_epochs 5
# → 25 epochs arch learning (acceptable)
```

### Nếu muốn training CHẬM hơn nhưng chắc chắn hội tụ:
```powershell
--epochs 100
--arch_after 5
--transformer_warmup_epochs 10
# → 90 epochs arch learning (overkill nhưng safe)
```

### Nếu dataset LỚN (như ISIC2018):
```powershell
--epochs 80
--arch_after 10
--transformer_warmup_epochs 10
# → 70 epochs arch learning
```

---

## 🔍 Cách kiểm tra sau training

### 1. Verify gradient flow
```bash
grep "TransGrad" search_exp/*/run.log
```
Nên thấy: `TransGrad: 0.000XXX` (non-zero)

### 2. Verify probability changes
```bash
python verify_fix.py search_exp/UnetLayer9/cvc/*/run.log
```
Nên thấy: "✅ THÀNH CÔNG: Transformer connections đang học!"

### 3. Check convergence
```bash
grep "TransON" search_exp/*/run.log | tail -10
```
Nên thấy probabilities rõ ràng (0.8+ hoặc 0.2-), không còn ~0.5

---

## 📝 Tóm tắt

| Aspect | Quick Test | Full Training |
|--------|-----------|---------------|
| **Script** | `run_quick_test.ps1` | `run.ps1` |
| **Mục đích** | Verify fix | Train final model |
| **Epochs** | 12 | 50 |
| **Thời gian** | 1-2h | 5-8h |
| **Khi nào dùng** | Lần đầu sau fix | Sau khi verify OK |
| **Kết quả** | Xác nhận gradient flow | Model hoàn chỉnh |
