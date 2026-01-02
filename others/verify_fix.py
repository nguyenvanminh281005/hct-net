"""
Script để verify rằng transformer connections đang học được
Chạy script này sau khi training 1 epoch để kiểm tra
"""

import torch
import re
import sys

def check_training_log(log_path):
    """Kiểm tra log file để verify transformer learning"""
    print("=" * 80)
    print("VERIFYING TRANSFORMER LEARNING FROM LOG")
    print("=" * 80)
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Tìm tất cả TransON values
    trans_on_pattern = r"TransON: \[(.*?)\]"
    matches = re.findall(trans_on_pattern, content)
    
    if not matches:
        print("❌ Không tìm thấy TransON trong log!")
        return False
    
    print(f"\n📊 Tìm thấy {len(matches)} epochs với TransON data\n")
    
    # Parse các giá trị
    epochs_data = []
    for i, match in enumerate(matches[:min(20, len(matches))]):  # Lấy 20 epochs đầu
        values = [float(v.strip().strip("'")) for v in match.split(',')]
        epochs_data.append((i, values))
        print(f"Epoch {i:2d}: {[f'{v:.4f}' for v in values]}")
    
    # Kiểm tra xem có thay đổi không
    print("\n" + "=" * 80)
    print("PHÂN TÍCH KẾT QUẢ")
    print("=" * 80)
    
    if len(epochs_data) < 2:
        print("⚠️  Chưa đủ data để phân tích (cần ít nhất 2 epochs)")
        return False
    
    first_epoch = epochs_data[0][1]
    last_epoch = epochs_data[-1][1]
    
    # Tính độ thay đổi
    max_change = max(abs(first_epoch[i] - last_epoch[i]) for i in range(len(first_epoch)))
    
    print(f"\n🔍 Giá trị đầu tiên (Epoch 0): {[f'{v:.4f}' for v in first_epoch]}")
    print(f"🔍 Giá trị cuối cùng (Epoch {len(epochs_data)-1}): {[f'{v:.4f}' for v in last_epoch]}")
    print(f"\n📈 Độ thay đổi lớn nhất: {max_change:.6f}")
    
    # Đánh giá
    if max_change < 0.001:
        print("\n❌ THẤT BẠI: Giá trị hầu như không đổi!")
        print("   → Transformer connections CHƯA học được")
        print("   → Vui lòng kiểm tra lại code có a_loss.backward()")
        return False
    elif max_change < 0.01:
        print("\n⚠️  CẢNH BÁO: Giá trị thay đổi rất nhỏ")
        print("   → Có thể learning rate quá thấp")
        print("   → Hoặc cần training thêm epochs")
        return True
    else:
        print("\n✅ THÀNH CÔNG: Transformer connections đang học!")
        print("   → Giá trị thay đổi theo thời gian")
        print("   → Fix đã hoạt động đúng!")
        return True

def check_gradient_in_code():
    """Kiểm tra code có a_loss.backward() không"""
    print("\n" + "=" * 80)
    print("CHECKING CODE FOR FIXES")
    print("=" * 80)
    
    train_file = "hct_net/train_CVCDataset.py"
    
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check 1: a_loss.backward()
        if "a_loss.backward()" in content:
            print("✅ Found: a_loss.backward()")
        else:
            print("❌ MISSING: a_loss.backward()")
            print("   → Cần thêm a_loss.backward() trước optimizer_arch.step()")
        
        # Check 2: alphas_transformer_connections.grad initialization
        if "alphas_transformer_connections.grad = torch.zeros_like" in content:
            print("✅ Found: alphas_transformer_connections.grad initialization")
        else:
            print("❌ MISSING: alphas_transformer_connections.grad initialization")
            print("   → Cần thêm gradient initialization cho transformer alphas")
        
        # Check 3: transformer_loss in a_loss computation
        if "transformer_loss" in content and "a_loss = a_loss +" in content:
            print("✅ Found: transformer_loss được thêm vào a_loss")
        else:
            print("⚠️  WARNING: transformer_loss có thể chưa được thêm vào a_loss")
        
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {train_file}")
        return False
    
    return True

def main():
    print("\n" + "=" * 80)
    print(" TRANSFORMER LEARNING VERIFICATION TOOL")
    print("=" * 80)
    
    # Check code
    code_ok = check_gradient_in_code()
    
    # Check log nếu có
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
        print(f"\n📁 Checking log file: {log_path}")
        log_ok = check_training_log(log_path)
        
        print("\n" + "=" * 80)
        print("FINAL VERDICT")
        print("=" * 80)
        
        if code_ok and log_ok:
            print("✅ ✅ ✅  ALL CHECKS PASSED!")
            print("Transformer connections đang học đúng cách!")
        elif code_ok and not log_ok:
            print("⚠️  Code đã được fix nhưng training chưa thành công")
            print("Hãy training thêm vài epochs và kiểm tra lại")
        else:
            print("❌ Vẫn còn vấn đề cần fix!")
            print("Vui lòng xem hướng dẫn trong BUG_FIX_TRANSFORMER_LEARNING.md")
    else:
        print("\n💡 Tip: Chạy với log file để kiểm tra kết quả training:")
        print("   python verify_fix.py path/to/run.log")

if __name__ == "__main__":
    main()
