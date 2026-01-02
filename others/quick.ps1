# Quick test run to verify transformer learning fix
# This is a SHORT run to quickly verify the fix is working

Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host "QUICK TEST - VERIFY TRANSFORMER LEARNING FIX" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host ""

Write-Host "⚡ This is a QUICK TEST run (12 epochs)" -ForegroundColor Yellow
Write-Host "   Purpose: Verify that transformer probabilities are changing" -ForegroundColor Yellow
Write-Host ""

Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  - transformer_init_bias: 0.3" -ForegroundColor White
Write-Host "  - transformer_warmup_epochs: 3" -ForegroundColor White
Write-Host "  - arch_after: 3 (start ASAP)" -ForegroundColor White
Write-Host "  - epochs: 12 (QUICK TEST)" -ForegroundColor White
Write-Host ""

Write-Host "What to look for:" -ForegroundColor Cyan
Write-Host "  1. Initial TransON: ~[0.646, 0.646, 0.646, 0.646]" -ForegroundColor Green
Write-Host "  2. TransGrad: NON-ZERO values (e.g., 0.000312)" -ForegroundColor Green
Write-Host "  3. TransON CHANGES after epoch 3+ (even small changes OK)" -ForegroundColor Green
Write-Host "  4. No errors during backward pass" -ForegroundColor Green
Write-Host ""

Write-Host "Timeline:" -ForegroundColor Cyan
Write-Host "  Epoch 0-2:  Weight only" -ForegroundColor Gray
Write-Host "  Epoch 3-11: Weight + Arch (9 epochs of learning)" -ForegroundColor Gray
Write-Host ""

Write-Host "After this test:" -ForegroundColor Magenta
Write-Host "  ✓ If TransON changes → Run full training (run.ps1)" -ForegroundColor Green
Write-Host "  ✗ If TransON stays 0.646 → Check BUG_FIX_TRANSFORMER_LEARNING.md" -ForegroundColor Red
Write-Host ""

Write-Host "Press any key to start quick test..." -ForegroundColor Magenta
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

python .\hct_net\train_CVCDataset.py `
    --resume None `
    --dataset cvc `
    --epochs 12 `
    --model UnetLayer9 `
    --layers 9 `
    --arch_after 3 `
    --transformer_warmup_epochs 3 `
    --transformer_init_bias 0.3 `
    --fix_transformer_arch False `
    --transformer_connection_weight 1.0 `

Write-Host ""
Write-Host ("=" * 80) -ForegroundColor Green
Write-Host "TRAINING COMPLETED" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Green
Write-Host ""