# Run training with new transformer learning settings
# This script uses the recommended parameters for transformer exploration

Write-Host "="*80 -ForegroundColor Green
Write-Host "STARTING TRANSFORMER LEARNING EXPERIMENT" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green
Write-Host ""

Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  - transformer_init_bias: 0.3 (64.6% initial ON probability)" -ForegroundColor Yellow
Write-Host "  - fix_transformer_arch: False (let transformers learn)" -ForegroundColor Yellow
Write-Host "  - transformer_warmup_epochs: 5" -ForegroundColor Yellow
Write-Host "  - Total epochs: 50 (RECOMMENDED)" -ForegroundColor Yellow
Write-Host "  - arch_after: 5 (start architecture training after warmup)" -ForegroundColor Yellow
Write-Host ""

Write-Host "Expected behavior:" -ForegroundColor Cyan
Write-Host "  ✓ Transformers start with ~65% ON probability" -ForegroundColor Green
Write-Host "  ✓ Transformers are NOT fixed early" -ForegroundColor Green
Write-Host "  ✓ Transformers can learn from performance" -ForegroundColor Green
Write-Host "  ✓ Monitor TransGrad for gradient flow" -ForegroundColor Green
Write-Host "  ✓ Sufficient epochs (45 epochs of arch learning)" -ForegroundColor Green
Write-Host ""

Write-Host "Timeline:" -ForegroundColor Cyan
Write-Host "  Epoch 0-4:  Weight training only (warmup)" -ForegroundColor Gray
Write-Host "  Epoch 5-9:  Arch training during transformer warmup" -ForegroundColor Gray
Write-Host "  Epoch 10-49: Full architecture + weight learning (40 epochs)" -ForegroundColor Gray
Write-Host ""

Write-Host "Press any key to start training..." -ForegroundColor Magenta
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

python .\hct_net\train_CVCDataset.py `
    --resume None `
    --dataset cvc `
    --epochs 50 `
    --model UnetLayer9 `
    --layers 9 `
    --arch_after 5 `
    --transformer_warmup_epochs 5 `
    --transformer_init_bias 0.3 `
    --fix_transformer_arch False `
    --transformer_connection_weight 1.0 `
    --complexity_weight 0.2 `
    --arch_complexity_weight 0.1 `

Write-Host ""
Write-Host "="*80 -ForegroundColor Green
Write-Host "TRAINING COMPLETED" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Green
Write-Host ""
Write-Host "Check logs for:" -ForegroundColor Cyan
Write-Host "  1. Initial ON probabilities (~0.646)" -ForegroundColor Yellow
Write-Host "  2. TransGrad values (should be non-zero)" -ForegroundColor Yellow
Write-Host "  3. TransON probabilities changing over epochs" -ForegroundColor Yellow
Write-Host "  4. [LEARNING] status (no [FIXED] before epoch 15)" -ForegroundColor Yellow
