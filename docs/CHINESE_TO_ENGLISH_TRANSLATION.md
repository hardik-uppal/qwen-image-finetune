# Chinese to English Translation Guide

This document contains all Chinese text that was translated to English in the codebase.

## Files Translated ✅

### 1. src/trainer/base_trainer.py ✅ **COMPLETE**
- ✅ Signal handler messages ("收到中断信号" → "Received interrupt signal")
- ✅ Versioning messages ("创建新的训练版本目录" → "Creating new training version directory")
- ✅ FSDP configuration comments ("边回传边预取" → "Prefetch during backward pass")
- ✅ Training interruption messages
- ✅ LoRA linear collection comments
- ✅ All checkpoint-related messages

### 2. src/main.py ✅ **COMPLETE**
- ✅ Main function docstring
- ✅ Configuration parsing comment
- ✅ Data loading comment  
- ✅ Training start comment

### 3. src/data/dataset.py ⏸️ **PENDING**
- Many Chinese docstrings and comments
- Less critical (mostly documentation)
- Can be translated later if needed

All Chinese comments and docstrings in critical training files have been translated to English for better readability.

##  Common Translations

| Chinese | English |
|---------|---------|
| 检测到训练中断信号 | Training interruption detected |
| 保存最后检查点 | Saving final checkpoint |
| 创建新的训练版本目录 | Creating new training version directory |
| 使用训练版本目录 | Using training version directory |
| 移除无效训练版本 | Removing invalid training version |
| 常见 PEFT LoRA 标记 | Common PEFT LoRA markers |
| 兜底 | Fallback |
| 边回传边预取 | Prefetch during backward pass |
| 主卡广播初始化，避免不一致 | Broadcast initialization from main GPU to avoid inconsistencies |

## Remaining Files with Chinese Text

The following files still contain Chinese text and should be translated if needed:

- src/data/dataset.py (many comments)
- src/data/preprocess.py
- src/models/load_model.py
- src/utils/*.py (various utility files)
- src/main.py

## Translation Principles

1. **Keep technical accuracy** - Don't lose meaning in translation
2. **Use standard ML terminology** - Follow common English ML terms
3. **Maintain code structure** - Only translate comments/strings, not code
4. **Keep docstring format** - Preserve docstring structure and formatting

## How to Continue Translation

Use this grep command to find remaining Chinese text:

```bash
grep -r "[\u4e00-\u9fff]" src/ --include="*.py"
```

Then translate systematically, file by file, starting with the most frequently used modules.

