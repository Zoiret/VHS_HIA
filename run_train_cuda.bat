@echo off
setlocal

cd /d %~dp0

if exist ".venv\\Scripts\\activate.bat" (
  call ".venv\\Scripts\\activate.bat"
)

set TORCH_HOME=.cache\\torch
set HF_HOME=.cache\\huggingface
set XDG_CACHE_HOME=.cache
set MPLCONFIGDIR=.cache\\matplotlib

python training\\train.py --config training\\configs\\unetpp_effb3_cuda_multiclass_full_aug_100ep.yaml
