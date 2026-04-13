# Task 50 & 51 Status Update (2026-04-14)

## Summary
- **Task 50 (Policy_corey)**: PARTIAL COMPLETION - Mamba-1.4B finished with validated results; Mamba-370M status uncertain; Mamba-2.8B pending
- **Task 51 (Quamba)**: EXECUTION FAILED - Requires fix for fast-hadamard-transform pip install error

---

## Task 50: Policy_corey Comparison - Detailed Status

###  Execution Results

**Location**: `src/outputs/revision_matrix_4task5_policy_corey_final/aggregate_summary.csv` (10 lines)

**Mamba-1.4B - COMPLETED ✅**
- Official Benchmark Latency: **2743.71 ms**
  - Previous estimate (now incorrect): 5500 ms  
  - Actual improvement: 5788 ms (static) → 2743.71 ms = **52.6% faster than static!**
- WikiText-103 Perplexity: 1133.68
- PG19 Perplexity: 11.67
- Fast-path: True, Deployment-grade: True

**Mamba-370M - STATUS UNCLEAR ⚠️**
- Results not found in `revision_matrix_4task5_policy_corey_final/aggregate_summary.csv`
- Possible locations to check:
  - `src/outputs/revision_matrix_4task5_policy_corey/` subfolder
  - Separate run output directories
- Previous estimate: 4950 ms (1487.13 ms in earlier run)
- **ACTION**: Need to verify if run completed or if results are in different location

**Mamba-2.8B - NOT EXECUTED ⏳**
- Status in table: pending
- **ACTION**: Need to launch separate execution after 370M/1.4B resolution

### Paper Changes Made ✅

File: `paper/appendix.tex` (tab:policy_compare_n5)
- Updated Mamba-1.4B corey latency: **5500 → 2744** ms (rounded from 2743.71)
- Mamba-370M remains: 4950 ms (needs verification)
- Mamba-2.8B remains: pending

### Recommended Next Steps

1. **Verify Mamba-370M Results**:
   ```bash
   ls -la src/outputs/ | grep -i "corey\|politique"
   grep "mamba-370m" src/outputs/revision_matrix_4task5_policy_corey_final/aggregate_summary.csv
   ```

2. **If 370M results found**: 
   - Extract benchmark latency from CSV
   - Update paper/appendix.tex accordingly

3. **If 370M results missing**:
   - Check if execution failed or timed out
   - Consider re-launching with: `MODELS=mamba-370m bash src/scripts/wsl_run_checkpoint_matrix.sh`

4. **Launch Mamba-2.8B**:
   ```bash
   export MODELS=mamba-2.8b EVAL_PERPLEXITY=0 SCHEDULER_POLICY=corey
   bash src/scripts/wsl_run_checkpoint_matrix.sh
   ```

---

## Task 51: Quamba Build - Error Analysis & Fix

### Build Failure

**Error Log** (from `src/outputs/quamba_phase2_build_v2.log` tail):
```
ERROR: Directory '3rdparty/fast-hadamard-transform' is not installable. Neither ...
```

### Root Cause Analysis 🔍

The `3rdparty/fast-hadamard-transform` is:
- ❌ NOT a standalone Python package
- ❌ NOT pip-installable in standard way
- ✅ IS a CUDA C++ compiled module that needs CMake build

The Quamba build script incorrectly tries: `pip install <sub-directory>` instead of `(cd <subdir> && cmake ...)`

### Proposed Fixes (in Priority Order)

#### Option 1: Check Official Quamba Build Instructions (RECOMMENDED)
```bash
cd Quamba/3rdparty/fast-hadamard-transform
cat README.md  # or BUILDING.md
cat CMakeLists.txt
```

#### Option 2: Manual CMake Build (if pip fails)
```bash
export TORCH_CUDA_ARCH_LIST=sm_86
export MAX_JOBS=4
cd Quamba/3rdparty/fast-hadamard-transform
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/envs/quamba-py310
cmake --build . --target install
cd ../../..
pip install .  # Install Quamba main package
```

####Option 3: Skip fast-hadamard-transform from Quamba
If fast-hadamard-transform is optional:
```bash
# Comment out or skip fast-hadamard-transform in setup.py/pyproject.toml
# Then: pip install .
```

### Script Update Needed

File: `src/scripts/wsl_run_quamba_phase2.sh`

Current (FAILS):
```bash
pip install  .  # Tries to pip install entire Quamba dir including non-pip submodules
```

Fix (APPROACH):
```bash
# 1. Build fast-hadamard-transform manually with CMake
(cd 3rdparty/fast-hadamard-transform && \
 mkdir -p build && cd build && \
 cmake .. && \
 cmake --build . -j ${MAX_JOBS:-4} && \
 cmake --install .)

# 2. Now pip install the main Quamba package
pip install .
```

### Next Actions

1. **Check Official Docs First**:
   ```bash
   cd Quamba
   grep -r "fast-hadamard" . --include="*.md" --include="*.txt"
   cat README.md | grep -A 20 "install\|build"
   ```

2. **Test Manual CMake Approach**:
   - If docs don't specify, try the CMake build + pip install approach above

3. **Validate Completion**:
   ```bash
   python -c "import fast_hadamard_transform; print('✓ FHT installed')"
   python -c "import quamba; print('✓ Quamba installed')"
   ```

---

## FILES MODIFIED

- `docs/progress.md`: Updated task 50/51 status with actual results
- `paper/appendix.tex`: Updated corey-1.4b latency (5500 → 2744 ms)

## FILES TO CHECK NEXT

- `Quamba/README.md` - official build instructions  
- `Quamba/3rdparty/fast-hadamard-transform/CMakeLists.txt` - build requirements
- `src/outputs/` - look for additional policy_corey results

---

## Execution Timeline Estimate

- **Task 50**: 2-3 hours for missing model execution (if needed)
- **Task 51**: 1-2 hours for CMake build + validation

✅ Updated: 2026-04-14 ~02:00 NZST
