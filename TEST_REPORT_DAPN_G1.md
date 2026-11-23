# Báo cáo kiểm thử CCVC-DAPN — Giai đoạn 1

Ngày 23/11/2026 · CPU (torch 2.13.0+cpu, numpy 2.2.6, python 3.12.9), dữ liệu giả (16 train / 4 val, crop 64).

Lệnh: `bash tools/tests_dapn/run_cpu_tests_dapn.sh <thư mục tạm>` → **48 passed, 0 failed** (trong đó `test_dapn_unit.py`: 30 kiểm tra, 0 failed).

Chưa có trong giai đoạn 1 (để giai đoạn 2): test 13 (`diagnose_entanglement.py`), test 17 (`--dapg-pos c4` chạy vài iteration; code đã có, chưa kiểm thử).

## Đầu ra verify_origin

```
verify_origin (d:\Claude\Semi-CCVC\CCVC-DAPN)
  step           result detail
  1 integrity    PASS  78 origin files, all identical to b44dc57
verify_origin: PASS
  step           result detail
  2 behaviour    PASS  CCVC_aug.py: 5 iterations, losses and weights bit-identical to the golden file (torch 2.13.0+cpu, numpy 2.2.6, python 3.12.9)
  3 superset     PASS  CCVC_aug_dapn.py --use_dapg False == golden file (losses and weights bit-identical)
verify_origin: PASS
```

## Kết quả từng kiểm tra

```
=== 0. synthetic data ===
PASS  fake data: 16 train / 4 val
=== 1. verify_origin step 1 (integrity); one byte changed in a copy -> FAIL ===
PASS  integrity PASS on this repository
PASS  modified origin file -> FAIL (exit 1)
PASS  the modified file is named
PASS  missing origin file -> FAIL and named
=== 2/4. golden (from tag ccvc-origin): CCVC_aug.py and CCVC_aug_dapn.py --use_dapg False ===
PASS  step 2: CCVC_aug.py == golden (losses of 5 iterations + weights)
PASS  step 3: CCVC_aug_dapn.py --use_dapg False == golden
=== 5. initialisation: ccvc_dapn and CCVC_aug.py, same seed -> origin part bit-identical ===
PASS  origin-part checksum identical; DAPN model holds da1 + da2
=== 6. check_dapn_diff ===
PASS  CCVC_aug_dapn.py: every difference marked [DAPN]
PASS  one unmarked line -> FAIL
=== 7. DAPN copy == original DAPN files (main_final 33ad638) ===
PASS  DAPN copy == original (assign/proto/quality/loss/grad), max abs diff 0
PASS  assign / proto / quality / loss / gradients within 1e-6
=== 8-12. unit tests (presets, both sub-networks, gradients, optimizer, masks, fp32, evaluation) ===
PASS  ccvc_dapn: da1 on branch 1 and da2 on branch 2
PASS  ccvc_dapn: checkpoint (state_dict) holds da1.* and da2.* (['da1', 'da2'])
PASS  da1 / da2: separate prototype sets (64 x 304 each)
PASS  ccvc_dapn: DA called on 3 passes x 2 branches ([('l', 1), ('l', 2), ('u_strong', 1), ('u_strong', 2), ('u_raw', 1), ('u_raw', 2)])
PASS  ccvc_dapn: L^l > 0 and L^u > 0 (L^l 0.7183 L^u 0.7280)
PASS  ccvc_dapn: L_dapg_b1 > 0 and L_dapg_b2 > 0 (labeled and unlabeled passes) ({'l': 0.718274712562561, 'u_strong': 0.7300077080726624, 'u_raw': 0.7260649800300598})
PASS  L_DAPG = 1/2 (L^l + L^u), L^u = 1/2 (strong + raw)
PASS  DAPG of branch 1 reaches branch-1 parameters
PASS  DAPG of branch 1 does not touch branch-2 parameters
PASS  DAPG of branch 2 reaches branch-2 parameters
PASS  DAPG of branch 2 does not touch branch-1 parameters
PASS  DAPG of branch 1 does not touch da2; branch 2 does not touch da1
PASS  da1.prototypes and da1.quality_net receive gradients
PASS  da2.prototypes and da2.quality_net receive gradients
PASS  optimizer code of CCVC_aug_dapn.py == origin CCVC_aug.py
PASS  optimizer: every da1/da2 parameter in group 2 (lr base x lr_network = x10) (lr 0.01)
PASS  ccvc_dapn_l: L^u = 0 and DA not called on unlabeled passes ([('l', 1), ('l', 2)])
PASS  ccvc_dapn_b1: da1 only, no da2
PASS  ccvc_dapn_b1: DA only on branch 1
PASS  ccvc_dapn_off: no DA module
PASS  mask: pixels labelled 255 (padding / unknown) excluded (380 of 512)
PASS  mask all 255 -> pass skipped, loss 0, no error
PASS  skipped pass keeps a (zero) gradient on every DA parameter (DDP)
PASS  NaN feature -> loss 0, counted as nan skip
PASS  DAPG computed in fp32 under autocast (half input) (torch.float32)
PASS  valid.evaluate and evaluate_dapn: same three mIoU and conflict ratio (3.2192 2.3776 4.2080)
PASS  agree_wrong + agree_right + conflict = 1 (1.000000000000)
PASS  err_b1 = 1 - acc_all of branch 1
PASS  confusion_b1: 8x8, total = valid pixels, diagonal = correct pixels
PASS  confusion_agree_wrong: zero diagonal, total = agree_wrong pixels
PASS  test_dapn_unit.py: 0 failed
=== 3. origin path imports no DAPN file (sys.modules at exit of every python process) ===
PASS  run_baseline.sh --dryrun exit 0
PASS  run_baseline.sh: training process ran (module log written)
PASS  run_baseline.sh: no *dapn* module imported []
PASS  run_dapn.sh --method ccvc_origin --dryrun exit 0
PASS  ccvc_origin: training script is CCVC_aug.py
PASS  ccvc_origin: CCVC_aug.py process imports no *dapn* module
PASS  control: ccvc_dapn process does import model.dapn / valid_dapn / run_utils_dapn
=== full runs (1 epoch): run_baseline.sh ccvc_aug and run_dapn.sh ccvc_dapn, split 1/4 ===
PASS  run_baseline.sh split 1/4 exit 0
PASS  run_dapn.sh split 1/4 (ccvc_dapn) exit 0
PASS  DAPN on both sub-networks end to end: final.pth has da1.* and da2.*; train_log has b1 and b2 columns filled
PASS  DAPN outputs: eval_log agreement columns, final_eval agreement metrics, confusion CSVs, config keys
=== 12. eval_ckpt.py: origin and DAPN checkpoints reproduce final_eval.json ===
PASS  origin checkpoint: three mIoU = final_eval.json (+-0.01)
PASS  origin checkpoint recognised as origin
PASS  DAPN checkpoint: three mIoU = final_eval.json (+-0.01)
PASS  DAPN checkpoint: da1, da2 ignored at evaluation
PASS  eval_ckpt agreement metrics = final_eval.json of the DAPN run
=== 14. output isolation and restore_output_dapn.py ===
PASS  origin results only under exp/oem, DAPN results only under exp/oem_dapn
PASS  restore: exit 0
PASS  restore: DAPN run restored
PASS  restore: origin results NOT copied
PASS  restore: warning about the origin results
=== 15. partitions vs the baseline run (--baseline-part-root) ===
PASS  identical split -> run (exit 0)
PASS  logged: identical to baseline
PASS  one line different -> stop (exit 2)
PASS  message: split differs from the baseline run
=== 16. resume ccvc_dapn (2 epochs, session limit after epoch 1) ===
PASS  stopped for resume (exit 3), last.pth written
PASS  last.pth holds da1.* and da2.*
PASS  w_dapg changed -> resume blocked
PASS  same command -> resumes and finishes (exit 0)
PASS  resumed at epoch 1 with da* restored from last.pth
PASS  train_log.csv continues to the last iteration, no duplicates
=== 18. Kaggle package: unzip into an empty directory -> verify_origin PASS, run_dapn.sh --dryrun runs ===
PASS  package built
PASS  zip contains origin + DAPN files, no .git
PASS  unzipped: verify_origin step 1 PASS
PASS  unzipped: run_dapn.sh --dryrun exit 0
PASS  unzipped run records the package commit
==== run_cpu_tests_dapn: 48 passed, 0 failed ====
```
