# Báo cáo kiểm thử CCVC-DAPN — Giai đoạn 1 + 2

Ngày 23/11/2026 · CPU (torch 2.13.0+cpu, numpy 2.2.6, python 3.12.9), dữ liệu giả (16 train / 4 val), mô hình huấn luyện 1–2 epoch.

Lệnh: `bash tools/tests_dapn/run_cpu_tests_dapn.sh <thư mục tạm>` → **73 passed, 0 failed**. Trong đó `test_dapn_unit.py`: 34 kiểm tra; `test_diagnose.py`: 12 kiểm tra; 0 failed.

Phủ toàn bộ test 1–18 của spec 9.1, DAPN trên cả hai mạng con, và đo giai đoạn 2 chạy tự động ngay sau khi train trong cùng lệnh `run_dapn.sh` (diagnose run + baseline, eval_ckpt baseline, results_summary.csv; bỏ qua bước đã có; dừng sạch khi hết giờ phiên rồi làm nốt ở lần gọi sau).

Số liệu mIoU / độ thuần trong log chỉ để kiểm tra đường chạy (mô hình 1 epoch trên dữ liệu giả), không mang ý nghĩa thực nghiệm.

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
PASS  c4: da1 / da2 on layer4 (2048 channels)
PASS  c4, training: pred/feature identical to the origin forward, plus c4_1 / c4_2
PASS  c4, evaluation: exactly the origin outputs (no c4 keys)
PASS  c4: DAPG on both branches, every pass computed, grads on da1 and da2 (L 0.7500)
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
=== stage-2 measurements run by the same command right after training (tools/post_run_dapn.py) ===
PASS  ccvc_dapn run: diagnose_b1.json and diagnose_b2.json written after training
PASS  baseline (run_baseline.sh 1/4, same seed): eval_ckpt.json + diagnose_b1/b2.json written
PASS  results_summary.csv in exp/oem_dapn with the origin and the DAPN row
PASS  summary table printed, analysis complete, exit 0
PASS  same command again: exit 0, every step already done (nothing recomputed)
PASS  no session time left: step not started, message to rerun, exit 0
PASS  next call of the same command finishes the missing step only
PASS  --no-analysis: skipped
=== 12. eval_ckpt.py: origin and DAPN checkpoints reproduce final_eval.json ===
PASS  origin checkpoint: three mIoU = final_eval.json (+-0.01)
PASS  origin checkpoint recognised as origin
PASS  origin: eval_ckpt.json written next to the checkpoint, agreement metrics sum to 1
PASS  DAPN checkpoint: three mIoU = final_eval.json (+-0.01)
PASS  DAPN checkpoint: da1, da2 ignored at evaluation
PASS  eval_ckpt agreement metrics = final_eval.json of the DAPN run
=== 13. diagnose_entanglement.py: constructed features, then origin and DAPN checkpoints ===
PASS  sampling: 6000 pixels (quota per image, then subsample) ((6000, 304))
PASS  every pixel assigned once per EM seed
PASS  group shares sum to 1
PASS  separable features -> purity ~ 1 (all pixels, every group) (1.0000)
PASS  separable -> no pixel in a mixed cluster
PASS  separable -> agree_wrong not in a pure cluster of the predicted class
PASS  3 EM seeds -> mean and std reported
PASS  per class: 8 classes, purity ~ 1
PASS  near_boundary = share of boundary pixels (~0.25)
PASS  entangled (random) features -> low purity, mostly mixed clusters (0.140)
PASS  agree_wrong pixels carrying the predicted class feature -> in pure cluster of the predicted class (1.000)
PASS  boundary band: radius 3 around the class edge, padding is not an edge
PASS  test_diagnose.py: 0 failed (separable features -> purity ~ 1)
PASS  origin checkpoint, branch 1: diagnose_b1.json + per_class.csv
PASS  origin checkpoint, branch 2: diagnose_b2.json + per_class.csv
PASS  DAPN checkpoint, branch 1: runs (da1/da2 ignored)
PASS  diagnose outputs: 3 EM seeds, shares sum to 1, purity in [0,1], 8 classes per CSV
=== collect_results_dapn.py: origin + DAPN runs in one table ===
PASS  collect: exit 0
PASS  collect: origin row (agree_* from eval_ckpt.json, diagnose columns) and DAPN row (agree_* from final_eval.json)
PASS  collect: mean +- std over seeds printed
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
=== 17. --dapg-pos c4: a few iterations end to end ===
PASS  run_dapn.sh --dapg-pos c4 --dryrun exit 0
PASS  c4: own run directory, config dapg_pos c4, DAPG on both branches (2048-d DA)
=== 7. dry run vs baseline (trial_report.json: s/iter, memory, hours, % increase) ===
PASS  1/4 dry run exit 0
PASS  1/4: compared with the origin dry run of run_baseline.sh (exp/oem/.../ccvc_aug/.../trial)
PASS  comparison table printed
PASS  1/8 dry run (no baseline dry run yet) exit 0
PASS  1/8: extra ccvc_dapn_off dry run made and used as the baseline
PASS  1/8: trial_report.json holds the comparison with ccvc_dapn_off
=== 18. Kaggle package: unzip into an empty directory -> verify_origin PASS, run_dapn.sh --dryrun runs ===
PASS  package built
PASS  zip contains origin + DAPN files, no .git
PASS  unzipped: verify_origin step 1 PASS
PASS  unzipped: run_dapn.sh --dryrun exit 0
PASS  unzipped run records the package commit
==== run_cpu_tests_dapn: 73 passed, 0 failed ====
```

---

## Chẩn đoán origin 0a/0b trên Kaggle — `tools/run_diag_origin.sh` (24/11/2026)

`bash tools/tests_dapn/test_run_diag_origin.sh <work_dir>` — CPU, dữ liệu giả lập, hai run origin ngắn (`run_baseline.sh` 1/4, 1/8) đặt như input Kaggle (cây chỉ đọc giả lập, kiểm sha256 trước/sau).

```
=== 0. synthetic data + two short origin runs ===             4 PASS
=== 1. --quick (auto-discovered paths, preflight) ===          4 PASS
=== 2. full run: 6 steps, 6 output dirs, summary A-C, input untouched ===   8 PASS
=== 3. preflight failures (missing final_eval.json, val.txt line count, two different final.pth for one split,
       identical copies counted once, --run-1-4 resolves, missing val image) ===   6 PASS
=== 4. corrupted 1/8 checkpoint: 1/8 steps fail, 1/4 steps run, summary 'chưa có', exit 4 ===   4 PASS
=== 4b. one split per session (--split 1/4 with only the 1/4 run attached, then --split 1/8; --quick per split;
        merge of the two outputs -> one summary) ===   7 PASS
=== 5. verify_origin step 1, check_dapn_diff, test_diagnose.py ===   3 PASS
test_run_diag_origin: 36 passed, 0 failed
```
`eval_ckpt.py`, `diagnose_entanglement.py` và mọi file của `ORIGIN_MANIFEST.sha256` không đổi.
