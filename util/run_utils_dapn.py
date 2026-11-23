"""Run bookkeeping of CCVC_aug_dapn.py (spec 3.4, 5.1, 5.2, 8). util/run_utils.py is not modified.

RunContextDAPN(RunContext) adds, without changing what the origin RunContext writes:
- config.json: code_version, commits (this repo / origin b44dc57 / DAPN 33ad638 main_final), every DA/DAPG
  parameter, verify_origin result, deviations from the original DAPN;
- resume is blocked when a DAPN parameter differs from the saved config.json (on top of the origin keys);
- train_log.csv: L_dapg, L_dapg_w, L_dapg_l, L_dapg_u, per branch intra/inter/quality/n_valid_proto/q_mean,
  L_dapg_b1/b2, cc_ratio_loss (CC ratio with the mask the loss uses: gamma vs raw logits), dapg_nan_skips;
- eval_log.csv: agree_wrong_ratio, agree_right_ratio, conflict_ratio, err_b1, agree_wrong_<class>;
- final_eval.json: the same agreement metrics; confusion_b1.csv and confusion_agree_wrong.csv (last epoch).
Evaluation uses valid_dapn.evaluate_dapn (same three mIoU as valid.evaluate).
"""
import json
import os

import torch

import util.run_utils as ru
from util.run_utils import (RunContext, CsvLog, get_logger, count_lines, exit_code, read_json, write_json,  # noqa: F401
                            pseudo_stats, git_commit)
from util.dist_helper import all_reduce_sum
from valid_dapn import evaluate_dapn, write_confusion, AGREE_KEYS

ORIGIN_COMMIT = 'b44dc5707d98e12d8b015697c4f2dd66019c5ebb'
DAPN_REPO = 'https://github.com/HuanLuongMinh/DAPN-for-VHR-Remote-Sensing-Segmentation'
DAPN_COMMIT = '33ad6387068f78240e8b04f1dbddbc8dd6dc5d6c'
DAPN_BRANCH = 'main_final'

DAPN_KEYS = ['use_dapg', 'dapg_on', 'dapg_branches', 'dapg_pos', 'w_dapg', 'da_max_groups', 'da_temperature',
             'da_num_iters', 'da_init', 'da_min_quality', 'da_ema_decay', 'dapg_margin', 'dapg_lambda_inter',
             'dapg_lambda_quality']
DAPN_RESUME_KEYS = ['code_version'] + DAPN_KEYS

DEVIATIONS_FROM_DAPN = [
    'No boundary-aware loss (decision: DAPG only).',
    'Optimizer: da1/da2 in the SGD group of the non-backbone parameters (lr x10, momentum 0.9, wd 1e-4), '
    'not AdamW lr 6e-5 as in DAPN.',
    'Pixel mask: pixels whose nearest-downsampled label is 255 are excluded before the DA module '
    '(padding; class 0 unknown on labeled images); the original uses every pixel.',
    'DA position: CCVC discrepancy feature (304 ch, branch 2 after mapping + dropout) by default; '
    'DAPN UNetFormer uses before_fusion.',
    'One DA module per CCVC sub-network (da1, da2); applied to labeled and unlabeled (strong+CutMix, raw) passes.',
    'Loss weight w_dapg = 0.2 on top of w_CE = 2 (keeps the DAPN CE : DAPG ratio 1 : 0.1).',
    'DA seed: created under torch.random.fork_rng with seed + 1000 (global RNG stream untouched).',
    'Non-finite features or a NaN raised by DAPGLoss -> loss 0 for that pass (counted), training continues.',
]

TRAIN_EXTRA = ['L_dapg', 'L_dapg_w', 'L_dapg_l', 'L_dapg_u', 'L_dapg_b1', 'L_dapg_b2',
               'intra_b1', 'inter_b1', 'quality_b1', 'n_valid_proto_b1', 'q_mean_b1',
               'intra_b2', 'inter_b2', 'quality_b2', 'n_valid_proto_b2', 'q_mean_b2',
               'cc_ratio_loss', 'dapg_nan_skips']
WIN_KEYS = ['L_dapg', 'L_dapg_l', 'L_dapg_u', 'L_dapg_b1', 'L_dapg_b2',
            'intra_b1', 'inter_b1', 'quality_b1', 'n_valid_proto_b1', 'q_mean_b1',
            'intra_b2', 'inter_b2', 'quality_b2', 'n_valid_proto_b2', 'q_mean_b2']


class _MergeCsv(CsvLog):
    """CsvLog whose rows are completed by extra() before writing (the origin code builds the base row)."""

    def __init__(self, path, fields, extra):
        super(_MergeCsv, self).__init__(path, fields)
        self.extra = extra

    def write(self, row):
        row = dict(row)
        row.update(self.extra() or {})
        super(_MergeCsv, self).write(row)


def _repo_commit():
    return git_commit(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


class RunContextDAPN(RunContext):

    def __init__(self, args, cfg, logger, script):
        super(RunContextDAPN, self).__init__(args, cfg, logger, script)
        self._dwin = None
        self._dapg_last = None
        self._row_extra = None
        self._eval_extra = None
        self._last_dapn_res = None
        self.nan_skips_total = 0
        self.train_csv = _MergeCsv(self.train_csv.path, self.train_csv.fields + TRAIN_EXTRA,
                                   lambda: self._row_extra)
        cls = [c.replace(' ', '_') for c in self.class_names]
        self.eval_csv = _MergeCsv(self.eval_csv.path,
                                  self.eval_csv.fields + list(AGREE_KEYS) + ['agree_wrong_%s' % c for c in cls],
                                  lambda: self._eval_extra)

    # ------------------------------------------------------------------ config / resume
    def dapn_config(self):
        a = self.args
        commit, dirty = _repo_commit()
        cfg = {k: getattr(a, k, None) for k in DAPN_KEYS}
        cfg.update({
            'code_version': 'dapn', 'ccvc_dapn_commit': commit, 'ccvc_dapn_dirty': dirty,
            'origin_commit': ORIGIN_COMMIT, 'dapn_repo': DAPN_REPO, 'dapn_commit': DAPN_COMMIT,
            'dapn_branch': DAPN_BRANCH, 'da_use_quality_gate': True, 'da_use_mask_predictor': False,
            'dapg_formula': 'loss = w_CE*L_sup + w_con*L_con + w_dis*L_dis + w_dapg*L_DAPG; '
                            'L_DAPG = 1/2(L^l + L^u) (lu) | L^l (l); L_pass = 1/2(b1 + b2) (both) | b1',
            'deviations_from_dapn': DEVIATIONS_FROM_DAPN,
            'verify_origin': self.meta.get('verify_origin'),
        })
        return cfg

    def start(self, model, optimizer, scaler, n_labeled, n_unlabeled, valset, iters_per_epoch):
        a = self.args
        dcfg = self.dapn_config()
        cfg_path = os.path.join(self.save, 'config.json')
        if (not self.trial) and os.path.exists(os.path.join(self.save, ru.CKPT_LAST)):
            old = read_json(cfg_path, {})
            diff = [(k, old.get(k), dcfg.get(k)) for k in DAPN_RESUME_KEYS if old.get(k) != dcfg.get(k)]
            if diff:
                raise RuntimeError('resume blocked: DAPN configuration differs from %s\n%s\n'
                                   'Use the original settings, or --force to start this run again.' % (
                                       cfg_path, '\n'.join('  %s: saved=%r now=%r' % d for d in diff)))
        self.meta = dict(self.meta)
        self.meta.update(dcfg)                       # origin start() copies meta into config.json
        out = super(RunContextDAPN, self).start(model, optimizer, scaler, n_labeled, n_unlabeled, valset,
                                                iters_per_epoch)
        if self.main:
            das = ', '.join('da%d (branch %d)' % (b, b) for b in sorted(_net(model).da_modules())) or 'none'
            self.logger.info('DAPN: use_dapg %s | dapg_on %s | dapg_branches %s -> %s | dapg_pos %s | w_dapg %g' % (
                a.use_dapg, a.dapg_on, a.dapg_branches, das, a.dapg_pos, a.w_dapg))
        return out

    # ------------------------------------------------------------------ per iteration
    def record_dapg(self, stats):
        """Called by the training script after the DAPG loss of this iteration (before after_iter)."""
        self._dapg_last = stats

    def after_iter(self, epoch, i, iters, optimizer, loss, loss_CE, loss_con, loss_dis, loss_con_raw, loss_dis_raw,
                   included, pred1=None, pred2=None, valid=None):
        a = self.args
        st = self._dapg_last
        self._dapg_last = None
        if self._dwin is None:
            self._dwin = {'n': 0, 'sums': {}, 'cnt': {}, 'cc': torch.zeros(2, dtype=torch.float64)}
        w = self._dwin
        w['n'] += 1
        if st is not None:
            self.nan_skips_total += st['nan_skips']
            for k in WIN_KEYS:
                if st.get(k) is not None:
                    w['sums'][k] = w['sums'].get(k, 0.0) + st[k]
                    w['cnt'][k] = w['cnt'].get(k, 0) + 1
        if pred1 is not None:
            ps = pseudo_stats(pred1, pred2, valid, a.conf_threshold, False, self.nclass).cpu()
            w['cc'] += torch.stack([ps[0], ps[2] + ps[3]])

        if self.global_iter < 10 and self.main and st is not None:
            self.logger.info('[iter %d] L_DAPG %.4f (x w_dapg %g = %.4f) | L^l %.4f | L^u %.4f | b1 %s | b2 %s' % (
                iters, st['L_dapg'], a.w_dapg, a.w_dapg * st['L_dapg'], st['L_dapg_l'], st['L_dapg_u'],
                _f(st.get('L_dapg_b1')), _f(st.get('L_dapg_b2'))))

        nxt = iters + 1
        will_write = nxt % a.log_every == 0 or nxt == self.total_iters
        if will_write:
            cc = all_reduce_sum(w['cc'].to(a.device), a).cpu()
            row = {k: (w['sums'][k] / w['cnt'][k]) if w['cnt'].get(k) else '' for k in WIN_KEYS}
            row['L_dapg_w'] = a.w_dapg * row['L_dapg'] if row['L_dapg'] != '' else ''
            row['cc_ratio_loss'] = float(cc[1]) / (2 * max(float(cc[0]), 1.0)) if pred1 is not None else ''
            row['dapg_nan_skips'] = self.nan_skips_total
            self._row_extra = row
        done = super(RunContextDAPN, self).after_iter(epoch, i, iters, optimizer, loss, loss_CE, loss_con, loss_dis,
                                                      loss_con_raw, loss_dis_raw, included, pred1, pred2, valid)
        if will_write:
            self._row_extra = None
            self._dwin = None
        return done

    # ------------------------------------------------------------------ evaluation
    def run_eval(self, model, valloader, preferred_mode, epoch, iters, title_prefix=''):
        """Origin run_eval with valid.evaluate replaced by evaluate_dapn for this call only."""
        import valid
        orig = valid.evaluate
        cls = [c.replace(' ', '_') for c in self.class_names]

        def _evaluate(*args, **kw):
            res = evaluate_dapn(*args, **kw)
            extra = {k: res[k] for k in AGREE_KEYS}
            extra.update({'agree_wrong_%s' % c: res['agree_wrong_class'][n] for c, n in zip(cls, self.class_names)})
            self._eval_extra = extra
            return res

        valid.evaluate = _evaluate
        try:
            res = super(RunContextDAPN, self).run_eval(model, valloader, preferred_mode, epoch, iters, title_prefix)
        finally:
            valid.evaluate = orig
            self._eval_extra = None
        self._last_dapn_res = res
        self.logger.info('***** agreement on val: agree_wrong %.4f | agree_right %.4f | conflict %.4f | err_b1 %.4f'
                         % tuple(res[k] for k in AGREE_KEYS))
        return res

    def write_final(self, epoch, iters, res):
        super(RunContextDAPN, self).write_final(epoch, iters, res)
        p = os.path.join(self.save, 'final_eval.json')
        out = read_json(p, {})
        out['code_version'] = 'dapn'
        if res is not None and 'agree_wrong_ratio' in res:
            for k in AGREE_KEYS:
                out[k] = res[k]
            out['agree_wrong_class'] = res['agree_wrong_class']
            write_confusion(os.path.join(self.save, 'confusion_b1.csv'), res['confusion_b1'], self.class_names)
            write_confusion(os.path.join(self.save, 'confusion_agree_wrong.csv'), res['confusion_agree_wrong'],
                            self.class_names)
        out['dapg'] = {k: self.config.get(k) for k in DAPN_KEYS}
        out['dapg_nan_skips'] = self.nan_skips_total
        write_json(p, out)


def _net(model):
    return model.module if hasattr(model, 'module') else model


def _f(v):
    return 'n/a' if v is None else '%.4f' % v
