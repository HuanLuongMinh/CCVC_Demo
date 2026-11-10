"""Run bookkeeping shared by CCVC_aug.py and CCVC_no_aug.py (OEM / Kaggle reproduction).

Everything here is outside the method itself: config.json, resume from last.pth, CSV logs, evaluation
scheduling with OOM fallback, example panels, plots, trial report and the Kaggle session limit.
The training scripts call a handful of hooks on RunContext; the loss code stays upstream's.
"""
import csv
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from util.dist_helper import all_gather_object, all_reduce_sum, barrier, broadcast_object, unwrap

CKPT_LAST = 'last.pth'
CKPT_FINAL = 'final.pth'
KAGGLE_SESSION_HOURS = 12.0

# keys that must not change when resuming (anything that changes the optimisation)
RESUME_KEYS = ['split', 'split_seed', 'full_unlabeled', 'method', 'script', 'lambda_preset', 'seed',
               'epochs', 'batch_size', 'crop_size', 'base_lr', 'lr_backbone', 'lr_network', 'mul_scheduler',
               'w_CE', 'w_con', 'w_dis', 'w_confident', 'conf_threshold', 'conf_prob', 'cc_mask_mixed', 'mode_confident',
               'mode_mapping', 'use_MLP', 'use_norm', 'use_dropout', 'dropout', 'use_con', 'use_dis', 'sup_pure',
               'amp', 'world_size', 'backbone', 'n_labeled', 'n_unlabeled', 'n_val', 'iters_per_epoch',
               'labeled_list_sha1', 'unlabeled_list_sha1']

DEVIATIONS_FROM_PAPER = [
    'Mixed precision (AMP, fp16 forward): FP32 needs ~17-21 GB/GPU for R101/512/batch 2 > 16 GB T4.',
    'Smaller effective batch: 2 labeled + 2 unlabeled images per GPU per iteration (paper: 24); lr NOT scaled.',
    'Evaluation every EVAL_EVERY epochs (+ last epoch) instead of every epoch; no effect on training.',
    'No full-val prediction dump (upstream evaluate_save); only 4 fixed val examples per evaluation.',
    'OEM instead of Pascal VOC; lambda preset chosen by LAMBDA_PRESET (default blended 2/2/1).',
]


# ----------------------------------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------------------------------
def get_logger(args):
    logger = logging.getLogger('ccvc_run')
    logger.handlers.clear()
    logger.propagate = False
    if args.local_rank <= 0:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter('[%(asctime)s] %(message)s', '%m-%d %H:%M:%S')
        for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(args.save_path, 'log.txt'))):
            h.setFormatter(fmt)
            logger.addHandler(h)
    else:
        logger.setLevel(logging.WARNING)
        logger.addHandler(logging.NullHandler())
    return logger


def write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.device):
        return str(o)
    return str(o)


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def file_sha1(path):
    import hashlib
    if not path or not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()


def count_lines(path):
    if not path or not os.path.exists(path):
        return 0
    with open(path, encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def git_commit(cwd='.'):
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=cwd, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=cwd,
                               capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None, bool(dirty.stdout.strip())
    except Exception:  # noqa: BLE001
        return None, None


def disk_free_gb(path):
    try:
        return shutil.disk_usage(path).free / 1024 ** 3
    except OSError:
        return float('nan')


class CsvLog(object):
    """Append-only CSV with a fixed header; rows past a resume point can be dropped."""

    def __init__(self, path, fields):
        self.path, self.fields = path, list(fields)

    def write(self, row):
        new = not os.path.exists(self.path)
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction='ignore')
            if new:
                w.writeheader()
            w.writerow({k: _fmt(row.get(k)) for k in self.fields})

    def truncate(self, keep):
        """Keep only rows for which keep(row) is True (used after a resume)."""
        if not os.path.exists(self.path):
            return 0
        with open(self.path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        kept = [r for r in rows if keep(r)]
        with open(self.path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(kept)
        return len(rows) - len(kept)

    def rows(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path, newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f))


def _fmt(v):
    if isinstance(v, float):
        return '%.6g' % v
    return v


def rng_state():
    st = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        st['cuda'] = torch.cuda.get_rng_state()
    return st


def set_rng_state(st):
    random.setstate(st['python'])
    np.random.set_state(st['numpy'])
    torch.set_rng_state(st['torch'])
    if 'cuda' in st and torch.cuda.is_available():
        torch.cuda.set_rng_state(st['cuda'])


def atomic_torch_save(obj, path):
    """Write to path.tmp, fsync, then rename: a kill during the save leaves the previous file intact."""
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def pseudo_stats(pred1, pred2, valid, gamma, use_prob, nclass):
    """Statistics of the pseudo-labels of the two branches on unlabeled pixels (no gradient).

    pred1/pred2: logits of the pass that produces the pseudo-labels; valid: bool mask (no padding/ignore).
    Returns a float64 tensor [n_valid, n_conflict, n_cc1, n_cc2, hist1 (nclass), hist2 (nclass)].
    CC = conflicting and confident (max softmax > gamma when use_prob, raw logit otherwise), as in
    vote_threshold_label_selection.
    """
    with torch.no_grad():
        p1 = pred1.float().softmax(dim=1) if use_prob else pred1.float()
        p2 = pred2.float().softmax(dim=1) if use_prob else pred2.float()
        c1, l1 = p1.max(dim=1)
        c2, l2 = p2.max(dim=1)
        conflict = (l1 != l2) & valid
        out = torch.zeros(4 + 2 * nclass, dtype=torch.float64, device=pred1.device)
        out[0] = valid.sum()
        out[1] = conflict.sum()
        out[2] = (conflict & (c1 > gamma)).sum()
        out[3] = (conflict & (c2 > gamma)).sum()
        out[4:4 + nclass] = torch.bincount(l1[valid], minlength=nclass)[:nclass].double()
        out[4 + nclass:] = torch.bincount(l2[valid], minlength=nclass)[:nclass].double()
    return out


# ----------------------------------------------------------------------------------------------
# run context
# ----------------------------------------------------------------------------------------------
class RunContext(object):

    def __init__(self, args, cfg, logger, script):
        self.args, self.cfg, self.logger, self.script = args, cfg, logger, script
        self.main = args.local_rank <= 0
        self.nclass = cfg['nclass']
        self.class_names = cfg.get('class_names', ['c%d' % c for c in range(self.nclass)])
        self.trial = args.trial_iters > 0
        self.save = args.save_path
        self.session_start = args.session_start or time.time()
        self.start_epoch, self.global_iter, self.best = 0, 0, (0.0, 0.0, 0.0)
        self.eval_mode_state = None
        self.trial_done = False
        self.iter_times = []
        self._t_last = None
        self._win = None
        self._first_epoch_of_session = True
        self.pseudo_acc = torch.zeros(4 + 2 * self.nclass, dtype=torch.float64)
        self.meta = read_json(args.run_meta, {}) if args.run_meta else {}

        self.train_csv = CsvLog(os.path.join(self.save, 'train_log.csv'), [
            'epoch', 'iteration', 'lr_backbone', 'lr_head', 'L_sup', 'L_con', 'L_dis', 'L_total',
            'conflict_ratio', 'cc_ratio', 's_per_iter'])
        cls = [c.replace(' ', '_') for c in self.class_names]
        self.eval_csv = CsvLog(os.path.join(self.save, 'eval_log.csv'),
                               ['epoch', 'iteration', 'eval_mode', 'eval_time_s', 'miou_b1', 'miou_b2', 'miou_ave',
                                'val_conflict_ratio']
                               + ['iou_b1_%s' % c for c in cls] + ['iou_b2_%s' % c for c in cls]
                               + ['iou_ave_%s' % c for c in cls])
        self.pseudo_csv = CsvLog(os.path.join(self.save, 'pseudo_label_dist.csv'),
                                 ['epoch', 'iteration', 'branch', 'n_pixels'] + ['pct_%s' % c for c in cls])
        self.conf_csv = CsvLog(os.path.join(self.save, 'confidence_reliability.csv'),
                               ['epoch', 'iteration', 'gamma', 'n_valid', 'conf_ratio', 'acc_conf', 'acc_nonconf',
                                'acc_all'])

    # ------------------------------------------------------------------------------------------
    # start: config.json, resume, summary
    # ------------------------------------------------------------------------------------------
    def start(self, model, optimizer, scaler, n_labeled, n_unlabeled, valset, iters_per_epoch):
        a = self.args
        self.iters_per_epoch = iters_per_epoch
        self.total_iters = iters_per_epoch * a.epochs
        self.valset_ids = valset.ids
        rs = np.random.RandomState(a.vis_seed)
        pick = sorted(rs.choice(len(valset.ids), size=min(a.vis_num, len(valset.ids)), replace=False).tolist())
        self.vis_ids = [valset.ids[k] for k in pick]

        commit, dirty = git_commit(os.path.dirname(os.path.abspath(__file__)) + '/..')
        upstream = None
        up_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'UPSTREAM_COMMIT')
        if os.path.exists(up_file):
            upstream = open(up_file).read().strip()
        gpus = [torch.cuda.get_device_name(k) for k in range(torch.cuda.device_count())] \
            if a.device.type == 'cuda' else ['cpu']
        cfg_now = dict(self.meta)
        cfg_now.update({
            'script': self.script, 'backbone': a.backbone, 'seed': a.seed, 'epochs': a.epochs,
            'batch_size': a.batch_size, 'crop_size': a.crop_size, 'base_lr': a.base_lr,
            'lr_backbone': a.lr_backbone, 'lr_network': a.lr_network, 'mul_scheduler': a.mul_scheduler,
            'momentum': 0.9, 'weight_decay': 1e-4, 'w_CE': a.w_CE, 'w_con': a.w_con, 'w_dis': a.w_dis,
            'w_confident': a.w_confident, 'conf_threshold': a.conf_threshold, 'conf_prob': a.conf_prob,
            'cc_mask_mixed': getattr(a, 'cc_mask_mixed', None),
            'mode_confident': a.mode_confident, 'mode_mapping': a.mode_mapping, 'use_MLP': a.use_MLP,
            'use_norm': a.use_norm, 'use_dropout': a.use_dropout, 'dropout': a.dropout,
            'use_con': getattr(a, 'use_con', True), 'use_dis': getattr(a, 'use_dis', True),
            'sup_pure': getattr(a, 'sup_pure', False), 'use_SPL': a.use_SPL, 'mode_criterion': getattr(a, 'mode_criterion', 'CE'),
            'amp': a.amp, 'world_size': a.world_size, 'ddp': a.ddp, 'dist_backend': a.dist_backend,
            'effective_batch_labeled': a.batch_size * a.world_size,
            'effective_batch_unlabeled': 0 if getattr(a, 'sup_pure', False) else a.batch_size * a.world_size,
            'n_labeled': n_labeled, 'n_unlabeled': n_unlabeled, 'n_val': len(valset.ids),
            'iters_per_epoch': iters_per_epoch, 'total_iters': self.total_iters,
            'eval_every': a.eval_every, 'n_evals': len(self.eval_epochs()),
            'max_session_hours': a.max_session_hours, 'vis_ids': self.vis_ids,
            'labeled_list': a.labeled_id_path, 'unlabeled_list': a.unlabeled_id_path,
            'labeled_list_sha1': file_sha1(a.labeled_id_path), 'unlabeled_list_sha1': file_sha1(a.unlabeled_id_path),
            'val_list': a.val_id_path, 'data_root': self.cfg.get('data_root'),
            'upstream_commit': upstream, 'code_commit': commit, 'code_dirty': dirty,
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpus': gpus,
            'python': sys.version.split()[0], 'trial_iters': a.trial_iters, 'log_every': a.log_every,
            'class_names': self.class_names, 'palette': self.cfg.get('palette'),
            'deviations_from_paper': DEVIATIONS_FROM_PAPER,
        })
        self.config = cfg_now
        cfg_path = os.path.join(self.save, 'config.json')
        last = os.path.join(self.save, CKPT_LAST)
        resume = (not self.trial) and os.path.exists(last)

        if resume:
            old = read_json(cfg_path, {})
            diff = [(k, old.get(k), cfg_now.get(k)) for k in RESUME_KEYS if old.get(k) != cfg_now.get(k)]
            if diff:
                msg = '\n'.join('  %s: saved=%r now=%r' % d for d in diff)
                raise RuntimeError('resume blocked: configuration differs from %s\n%s\n'
                                   'Use the original settings, or FORCE=1 to start this run again.' % (cfg_path, msg))
            ck = torch.load(last, map_location='cpu', weights_only=False)
            unwrap(model).load_state_dict(ck['model'])
            optimizer.load_state_dict(ck['optimizer'])
            scaler.load_state_dict(ck['scaler'])
            self.start_epoch = ck['epoch'] + 1
            self.global_iter = ck['iteration']
            self.best = tuple(ck['best'])
            self.eval_mode_state = ck.get('eval_mode_state')
            self.pseudo_acc = ck.get('pseudo_acc', self.pseudo_acc)
            if ck.get('rng') and len(ck['rng']) == a.world_size:
                set_rng_state(ck['rng'][max(a.local_rank, 0)])
            if self.main:
                removed = self.train_csv.truncate(lambda r: int(r['iteration']) <= self.global_iter)
                for log in (self.eval_csv, self.pseudo_csv, self.conf_csv):
                    removed += log.truncate(lambda r: int(r['epoch']) <= ck['epoch'])
                old.setdefault('resumes', []).append({'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                                                      'from_epoch': self.start_epoch,
                                                      'code_commit': commit, 'gpus': gpus,
                                                      'dist_backend': a.dist_backend,
                                                      'dropped_log_rows': removed})
                write_json(cfg_path, old)
        elif self.main:
            write_json(cfg_path, cfg_now)
        if self.main:
            self.write_labeled_baseline()
        barrier(a)
        if self.main:
            self.print_summary(resume)
        return self.start_epoch, self.best

    def write_labeled_baseline(self):
        """Reference row of pseudo_label_dist.csv: class distribution of the labeled ground truth
        (computed by tools/prepare_oem.py from labeled.txt only; unlabeled ground truth is never read)."""
        src = self.meta.get('labeled_class_dist')
        if not src or not os.path.exists(src) or self.config['effective_batch_unlabeled'] == 0:
            return                              # sup_pure / full-none: no pseudo-labels to compare
        dist = read_json(src)
        shutil.copyfile(src, os.path.join(self.save, 'labeled_class_dist.json'))
        if any(r['branch'] == 'labeled_gt' for r in self.pseudo_csv.rows()):
            return
        row = {'epoch': -1, 'iteration': 0, 'branch': 'labeled_gt', 'n_pixels': dist['n_pixels']}
        row.update({'pct_%s' % c.replace(' ', '_'): dist['pct'][c] for c in self.class_names})
        self.pseudo_csv.write(row)

    def eval_epochs(self):
        a = self.args
        return sorted({e for e in range(a.epochs) if (e + 1) % a.eval_every == 0} | {a.epochs - 1})

    def print_summary(self, resume):
        a, c, L = self.args, self.config, self.logger.info
        L('=' * 78)
        L('split %s | method %s (%s) | lambda %s | seed %s | split_seed %s' % (
            c.get('split'), c.get('method'), self.script, c.get('lambda_preset'), a.seed, c.get('split_seed')))
        L('images: labeled %d | unlabeled %d | val %d | train total %s' % (
            c['n_labeled'], c['n_unlabeled'], c['n_val'], c.get('n_train')))
        L('GPUs %d (%s, backend %s) | batch/GPU/loader %d | effective batch %d labeled + %d unlabeled | AMP %s' % (
            a.world_size, ', '.join(c['gpus']), a.dist_backend, a.batch_size, c['effective_batch_labeled'],
            c['effective_batch_unlabeled'], a.amp))
        L('iter/epoch %d | epochs %d | total iterations %d | evaluations %d (every %d epochs + last)' % (
            self.iters_per_epoch, a.epochs, self.total_iters, c['n_evals'], a.eval_every))
        if c.get('split') == 'full':
            L('note: SPLIT=full -> N_labeled = N_train, so 1 epoch over the (replicated) labeled set equals '
              '1 pass over the labeled set; full/same and full/none have the same number of iterations')
        if self.trial:
            L('TRIAL: %d iterations then one evaluation -> %s' % (a.trial_iters, self.save))
        elif resume:
            L('RESUME from epoch %d (iteration %d done, %d left)' % (
                self.start_epoch, self.global_iter, self.total_iters - self.global_iter))
        else:
            L('new run -> %s' % self.save)
        self.print_estimate()
        L('=' * 78)

    def print_estimate(self, s_iter=None, source=None):
        a = self.args
        t_eval = None
        if s_iter is None:
            rows = self.train_csv.rows()
            if rows:
                vals = [float(r['s_per_iter']) for r in rows[-20:] if r.get('s_per_iter')]
                if vals:
                    s_iter, source = float(np.median(vals)), 'train_log.csv of this run'
        trial = read_json(os.path.join(self.save, 'trial', 'trial_report.json'))
        if s_iter is None and trial:
            s_iter, source = trial['s_per_iter_median'], 'trial/trial_report.json'
        ev = [float(r['eval_time_s']) for r in self.eval_csv.rows() if r.get('eval_time_s')]
        if ev:
            t_eval = float(np.median(ev))
        elif trial:
            t_eval = trial.get('eval_time_s')
        if s_iter is None:
            self.logger.info('Kaggle sessions: no estimate yet (no s/iter measured) - printed after %d iterations '
                             '(or run TRIAL=1 first)' % a.log_every)
            return
        rem_iters = self.total_iters - self.global_iter
        rem_evals = len([e for e in self.eval_epochs() if e >= self.start_epoch])
        hours = (rem_iters * s_iter + rem_evals * (t_eval or 0.0)) / 3600.0
        limit = a.max_session_hours if a.max_session_hours > 0 else KAGGLE_SESSION_HOURS
        self.logger.info('estimate: %.3f s/iter (%s) | eval %s | remaining %d iter + %d evals = %.1f h '
                         '-> %d Kaggle session(s) of %.2f h' % (
                             s_iter, source, '%.0f s' % t_eval if t_eval else 'not measured (ignored)',
                             rem_iters, rem_evals, hours, max(1, math.ceil(hours / limit)), limit))

    # ------------------------------------------------------------------------------------------
    # per iteration
    # ------------------------------------------------------------------------------------------
    def after_iter(self, epoch, i, iters, optimizer, loss, loss_CE, loss_con, loss_dis, loss_con_raw, loss_dis_raw,
                   included, pred1=None, pred2=None, valid=None):
        """Logging after optimizer.step(). Returns True when the trial is finished (break)."""
        a = self.args
        now = time.time()
        if self.trial and a.device.type == 'cuda':
            torch.cuda.synchronize()
            now = time.time()
        dt = None if self._t_last is None else now - self._t_last
        self._t_last = now
        if dt is not None:
            self.iter_times.append(dt)

        if pred1 is not None:
            st = pseudo_stats(pred1, pred2, valid, a.conf_threshold, a.conf_prob, self.nclass).cpu()
            self.pseudo_acc += st
        else:
            st = torch.zeros(4 + 2 * self.nclass, dtype=torch.float64)
        if self._win is None:
            self._win = {'n': 0, 'sup': 0.0, 'con': 0.0, 'dis': 0.0, 'tot': 0.0, 'st': torch.zeros_like(st),
                         'dt': 0.0, 'n_dt': 0}
        w = self._win
        w['n'] += 1
        if dt is not None:
            w['dt'] += dt
            w['n_dt'] += 1
        loss, loss_CE, loss_con, loss_dis, loss_con_raw, loss_dis_raw = (
            float(t.detach()) if torch.is_tensor(t) else float(t)
            for t in (loss, loss_CE, loss_con, loss_dis, loss_con_raw, loss_dis_raw))
        w['sup'] += loss_CE
        w['con'] += loss_con
        w['dis'] += loss_dis
        w['tot'] += loss
        w['st'] += st

        if self.global_iter < 10 and self.main:
            self.logger.info(
                '[iter %d] L_total %.4f = L_sup %.4f%s + L_con %.4f (raw %.4f, w_con %g)%s + L_dis %.4f (raw %.4f, w_dis %g)%s'
                % (iters, float(loss), float(loss_CE), '' if included[0] else ' [not added]', float(loss_con),
                   float(loss_con_raw), a.w_con, '' if included[1] else ' [not added]', float(loss_dis),
                   float(loss_dis_raw), a.w_dis, '' if included[2] else ' [not added]'))

        self.global_iter = iters + 1
        if self.global_iter % a.log_every == 0 or self.global_iter == self.total_iters:
            stw = all_reduce_sum(w['st'].to(a.device), a).cpu()
            if self.main:
                nv = max(float(stw[0]), 1.0)
                self.train_csv.write({
                    'epoch': epoch, 'iteration': self.global_iter,
                    'lr_backbone': optimizer.param_groups[0]['lr'], 'lr_head': optimizer.param_groups[-1]['lr'],
                    'L_sup': w['sup'] / w['n'], 'L_con': w['con'] / w['n'], 'L_dis': w['dis'] / w['n'],
                    'L_total': w['tot'] / w['n'],
                    'conflict_ratio': float(stw[1]) / nv if pred1 is not None else '',
                    'cc_ratio': (float(stw[2]) + float(stw[3])) / (2 * nv) if pred1 is not None else '',
                    's_per_iter': w['dt'] / w['n_dt'] if w['n_dt'] else '',
                })
                if self.global_iter == a.log_every and not self.trial and self.iter_times:
                    self.print_estimate(float(np.median(self.iter_times)), 'first %d iterations' % a.log_every)
            self._win = None

        if self.trial and self.global_iter >= a.trial_iters:
            self.trial_done = True
        return self.trial_done

    # ------------------------------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------------------------------
    def should_eval(self, epoch):
        """Called by every rank at the end of an epoch; also reduces the pseudo-label histograms."""
        if epoch not in self.eval_epochs():
            return False
        # clone: on a single device .to()/.cpu() return the same tensor, which zero_() would clear
        self._pseudo_total = all_reduce_sum(self.pseudo_acc.clone().to(self.args.device), self.args).cpu()
        self.pseudo_acc.zero_()
        return True

    def run_eval(self, model, valloader, preferred_mode, epoch, iters, title_prefix=''):
        """Rank 0 only. Evaluates the whole val set, writes CSVs, panel and plots; returns the result."""
        from valid import evaluate
        a = self.args
        mode = self.eval_mode_state or preferred_mode
        net = unwrap(model)
        if a.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(a.device)
        t0 = time.time()
        try:
            res = evaluate(a.local_rank, net, valloader, mode, a, self.cfg, vis_ids=self.vis_ids,
                           gamma=a.conf_threshold)
        except torch.cuda.OutOfMemoryError:
            self.logger.info('[eval] CUDA OOM in %s mode -> switching to sliding_window (window %d, stride 2/3) '
                             'for this and all later evaluations' % (mode, a.crop_size))
            torch.cuda.empty_cache()
            mode = self.eval_mode_state = 'sliding_window'
            t0 = time.time()
            res = evaluate(a.local_rank, net, valloader, mode, a, self.cfg, vis_ids=self.vis_ids,
                           gamma=a.conf_threshold)
        res['eval_time_s'] = time.time() - t0
        res['mode'] = mode
        if a.device.type == 'cuda':
            res['eval_peak_allocated_gb'] = torch.cuda.max_memory_allocated(a.device) / 1024 ** 3
            res['eval_peak_reserved_gb'] = torch.cuda.max_memory_reserved(a.device) / 1024 ** 3
            torch.cuda.empty_cache()
        model.train()

        cls = [c.replace(' ', '_') for c in self.class_names]
        row = {'epoch': epoch, 'iteration': iters, 'eval_mode': mode, 'eval_time_s': res['eval_time_s'],
               'miou_b1': res['IOU1'], 'miou_b2': res['IOU2'], 'miou_ave': res['IOU_ave'],
               'val_conflict_ratio': res['val_conflict_ratio']}
        for tag, key in (('b1', 'iou_class1'), ('b2', 'iou_class2'), ('ave', 'iou_class_ave')):
            for c, v in zip(cls, res[key]):
                row['iou_%s_%s' % (tag, c)] = 100.0 * float(v)
        self.eval_csv.write(row)
        cr = res['confidence']
        self.conf_csv.write(dict(cr, epoch=epoch, iteration=iters, gamma=a.conf_threshold))

        pt = getattr(self, '_pseudo_total', None)
        if pt is not None and float(pt[0]) > 0:
            for b, off in ((1, 4), (2, 4 + self.nclass)):
                h = pt[off:off + self.nclass]
                r = {'epoch': epoch, 'iteration': iters, 'branch': b, 'n_pixels': int(h.sum())}
                r.update({'pct_%s' % c: 100.0 * float(v) / max(float(h.sum()), 1.0) for c, v in zip(cls, h)})
                self.pseudo_csv.write(r)
        self._pseudo_total = None

        try:
            from util.vis import save_panel
            title = '%s%s · %s · Epoch %d · Iter %d | mIoU val (%d ảnh): nhánh 1 = %.2f (dự đoán chính) · ' \
                    'nhánh 2 = %.2f · TB = %.2f' % (title_prefix, self.config.get('split'), self.config.get('method'),
                                                    epoch + 1, iters, len(self.valset_ids), res['IOU1'], res['IOU2'],
                                                    res['IOU_ave'])
            os.makedirs(os.path.join(self.save, 'vis'), exist_ok=True)
            save_panel(res['vis'], self.cfg, title,
                       os.path.join(self.save, 'vis', 'epoch_%03d_iter_%06d.png' % (epoch + 1, iters)))
        except Exception as e:  # noqa: BLE001 - drawing must never stop training
            self.logger.info('[vis] panel failed: %s: %s' % (type(e).__name__, e))
        self.plot()
        return res

    def plot(self):
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
            from plot_run import plot_run
            plot_run(self.save, quiet=True)
        except Exception as e:  # noqa: BLE001
            self.logger.info('[plot] failed: %s: %s' % (type(e).__name__, e))

    # ------------------------------------------------------------------------------------------
    # end of epoch: checkpoint, final, session limit
    # ------------------------------------------------------------------------------------------
    def end_epoch(self, epoch, iters, model, optimizer, scaler, best, last_eval=None, epoch_time=None,
                  eval_time=None):
        """Called by every rank after the (optional) evaluation. Returns True to stop (session limit)."""
        a = self.args
        self.best = tuple(best)
        if self.trial:
            return False
        if epoch == a.epochs - 1:
            if self.main:
                torch.save(unwrap(model).state_dict(), os.path.join(self.save, CKPT_FINAL))
                self.write_final(epoch, iters, last_eval)
                write_json(os.path.join(self.save, 'progress.json'),
                           {'epochs_done': epoch + 1, 'epochs': a.epochs, 'iteration': iters + 1, 'finished': True,
                            'time': time.strftime('%Y-%m-%d %H:%M:%S')})
                if os.path.exists(os.path.join(self.save, CKPT_LAST)):
                    os.remove(os.path.join(self.save, CKPT_LAST))
                self.logger.info('DONE: final.pth + final_eval.json written to %s' % self.save)
            barrier(a)
            return False

        rngs = all_gather_object(rng_state(), a)
        if self.main:
            atomic_torch_save({'model': unwrap(model).state_dict(), 'optimizer': optimizer.state_dict(),
                               'scaler': scaler.state_dict(), 'epoch': epoch, 'iteration': iters + 1,
                               'best': list(self.best), 'eval_mode_state': self.eval_mode_state,
                               'pseudo_acc': self.pseudo_acc, 'rng': rngs, 'world_size': a.world_size},
                              os.path.join(self.save, CKPT_LAST))
            # small side file so tools/restore_output.py can compare runs without opening last.pth
            write_json(os.path.join(self.save, 'progress.json'),
                       {'epochs_done': epoch + 1, 'epochs': a.epochs, 'iteration': iters + 1,
                        'time': time.strftime('%Y-%m-%d %H:%M:%S')})
            elapsed_h = (time.time() - self.session_start) / 3600.0
            self.logger.info('[ckpt] epoch %d/%d saved (last.pth) | session %.2f h | free disk %.1f GB' % (
                epoch + 1, a.epochs, elapsed_h, disk_free_gb(self.save)))
            if self._first_epoch_of_session and epoch_time is not None and a.max_session_hours > 0:
                margin = (KAGGLE_SESSION_HOURS - a.max_session_hours) * 3600
                need = epoch_time + (eval_time or 0.0)
                if need > margin:
                    self.logger.info('[session] WARNING: one epoch + eval takes %.0f min but MAX_SESSION_HOURS=%.2f '
                                     'leaves only %.0f min before the 12 h Kaggle limit; consider lowering it' % (
                                         need / 60, a.max_session_hours, margin / 60))
        self._first_epoch_of_session = False
        stop = False
        if self.main and a.max_session_hours > 0:
            stop = (time.time() - self.session_start) / 3600.0 > a.max_session_hours
        stop = broadcast_object(stop, a)
        if stop and self.main:
            self.logger.info('[session] past MAX_SESSION_HOURS=%.2f h: epoch %d finished and saved. '
                             'STOPPED FOR RESUME: %d epoch(s) left - run the same command in the next session.' % (
                                 a.max_session_hours, epoch + 1, a.epochs - epoch - 1))
        return stop

    def write_final(self, epoch, iters, res):
        rows = self.eval_csv.rows()
        best = max(rows, key=lambda r: float(r['miou_b1'])) if rows else None
        out = {
            'split': self.config.get('split'), 'full_unlabeled': self.config.get('full_unlabeled'),
            'split_seed': self.config.get('split_seed'), 'method': self.config.get('method'),
            'lambda_preset': self.config.get('lambda_preset'), 'seed': self.args.seed,
            'n_labeled': self.config['n_labeled'], 'n_unlabeled': self.config['n_unlabeled'],
            'n_train': self.config.get('n_train'), 'n_val': self.config['n_val'],
            'epoch': epoch + 1, 'iteration': iters + 1, 'eval_mode': res['mode'] if res else None,
            'main_metric': 'miou_b1 at the last epoch',
            'miou_b1': res['IOU1'], 'miou_b2': res['IOU2'], 'miou_ave': res['IOU_ave'],
            'iou_b1': dict(zip(self.class_names, (100.0 * np.asarray(res['iou_class1'])).tolist())),
            'iou_b2': dict(zip(self.class_names, (100.0 * np.asarray(res['iou_class2'])).tolist())),
            'iou_ave': dict(zip(self.class_names, (100.0 * np.asarray(res['iou_class_ave'])).tolist())),
            'best_on_val_reference_only': ({'epoch': int(best['epoch']) + 1, 'iteration': int(best['iteration']),
                                            'miou_b1': float(best['miou_b1'])} if best else None),
        }
        write_json(os.path.join(self.save, 'final_eval.json'), out)

    # ------------------------------------------------------------------------------------------
    # trial
    # ------------------------------------------------------------------------------------------
    def finish_trial(self, model, valloader, preferred_mode, epoch, iters):
        a = self.args
        mem = {}
        if a.device.type == 'cuda':
            mem = {'train_peak_allocated_gb': torch.cuda.max_memory_allocated(a.device) / 1024 ** 3,
                   'train_peak_reserved_gb': torch.cuda.max_memory_reserved(a.device) / 1024 ** 3,
                   'gpu': torch.cuda.get_device_name(a.device)}
        mems = all_gather_object(mem, a)
        self.should_eval(a.epochs - 1)          # reduce the pseudo-label histograms on every rank
        if self.main:
            res = self.run_eval(model, valloader, preferred_mode, epoch, iters, title_prefix='TRIAL · ')
            times = self.iter_times
            skip = 20 if len(times) > 20 else 0
            used = times[skip:] or times
            s_mean, s_med = float(np.mean(used)), float(np.median(used))
            ipe = self.iters_per_epoch
            total = ipe * a.epochs
            n_evals = math.ceil(a.epochs / a.eval_every)
            hours = (total * s_med + n_evals * res['eval_time_s']) / 3600.0
            limit = a.max_session_hours if a.max_session_hours > 0 else KAGGLE_SESSION_HOURS
            for k, m in enumerate(mems):
                if k == 0 and 'eval_peak_allocated_gb' in res:
                    m['eval_peak_allocated_gb'] = res['eval_peak_allocated_gb']
                    m['eval_peak_reserved_gb'] = res['eval_peak_reserved_gb']
            c = self.config
            rep = {
                'split': c.get('split'), 'method': c.get('method'), 'trial_iters': a.trial_iters,
                'n_train': c.get('n_train'), 'n_labeled': c['n_labeled'], 'n_unlabeled': c['n_unlabeled'],
                'n_val': c['n_val'], 'gpus': c['gpus'], 'world_size': a.world_size,
                'dist_backend': a.dist_backend, 'batch_per_gpu_per_loader': a.batch_size,
                'effective_batch_labeled': c['effective_batch_labeled'],
                'effective_batch_unlabeled': c['effective_batch_unlabeled'], 'amp': a.amp,
                'eval_mode': res['mode'], 'iters_measured': len(used), 'warmup_iters_excluded': skip,
                's_per_iter_mean': s_mean, 's_per_iter_median': s_med,
                'memory_per_gpu': mems, 'eval_time_s': res['eval_time_s'],
                'miou_b1': res['IOU1'], 'miou_b2': res['IOU2'], 'miou_ave': res['IOU_ave'],
                'projection': {
                    'iters_per_epoch': ipe, 'epochs': a.epochs, 'total_iters': total,
                    'n_evals': n_evals, 'eval_every': a.eval_every,
                    'formula': 'hours = total_iters * s_per_iter_median + n_evals * eval_time_s',
                    'total_hours': hours, 'session_hours': limit,
                    'kaggle_sessions': max(1, math.ceil(hours / limit)),
                },
            }
            write_json(os.path.join(self.save, 'trial_report.json'), rep)
            L = self.logger.info
            L('-' * 78)
            L('TRIAL REPORT (%s)' % os.path.join(self.save, 'trial_report.json'))
            L('  images  train %s | labeled %d | unlabeled %d | val %d' % (rep['n_train'], rep['n_labeled'],
                                                                         rep['n_unlabeled'], rep['n_val']))
            L('  GPUs    %s x %d (%s) | batch/GPU %d | effective %d+%d | AMP %s | eval %s' % (
                ', '.join(rep['gpus']), a.world_size, a.dist_backend, a.batch_size,
                rep['effective_batch_labeled'], rep['effective_batch_unlabeled'], a.amp, rep['eval_mode']))
            L('  speed   %.3f s/iter mean, %.3f median (%d iters, first %d excluded)' % (s_mean, s_med, len(used), skip))
            for k, m in enumerate(mems):
                if m:
                    L('  GPU%d    train peak %.2f GB alloc / %.2f GB reserved%s' % (
                        k, m['train_peak_allocated_gb'], m['train_peak_reserved_gb'],
                        ' | eval peak %.2f / %.2f GB' % (m['eval_peak_allocated_gb'], m['eval_peak_reserved_gb'])
                        if 'eval_peak_allocated_gb' in m else ''))
            L('  eval    %.0f s | mIoU b1 %.2f b2 %.2f ave %.2f (sanity check only)' % (
                res['eval_time_s'], res['IOU1'], res['IOU2'], res['IOU_ave']))
            L('  full    %d iter/epoch x %d epochs = %d iters + %d evals -> %.1f h -> %d session(s) of %.2f h' % (
                ipe, a.epochs, total, n_evals, hours, rep['projection']['kaggle_sessions'], limit))
            L('-' * 78)
        barrier(a)


def exit_code(save_path, trial):
    """0 = finished (or trial done), 3 = stopped for resume, 1 = nothing usable written."""
    if trial:
        return 0 if os.path.exists(os.path.join(save_path, 'trial_report.json')) else 1
    if os.path.exists(os.path.join(save_path, 'final_eval.json')):
        return 0
    if os.path.exists(os.path.join(save_path, CKPT_LAST)):
        return 3
    return 1
