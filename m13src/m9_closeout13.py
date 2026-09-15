"""M9 six-only close-out with M9 statistics and the reviewed M13 scoring helpers.

No reserved access, no post-tag scoring continuation. --recover reads only complete,
identity-bound saved scores. M9's R3 bridge amendment is read from its own registry.
"""
import argparse
import json
from pathlib import Path

import access13 as A
import score13 as S


def production():
    return A.Config(registry_path=A.REPO / 'm9/final_run_registry.json',
                    result_path=A.REPO / 'results/m9_final_run.json',
                    scores_dir=A.REPO / 'results/m9_final_scores',
                    ledger_path=A.REPO / 'm9/LEDGER.md',
                    freeze_path=A.REPO / 'm9/FREEZE.json',
                    lock_path=A.REPO / 'work/m9final.lock', spent_tag='m9-six-spent',
                    ledger_prefix='m9', serving_parity_artifacts=(),
                    build_record_path=None, decision_layer='m9')


def config_for_scoring(conf):
    from m9src.final_stats import SIX
    out = dict(conf)
    datasets = conf['datasets']
    if len(datasets) != 6 or set(datasets) != set(SIX):
        raise ValueError('M9 requires the six canonical registered datasets')
    out['partitions'] = {'all6': datasets}
    if conf['bridge'].get('dataset_mean_abs_delta_max') != 0.003:
        raise ValueError('M9 R3 bridge amendment missing')
    return out


def code_identity():
    return A.sha_json({'shared': S.code_identity(),
                      'closeout': A.sha256_file(__file__),
                      'stats': A.sha256_file(A.REPO / 'm9src/final_stats.py'),
                      'legacy_decision': A.sha256_file(A.REPO / 'm9src/final9.py'),
                      'student': A.sha256_file(A.REPO / 'm9src/nano.py'),
                      'bootstrap': A.sha256_file(A.REPO / 'm7src/boot.py')})


def load_student(cfg, freeze):
    if cfg.load_student:
        return cfg.load_student(freeze)
    if freeze['student'] != 'bge-small-en-v1.5':
        raise ValueError('Unregistered M9 student')
    adapted = dict(freeze, student_key=freeze['student'],
                   query_prefix=freeze['student_query_prefix'])
    return S.M9Student(adapted, repo=cfg.repo)


def validate_saved(cfg, conf, manifest, scored):
    if A.remote_tag_commit(cfg) != manifest.get('begin_commit'):
        raise ValueError('M9 origin tag no longer binds the recorded BEGIN commit')
    if manifest.get('datasets') != conf['datasets']:
        raise ValueError('M9 manifest dataset order changed')
    if set(scored) != set(conf['datasets']):
        raise ValueError('Incomplete M9 scores; no protected reread is allowed')
    if manifest['code_sha256'] != code_identity():
        raise ValueError('M9 scoring code identity changed')
    if manifest['registry_sha256'] != A.sha256_file(cfg.registry_path):
        raise ValueError('M9 registry changed')
    if manifest['comparator_sha256'] != A.sha256_file(cfg.perquery_path):
        raise ValueError('M9 comparator changed')
    freeze = S._freeze_blob(cfg)
    if manifest['freeze_sha256'] != freeze['_sha256']:
        raise ValueError('M9 checkpoint changed')
    if manifest['freeze_file_sha256'] != A.sha256_file(cfg.freeze_path):
        raise ValueError('M9 freeze metadata changed')
    if manifest['system'] != conf['contrasts']['C1']['a']:
        raise ValueError('M9 candidate system changed')
    qids = S.comparator_qids(cfg, conf['datasets'])
    for ds, row in scored.items():
        if S.row_identity_problems(ds, row, manifest) or row.get('bridge_ok') is not True:
            raise ValueError(f'{ds}: M9 score identity or bridge failed')
        if set(row['scores']) != set(qids[ds]):
            raise ValueError(f'{ds}: M9 saved query set differs')


def finalize(cfg, conf, manifest, scored):
    import final9
    validate_saved(cfg, conf, manifest, scored)
    candidate = manifest['system']
    rows = S.comparator_rows(cfg, conf, conf['datasets'])
    rows[candidate] = {ds: scored[ds]['scores'] for ds in conf['datasets']}
    decision = final9.decide(rows, conf)
    blob = {'candidate': candidate, 'datasets': conf['datasets'],
            'registry_sha256': manifest['registry_sha256'],
            'freeze_sha256': manifest['freeze_sha256'],
            'begin_commit': manifest['begin_commit'], 'spent_tag': cfg.spent_tag,
            'bridge': {ds: scored[ds]['bridge'] for ds in conf['datasets']},
            'decision_record': decision, 'end_status': 'COMPLETE_SIX_ONLY',
            'scope': 'M13-authorized M9 six-only close-out; no separate reserved access.',
            'provenance': json.loads(cfg.freeze_path.read_text()).get('provenance'),
            'written': A.utcnow()}
    A.write_atomic(cfg.result_path, json.dumps(blob, indent=2))
    A.ledger_append(cfg, f"- {A.utcnow()} — **FINAL-RUN-END** M9 six-only "
                    f"{A.sha256_file(cfg.result_path)}; original two-build-lock disclosure applies.")
    ok, command, error = A.commit_and_push(cfg, [cfg.ledger_path, cfg.result_path,
                                               cfg.scores_dir], 'm9: publish six-only close-out')
    if not ok:
        raise RuntimeError(f'Result not durable: {command}: {error}')
    print(json.dumps({'outcome': decision['decision'], 'end_status': blob['end_status']}),
          flush=True)
    return 0


def run(cfg=None, preflight_only=False, recover=False):
    cfg = cfg or production()
    conf = json.loads(cfg.registry_path.read_text())
    scoring = config_for_scoring(conf)
    A.seal_protected_paths(fatal=True)
    A.acquire_lock(cfg)
    if preflight_only:
        return S.preflight_report(cfg, scoring)
    exists, where = A.spent_tag_exists(cfg, conf)
    if recover:
        if not exists or where != 'origin':
            raise ValueError('Recovery requires the durable M9 spent tag')
        manifest = json.loads(cfg.state_path.read_text())
        return finalize(cfg, conf, manifest, S.persisted(cfg, conf['datasets']))
    if exists:
        raise ValueError('M9 access already spent; no scoring continuation')
    problems = A.preflight(cfg, scoring)
    if problems:
        raise ValueError('M9 preflight refused: ' + '; '.join(problems))
    freeze = S._freeze_blob(cfg)
    candidate = conf['contrasts']['C1']['a']
    manifest = {}

    def write_manifest(begin):
        manifest.update(S.run_manifest(cfg, scoring, freeze, begin, candidate))
        manifest['code_sha256'] = code_identity()
        manifest['freeze_file_sha256'] = A.sha256_file(cfg.freeze_path)
        manifest['scope'] = 'M9 six-only; no post-tag continuation or reserved access'
        A.write_atomic(cfg.state_path, json.dumps(manifest, indent=2))

    A.spend_access(cfg, freeze['_sha256'], before_tag=write_manifest)
    problems = S.payload_checks(cfg, scoring, conf['datasets'])
    if problems:
        raise ValueError('M9 payload check failed: ' + '; '.join(problems))
    student, anchor = load_student(cfg, freeze), S._anchor(cfg)
    qids = S.comparator_qids(cfg, conf['datasets'])
    frozen_anchor = S.comparator_rows(cfg, conf, conf['datasets'])[conf['bridge']['anchor']]
    scored = {}
    for ds in conf['datasets']:
        scored[ds] = S.score_dataset(cfg, scoring, ds, student, anchor, qids[ds],
                                    frozen_anchor[ds], manifest)
    return finalize(cfg, conf, manifest, scored)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--recover', action='store_true')
    args = parser.parse_args()
    raise SystemExit(run(preflight_only=args.preflight_only, recover=args.recover))
