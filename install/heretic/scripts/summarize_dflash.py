#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Summarize rank-0 DFlash evidence after an approved run; never calls the API."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

ANSI = re.compile(r'\x1b\[[0-9;]*m')
TARGET = re.compile(r'adapter_slot=(-?\d+).*adapter_id=(\d+).*rows=(\d+).*heretic=(true|false)')
ACCEPT = re.compile(r'accepted=(\d+)/(\d+)')
TIMING = re.compile(r'verify=([\d.]+)ms propose=([\d.]+)ms \(K=(\d+), accepted=(\d+)\)')


def record_acceptance(model, block, accepted, offered, source, number, errors):
    if offered != block['width'] - 1 or not 0 <= accepted <= offered or offered == 0:
        errors.append(f'line {number}: inconsistent acceptance width/count')
        return False
    counts = (accepted, offered)
    if block['acceptance'] is not None:
        if block['acceptance'] != counts:
            errors.append(f'line {number}: debug/timing acceptance counts disagree')
            return False
        return True
    block['acceptance'] = counts
    model['verified_blocks'] += 1
    model['accepted_drafts'] += accepted
    model['offered_drafts'] += offered
    model['acceptance_sources'][source] += 1
    kind = 'zero' if accepted == 0 else 'full' if accepted == offered else 'partial'
    model['acceptance_classes'][kind] += 1
    return True


def summarize(lines, heretic_id):
    models = {name: {'target_widths': Counter(), 'verified_blocks': 0,
                     'accepted_drafts': 0, 'offered_drafts': 0, 'acceptance_classes': Counter(),
                     'acceptance_sources': Counter(),
                     'step_timings': []} for name in ['glm53-base', 'heretic']}
    current = None
    errors = []
    unassigned = 0
    for number, raw in enumerate(lines, 1):
        line = ANSI.sub('', raw)
        if 'GLM DFlash target' in line:
            match = TARGET.search(line)
            if not match:
                errors.append(f'line {number}: malformed target evidence')
                current = None
                continue
            slot, identity, width = map(int, match.groups()[:3])
            adapted = match.group(4) == 'true'
            if (slot, identity, adapted) == (-2, 0, False):
                name = 'glm53-base'
            elif (slot, identity, adapted) == (0, heretic_id, True) and heretic_id != 0:
                name = 'heretic'
            else:
                errors.append(f'line {number}: unexpected target identity {slot}/{identity}/{adapted}')
                current = None
                continue
            if not 1 <= width <= 8:
                errors.append(f'line {number}: invalid target width {width}')
                current = None
                continue
            models[name]['target_widths'][width] += 1
            current = {'name': name, 'width': width, 'acceptance': None}
        elif 'K=γ verify:' in line:
            match = ACCEPT.search(line)
            if not match or current is None:
                unassigned += 1
                continue
            accepted, offered = map(int, match.groups())
            record_acceptance(models[current['name']], current, accepted, offered,
                              'verify_debug', number, errors)
        elif 'DFLASH STEP_TIMING:' in line:
            match = TIMING.search(line)
            if match and current and int(match.group(3)) == current['width']:
                model = models[current['name']]
                # INFO timing logs use the same num_accepted as the DEBUG
                # verdict, with K=tokens.len()=drafts.len()+1. They remain
                # available when the scheduler's DEBUG target is filtered.
                if not record_acceptance(model, current, int(match.group(4)),
                                         current['width'] - 1, 'step_timing', number, errors):
                    continue
                model['step_timings'].append({
                    'verify_ms': float(match.group(1)), 'propose_ms': float(match.group(2)),
                    'rows': int(match.group(3)), 'accepted_drafts': int(match.group(4))})
            else:
                unassigned += 1
    for model in models.values():
        model['draft_acceptance_ratio'] = (model['accepted_drafts'] / model['offered_drafts']
                                           if model['offered_drafts'] else None)
        model['missing_acceptance_classes'] = [kind for kind in ['zero', 'partial', 'full']
                                               if not model['acceptance_classes'][kind]]
        model['target_blocks_without_acceptance_record'] = sum(
            count for width, count in model['target_widths'].items() if width > 1
        ) - model['verified_blocks']
    return {'schema': 'onlineheretic.dflash-log-evidence/v1', 'models': models,
            'errors': errors, 'unassigned_records': unassigned,
            'both_models_exercised_dflash': not errors and unassigned == 0
                and all(model['verified_blocks'] > 0 for model in models.values()),
            'limitations': ['Log evidence does not prove numerical correctness or speed parity.',
                            'Acceptance ratios cover recorded nonterminal verify steps; early sequence-finish returns omit verdict/timing logs.',
                            'Missing zero/partial/full classes need more coverage before claiming rollback coverage.',
                            'Use this on the rank-0 log of the single-active-sequence candidate only.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--heretic-id', type=int, default=12003015800042089565)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.log.read_text(errors='replace').splitlines(), args.heretic_id)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['both_models_exercised_dflash'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
