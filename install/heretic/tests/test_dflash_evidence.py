#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Offline evidence parser checks with mixed-owner and missing-path regressions."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('summarize_dflash',
    Path(__file__).resolve().parents[1] / 'scripts/summarize_dflash.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def target(slot, identity, rows, adapted):
    return f'DEBUG GLM DFlash target adapter_slot={slot} adapter_id={identity} rows={rows} position=100 heretic={adapted}'


class Evidence(unittest.TestCase):
    def test_correct_identity_and_acceptance_are_assigned(self):
        lines = [target(-2, 0, 8, 'false'), 'K=γ verify: γ=8 accepted=7/7 (100%) seq_len=108',
                 'DFLASH STEP_TIMING: verify=10.0ms propose=2.0ms (K=8, accepted=7)',
                 target(0, 99, 4, 'true'), 'K=γ verify: γ=4 accepted=1/3 (33%) seq_len=104']
        result = module.summarize(lines, 99)
        self.assertTrue(result['both_models_exercised_dflash'])
        self.assertEqual(result['models']['glm53-base']['accepted_drafts'], 7)
        self.assertEqual(result['models']['heretic']['acceptance_classes']['partial'], 1)
        self.assertIn('zero', result['models']['heretic']['missing_acceptance_classes'])
        self.assertEqual(len(result['models']['glm53-base']['step_timings']), 1)

    def test_empty_and_single_row_logs_do_not_claim_dflash(self):
        for lines in [[], [target(-2, 0, 1, 'false'), target(0, 99, 1, 'true')]]:
            self.assertFalse(module.summarize(lines, 99)['both_models_exercised_dflash'])

    def test_wrong_identity_and_impossible_counts_fail(self):
        for line in [target(0, 98, 8, 'true'), target(-2, 99, 8, 'false'),
                     target(0, 99, 9, 'true'), 'GLM DFlash target malformed']:
            self.assertTrue(module.summarize([line], 99)['errors'])
        for count in ['9/7', '2/2']:
            report = module.summarize([target(0, 99, 8, 'true'), f'K=γ verify: accepted={count}'], 99)
            self.assertTrue(report['errors'])

    def test_unassigned_acceptance_and_missing_owner_are_not_success(self):
        report = module.summarize(['K=γ verify: accepted=1/7'], 99)
        self.assertEqual(report['unassigned_records'], 1)
        report = module.summarize([target(-2, 0, 2, 'false'), 'K=γ verify: accepted=0/1'], 99)
        self.assertFalse(report['both_models_exercised_dflash'])
        self.assertEqual(report['models']['glm53-base']['acceptance_classes']['zero'], 1)

    def test_info_timings_supply_acceptance_when_debug_is_filtered(self):
        lines = [target(-2, 0, 8, 'false'),
                 'DFLASH STEP_TIMING: verify=10.0ms propose=2.0ms (K=8, accepted=7)',
                 target(0, 99, 4, 'true'),
                 'DFLASH STEP_TIMING: verify=15.0ms propose=3.0ms (K=4, accepted=0)']
        result = module.summarize(lines, 99)
        self.assertTrue(result['both_models_exercised_dflash'])
        self.assertEqual(result['models']['glm53-base']['verified_blocks'], 1)
        self.assertEqual(result['models']['glm53-base']['offered_drafts'], 7)
        self.assertEqual(result['models']['heretic']['acceptance_classes']['zero'], 1)
        self.assertEqual(result['models']['heretic']['acceptance_sources']['step_timing'], 1)

    def test_two_log_formats_count_each_target_block_once(self):
        for records in [
            ['K=γ verify: accepted=2/3',
             'DFLASH STEP_TIMING: verify=10.0ms propose=2.0ms (K=4, accepted=2)'],
            ['DFLASH STEP_TIMING: verify=10.0ms propose=2.0ms (K=4, accepted=2)',
             'K=γ verify: accepted=2/3'],
        ]:
            result = module.summarize([target(0, 99, 4, 'true')] + records, 99)
            model = result['models']['heretic']
            self.assertFalse(result['errors'])
            self.assertEqual(model['verified_blocks'], 1)
            self.assertEqual(model['accepted_drafts'], 2)
            self.assertEqual(len(model['step_timings']), 1)

    def test_timing_counts_must_match_the_target_and_debug_record(self):
        for rows, accepted in [(4, 4), (3, 1), (1, 0)]:
            result = module.summarize([target(0, 99, 4, 'true'),
                f'DFLASH STEP_TIMING: verify=10.0ms propose=2.0ms (K={rows}, accepted={accepted})'], 99)
            self.assertTrue(result['errors'] or result['unassigned_records'])
            self.assertEqual(result['models']['heretic']['verified_blocks'], 0)
        result = module.summarize([target(0, 99, 4, 'true'),
            'K=γ verify: accepted=2/3',
            'DFLASH STEP_TIMING: verify=10.0ms propose=2.0ms (K=4, accepted=1)'], 99)
        self.assertTrue(result['errors'])


if __name__ == '__main__':
    unittest.main()
