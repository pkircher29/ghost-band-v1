import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'cloud'), str(ROOT / 'app')]

import alignment  # noqa: E402
import lyrics_lookup  # noqa: E402
import phrasing  # noqa: E402

RATE = 16000


def tone(signal, start, end, amp=0.3, freq=220):
    t = np.arange(int(start * RATE), int(end * RATE))
    signal[t] += amp * np.sin(2 * np.pi * freq * t / RATE)


class EdgeRefinement(unittest.TestCase):
    def refine(self, signal, words):
        env = alignment.envelope(signal, RATE)
        for w in words:
            w.setdefault('block_start', 0.0)
            w.setdefault('block_end', len(signal) / RATE)
        alignment.refine_edges(words, env)
        return words

    def test_sustained_vowel_extends_end_and_onset_moves_back(self):
        sig = np.random.default_rng(0).normal(0, 1e-4, 4 * RATE).astype(np.float32)
        tone(sig, 1.00, 1.80)      # word sung 1.00-1.80 with a held vowel
        tone(sig, 2.50, 3.00)
        w = self.refine(sig, [{'ctc_start': 1.06, 'ctc_end': 1.22}, {'ctc_start': 2.55, 'ctc_end': 2.70}])
        self.assertAlmostEqual(w[0]['start'], 1.00, delta=0.02)
        self.assertAlmostEqual(w[0]['end'], 1.80, delta=0.03)
        self.assertEqual(w[0]['end_source'], 'sustain')
        self.assertAlmostEqual(w[1]['end'], 3.00, delta=0.03)

    def test_start_in_silence_moves_to_first_sound(self):
        sig = np.random.default_rng(1).normal(0, 1e-4, 3 * RATE).astype(np.float32)
        tone(sig, 1.10, 1.50)
        w = self.refine(sig, [{'ctc_start': 1.00, 'ctc_end': 1.30}])
        self.assertAlmostEqual(w[0]['start'], 1.10, delta=0.02)

    def test_legato_words_never_overlap(self):
        sig = np.random.default_rng(2).normal(0, 1e-4, 3 * RATE).astype(np.float32)
        tone(sig, 0.50, 2.00)      # two words sung on one continuous breath
        w = self.refine(sig, [{'ctc_start': 0.52, 'ctc_end': 0.70}, {'ctc_start': 1.20, 'ctc_end': 1.40}])
        self.assertLessEqual(w[0]['end'], w[1]['start'])
        self.assertGreater(w[0]['end'], 1.1)   # held until the next word, not cut at CTC end
        self.assertAlmostEqual(w[1]['end'], 2.00, delta=0.03)

    def test_energy_drop_ends_word_before_quiet_tail(self):
        sig = np.random.default_rng(3).normal(0, 1e-4, 3 * RATE).astype(np.float32)
        tone(sig, 0.50, 1.00, amp=0.5)
        tone(sig, 1.00, 2.00, amp=0.5 * 10 ** (-26 / 20))   # bleed 26 dB down
        w = self.refine(sig, [{'ctc_start': 0.55, 'ctc_end': 0.70}])
        self.assertAlmostEqual(w[0]['end'], 1.00, delta=0.04)


    def test_held_first_vowel_starts_at_its_attack(self):
        sig = np.random.default_rng(4).normal(0, 1e-4, 4 * RATE).astype(np.float32)
        tone(sig, 1.00, 2.60)      # "Buuuurning": CTC only finds the tail at 2.0
        w = self.refine(sig, [{'ctc_start': 2.00, 'ctc_end': 2.40, 'line': 0}])
        self.assertAlmostEqual(w[0]['start'], 1.00, delta=0.02)
        self.assertEqual(w[0]['start_source'], 'held-onset')

    def test_mid_line_word_does_not_reach_back(self):
        sig = np.random.default_rng(5).normal(0, 1e-4, 4 * RATE).astype(np.float32)
        tone(sig, 1.00, 2.60)
        w = self.refine(sig, [{'ctc_start': 1.00, 'ctc_end': 1.30, 'line': 0},
                              {'ctc_start': 2.00, 'ctc_end': 2.40, 'line': 0}])
        self.assertAlmostEqual(w[1]['start'], 2.00, delta=0.16)

    def test_note_release_ends_word_despite_quiet_tail(self):
        sig = np.random.default_rng(6).normal(0, 1e-4, 3 * RATE).astype(np.float32)
        tone(sig, 0.50, 1.00, amp=0.5)
        tone(sig, 1.00, 1.60, amp=0.5 * 10 ** (-15 / 20))   # reverb-like tail 15 dB down
        w = self.refine(sig, [{'ctc_start': 0.55, 'ctc_end': 0.70}])
        self.assertAlmostEqual(w[0]['end'], 1.00, delta=0.04)


class Text(unittest.TestCase):
    def test_spoken_forms(self):
        self.assertEqual(alignment.spoken("I've"), "i've")
        self.assertEqual(alignment.spoken('99'), 'ninety nine')
        self.assertEqual(alignment.spoken('1984'), 'nineteen eighty four')
        self.assertEqual(alignment.spoken('Rock & Roll'), 'rock and roll')
        self.assertEqual(alignment.spoken('“Hello,”'), 'hello')

    def test_backing_vocals_marked(self):
        words = alignment.tokenize_lines(["And what's the worst you take (Worst you take)"])
        self.assertEqual([w['backing'] for w in words], [False] * 6 + [True] * 3)

    def test_anchors_ignore_lone_short_matches(self):
        words = alignment.tokenize_lines(['so long and goodnight', 'you'])
        asr = [{'text': t, 'start': i, 'end': i + .5} for i, t in enumerate(['so', 'long', 'and', 'goodnight'])]
        anchors = alignment.anchor_words(words, asr)
        self.assertTrue(all(anchors[:4]))
        self.assertIsNone(anchors[4])

    def test_lines_without_anchor_use_shifted_lrc(self):
        words = alignment.tokenize_lines(['alpha beta gamma', 'delta epsilon', 'zeta eta theta'])
        anchors = [{'start': 10.0, 'end': 10.3}, {'start': 10.4, 'end': 10.7}, None, None, None,
                   {'start': 20.0, 'end': 20.3}, {'start': 20.4, 'end': 20.6}, {'start': 20.7, 'end': 21}]
        times = alignment.estimate_lines(words, anchors, 3, lrc_times=[9.0, 14.0, 19.0], duration=60)
        self.assertAlmostEqual(times[1][0], 15.0, delta=0.01)   # LRC 14.0 + measured +1.0 offset

    def test_wer(self):
        r = alignment.word_error_rate(['so', 'long', 'and', 'goodnight'], ['so', 'long', 'goodnight'])
        self.assertEqual(r['errors'], 1)

    def test_compounds_and_optional_echoes(self):
        r = alignment.word_error_rate(['when', 'every', 'star', 'fall'], ['when', 'every', 'starfall'])
        self.assertEqual(r['errors'], 0)
        r = alignment.lyric_check(["And what's the worst you take (Worst you take)"],
                                  "and what's the worst you take".split())
        self.assertEqual(r['errors'], 0)
        r = alignment.lyric_check(["And what's the worst you take (Worst you take)"],
                                  "and what's the worst you take worst you take".split())
        self.assertEqual(r['errors'], 0)
        r = alignment.lyric_check(["And what's the worst you take"], "and what's the first you take".split())
        self.assertEqual(r['errors'], 1)

    def test_lrc_output(self):
        words = [{'text': 'Long', 'line': 0, 'start': 0.39, 'end': 0.9}, {'text': 'ago', 'line': 0, 'start': 0.95, 'end': 2.5}]
        self.assertEqual(alignment.to_lrc(words, ['Long ago']), '[00:00.39]<00:00.39>Long <00:00.95>ago <00:02.50>\n')


class Lookup(unittest.TestCase):
    def test_filename_guess(self):
        self.assertEqual(lyrics_lookup.guess_artist_title('My_Chemical_Romance_-_Helena (Official Video).mp3'),
                         ('My Chemical Romance', 'Helena'))

    def test_parse_lrc_repeats(self):
        lines, times = lyrics_lookup.parse_lrc('[00:01.00]a\n[00:05.00][00:20.50]chorus\n[00:09.00]b\n[00:30.00]')
        self.assertEqual(lines, ['a', 'chorus', 'b', 'chorus'])
        self.assertEqual(times, [1.0, 5.0, 9.0, 20.5])


class Phrasing(unittest.TestCase):
    def test_repeated_stanza_is_chorus_and_lines_kept(self):
        def line(text, start):
            return {'text': text, 'start': start, 'end': start + 2.5}
        lines = [line('verse one a', 0), line('verse one b', 3), line('so long', 9), line('and goodnight', 12),
                 line('verse two a', 20), line('verse two b', 23), line('so long', 29), line('and goodnight', 32)]
        out = phrasing.build(lines)
        self.assertEqual(out['lyrics'].count('[Chorus]'), 2)
        self.assertEqual(out['lyrics'].count('[Verse]'), 2)
        self.assertIn('so long\nand goodnight', out['lyrics'])

    def test_structure_labels(self):
        lab = '0.0 8.0 verse\n8.0 16.0 chorus\n'
        lines = [{'text': 'x y', 'start': 1, 'end': 3}, {'text': 'z w', 'start': 9, 'end': 11}]
        out = phrasing.build(lines, lab)
        self.assertTrue(out['lyrics'].startswith('[Verse]\nx y\n\n[Chorus]\nz w'))


if __name__ == '__main__':
    unittest.main()
