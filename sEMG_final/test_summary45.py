"""Small synthetic fixtures for the prediction-file validator; no training."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from summarize45 import read_run


class PredictionValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        self.rows=[{'path':f'{label}/trial.csv','window':str(w),'true':label,'predicted':label}
                   for label in 'ABCDE' for w in range(19)]
        self.metrics={'accuracy':1.,'precision_macro':1.,'recall_macro':1.,'f1_macro':1.,
                      'confusion_matrix':[[19 if i==j else 0 for j in range(5)] for i in range(5)],
                      'n_evaluation_windows':95,'n_evaluation_trials':5}
        self.save()
    def tearDown(self): self.temporary.cleanup()
    def save(self):
        (self.root/'metrics.json').write_text(json.dumps(self.metrics))
        with (self.root/'predictions.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(self.rows[0]));writer.writeheader();writer.writerows(self.rows)
    def test_verified_fixture(self):
        result,rows=read_run(self.root)
        self.assertEqual(result['accuracy'],1.)
        self.assertEqual(len(rows),95)
    def test_rejects_changed_metric(self):
        self.metrics['f1_macro']=.5;self.save()
        with self.assertRaises(ValueError): read_run(self.root)
    def test_rejects_duplicate_window(self):
        self.rows[1]=self.rows[0].copy();self.save()
        with self.assertRaises(ValueError): read_run(self.root)
    def test_rejects_wrong_trial_membership(self):
        self.metrics['config']={'source_signal_hashes':[f'fixture-{c}' for c in 'ABCDE'],
                                'evaluation_indices':list(range(5))}
        manifest={i:{'path':f'{c}/trial.csv','subject':c,'signal_sha256':f'fixture-{c}'}
                  for i,c in enumerate('ABCDE')}
        self.save();read_run(self.root,manifest)
        for row in self.rows:
            if row['true']=='A': row['path']='A/another_trial.csv'
        self.save()
        with self.assertRaises(ValueError): read_run(self.root,manifest)
    def test_rejects_unknown_label(self):
        self.rows[0]['predicted']='F';self.save()
        with self.assertRaises(ValueError): read_run(self.root)


if __name__=='__main__': unittest.main()
