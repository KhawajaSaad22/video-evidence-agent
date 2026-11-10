import unittest
import numpy as np
from video_search import candidates,temporal_iou,normalize
from evidence import search

class Tests(unittest.TestCase):
    def test_iou(self): self.assertAlmostEqual(temporal_iou((0,2),(1,3)),1/3)
    def test_disjoint(self): self.assertEqual(temporal_iou((0,1),(2,3)),0)
    def test_candidates(self):
        result=candidates(np.array([0,2,4]),np.array([.1,.9,.5]),5,2,2)
        self.assertEqual(result[0]['timestamp'],2); self.assertLessEqual(result[-1]['end'],5)
    def test_normalization(self): np.testing.assert_allclose(normalize(np.array([[3.,4.]])),[[.6,.8]])
    def test_bad_interval(self):
        with self.assertRaises(ValueError): search([dict(video_id='v',start=2,end=1,description='battery')],'battery')
    def test_retrieval_metric(self):
        from evaluate import evaluate
        result=evaluate([dict(relevant_intervals=[[4,8]],predictions=[dict(start=5,end=7)])])
        self.assertEqual(result[1]['recall_at_3'],1)
        self.assertEqual(result[2]['recall_at_3'],0)
    def test_verifier_budget_and_abstention(self):
        from verifier import inspect
        from unittest.mock import patch
        from PIL import Image
        with patch('verifier.read_frame',return_value=Image.new('RGB',(8,8))):
            result=inspect('battery','unused',[dict(start=0,end=2)],lambda *a:dict(decision='inspect_more',observation='unclear',evidence_frames=[]),frame_budget=4)
        self.assertEqual(result['frames_inspected'],4); self.assertTrue(result['abstained'])
        self.assertEqual(result['results'][0]['assessment']['decision'],'unknown')
    def test_verifier_rejects_unknown_evidence(self):
        from verifier import validate
        with self.assertRaises(ValueError): validate(dict(decision='supported',observation='visible',evidence_frames=[5]),2)
if __name__=='__main__': unittest.main()
