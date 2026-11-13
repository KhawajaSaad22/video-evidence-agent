import unittest
import numpy as np
from video_search import candidates,temporal_iou,normalize
from evidence import search

class Tests(unittest.TestCase):
    def test_failed_build_is_explicit_and_cannot_be_queried(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        from video_search import build,query
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);video=root/'source.mp4';video.write_bytes(b'fixture');output=root/'index'
            with patch('video_search.video_info',return_value=dict(fps=10,count=20,duration=2.)),patch('video_search.Vision',side_effect=RuntimeError('model unavailable')):
                with self.assertRaises(RuntimeError):build(video,output)
            self.assertEqual(json.loads((output/'run.json').read_text())['status'],'failed')
            self.assertFalse((output/'index.json').exists())
            with self.assertRaisesRegex(ValueError,'incomplete or failed'):query(output,'surface')
    def test_completed_build_checks_features_and_source_identity(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        from types import SimpleNamespace
        from PIL import Image
        from video_search import build,load_features
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);video=root/'source.mp4';video.write_bytes(b'fixture');output=root/'index'
            vision=SimpleNamespace(device='cpu',images=lambda images:np.ones((len(images),512),dtype='float32')/np.sqrt(512))
            with patch('video_search.video_info',return_value=dict(fps=10,count=20,duration=2.)),patch('video_search.Vision',return_value=vision),patch('video_search.read_frame',return_value=Image.new('RGB',(8,8))):
                metadata=build(video,output,step=1)
            self.assertEqual(json.loads((output/'run.json').read_text())['status'],'completed')
            self.assertEqual(load_features(output,metadata)[1].shape,(2,512))
            with (output/'features.npz').open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'archive changed'):load_features(output,metadata)
    def test_legacy_feature_schema_rejects_bad_shapes_times_and_values(self):
        import tempfile
        from pathlib import Path
        from video_search import load_features,MODEL
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);meta=dict(version=1,model=MODEL,fps=10,duration=3.,step=1.)
            for times,vectors in (([0,1],np.ones((1,512))),([1,0],np.ones((2,512))),([0,3],np.ones((2,512))),([0,1],np.full((2,512),np.nan))):
                np.savez(root/'features.npz',times=np.array(times),vectors=vectors)
                with self.assertRaises(ValueError):load_features(root,meta)
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
    def test_negative_queries_are_scored_separately(self):
        from evaluate import evaluate
        result=evaluate([dict(relevant_intervals=[],predictions=[]),
                         dict(relevant_intervals=[],predictions=[dict(start=0,end=1)])])
        self.assertIsNone(result[0]['recall_at_3'])
        self.assertEqual(result[-1]['negative_false_positive_rate'],.5)
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

    def test_second_pass_preserves_chronological_image_references(self):
        from verifier import inspect
        from unittest.mock import patch
        seen=[]
        def verify(question,images,times):
            self.assertEqual(times,sorted(times))
            self.assertEqual(images,times)
            seen.append(list(times))
            return dict(decision='inspect_more' if len(seen)==1 else 'supported',
                        observation='visible',evidence_frames=[1])
        with patch('verifier.read_frame',side_effect=lambda video,t:t):
            result=inspect('event','unused',[dict(start=0,end=8)],verify,6)
        self.assertEqual(len(seen),2)
        self.assertEqual(result['results'][0]['frame_times'],seen[-1])
        self.assertEqual(result['results'][0]['frame_times'][1],2.5)
if __name__=='__main__': unittest.main()
