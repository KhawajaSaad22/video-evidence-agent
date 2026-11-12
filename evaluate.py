"""Evaluate timestamp retrieval from saved predictions; no model downloads required."""
import argparse
import json
from video_search import temporal_iou

def evaluate(rows):
    if not rows: raise ValueError('No evaluation rows')
    positive=[row for row in rows if row['relevant_intervals']]
    negative=[row for row in rows if not row['relevant_intervals']]
    metrics=[]
    for threshold in (.3,.5,.7):
        hits=0
        for row in positive:
            targets=row['relevant_intervals']
            for start,end in targets:
                if not 0<=start<end: raise ValueError('Invalid ground-truth interval')
            candidates=row['predictions'][:3]
            hits+=any(temporal_iou((c['start'],c['end']),t)>=threshold for c in candidates for t in targets)
        metrics.append(dict(iou=threshold,recall_at_3=hits/len(positive) if positive else None))
    metrics.append(dict(answerable_queries=len(positive),negative_queries=len(negative),
                        negative_false_positive_rate=sum(bool(row['predictions']) for row in negative)/len(negative) if negative else None,
                        note='Abstention means no emitted predictions; raw retrieval needs calibration.'))
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('manifest'); a=p.parse_args()
    with open(a.manifest,encoding='utf-8') as f: rows=json.load(f)
    print(json.dumps(evaluate(rows),indent=2))
