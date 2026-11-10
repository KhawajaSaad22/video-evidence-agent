"""Validated lexical evidence-store scaffold; no visual model or agent yet."""
import argparse
import json
import re

def search(records, query, limit=5):
    terms=set(re.findall(r'\w+',query.lower()))
    if not terms: raise ValueError('Question is empty')
    validated=[]
    for record in records:
        if not isinstance(record.get('video_id'),str) or not record['video_id']: raise ValueError('Missing video ID')
        if not 0 <= record['start'] < record['end']: raise ValueError('Invalid time interval')
        words=set(re.findall(r'\w+',record['description'].lower()))
        score=len(terms & words)
        if score: validated.append({**record,'score':score})
    return sorted(validated,key=lambda r:-r['score'])[:limit]

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('manifest'); p.add_argument('query'); a=p.parse_args()
    with open(a.manifest,encoding='utf-8') as f: rows=json.load(f)
    print(json.dumps(search(rows,a.query),indent=2))
