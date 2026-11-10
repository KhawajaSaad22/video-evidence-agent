"""Local visual evidence retrieval with a budgeted coarse-to-fine controller."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

MODEL='openai/clip-vit-base-patch32'

def normalize(x): return x / np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

class Vision:
    def __init__(self,device='auto'):
        from transformers import CLIPModel, AutoTokenizer
        self.device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
        self.model=CLIPModel.from_pretrained(MODEL).to(self.device).eval()
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL)

    @torch.inference_mode()
    def images(self,images):
        tensors=[]
        for image in images:
            image=image.convert('RGB'); w,h=image.size
            scale=224/min(w,h); image=image.resize((round(w*scale),round(h*scale)),Image.Resampling.BICUBIC)
            w,h=image.size; image=image.crop(((w-224)//2,(h-224)//2,(w+224)//2,(h+224)//2))
            tensors.append(torch.tensor(np.array(image),dtype=torch.float32).permute(2,0,1)/255)
        x=torch.stack(tensors).to(self.device)
        mean=x.new_tensor([.48145466,.4578275,.40821073])[None,:,None,None]
        std=x.new_tensor([.26862954,.26130258,.27577711])[None,:,None,None]
        out=self.model.vision_model(pixel_values=(x-mean)/std).pooler_output
        return F.normalize(self.model.visual_projection(out),dim=-1).cpu().numpy()

    @torch.inference_mode()
    def text(self,query):
        inputs=self.tokenizer([query],return_tensors='pt',truncation=True,max_length=77).to(self.device)
        out=self.model.text_model(**inputs).pooler_output
        return F.normalize(self.model.text_projection(out),dim=-1)[0].cpu().numpy()

def video_info(path):
    import cv2
    cap=cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened(): raise ValueError(f'Cannot decode video: {path}')
        fps=cap.get(cv2.CAP_PROP_FPS); count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps<=0 or count<=0: raise ValueError('Video needs a valid frame rate and frame count')
        return dict(fps=fps,count=int(count),duration=count/fps)
    finally: cap.release()

def read_frame(path,seconds):
    import cv2
    cap=cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC,float(seconds)*1000)
        ok,frame=cap.read()
        if not ok: raise ValueError(f'Cannot decode frame at {seconds:.3f}s')
        return Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    finally: cap.release()

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def build(video,output,step=2.,max_frames=600,device='auto'):
    if step<=0 or max_frames<1: raise ValueError('Positive step and frame limit required')
    video=Path(video).resolve(); output=Path(output)
    if output.exists(): raise ValueError('Use a new index folder to avoid overwriting an index')
    info=video_info(video); times=np.arange(0,info['duration'],step)
    if len(times)>max_frames: raise ValueError('Frame budget exceeded; increase --step or --max-frames explicitly')
    vision=Vision(device); output.mkdir(parents=True); (output/'frames').mkdir()
    vectors=[]
    for start in range(0,len(times),8):
        images=[read_frame(video,t) for t in times[start:start+8]]
        vectors.extend(vision.images(images))
        for i,im in enumerate(images,start): im.save(output/'frames'/f'{i:06}.jpg')
    np.savez_compressed(output/'features.npz',vectors=np.array(vectors),times=times)
    metadata=dict(version=1,model=MODEL,video=str(video),sha256=digest(video),step=step,**info)
    (output/'index.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata

def candidates(times,scores,duration,step,top_k=3):
    result=[]
    for i in np.argsort(-np.asarray(scores)):
        t=float(times[i]); start=max(0,t-step/2); end=min(duration,t+step/2)
        if any(max(start,r['start'])<min(end,r['end']) for r in result): continue
        result.append(dict(start=start,end=end,timestamp=t,similarity=float(scores[i]),frame_index=int(i)))
        if len(result)==top_k: break
    return result

def query(index,question,top_k=3,refine_budget=12,min_score=None,device='auto'):
    if not question.strip(): raise ValueError('Enter a question')
    if top_k<1 or refine_budget<0: raise ValueError('Invalid retrieval budgets')
    index=Path(index); meta=json.loads((index/'index.json').read_text(encoding='utf-8'))
    if meta.get('version')!=1 or meta['model']!=MODEL: raise ValueError('Unsupported index')
    if digest(meta['video'])!=meta['sha256']: raise ValueError('Video changed; rebuild its index')
    with np.load(index/'features.npz',allow_pickle=False) as f: times=f['times']; vectors=f['vectors']
    vision=Vision(device); text=vision.text(question); scores=vectors@text
    proposed=candidates(times,scores,meta['duration'],meta['step'],top_k)
    trace=[dict(action='retrieve',frames=len(times),candidates=len(proposed))]
    refined=[]; remaining=refine_budget; read_count=0
    # Deterministic controller: allocate the remaining budget fairly among intervals.
    # This is not an LLM planner and does not verify actions or their order.
    for n,record in enumerate(proposed):
        count=remaining//(len(proposed)-n)
        if count:
            sample_times=np.linspace(record['start'],record['end'],count+2)[1:-1]
            images=[read_frame(meta['video'],t) for t in sample_times]
            local=vision.images(images)@text; best=int(np.argmax(local))
            read_count+=count; remaining-=count
            record={**record,'refined_timestamp':float(sample_times[best]),'refined_similarity':float(local[best])}
            trace.append(dict(action='inspect_interval',start=record['start'],end=record['end'],frames=count))
        refined.append(record)
    refined.sort(key=lambda r:-r.get('refined_similarity',r['similarity']))
    if min_score is not None: refined=[r for r in refined if r.get('refined_similarity',r['similarity'])>=min_score]
    return dict(question=question,video=meta['video'],results=refined,trace=trace,
                additional_frames=read_count,abstained=not bool(refined),
                warning='Visual similarity candidates, not verified answers. No calibrated confidence or action-order reasoning.')

def temporal_iou(a,b):
    intersection=max(0,min(a[1],b[1])-max(a[0],b[0])); union=max(a[1],b[1])-min(a[0],b[0])
    return intersection/union if union>0 else 0.

def main():
    p=argparse.ArgumentParser(description=__doc__); commands=p.add_subparsers(dest='command',required=True)
    b=commands.add_parser('index'); b.add_argument('video'); b.add_argument('--output',required=True)
    b.add_argument('--step',type=float,default=2.); b.add_argument('--max-frames',type=int,default=600)
    q=commands.add_parser('search'); q.add_argument('index'); q.add_argument('question')
    q.add_argument('--top-k',type=int,default=3); q.add_argument('--refine-budget',type=int,default=12)
    q.add_argument('--min-score',type=float); q.add_argument('--output')
    for parser in (b,q): parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=vars(p.parse_args()); cmd=a.pop('command')
    if cmd=='index': result=build(**a)
    else:
        output=a.pop('output'); result=query(**a)
        if output: Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
