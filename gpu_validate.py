"""Run real cached CLIP image and text encoders on the selected device."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--allow-download',action='store_true')
    args=p.parse_args();out=Path(args.output)
    if out.exists():p.error('Choose a new report path')
    os.environ['HF_HUB_OFFLINE']='0' if args.allow_download else '1'
    import numpy as np
    import torch
    from PIL import Image
    from video_search import Vision,MODEL
    model=Vision(args.device)
    assert next(model.model.parameters()).device.type==args.device
    images=model.images([Image.new('RGB',(240,240),'red'),Image.new('RGB',(240,240),'blue')])
    text=model.text('a blue surface')
    assert images.shape==(2,len(text)) and np.isfinite(images).all() and np.isfinite(text).all()
    assert np.allclose(np.linalg.norm(images,axis=1),1,atol=1e-3)
    report=dict(device=model.device,model_device=str(next(model.model.parameters()).device),model=MODEL,
                torch=torch.__version__,cuda_runtime=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
                image_features=list(images.shape),text_features=list(text.shape),similarities=(images@text).tolist(),
                note='Real CLIP encoding on generated images/text; not video-query accuracy or throughput.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
