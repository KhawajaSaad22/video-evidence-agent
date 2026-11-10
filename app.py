from pathlib import Path
import streamlit as st
from video_search import query

st.set_page_config(page_title='Video evidence search',layout='wide')
st.title('Find visual evidence in videos')
st.caption('Coarse-to-fine visual retrieval. Candidates require review; no automatic factual answer is generated.')
index=st.text_input('Index folder','artifacts/demo')
question=st.text_input('Describe what to find','a blue circle')
budget=st.slider('Additional inspection frames',0,48,12)
verifier_model=st.text_input('Optional installed Ollama vision model','')
st.caption('Leave blank for visual retrieval only. Vision verification uses the local Ollama service.')
if st.button('Search'):
    try:
        with st.spinner('Searching visual evidence…'): result=query(index,question,refine_budget=0 if verifier_model else budget)
        st.warning(result['warning'])
        for rank,r in enumerate(result['results'],1):
            st.write(f"{rank}. {r['start']:.2f}–{r['end']:.2f}s | similarity {r.get('refined_similarity',r['similarity']):.3f}")
            st.image(str(Path(index)/'frames'/f"{r['frame_index']:06}.jpg"),caption='Cached coarse candidate frame')
        if result['abstained']: st.info('No candidates passed the configured filter.')
        with st.expander('Inspection trace'): st.json(result['trace'])
        if verifier_model:
            if budget<1: raise ValueError('Verification needs at least one inspection frame')
            from verifier import inspect,OllamaVerifier
            with st.spinner('Assessing candidate evidence…'):
                verified=inspect(question,result['video'],result['results'],OllamaVerifier(verifier_model),budget)
            st.subheader('Model assessments — review the evidence')
            st.json(verified)
        st.caption('Use the timestamps in your original video player. Some browsers cannot play the AVI smoke clip.')
    except Exception as exc: st.error(str(exc))
