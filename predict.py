from pathlib import Path
import joblib
import pandas as pd
from features import extract, FEATURE_VERSION
ROOT=Path(__file__).resolve().parent

def load_models():
    bundles={t:joblib.load(ROOT/'models'/f'{t}.joblib') for t in ('grading','ripeness')}
    for task,b in bundles.items():
        if b.get('feature_version')!=FEATURE_VERSION:
            raise ValueError(f'{task}: model/feature version mismatch. Run main.py --retrain.')
    return bundles

def predict(img,models):
    f,img,mask,defect=extract(img); pred={}
    for task,bundle in models.items():
        if bundle.get('feature_version')!=FEATURE_VERSION: raise ValueError('Model version mismatch')
        x=pd.DataFrame([[f[c] for c in bundle['columns']]],columns=bundle['columns']); pred[task]=bundle['model'].predict(x)[0]
    return pred,f,img,mask,defect
