"""Fail-closed readiness/model selection for the requested step-6 -> step-7 handoff."""
import json,math
from pathlib import Path


def select_completed_checkpoint(root,series):
    root=Path(root).resolve();series=Path(series).resolve()
    if not series.is_relative_to(root):raise ValueError('Series outside project')
    state=json.loads((series/'series.json').read_text())
    if state['status']!='completed':raise ValueError('Step 6 has not completed successfully')
    for label in ('A3','A4','A5'):
        item=state['experiments'].get(label,{})
        if item.get('status')!='completed' or item.get('exit_code')!=0:raise ValueError('Incomplete step-6 experiment '+label)
    summary=json.loads((series/'summary.json').read_text());results=summary['results']
    if set(results)!=set(('A1','A2','A3','A4','A5')):raise ValueError('Incomplete model comparison')
    verified={}
    for label,item in results.items():
        checkpoint=(root/item['checkpoint']).resolve()
        if not checkpoint.is_relative_to(root):raise ValueError('Checkpoint outside project')
        folder=checkpoint.parent
        if label in ('A3','A4','A5') and folder!=(series/label):raise ValueError('Experiment checkpoint does not belong to the completed series')
        report=json.loads((folder/'report.json').read_text());best=json.loads((folder/'best.json').read_text())
        if report['status']!='completed' or report.get('exit_code')!=0:raise ValueError('Model reference not completed: '+label)
        score=float(best['score'])
        if not math.isfinite(score) or score!=item['single_macro_mae']:raise ValueError('Selection score mismatch')
        metrics=best['metrics']['single']['per_dataset']
        if set(metrics)!=set(('FSC147','VisDrone','UA_DETRAC')):raise ValueError('Missing source metrics')
        if not math.isclose(score,sum(v['mae'] for v in metrics.values())/3,abs_tol=1e-10):raise ValueError('Invalid macro score')
        for source,expected in [('FSC147',1286),('VisDrone',548),('UA_DETRAC',8323)]:
            if metrics[source]['images']!=expected:raise ValueError('Incomplete source validation')
        verified[label]={'checkpoint':str(checkpoint.relative_to(root)),'score':score,'checkpoint_sha256':best['checkpoint_sha256']}
    winner=min(verified,key=lambda k:verified[k]['score'])
    if winner!=summary['best_by_natural_source_macro_mae']:raise ValueError('Summary winner does not match verified metrics')
    return {'selected_model':winner,**verified[winner],'candidates':verified,'selection_metric':'source natural validation single macro MAE','target_data_used':False}
