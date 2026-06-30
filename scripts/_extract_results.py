import json
files = ['modma_eeg_baseline.json', 'modma_eeg_baseline_v2.json', 'modma_eeg_baseline_v3.json', 'modma_audio_baseline.json', 'modma_multimodal_baseline.json']
for f in files:
    try:
        with open(f'results/{f}') as fh:
            d = json.load(fh)
        print('=== ' + f + ' ===')
        if 'best' in d:
            best = d['best']
            cfg = best.get('config', '')
            bacc = best.get('mean_bacc', 0)
            acc = best.get('mean_acc', 0)
            print('  best: ' + cfg + ' bacc=' + str(round(bacc, 3)) + ' acc=' + str(round(acc, 3)))
        if 'n_subjects' in d:
            print('  n_subjects: ' + str(d['n_subjects']) + ' n_MDD: ' + str(d.get('n_MDD','?')) + ' n_HC: ' + str(d.get('n_HC','?')))
    except Exception as e:
        print('Error ' + f + ': ' + str(e))
