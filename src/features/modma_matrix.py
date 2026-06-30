"""MODMA multimodal feature matrix builder.
Combines EEG, audio, and psychometric features into a single matrix
ready for ML training, with subject-level labels.
"""
import warnings
import numpy as np

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def build_multimodal_matrix(eeg_subs, eeg_X, eeg_feats,
                            audio_subs, audio_X, audio_feats,
                            psych_subs, psych_X,
                            subject_group_dict,
                            min_subjects_with_data=1):
    """Build multimodal feature matrix.
    Returns: (X, y, subjects, feature_names)
    """
    eeg_dict = dict(zip(eeg_subs, eeg_X))
    audio_dict = dict(zip(audio_subs, audio_X))
    psych_dict = dict(zip(psych_subs, psych_X))

    # Audio ID mapping: 02010XXX -> sub-XXX
    audio_to_part = {a: f"sub-{a[5:]}" for a in audio_subs}

    common = []
    for e_sub in eeg_subs:
        a_sub = None
        for a_sub_orig in audio_subs:
            if audio_to_part.get(a_sub_orig) == e_sub:
                a_sub = a_sub_orig
                break
        if a_sub is None:
            continue
        g = subject_group_dict.get(e_sub)
        if g in ('MDD', 'HC'):
            common.append((e_sub, a_sub, 1 if g == 'MDD' else 0))

    if not common:
        return None, None, None, None

    common = sorted(common, key=lambda x: x[0])
    eeg_ids = [c[0] for c in common]
    audio_ids = [c[1] for c in common]
    y = np.array([c[2] for c in common])

    eeg_rows = np.array([eeg_dict[s] for s in eeg_ids])
    audio_rows = np.array([audio_dict[s] for s in audio_ids])
    psych_rows = np.array([psych_dict.get(s, np.zeros(6)) for s in eeg_ids])

    # Impute, standardize per modality
    eeg_imp = SimpleImputer(strategy='constant', fill_value=0.0).fit_transform(eeg_rows)
    audio_imp = SimpleImputer(strategy='constant', fill_value=0.0).fit_transform(audio_rows)
    psych_imp = SimpleImputer(strategy='constant', fill_value=0.0).fit_transform(psych_rows)

    eeg_sc = StandardScaler().fit_transform(eeg_imp)
    audio_sc = StandardScaler().fit_transform(audio_imp)
    psych_sc = StandardScaler().fit_transform(psych_imp)

    X = np.hstack([eeg_sc, audio_sc, psych_sc])
    feat_names = [f'eeg_{f}' for f in eeg_feats] + [f'audio_{f}' for f in audio_feats] + [f'psych_{f}' for f in ['gender','age','education','PHQ-9','GAD-7','PSQI']]

    return X, y, np.array(eeg_ids), feat_names
