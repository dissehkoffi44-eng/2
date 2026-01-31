# RCDJ228 SNIPER M3 - VERSION FUSIONNÉE (MOTEUR CODE 2 + ROBUSTESSE CODE 1)
# Avec détection moment modulation + % en target + fin en target
# + Conseils de mix harmonique basés sur la checklist
# + CONSEIL RAPIDE MIX dans le rapport Telegram (version ultra-résumée)
# MODIFS : Améliorations pour précision harmonique et perception humaine

import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import os
import requests
import gc
import json
import streamlit.components.v1 as components
from scipy.signal import butter, lfilter
from datetime import datetime
from pydub import AudioSegment
from hmmlearn import hmm  # AJOUT : Pour modèle HMM (pip install hmmlearn si besoin, mais assumez disponible)

# --- FORCE FFMPEG PATH (WINDOWS FIX) ---
if os.path.exists(r'C:\ffmpeg\bin'):
    os.environ["PATH"] += os.pathsep + r'C:\ffmpeg\bin'

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="RCDJ228 MUSIC SNIPER", page_icon="🎯", layout="wide")

# Récupération des secrets
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- RÉFÉRENTIELS HARMONIQUES ---
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MODES = ['major', 'minor', 'mixolydian', 'dorian', 'phrygian', 'lydian', 'locrian', 'aeolian', 'ionian']  # Ajout des modes étendus
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in MODES]

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
    # Note: Pour modes étendus, mapper à maj/min équivalents pour Camelot
}

# AJOUT : Profils étendus pour modes (basés sur intervalles théoriques, adaptés de Krumhansl)
PROFILES = {
    "krumhansl": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
        "mixolydian": [6.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 5.0, 2.0, 3.5, 4.0, 2.0],  # Flattened 7th boosted
        "dorian": [6.0, 2.0, 4.5, 5.0, 2.0, 4.0, 2.0, 5.0, 3.5, 2.0, 4.0, 2.0],    # Minor with raised 6th
        "phrygian": [6.0, 4.5, 3.5, 5.0, 2.0, 4.0, 2.0, 5.0, 3.5, 2.0, 2.0, 2.0],   # Flattened 2nd
        "lydian": [6.0, 2.0, 3.5, 2.0, 5.0, 4.0, 2.0, 5.0, 2.0, 3.5, 2.0, 4.0],     # Raised 4th
        "locrian": [6.0, 4.0, 3.5, 5.0, 4.0, 2.0, 2.0, 5.0, 3.5, 2.0, 2.0, 2.0],    # Diminished feel
        "aeolian": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],  # Same as minor
        "ionian": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]    # Same as major
    },
    # Ajouter Temperley et Bellman de manière similaire, en adaptant pour chaque mode (fallback si non défini)
    "temperley": { ... },  # Copiez et adaptez comme ci-dessus
    "bellman": { ... }     # Idem
}

# ... (Le reste du code pour get_neighbor_camelot, get_mixing_advice, get_mode_intervals, get_diatonic_chords, etc. reste identique)

# MODIF : Fonction de filtrage avec poids énergétique pour perception humaine
def apply_sniper_filters(y, sr):
    y_harm = librosa.effects.harmonic(y, margin=4.0)
    nyq = 0.5 * sr
    low = 80 / nyq
    high = 5000 / nyq
    b, a = butter(4, [low, high], btype='band')
    y_filt = lfilter(b, a, y_harm)
    # AJOUT : Calculer énergie pour pondérer segments (oreille ignore sections faibles)
    rmse = librosa.feature.rms(y=y_filt)
    return y_filt, rmse

# ... (get_bass_priority reste identique)

# MODIF : Solve_key avec profils étendus et bonus perceptuels renforcés
def solve_key_sniper(chroma_vector, bass_vector, energy_weight=1.0):
    best_overall_score = -1
    best_key = "Unknown"
    
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    bv = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-6)
    
    key_scores = {f"{NOTES_LIST[i]} {mode}": [] for mode in MODES for i in range(12)}
    
    for p_name, p_data in PROFILES.items():
        for mode in MODES:
            # MODIF : Utiliser profil spécifique si disponible, sinon fallback
            profile_mode = mode if mode in p_data else ("major" if mode in ['ionian', 'mixolydian', 'lydian'] else "minor")
            for i in range(12):
                score = np.corrcoef(cv, np.roll(p_data[profile_mode], i))[0, 1]
                
                # MODIF : Bonus renforcés pour perception (leading tone, dominant, third/fifth)
                dom_idx = (i + 7) % 12
                leading_tone = (i + 11) % 12
                if 'minor' in mode or mode in ['aeolian', 'dorian', 'phrygian', 'locrian']:
                    if cv[leading_tone] > 0.35: score *= 1.4  # Boosté
                    if cv[dom_idx] > 0.5: score *= 1.2
                else:
                    if cv[i] > 0.75 and cv[dom_idx] > 0.65: score *= 1.15
                
                if bv[i] > 0.65: score += (bv[i] * 0.3)  # Bass plus pondéré
                
                third_idx = (i + 4 if 'major' in mode else i + 3) % 12
                if cv[third_idx] > 0.55: score += 0.2  # Tierce boostée
                
                fifth_idx = (i + 7) % 12
                if cv[fifth_idx] > 0.55: score += 0.15
                
                # AJOUT : Pondération par énergie (sections fortes plus importantes)
                score *= energy_weight
                
                key_name = f"{NOTES_LIST[i]} {mode}"
                key_scores[key_name].append(score)
    
    for key_name, scores in key_scores.items():
        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_overall_score:
                best_overall_score = avg_score
                best_key = key_name
    
    return {"key": best_key, "score": best_overall_score}

# ... (seconds_to_mmss, test_chord_consonance restent identiques)

# MODIF MAJEURE : Process_audio avec segmentation structurelle et HMM
def process_audio_precision(file_bytes, file_name, _progress_callback=None):
    # ... (Lecture audio identique)
    
    duration = librosa.get_duration(y=y, sr=sr)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_filt, rmse = apply_sniper_filters(y, sr)  # MODIF : Récup énergie

    # AJOUT : Segmentation structurelle (détecter boundaries pour pondérer sections)
    novelty = librosa.onset.onset_strength_multi(y=y_filt, sr=sr)
    bounds = librosa.segment.agglomerative(novelty, k=5)  # 5 sections approx (intro/verse/chorus/etc.)
    section_weights = np.ones(len(bounds))  # Pondérer chorus (supposé sections centrales plus énergétiques)
    section_energies = [np.mean(rmse[bound_start:bound_end]) for bound_start, bound_end in zip(bounds[:-1], bounds[1:])]
    max_energy = max(section_energies)
    section_weights = [energy / max_energy for energy in section_energies]  # Poids 0-1

    step, timeline, votes = 6, [], Counter()
    segments = list(range(0, max(1, int(duration) - step), 2))
    total_segments = len(segments)
    
    chroma_segments = []  # Pour HMM
    
    for idx, start in enumerate(segments):
        # ... (Progress callback identique)
        
        idx_start, idx_end = int(start * sr), int((start + step) * sr)
        seg = y_filt[idx_start:idx_end]
        if len(seg) < 1000 or np.max(np.abs(seg)) < 0.01: continue
        
        # AJOUT : Trouver section et poids
        section_idx = np.searchsorted(bounds, start) - 1
        energy_weight = section_weights[section_idx] if section_idx < len(section_weights) else 1.0
        
        c_raw = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, n_chroma=24, bins_per_octave=24)
        c_avg = np.mean((c_raw[::2, :] + c_raw[1::2, :]) / 2, axis=1)
        b_seg = get_bass_priority(y[idx_start:idx_end], sr)
        res = solve_key_sniper(c_avg, b_seg, energy_weight)  # MODIF : Passer poids
        
        if res['score'] < 0.9: continue
        
        weight = 2.5 if (start < 10 or start > (duration - 15)) else 1.0  # Boosté
        votes[res['key']] += int(res['score'] * 100 * weight * energy_weight)  # AJOUT : * energy_weight
        
        timeline.append({"Temps": start, "Note": res['key'], "Conf": res['score']})
        chroma_segments.append(c_avg)  # Pour HMM

    if not votes:
        return None

    # AJOUT : HMM pour modéliser transitions (perception de continuité)
    if len(chroma_segments) > 10:
        model = hmm.GaussianHMM(n_components=3, covariance_type="diag")  # 3 états (clé princ, mod, retour)
        model.fit(np.array(chroma_segments))
        hidden_states = model.predict(np.array(chroma_segments))
        # Ajuster votes basé sur états (ex. : états stables boostés)
        for i, state in enumerate(hidden_states):
            if state == 0:  # Assume 0 = stable
                votes[timeline[i]['Note']] *= 1.2

    # ... (Le reste : most_common, mod_detected, modulation_time, etc. identique)

    # ... (Chroma_avg, top_notes, matches, ajustement final_key avec match boosté à 0.4 au lieu de 0.3)

    # ... (Génération accords, Telegram, etc. identique)

    return res_obj

# ... (Le reste de l'interface principale reste identique)
