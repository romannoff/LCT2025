import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks, butter, filtfilt
import warnings
import math
from collections import deque
warnings.filterwarnings('ignore')

def extract_ctg_features(bpm, uterus, features, patient_id=0, sampling_rate=4.0):
    """
    Извлекает комплекс признаков из данных фетального монитора.
    
    Parameters:
    -----------
    bpm_df : pd.DataFrame
        DataFrame с колонкой 'value' - значения ЧСС плода
    uterus_df : pd.DataFrame  
        DataFrame с колонкой 'value' - значения маточной активности
    features : dict
        Выявленные признаки
    patient_id : str
        Идентификатор пациента
    sampling_rate : float
        Частота дискретизации данных (Гц). По умолчанию 4 Гц (стандарт для КТГ)
    
    Returns:
    --------
    pd.DataFrame с одной строкой признаков для пациента
    """
    
    bpm_values = np.array(bpm)
    uterus_values = np.array(uterus)
    
    # Минимальная длина для анализа
    # min_length = int(10 * 60 * sampling_rate)  # 10 минут данных
    # if len(bpm_values) < min_length:
        # raise ValueError(f"Данные слишком короткие: {len(bpm_values)} точек, нужно минимум {min_length}")
    
    # features = {}
    # features['patient_id'] = patient_id
    
    # 1. БАЗОВЫЕ СТАТИСТИЧЕСКИЕ ПРИЗНАКИ ДЛЯ ЧСС
    features.update(_extract_basic_stats(bpm_values, 'bpm'))
    
    # 2. МЕДИЦИНСКИЕ ПАРАМЕТРЫ ЧСС
    features.update(_extract_medical_features(bpm_values, features, sampling_rate))
    
    # 3. ПРИЗНАКИ ВАРИАБЕЛЬНОСТИ
    features.update(_extract_variability_features(bpm_values, features, sampling_rate))
    
    # 4. ПРИЗНАКИ ДЕЦЕЛЕРАЦИЙ И АКЦЕЛЕРАЦИЙ
    # features.update(_extract_accel_decel_features(bpm_values, features, sampling_rate))
    
    # 5. СТАТИСТИЧЕСКИЕ ПРИЗНАКИ ДЛЯ МАТОЧНОЙ АКТИВНОСТИ
    features.update(_extract_basic_stats(uterus_values, 'uterus'))
    features.update(_extract_uterine_activity_features(uterus_values, features, sampling_rate))
    
    # 6. ПРИЗНАКИ ВЗАИМОСВЯЗИ ЧСС И МАТОЧНОЙ АКТИВНОСТИ
    features.update(_extract_correlation_features(bpm_values, uterus_values, sampling_rate))
    
    # 7. СПЕКТРАЛЬНЫЕ ПРИЗНАКИ
    features.update(_extract_spectral_features(bpm_values, sampling_rate))
    
    # 8. НЕЛИНЕЙНЫЕ ПРИЗНАКИ
    features.update(_extract_nonlinear_features(bpm_values))
    
    return features

def _extract_basic_stats(signal, prefix):
    """Базовые статистические признаки"""
    return {
        # f'{prefix}_mean': np.mean(signal),            ##
        # f'{prefix}_std': np.std(signal),              ##
        f'{prefix}_median': np.median(signal),        ##
        # f'{prefix}_min': np.min(signal),              ##
        # f'{prefix}_max': np.max(signal),              ##
        f'{prefix}_q25': np.percentile(signal, 25),   ##
        f'{prefix}_q75': np.percentile(signal, 75),   ##
        f'{prefix}_skew': stats.skew(signal),         ##
        f'{prefix}_kurtosis': stats.kurtosis(signal), ##
        # f'{prefix}_cv': np.std(signal) / np.mean(signal) if np.mean(signal) != 0 else 0  # коэффициент вариации
    }

def extract_basic_stats_streaming(new_value, features, events, state=None, prefix=""):
    """
    Базовые статистические признаки для потоковых данных.
    
    Args:
        new_value: Новое значение сигнала
        features: Итоговые статистики, доставаемые из рядов
        state: Состояние статистик (None при инициализации)
        prefix: Префикс для названий признаков
    
    Returns:
        tuple: (новое_состояние, словарь_статистик)
    """
    if state is None:
        # Инициализация состояния
        state = {
            'n': 0,
            'mean': 0.0,
            'm2': 0.0,  # для дисперсии
            'min_val': float('inf'),
            'max_val': float('-inf'),
            'tachycard_n': 0,
            'bradicard_n': 0
        }
    
    n_prev = state['n']
    n = n_prev + 1
    
    # Обновление минимума и максимума
    min_val = min(state['min_val'], new_value)
    max_val = max(state['max_val'], new_value)
    
    # Инкрементальное среднее
    delta = new_value - state['mean']
    mean = state['mean'] + delta / n
    
    # Инкрементальная дисперсия (алгоритм Уэлфорда)
    delta2 = new_value - mean
    m2 = state['m2'] + delta * delta2
    
    # Стандартное отклонение
    std = math.sqrt(m2 / n) if n > 1 else 0.0
    
    # Коэффициент вариации
    cv = std / mean if mean != 0 else 0



    # Тахикардия/брадикардия индикаторы
    if prefix == 'bpm':
        if new_value > 160:
            print(f'Повышенный ЧСС плода ({int(new_value)} уд./мин.)')
            events['tachycard'] = 1
            state['tachycard_n'] += 1
        if new_value < 110:
            print(f'Пониженный ЧСС плода ({int(new_value)} уд./мин.)')
            events['bradicard'] = 1 
            state['bradicard_n'] += 1  

    
    # Подготовка нового состояния
    new_state = {
        'n': n,
        'mean': mean,
        'm2': m2,
        'min_val': min_val,
        'max_val': max_val,
        'tachycard_n': state['tachycard_n'],
        'bradicard_n': state['bradicard_n']
    }


    features[f'{prefix}_mean'] = mean
    features[f'{prefix}_std'] = std
    features[f'{prefix}_min'] = min_val
    features[f'{prefix}_max'] = max_val
    features[f'{prefix}_cv'] = cv

    if prefix == 'bpm':
        features['tachycardia_percentage'] = state['tachycard_n'] / n * 100
        features['bradycardia_percentage'] = state['bradicard_n'] / n * 100
    
    # stats = {
    #     f'{prefix}_mean': mean,
    #     f'{prefix}_std': std,
    #     f'{prefix}_min': min_val,
    #     f'{prefix}_max': max_val,
    #     f'{prefix}_cv': cv,
    # }
    
    return new_state

def _extract_medical_features(bpm, features, sampling_rate):
    """Медицинские параметры ЧСС согласно клиническим протоколам"""
    # Базальный ритм (медиана за весь период, исключая экстремальные значения)
    q25, q75 = features['bpm_q25'], features['bpm_q75']
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    
    basal_mask = (bpm >= lower_bound) & (bpm <= upper_bound)
    basal_rate = np.median(bpm[basal_mask]) if np.any(basal_mask) else features['bpm_median']
    
    
    # tachycardia_perc = np.sum(bpm > 160) / len(bpm) * 100
    # bradycardia_perc = np.sum(bpm < 110) / len(bpm) * 100
    
    return {
        'basal_rate': basal_rate,
        # 'tachycardia_percentage': tachycardia_perc,
        # 'bradycardia_percentage': bradycardia_perc,
        'basal_rate_category': 1 if basal_rate > 160 else (2 if basal_rate < 110 else 0)  # 0-норма, 1-тахи, 2-бради
    }

def _extract_variability_features(bpm, features, sampling_rate):
    """Признаки вариабельности сердечного ритма"""
    # Долгосрочная вариабельность (LTV)
    # ltv = np.std(bpm)
    ltv = features['bpm_std']
    
    # Краткосрочная вариабельность (STV) - средняя разница между соседними точками
    stv = np.mean(np.abs(np.diff(bpm)))
    
    # Вариабельность в скользящих окнах
    window_size = int(3.75 * sampling_rate)  # 3.75 минуты для оценки осцилляций
    if len(bpm) > window_size:
        windows = len(bpm) - window_size + 1
        window_vars = [np.std(bpm[i:i+window_size]) for i in range(0, windows, int(sampling_rate))]
        oscillations_per_min = len([v for v in window_vars if v > 5]) / (len(bpm) / sampling_rate / 60)
    else:
        oscillations_per_min = 0
    
    return {
        'ltv': ltv,
        'stv': stv,
        'oscillations_per_min': oscillations_per_min,
        'variability_ratio': stv / ltv if ltv != 0 else 0
    }

def extract_accel_decel_features_streaming(bpm, features, sampling_rate, stats, events):
    """Выявление акцелераций и децелераций в потоке данных"""
    
    if features['bpm_median'] == 0:
        features['accelerations_count'] = 0
        features['decelerations_count'] = 0
        features['accelerations_per_min'] = 0
        features['decelerations_per_min'] = 0
        features['acceleration_total_duration'] = 0
        features['deceleration_total_duration'] = 0
        return stats


    stats['basal_rate'] = features['bpm_median']
    
    # Пороги для акцелераций/децелераций
    accel_threshold = stats['basal_rate'] + 15
    decel_threshold = stats['basal_rate'] - 15
    
    # Минимальная длительность в точках (15 секунд)
    min_duration_points = int(15 * sampling_rate)
    
    # Увеличиваем счетчик обработанных точек
    stats['total_points'] += 1
    current_index = stats['total_points'] - 1
    
    # Текущее состояние
    current_state = stats['current_state']
    current_duration = stats['current_duration']
    
    # Определяем тип текущего значения
    if bpm > accel_threshold:
        new_state = 'accel'
    elif bpm < decel_threshold:
        new_state = 'decel'
    else:
        new_state = 'normal'
    
    # Обработка изменения состояния
    if current_state != new_state:
        # Завершаем предыдущую серию если она была достаточно длинной
        if current_duration >= min_duration_points:
            duration_seconds = current_duration / sampling_rate
            
            if current_state == 'accel':
                features['accelerations_count'] += 1
                features['acceleration_total_duration'] += duration_seconds
                events['acceleration'] = 1
                print('Завершена акцелерация')
            elif current_state == 'decel':
                features['decelerations_count'] += 1
                features['deceleration_total_duration'] += duration_seconds
                events['deceleration'] = 1
                print('Завершена децелерация')
        
        # Начинаем новую серию
        stats['current_state'] = new_state
        stats['current_duration'] = 1
        stats['current_start_index'] = current_index
        
    else:
        # Продолжаем текущую серию
        stats['current_duration'] += 1
    
    # Расчет features в реальном времени
    total_duration_min = stats['total_points'] / sampling_rate / 60
    
    features['accelerations_per_min'] = (features['accelerations_count'] / total_duration_min 
                                    if total_duration_min > 0 else 0)
    features['decelerations_per_min'] = (features['decelerations_count'] / total_duration_min 
                                    if total_duration_min > 0 else 0)
    
    return stats

def _extract_accel_decel_features(bpm, features, sampling_rate):
    """Выявление акцелераций и децелераций"""
    # basal_rate = np.median(bpm)
    basal_rate = features['bpm_median']
    
    # Пороги для акцелераций/децелераций (15 уд/мин от базального ритма)
    accel_threshold = basal_rate + 15
    decel_threshold = basal_rate - 15
    
    # Минимальная длительность в точках (15 секунд)
    min_duration_points = int(15 * sampling_rate)
    
    accel_count = 0
    decel_count = 0
    accel_total_duration = 0
    decel_total_duration = 0
    
    i = 0
    while i < len(bpm):
        # Проверка акцелерации
        if bpm[i] > accel_threshold:
            duration = 1
            while i + duration < len(bpm) and bpm[i + duration] > accel_threshold:
                duration += 1
            
            if duration >= min_duration_points:
                accel_count += 1
                print('Выявлена акцелерация')
                accel_total_duration += duration / sampling_rate  # в секундах
            i += duration
        # Проверка децелерации
        elif bpm[i] < decel_threshold:
            duration = 1
            while i + duration < len(bpm) and bpm[i + duration] < decel_threshold:
                duration += 1
            
            if duration >= min_duration_points:
                decel_count += 1
                print('Выявлена децелерация')
                decel_total_duration += duration / sampling_rate  # в секундах
            i += duration
        else:
            i += 1
    
    total_duration_min = len(bpm) / sampling_rate / 60
    
    return {
        'accelerations_count': accel_count,
        'decelerations_count': decel_count,
        'accelerations_per_min': accel_count / total_duration_min if total_duration_min > 0 else 0,
        'decelerations_per_min': decel_count / total_duration_min if total_duration_min > 0 else 0,
        'acceleration_total_duration': accel_total_duration,
        'deceleration_total_duration': decel_total_duration
    }

def _extract_uterine_activity_features(uterus, features, sampling_rate):
    """Анализ маточной активности"""
    # Поиск сокращений (пиков выше порога)
    # threshold = np.mean(uterus) + np.std(uterus)
    threshold = features['uterus_mean'] + features['uterus_std']
    peaks, properties = find_peaks(uterus, height=threshold, distance=int(60*sampling_rate))
    
    contraction_count = len(peaks)
    contraction_amplitudes = properties['peak_heights'] if len(peaks) > 0 else [0]
    
    # Анализ интервалов между сокращениями
    intervals = np.diff(peaks) / sampling_rate if len(peaks) > 1 else [0]
    
    return {
        'uterine_contractions_count': contraction_count,
        'uterine_mean_amplitude': np.mean(contraction_amplitudes),
        'uterine_max_amplitude': np.max(contraction_amplitudes) if len(peaks) > 0 else 0,
        'uterine_mean_interval': np.mean(intervals) if len(intervals) > 0 else 0,
        'uterine_interval_std': np.std(intervals) if len(intervals) > 0 else 0
    }

def _extract_correlation_features(bpm, uterus, sampling_rate):
    """Признаки взаимосвязи ЧСС и маточной активности"""
    # Корреляция между сигналами
    if len(bpm) == len(uterus):
        correlation = np.corrcoef(bpm, uterus)[0, 1] if not (np.all(bpm == bpm[0]) or np.all(uterus == uterus[0])) else 0
    else:
        min_len = min(len(bpm), len(uterus))
        correlation = np.corrcoef(bpm[:min_len], uterus[:min_len])[0, 1] if min_len > 0 else 0
    
    # Задержка между пиком сокращения и минимальным значением ЧСС
    uterus_peaks, _ = find_peaks(uterus, height=np.mean(uterus) + np.std(uterus), distance=int(120*sampling_rate))
    delays = []
    
    for peak in uterus_peaks[:5]:  # анализируем первые 5 сокращений
        if peak + int(60*sampling_rate) < len(bpm):
            window = bpm[peak:peak + int(60*sampling_rate)]
            if len(window) > 0:
                min_idx = np.argmin(window)
                delays.append(min_idx / sampling_rate)  # задержка в секундах
    
    mean_delay = np.mean(delays) if delays else 0
    
    return {
        'bpm_uterus_correlation': correlation,
        'mean_contraction_delay': mean_delay,
        'delayed_decelerations_present': 1 if mean_delay > 10 else 0  # децелерации с задержкой >10 сек
    }

def _extract_spectral_features(bpm, sampling_rate):
    """Спектральные признаки вариабельности"""
    from scipy.signal import periodogram
    
    try:
        f, Pxx = periodogram(bpm, fs=sampling_rate)
        
        # Полосы частот для анализа вариабельности
        vlf_band = (0.003, 0.04)    # Very Low Frequency
        lf_band = (0.04, 0.15)      # Low Frequency  
        hf_band = (0.15, 0.4)       # High Frequency
        
        vlf_power = np.sum(Pxx[(f >= vlf_band[0]) & (f < vlf_band[1])])
        lf_power = np.sum(Pxx[(f >= lf_band[0]) & (f < lf_band[1])])
        hf_power = np.sum(Pxx[(f >= hf_band[0]) & (f < hf_band[1])])
        total_power = vlf_power + lf_power + hf_power
        
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
        
        return {
            'spectral_vlf_power': vlf_power,
            'spectral_lf_power': lf_power,
            'spectral_hf_power': hf_power,
            'spectral_total_power': total_power,
            'lf_hf_ratio': lf_hf_ratio,
            'spectral_entropy': stats.entropy(Pxx[Pxx > 0]) if np.any(Pxx > 0) else 0
        }
    except:
        return {k: 0 for k in ['spectral_vlf_power', 'spectral_lf_power', 'spectral_hf_power', 
                              'spectral_total_power', 'lf_hf_ratio', 'spectral_entropy']}

def _extract_nonlinear_features(bpm):
    """Нелинейные динамические признаки"""
    # Sample Entropy (упрощенная версия)
    def sample_entropy(signal, m=2, r=0.2):
        if len(signal) < m + 1:
            return 0
            
        std_signal = np.std(signal)
        if std_signal == 0:
            return 0
            
        r = r * std_signal
        
        def _maxdist(x, y):
            return np.max(np.abs(x - y))
        
        def _phi(m):
            patterns = [signal[i:i+m] for i in range(len(signal) - m + 1)]
            count = 0
            for i in range(len(patterns)):
                for j in range(i+1, len(patterns)):
                    if _maxdist(patterns[i], patterns[j]) <= r:
                        count += 1
            return count / (len(patterns) * (len(patterns) - 1) / 2) if len(patterns) > 1 else 0
        
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        
        return -np.log(phi_m1 / phi_m) if phi_m1 > 0 and phi_m > 0 else 0
    
    # sampen = sample_entropy(bpm)
    
    # Детерминированный хаос (упрощенно)
    diff_signal = np.diff(bpm)
    chaos_measure = np.std(diff_signal) / np.std(bpm) if np.std(bpm) > 0 else 0
    
    return {
        'chaos_measure': chaos_measure,
    }