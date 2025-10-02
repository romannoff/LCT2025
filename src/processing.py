import pandas as pd
import numpy as np
from collections import deque
import time
from catboost import CatBoostClassifier

from process_functions import extract_ctg_features, extract_basic_stats_streaming, extract_accel_decel_features_streaming

class StreamCTGAnalyzer:
    """
    Потоковый анализатор КТГ данных для предсказания гипоксии плода.
    Обновляет признаки каждые 10 секунд на основе накопленных данных.
    """
    
    def __init__(self, patient_id=0, sampling_rate=4.0, update_interval=10.0, graph_update_interval=1.0):
        self.sampling_rate = sampling_rate  
        self.update_interval = update_interval
        self.graph_update_interval = graph_update_interval
        self.last_update_time = None
        self.last_graph_update_time = None
        
        self.bpm_buffer = deque()
        self.uterus_buffer = deque()

        self.bpm_last_graph_index = 0
        self.uterus_last_graph_index = 0
        
        self.current_features = {'patient_id': patient_id, 'bpm_median': 0, 'ltv': 0}

        self.model = CatBoostClassifier()
        self.model.load_model('models/model.cbm')

        self.bpm_stats = None
        self.uterus_stats = None

        self.accel_deccel_stat = {
            'current_state': 'normal',
            'current_duration': 0,
            'current_start_index': 0,
            'basal_rate': 0,
            'total_points': 0,
            'accelerations_count': 0,
            'decelerations_count': 0,
            'acceleration_total_duration': 0.0,
            'deceleration_total_duration': 0.0
        }
        
    def add_measurement(self, value, value_type):
        """
        Добавляет новое измерение в соответствующий буфер.
        """
        if self.last_update_time is None:
            self.last_update_time = time.time()
            self.last_graph_update_time = time.time()

        events = dict()

        if value_type == 'bpm':
            self.bpm_buffer.append(value)
            self.bpm_stats = extract_basic_stats_streaming(value, self.current_features, events, self.bpm_stats, value_type) 
            self.accel_deccel_stat = extract_accel_decel_features_streaming(value, self.current_features, 4, self.accel_deccel_stat, events)

        elif value_type == 'uterus':
            self.uterus_buffer.append(value)
            self.uterus_stats = extract_basic_stats_streaming(value, self.current_features, events, self.uterus_stats, value_type)

        current_time = time.time()

        # ИСПРАВЛЕННАЯ ЛОГИКА ОБНОВЛЕНИЯ ГРАФИКОВ
        if (current_time - self.last_graph_update_time) >= self.graph_update_interval:
            # Отправляем медиану новых значений BPM, если они есть, иначе последнее известное значение
            if len(self.bpm_buffer) > self.bpm_last_graph_index:
                events['bpm_value'] = np.median(list(self.bpm_buffer)[self.bpm_last_graph_index:])
            elif self.bpm_buffer:
                events['bpm_value'] = self.bpm_buffer[-1]
            
            # Отправляем медиану новых значений Uterus, если они есть, иначе последнее известное значение
            if len(self.uterus_buffer) > self.uterus_last_graph_index:
                events['uterus_value'] = np.median(list(self.uterus_buffer)[self.uterus_last_graph_index:])
            elif self.uterus_buffer:
                events['uterus_value'] = self.uterus_buffer[-1]

            self.bpm_last_graph_index = len(self.bpm_buffer)
            self.uterus_last_graph_index = len(self.uterus_buffer)
            self.last_graph_update_time = time.time()

        if (current_time - self.last_update_time) >= self.update_interval:
            prob, risk_factors = self._update_all_features()
            events['prob'] = prob
            if risk_factors:
                events['risk_factors'] = risk_factors
            self.last_update_time = time.time()
        
        return events
    
    def _update_all_features(self):
        """
        Выделение признаков, ответ модели и определение факторов риска.
        Возвращает: (вероятность, список_факторов_риска)
        """
        if not self.bpm_buffer or not self.uterus_buffer:
            return 0.0, []

        start = time.time()
        self.current_features = extract_ctg_features(self.bpm_buffer, self.uterus_buffer, self.current_features)
        process_fin = time.time()
        print('Признаки посчитаны за:', process_fin - start)
        
        features_df = pd.DataFrame(self.current_features, index=[0])
        # for col in self.model.feature_names_:
            # if col not in features_df.columns:
                # features_df[col] = 0

        model_answer = self.model.predict_proba(features_df)[:, 1][0]

        final_answer = 'Возможна гипоксия плода' if model_answer > 0.5 else 'Гипоксия плода не обнаружена'
        print('Результат: ', model_answer, f'({final_answer})')  
        print('Модель дала ответ за:', time.time() - process_fin)
        print('')

        risk_factors = []
        if model_answer > 0.4:
            if self.current_features.get('decelerations_per_min', 0) > 0.3:
                risk_factors.append("Частые децелерации")
            if self.current_features.get('ltv', 10) < 5:
                risk_factors.append("Сниженная вариабельность ритма (LTV)")
            if self.current_features.get('basal_rate_category', 0) == 1:
                risk_factors.append("Базальная тахикардия (>160 уд/мин)")
            if self.current_features.get('basal_rate_category', 0) == 2:
                risk_factors.append("Базальная брадикардия (<110 уд/мин)")
            if self.current_features.get('tachycardia_percentage', 0) > 10:
                risk_factors.append("Эпизоды тахикардии")

        return model_answer, risk_factors