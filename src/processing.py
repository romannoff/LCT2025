import pandas as pd
import numpy as np
from collections import deque
import time
from catboost import CatBoostClassifier

from process_functions import extract_ctg_features, extract_basic_stats_streaming, extract_accel_decel_features_streaming
# from process_f2 import extract_ctg_features2


class StreamCTGAnalyzer:
    """
    Потоковый анализатор КТГ данных для предсказания гипоксии плода.
    Обновляет признаки каждые 10 секунд на основе накопленных данных.
    """
    
    def __init__(self, patient_id=0, sampling_rate=4.0, update_interval=10.0, graph_update_interval=1.0):
        # self.patient_id = patient_id
        self.sampling_rate = sampling_rate  
        self.update_interval = update_interval  # секунды между обновлениями
        self.graph_update_interval = graph_update_interval  # секунды между обновлениями
        self.last_update_time = None
        self.last_graph_update_time = None
        
        # Буферы данных
        self.bpm_buffer = deque()
        self.uterus_buffer = deque()

        self.bpm_last_graph_index = 0
        self.uterus_last_graph_index = 0
        
        # Текущие признаки
        self.current_features = {'patient_id': patient_id, 'bpm_median': 0}

        self.model = CatBoostClassifier()
        self.model.load_model('models/model.cbm')

        # Промежуточные статистики для потокового вычисления статистик
        self.bpm_stats = None
        self.uterus_stats = None

        self.accel_deccel_stat = {

            'current_state': 'normal',  # 'normal', 'accel', 'decel'
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

        if (current_time - self.last_graph_update_time) >= self.graph_update_interval:
            if len(self.bpm_buffer) == 0:
                events['bpm_value'] = 0
            elif len(self.bpm_buffer) == self.bpm_last_graph_index:
                events['bpm_value'] = self.bpm_buffer[-1]
            else:
                events['bpm_value'] = np.median(list(self.bpm_buffer)[self.bpm_last_graph_index:])
            
            if len(self.uterus_buffer) == 0:
                events['uterus_value'] = 0
            elif len(self.uterus_buffer) == self.uterus_last_graph_index:
                events['uterus_value'] = self.uterus_buffer[-1]
            else:
                events['uterus_value'] = np.median(list(self.uterus_buffer)[self.uterus_last_graph_index:])

            self.bpm_last_graph_index = len(self.bpm_buffer)
            self.uterus_last_graph_index = len(self.uterus_buffer)
            self.last_graph_update_time = time.time()

        # Если прошло определённое количество секунд, то считаем признаки и ответ модели
        if (current_time - self.last_update_time) >= self.update_interval:
            prob = self._update_all_features()
            events['prob'] = prob
            self.last_update_time = time.time()
        
        return events

    
    def _update_all_features(self):
        """
        Выделение признаков + ответ модели
        """
        start = time.time()
        self.current_features = extract_ctg_features(self.bpm_buffer, self.uterus_buffer, self.current_features)
        # e_1 = extract_ctg_features(self.bpm_buffer, self.uterus_buffer, self.current_features)
        # e_2 = extract_ctg_features2(self.bpm_buffer, self.uterus_buffer, self.current_features)
        # print(e_1)
        # print()
        # print(e_2)
        # print()
        process_fin = time.time()
        print('Признаки посчитаны за:', process_fin - start)

        model_answer = self.model.predict_proba(pd.DataFrame(self.current_features, index=[0]))[:, 1][0]
        # print(self.model.predict_proba(pd.DataFrame(e_1, index=[0]))[:, 1][0])
        # print(self.model.predict_proba(pd.DataFrame(e_2, index=[0]))[:, 1][0])

        final_answer = 'Возможна гипоксия плода' if model_answer > 0.5 else 'Гипоксия плода не обнаружена'
        print('Результат: ', model_answer, f'({final_answer})')  
        print('Модель дала ответ за:', time.time() - process_fin)
        print('')
        return model_answer


