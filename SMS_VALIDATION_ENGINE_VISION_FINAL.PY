# =============================================================================
# SMC INSTITUTIONAL LOGIC VALIDATION ENGINE v3.0 - VISUAL AI (FINAL)
# Otimizado para execução longa em CPU com qwen2.5vl
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum
import warnings
import logging
import time
import sys
import re
import json
import base64
import io
import os
from datetime import datetime
from PIL import Image

# Ollama client
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("ERRO: Pacote 'ollama' não instalado. Execute: pip install ollama")
    sys.exit(1)

# MetaTrader5 (opcional)
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# Statsmodels (opcional)
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('smc_vision_engine.log')
    ]
)

# =============================================================================
# 1. DATA ENGINE
# =============================================================================
class DataEngine:
    def __init__(self, df: pd.DataFrame, timeframe_name: str = 'base'):
        self.raw = df.copy()
        self.timeframe_name = timeframe_name
        self._validate_raw()
        self.df = self._clean_and_normalize(self.raw)

    def _validate_raw(self):
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in self.raw.columns]
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing}")
        if not isinstance(self.raw.index, pd.DatetimeIndex):
            try:
                self.raw.index = pd.to_datetime(self.raw.index)
            except:
                raise TypeError("Índice deve ser DatetimeIndex ou conversível.")

    def _clean_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        freq = pd.infer_freq(df.index)
        if freq:
            df = df.asfreq(freq, method='ffill')
        df.dropna(inplace=True)
        return df


# =============================================================================
# 2. STRUCTURE ENGINE
# =============================================================================
class SwingMethod(Enum):
    FRACTAL_5 = "fractal_5"
    ZIGZAG_PCT = "zigzag_pct"

@dataclass
class SwingPoint:
    timestamp: pd.Timestamp
    price: float
    type: str
    method: SwingMethod

class StructureEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def detect_swings_fractal(self, window: int = 5) -> List[SwingPoint]:
        highs = self.df['high'].values
        lows = self.df['low'].values
        index = self.df.index
        swings = []
        n = len(self.df)
        for i in range(window, n - window):
            if highs[i] == np.max(highs[i-window : i+window+1]):
                swings.append(SwingPoint(index[i], highs[i], 'HH', SwingMethod.FRACTAL_5))
            if lows[i] == np.min(lows[i-window : i+window+1]):
                swings.append(SwingPoint(index[i], lows[i], 'LL', SwingMethod.FRACTAL_5))
        swings.sort(key=lambda x: x.timestamp)
        for i in range(2, len(swings)):
            a, b, c = swings[i-2], swings[i-1], swings[i]
            if a.type == 'HH' and c.type == 'HH' and b.price < a.price and b.price < c.price:
                b.type = 'HL'
            elif a.type == 'LL' and c.type == 'LL' and b.price > a.price and b.price > c.price:
                b.type = 'LH'
        return swings

    def build_structure_series(self, swings: List[SwingPoint]) -> pd.DataFrame:
        struct_df = pd.DataFrame(index=self.df.index, columns=['HH', 'HL', 'LH', 'LL'])
        current = {'HH': np.nan, 'HL': np.nan, 'LH': np.nan, 'LL': np.nan}
        swings_sorted = sorted(swings, key=lambda x: x.timestamp)
        idx = 0
        n_swings = len(swings_sorted)
        for timestamp in self.df.index:
            while idx < n_swings and swings_sorted[idx].timestamp <= timestamp:
                swing = swings_sorted[idx]
                current[swing.type] = swing.price
                idx += 1
            struct_df.loc[timestamp] = current
        struct_df.ffill(inplace=True)
        return struct_df


# =============================================================================
# 3. DEFINIÇÕES SMC
# =============================================================================
def detect_bos_A(df: pd.DataFrame, struct_eng: StructureEngine, params: Dict) -> pd.Series:
    swings = struct_eng.detect_swings_fractal(window=params.get('window', 20))
    struct_df = struct_eng.build_structure_series(swings)
    bos = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        prev_hh = struct_df['HH'].iloc[i-1]
        prev_ll = struct_df['LL'].iloc[i-1]
        if pd.notna(prev_hh) and df['close'].iloc[i] > prev_hh:
            bos.iloc[i] = 1
        elif pd.notna(prev_ll) and df['close'].iloc[i] < prev_ll:
            bos.iloc[i] = -1
    return bos

def detect_bos_B(df: pd.DataFrame, struct_eng: StructureEngine, params: Dict) -> pd.Series:
    swings = struct_eng.detect_swings_fractal(window=params.get('window', 20))
    struct_df = struct_eng.build_structure_series(swings)
    bos = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        prev_hh = struct_df['HH'].iloc[i-1]
        prev_ll = struct_df['LL'].iloc[i-1]
        if pd.notna(prev_hh) and df['high'].iloc[i] > prev_hh:
            bos.iloc[i] = 1
        elif pd.notna(prev_ll) and df['low'].iloc[i] < prev_ll:
            bos.iloc[i] = -1
    return bos

def detect_bos_C(df: pd.DataFrame, struct_eng: StructureEngine, params: Dict) -> pd.Series:
    swings = struct_eng.detect_swings_fractal(window=params.get('window', 20))
    struct_df = struct_eng.build_structure_series(swings)
    bos = pd.Series(0, index=df.index)
    atr = (df['high'] - df['low']).rolling(params.get('atr_period', 14)).mean()
    min_disp = params.get('min_displacement_atr', 0.5) * atr
    for i in range(1, len(df)):
        prev_hh = struct_df['HH'].iloc[i-1]
        prev_ll = struct_df['LL'].iloc[i-1]
        if pd.notna(prev_hh) and df['close'].iloc[i] > prev_hh + min_disp.iloc[i]:
            bos.iloc[i] = 1
        elif pd.notna(prev_ll) and df['close'].iloc[i] < prev_ll - min_disp.iloc[i]:
            bos.iloc[i] = -1
    return bos

def detect_bos_D(df: pd.DataFrame, struct_eng: StructureEngine, params: Dict) -> pd.Series:
    swings = struct_eng.detect_swings_fractal(window=params.get('window', 20))
    struct_df = struct_eng.build_structure_series(swings)
    bos = pd.Series(0, index=df.index)
    sma_fast = df['close'].rolling(50).mean()
    sma_slow = df['close'].rolling(200).mean()
    trend_up = sma_fast > sma_slow
    for i in range(1, len(df)):
        prev_hh = struct_df['HH'].iloc[i-1]
        prev_ll = struct_df['LL'].iloc[i-1]
        if pd.notna(prev_hh) and df['close'].iloc[i] > prev_hh and trend_up.iloc[i]:
            bos.iloc[i] = 1
        elif pd.notna(prev_ll) and df['close'].iloc[i] < prev_ll and not trend_up.iloc[i]:
            bos.iloc[i] = -1
    return bos

def detect_fvg_A(df: pd.DataFrame, struct_eng: StructureEngine, params: Dict) -> pd.Series:
    fvg = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        prev_high = df['high'].iloc[i-1]
        prev_low = df['low'].iloc[i-1]
        curr_low = df['low'].iloc[i]
        curr_high = df['high'].iloc[i]
        if curr_low > prev_high:
            fvg.iloc[i] = 1
        elif curr_high < prev_low:
            fvg.iloc[i] = -1
    return fvg

def detect_fvg_B(df: pd.DataFrame, struct_eng: StructureEngine, params: Dict) -> pd.Series:
    fvg = pd.Series(0, index=df.index)
    if 'volume' not in df.columns:
        return fvg
    avg_vol = df['volume'].rolling(20).mean()
    for i in range(1, len(df)):
        prev_high = df['high'].iloc[i-1]
        prev_low = df['low'].iloc[i-1]
        curr_high = df['high'].iloc[i]
        curr_low = df['low'].iloc[i]
        vol_cond = df['volume'].iloc[i] > avg_vol.iloc[i]
        if curr_low > prev_high and vol_cond:
            fvg.iloc[i] = 1
        elif curr_high < prev_low and vol_cond:
            fvg.iloc[i] = -1
    return fvg

def detect_choch_A(df: pd.DataFrame, struct_eng: StructureEngine, params: Dict) -> pd.Series:
    swings = struct_eng.detect_swings_fractal(window=params.get('window', 20))
    struct_df = struct_eng.build_structure_series(swings)
    choch = pd.Series(0, index=df.index)
    for i in range(2, len(df)):
        prev_lh = struct_df['LH'].iloc[i-1]
        prev_hl = struct_df['HL'].iloc[i-1]
        if pd.notna(prev_lh) and df['high'].iloc[i] > prev_lh and df['close'].iloc[i] > prev_lh:
            choch.iloc[i] = 1
        if pd.notna(prev_hl) and df['low'].iloc[i] < prev_hl and df['close'].iloc[i] < prev_hl:
            choch.iloc[i] = -1
    return choch

def detect_liquidity_sweep_A(df: pd.DataFrame, struct_eng: StructureEngine, params: Dict) -> pd.Series:
    swings = struct_eng.detect_swings_fractal(window=params.get('window', 20))
    struct_df = struct_eng.build_structure_series(swings)
    sweep = pd.Series(0, index=df.index)
    reversal_bars = params.get('reversal_bars', 3)
    for i in range(1, len(df)-reversal_bars):
        prev_hh = struct_df['HH'].iloc[i-1]
        prev_ll = struct_df['LL'].iloc[i-1]
        if pd.notna(prev_ll) and df['low'].iloc[i] < prev_ll:
            future_high = df['high'].iloc[i+1:i+reversal_bars+1].max()
            if future_high > prev_ll:
                sweep.iloc[i] = 1
        if pd.notna(prev_hh) and df['high'].iloc[i] > prev_hh:
            future_low = df['low'].iloc[i+1:i+reversal_bars+1].min()
            if future_low < prev_hh:
                sweep.iloc[i] = -1
    return sweep

def detect_orderblock_A(df: pd.DataFrame, struct_eng: StructureEngine, params: Dict) -> pd.Series:
    ob = pd.Series(0, index=df.index)
    impulse_atr = params.get('impulse_atr', 1.5)
    atr = (df['high'] - df['low']).rolling(14).mean()
    for i in range(2, len(df)-1):
        if df['close'].iloc[i] < df['open'].iloc[i]:
            next_move = df['high'].iloc[i+1] - df['close'].iloc[i]
            if next_move > impulse_atr * atr.iloc[i]:
                ob.iloc[i] = 1
        elif df['close'].iloc[i] > df['open'].iloc[i]:
            next_move = df['close'].iloc[i] - df['low'].iloc[i+1]
            if next_move > impulse_atr * atr.iloc[i]:
                ob.iloc[i] = -1
    return ob


# =============================================================================
# 4. REGISTRO DE HIPÓTESES
# =============================================================================
@dataclass
class Hypothesis:
    concept: str
    variant: str
    detect_func: Optional[Callable]
    description: str
    parameters: Dict = field(default_factory=dict)
    generated_by_ai: bool = False
    validated: bool = False

class DefinitionsRegistry:
    def __init__(self):
        self.hypotheses: List[Hypothesis] = []

    def register(self, hyp: Hypothesis):
        self.hypotheses.append(hyp)

    def get_all(self) -> List[Hypothesis]:
        return self.hypotheses

    def get_by_concept(self, concept: str) -> List[Hypothesis]:
        return [h for h in self.hypotheses if h.concept == concept]

def build_registry() -> DefinitionsRegistry:
    registry = DefinitionsRegistry()
    registry.register(Hypothesis("BOS", "A", detect_bos_A, "Fechamento > HH/LL", {"window": 20}))
    registry.register(Hypothesis("BOS", "B", detect_bos_B, "Pavio > HH/LL", {"window": 20}))
    registry.register(Hypothesis("BOS", "C", detect_bos_C, "Fechamento + ATR", {"window": 20, "atr_period": 14, "min_displacement_atr": 0.5}))
    registry.register(Hypothesis("BOS", "D", detect_bos_D, "Rompimento + tendência SMA 50/200", {"window": 20}))
    registry.register(Hypothesis("FVG", "A", detect_fvg_A, "Gap entre velas", {}))
    registry.register(Hypothesis("FVG", "B", detect_fvg_B, "Gap + volume", {}))
    registry.register(Hypothesis("CHoCH", "A", detect_choch_A, "Rompe LH/HL e fecha além", {"window": 20}))
    registry.register(Hypothesis("Liquidity Sweep", "A", detect_liquidity_sweep_A, "Rompe extremo e reverte", {"window": 20, "reversal_bars": 3}))
    registry.register(Hypothesis("Order Block", "A", detect_orderblock_A, "Vela oposta antes de impulso", {"impulse_atr": 1.5}))
    return registry


# =============================================================================
# 5. GERADOR DE IMAGENS
# =============================================================================
class ChartImageGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def generate_event_chart(self, event_idx: int, context_bars: int = 30) -> Image.Image:
        start_idx = max(0, event_idx - context_bars)
        end_idx = min(len(self.df), event_idx + context_bars)
        df_slice = self.df.iloc[start_idx:end_idx]

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (date, row) in enumerate(df_slice.iterrows()):
            color = 'green' if row['close'] >= row['open'] else 'red'
            ax.plot([i, i], [row['low'], row['high']], color='black', linewidth=0.5)
            ax.plot([i, i], [row['open'], row['close']], color=color, linewidth=4)

        event_pos = event_idx - start_idx
        ax.axvline(x=event_pos, color='blue', linestyle='--', alpha=0.7, label='Evento')
        ax.set_title(f'Contexto do Evento SMC')
        ax.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return img


# =============================================================================
# 6. ANALISTA VISUAL COM OLLAMA
# =============================================================================
class OllamaVisionAnalyst:
    def __init__(self, model_name: str = "qwen2.5vl:7b"):
        self.model = model_name

    def analyze_smc_event(self, image: Image.Image, concept: str, variant: str) -> dict:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        prompt = f"""
        Analise este gráfico de velas focado no contexto de um evento '{concept} - {variant}'.
        O evento foi detectado quantitativamente na vela marcada com a linha azul pontilhada.
        Confirme se o padrão visual corresponde a um '{concept}' válido.
        Responda em JSON com as chaves:
        - "padrao_valido": true/false
        - "justificativa": string
        """

        try:
            response = ollama.generate(model=self.model, prompt=prompt, images=[img_base64])
            text = response['response']
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"padrao_valido": False, "justificativa": "Resposta não JSON"}
        except Exception as e:
            logging.error(f"Erro na análise visual: {e}")
            return {"padrao_valido": False, "justificativa": str(e)}


# =============================================================================
# 7. EVENT ENGINE (COM VALIDAÇÃO VISUAL E SALVAMENTO PARCIAL)
# =============================================================================
class EventEngine:
    def __init__(self, df: pd.DataFrame, registry: DefinitionsRegistry):
        self.df = df
        self.registry = registry
        self.events = pd.DataFrame(index=df.index)

    def detect_all(self, use_vision: bool = False, vision_model: str = "qwen2.5vl:7b") -> pd.DataFrame:
        struct_eng = StructureEngine(self.df)
        for hyp in self.registry.get_all():
            if hyp.detect_func is not None:
                signal = hyp.detect_func(self.df, struct_eng, hyp.parameters)
                self.events[f"{hyp.concept}_{hyp.variant}"] = signal

        if use_vision and OLLAMA_AVAILABLE:
            self._validate_with_vision(vision_model)

        return self.events

    def _validate_with_vision(self, vision_model: str):
        chart_gen = ChartImageGenerator(self.df)
        analyst = OllamaVisionAnalyst(model_name=vision_model)

        # Carrega progresso salvo, se existir
        progress_file = 'vision_progress.json'
        processed_events = {}
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                processed_events = json.load(f)
            logging.info(f"Progresso carregado: {len(processed_events)} eventos já processados.")

        for col in self.events.columns:
            if col.endswith('_visao_valida'):
                continue

            concept = col.split('_')[0]
            variant = col.split('_', 1)[1]
            event_indices = self.events.index[self.events[col] != 0].tolist()

            valid_series = pd.Series(0, index=self.df.index, dtype=int)
            total = len(event_indices)
            for i, idx in enumerate(event_indices):
                key = f"{col}_{idx}"
                if key in processed_events:
                    valid_series.loc[idx] = processed_events[key]
                    continue

                logging.info(f"Processando {col} ({i+1}/{total}): {idx}")
                try:
                    img = chart_gen.generate_event_chart(self.df.index.get_loc(idx))
                    result = analyst.analyze_smc_event(img, concept, variant)
                except Exception as e:
                    logging.error(f"Falha ao analisar {idx}: {e}")
                    result = {"padrao_valido": False, "justificativa": str(e)}

                val = 1 if result.get('padrao_valido', False) else -1
                valid_series.loc[idx] = val
                processed_events[key] = val

                # Salva progresso a cada 50 eventos
                if (i+1) % 50 == 0:
                    with open(progress_file, 'w') as f:
                        json.dump(processed_events, f)
                    logging.info(f"Progresso salvo: {i+1}/{total} eventos de {col}")

            self.events[f"{col}_visao_valida"] = valid_series

        # Salva final
        with open(progress_file, 'w') as f:
            json.dump(processed_events, f)
        logging.info("Validação visual concluída.")


# =============================================================================
# 8. TESTES ISOLADOS
# =============================================================================
class IsolatedTestEngine:
    def __init__(self, df: pd.DataFrame, events_df: pd.DataFrame):
        self.df = df
        self.events_df = events_df

    def compute_forward_returns(self, horizon: int = 20) -> pd.DataFrame:
        rets = pd.DataFrame(index=self.df.index)
        for h in range(1, horizon+1):
            rets[f'ret_{h}'] = np.log(self.df['close'].shift(-h) / self.df['close'])
        return rets

    def test_concept_variant(self, variant_col: str, forward_rets: pd.DataFrame) -> Dict:
        signal = self.events_df[variant_col]
        events_idx = signal[signal != 0].index
        if len(events_idx) == 0:
            return {'frequency': 0, 'expectancy': {}, 'n_events': 0}
        rets_at_events = forward_rets.loc[events_idx]
        expectancy = {}
        for col in rets_at_events.columns:
            if variant_col.startswith('BOS'):
                direction = signal.loc[events_idx]
                adj_ret = rets_at_events[col] * direction
                expectancy[col] = adj_ret.mean()
            else:
                expectancy[col] = rets_at_events[col].mean()
        return {
            'frequency': len(events_idx) / len(self.df),
            'expectancy': expectancy,
            'n_events': len(events_idx)
        }


# =============================================================================
# 9. CAUSALITY ENGINE
# =============================================================================
class CausalityEngine:
    def __init__(self, df: pd.DataFrame, events_df: pd.DataFrame):
        self.df = df
        self.events_df = events_df

    def granger_test(self, signal_col: str, maxlag: int = 5) -> Dict:
        if not STATSMODELS_AVAILABLE:
            return {'is_causal': False}
        data = pd.DataFrame({
            'signal': self.events_df[signal_col],
            'price': self.df['close'].pct_change().fillna(0)
        }).dropna()
        if len(data) < 20:
            return {'is_causal': False}
        try:
            gc_res = grangercausalitytests(data[['price', 'signal']], maxlag=maxlag, verbose=False)
            p_values = {lag: gc_res[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag+1)}
            best_lag = min(p_values, key=p_values.get)
            return {'p_values': p_values, 'best_lag': best_lag, 'is_causal': p_values[best_lag] < 0.05}
        except:
            return {'is_causal': False}


# =============================================================================
# 10. ROBUSTNESS ENGINE
# =============================================================================
class RobustnessEngine:
    def __init__(self, df: pd.DataFrame, events_df: pd.DataFrame):
        self.df = df
        self.events_df = events_df

    def walk_forward_analysis(self, signal_col: str, train_ratio=0.7, test_ratio=0.15) -> Dict:
        n = len(self.df)
        train_end = int(n * train_ratio)
        test_end = int(n * (train_ratio + test_ratio))
        train_events = self.events_df[signal_col].iloc[:train_end]
        test_events = self.events_df[signal_col].iloc[train_end:test_end]
        oos_events = self.events_df[signal_col].iloc[test_end:]
        return {
            'train_events': (train_events != 0).sum(),
            'test_events': (test_events != 0).sum(),
            'oos_events': (oos_events != 0).sum()
        }


# =============================================================================
# 11. EVALUATION ENGINE
# =============================================================================
class EvaluationEngine:
    def __init__(self, registry, isolated_results, causality_results, robustness_results):
        self.registry = registry
        self.isolated_results = isolated_results
        self.causality_results = causality_results
        self.robustness_results = robustness_results

    def generate_concept_report(self) -> pd.DataFrame:
        rows = []
        for hyp in self.registry.get_all():
            key = f"{hyp.concept}_{hyp.variant}"
            iso = self.isolated_results.get(key, {})
            caus = self.causality_results.get(key, {})
            exp = iso.get('expectancy', {})
            exp_t5 = exp.get('ret_5', None) if isinstance(exp, dict) else None
            rows.append({
                'Conceito': hyp.concept,
                'Variante': hyp.variant,
                'Frequência': iso.get('frequency', 0),
                'Expectativa (t+5)': exp_t5,
                'Causal (Granger)': caus.get('is_causal', False),
                'N Eventos': iso.get('n_events', 0)
            })
        return pd.DataFrame(rows)


# =============================================================================
# 12. PIPELINE PRINCIPAL
# =============================================================================
def run_validation_pipeline(df: pd.DataFrame, use_vision: bool = False, vision_model: str = "qwen2.5vl:7b") -> Dict:
    data_eng = DataEngine(df)
    clean_df = data_eng.df

    registry = build_registry()

    event_eng = EventEngine(clean_df, registry)
    events_df = event_eng.detect_all(use_vision=use_vision, vision_model=vision_model)

    iso_eng = IsolatedTestEngine(clean_df, events_df)
    forward_rets = iso_eng.compute_forward_returns(horizon=20)
    isolated_results = {}
    for col in events_df.columns:
        if not col.endswith('_visao_valida'):
            isolated_results[col] = iso_eng.test_concept_variant(col, forward_rets)

    caus_eng = CausalityEngine(clean_df, events_df)
    causality_results = {}
    for col in events_df.columns:
        if not col.endswith('_visao_valida'):
            causality_results[col] = caus_eng.granger_test(col)

    rob_eng = RobustnessEngine(clean_df, events_df)
    robustness_results = {}
    for col in events_df.columns:
        if not col.endswith('_visao_valida'):
            robustness_results[col] = rob_eng.walk_forward_analysis(col)

    eval_eng = EvaluationEngine(registry, isolated_results, causality_results, robustness_results)
    report = eval_eng.generate_concept_report()
    ranking = report.sort_values('Frequência', ascending=False)

    return {'report': report, 'ranking': ranking, 'events_df': events_df}


# =============================================================================
# 13. CARREGAMENTO MT5
# =============================================================================
def load_data_from_mt5(symbol: str, timeframe, num_bars: int = 3000) -> pd.DataFrame:
    if not MT5_AVAILABLE:
        raise ImportError("MetaTrader5 não instalado.")
    if not mt5.initialize():
        raise ConnectionError("Falha ao inicializar MT5.")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        raise ValueError(f"Dados não obtidos para {symbol}.")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}, inplace=True)
    return df[['open', 'high', 'low', 'close']]


# =============================================================================
# 14. EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SMC VALIDATION ENGINE v3.0 - FINAL (CPU)")
    print("=" * 60)

    usar_mt5 = input("Usar MetaTrader 5 para dados reais? (s/N): ").strip().lower() == 's'
    usar_visao = input("Usar validação visual com Ollama? (s/N): ").strip().lower() == 's'

    if usar_visao:
        modelo = input("Modelo Ollama (ex: qwen2.5vl:7b, llava:7b) [qwen2.5vl:7b]: ").strip()
        if not modelo:
            modelo = "qwen2.5vl:7b"
    else:
        modelo = "qwen2.5vl:7b"

    if usar_mt5:
        symbol = input("Símbolo (ex: EURUSD): ").strip() or "EURUSD"
        try:
            df = load_data_from_mt5(symbol, mt5.TIMEFRAME_H1, 3000)
            print(f"Dados MT5 carregados: {len(df)} linhas.")
        except Exception as e:
            print(f"Erro MT5: {e}. Usando dados sintéticos.")
            usar_mt5 = False

    if not usar_mt5:
        print("Gerando dados sintéticos...")
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=3000, freq='1H')
        price = 1.0
        prices = [price]
        for _ in range(1, len(dates)):
            price *= (1 + np.random.randn() * 0.001)
            prices.append(price)
        close = pd.Series(prices, index=dates)
        df = pd.DataFrame({
            'open': close.shift(1).fillna(close.iloc[0]),
            'high': close + np.random.uniform(0, 0.002, len(close)),
            'low': close - np.random.uniform(0, 0.002, len(close)),
            'close': close
        }, index=dates)
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 0.001, len(df))
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 0.001, len(df))
        print(f"Dados sintéticos: {len(df)} linhas.")

    print("Executando pipeline de validação...")
    results = run_validation_pipeline(df, use_vision=usar_visao, vision_model=modelo)

    print("\n=== RELATÓRIO DE CONCEITOS SMC ===")
    print(results['report'].to_string())

    print("\n=== RANKING (por frequência) ===")
    print(results['ranking'].to_string())

    if usar_visao:
        print("\n=== ESTATÍSTICAS DE VALIDAÇÃO VISUAL ===")
        for col in results['events_df'].columns:
            if col.endswith('_visao_valida'):
                validos = (results['events_df'][col] == 1).sum()
                invalidos = (results['events_df'][col] == -1).sum()
                print(f"{col}: {validos} válidos, {invalidos} inválidos")
        results['events_df'].to_csv('eventos_validados_visualmente.csv')
        print("Resultados salvos em 'eventos_validados_visualmente.csv'.")

    print("\nExecução concluída.")