import json
import os
from datetime import datetime
from typing import Dict, List, Any

class EvolutionLogger:
    def __init__(self, log_file: str = "evolution_log.json"):
        self.log_file = log_file
        self.history = self._load_history()

    def _load_history(self) -> List[Dict]:
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_history(self):
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def log_iteration(self, version: str, params: Dict[str, float],
                      total_cost: float, deviations: Dict[str, float],
                      anomalies: List[str], modification_made: str = ""):
        record = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "params": params.copy(),
            "total_cost": total_cost,
            "deviations": deviations.copy(),
            "anomalies": anomalies.copy(),
            "modification_made": modification_made,
        }
        self.history.append(record)
        self._save_history()

    # evolution_logger.py (修改后)
    def get_history_summary(self, last_n: int = None) -> str:
        if not self.history:
            return "No history records."
        records = self.history if last_n is None else self.history[-last_n:]
        lines = ["## Control Law Evolution History Summary (chronological)\n"]
        for rec in records:
            ver = rec["version"]
            cost = rec["total_cost"]
            params = rec["params"]
            dev = rec["deviations"]
            anomalies = rec["anomalies"]
            mod = rec.get("modification_made", "")

            lines.append(f"### {ver}")
            lines.append(f"- **Total Cost**: {cost:.4f}")
            lines.append(f"- **Best Parameters**: {', '.join([f'{k}={v:.4f}' for k, v in params.items()])}")
            lines.append(
                f"- **Final State Deviations**: Iu={dev.get('Iu', 0):+.4f}, Ia={dev.get('Ia', 0):+.4f}, LSu={dev.get('LSu', 0):+.4f}, LIu={dev.get('LIu', 0):+.4f}, LRu={dev.get('LRu', 0):+.4f}")
            if anomalies:
                lines.append(f"- **Anomalies**: {', '.join(anomalies)}")
            if mod:
                lines.append(f"- **Modification Made**: {mod}")
            lines.append("")
        return "\n".join(lines)

    def get_evolution_table(self) -> str:
        """生成跨版本代价-参数-偏差的紧凑Markdown表格"""
        if not self.history:
            return ""
        lines = ["## Cross-Version Evolution Table\n"]
        lines.append("| Version | Total Cost | Iu | Ia | LSu | LIu | LRu | Params |")
        lines.append("|---------|------------|-----|-----|------|------|------|--------|")
        for rec in self.history:
            dev = rec.get('deviations', {})
            lines.append(
                f"| {rec['version']} | {rec['total_cost']:.2f} | "
                f"{dev.get('Iu', 0):+.3f} | {dev.get('Ia', 0):+.3f} | "
                f"{dev.get('LSu', 0):+.3f} | {dev.get('LIu', 0):+.3f} | "
                f"{dev.get('LRu', 0):+.3f} | {len(rec['params'])} |"
            )
        return "\n".join(lines)

    def get_failure_patterns_summary(self) -> str:
        """提取跨版本重复出现的异常模式"""
        if not self.history:
            return ""
        pattern_map = {}
        for rec in self.history:
            for a in rec.get('anomalies', []):
                if 'exceeds' in a:
                    pattern = a.split('exceeds')[0].strip()
                elif 'out of bounds' in a:
                    pattern = 'Control value out of bounds'
                elif 'PARAM DECLARATION MISSING' in a:
                    pattern = 'Missing parameter declaration'
                elif 'CRASH' in a or 'Critical' in a:
                    pattern = 'Simulation crash'
                else:
                    continue
                if pattern not in pattern_map:
                    pattern_map[pattern] = {'first': rec['version'], 'last': rec['version'], 'count': 0}
                pattern_map[pattern]['last'] = rec['version']
                pattern_map[pattern]['count'] += 1

        lines = ["## Repeated Failure Patterns\n"]
        lines.append("| Pattern | First Seen | Last Seen | Occurrences |")
        lines.append("|---------|-----------|-----------|-------------|")
        has_data = False
        for pattern, info in pattern_map.items():
            if info['count'] >= 2:
                lines.append(f"| {pattern} | {info['first']} | {info['last']} | {info['count']} |")
                has_data = True
        return "\n".join(lines) if has_data else ""

    def get_early_insights(self, n_early: int = 3) -> str:
        """提取早期版本首次出现的问题"""
        if len(self.history) < n_early:
            return ""
        early = self.history[:n_early]
        lines = ["## Early Version Key Insights\n"]
        for rec in early:
            anomalies = rec.get('anomalies', [])
            key = [a for a in anomalies if 'exceeds' in a or 'CRASH' in a][:2]
            if key:
                lines.append(f"- {rec['version']}: {', '.join(key)}")
        costs = [(r['version'], r['total_cost']) for r in early if r['total_cost'] < 1e9]
        if costs:
            best = min(costs, key=lambda x: x[1])
            lines.append(f"- Best early version: {best[0]} (cost={best[1]:.2f})")
        return "\n".join(lines) if len(lines) > 2 else ""

    def get_layered_history(self, n_recent_full: int = 3) -> str:
        """生成分层历史注入的完整文本"""
        parts = []
        early = self.get_early_insights()
        if early:
            parts.append(early)
        table = self.get_evolution_table()
        if table:
            parts.append(table)
        patterns = self.get_failure_patterns_summary()
        if patterns:
            parts.append(patterns)
        # 最近N轮详细信息
        if n_recent_full > 0 and self.history:
            recent = self.history[-n_recent_full:]
            parts.append(f"\n## Recent {len(recent)} Versions (Detailed)\n")
            for rec in recent:
                parts.append(f"### {rec['version']}")
                parts.append(f"- **Total Cost**: {rec['total_cost']:.4f}")
                parts.append(f"- **Parameters**: {', '.join([f'{k}={v:.4f}' for k, v in rec['params'].items()])}")
                parts.append(
                    f"- **Deviations**: Iu={rec['deviations'].get('Iu', 0):+.4f}, "
                    f"Ia={rec['deviations'].get('Ia', 0):+.4f}, "
                    f"LSu={rec['deviations'].get('LSu', 0):+.4f}, "
                    f"LIu={rec['deviations'].get('LIu', 0):+.4f}, "
                    f"LRu={rec['deviations'].get('LRu', 0):+.4f}"
                )
                if rec.get('anomalies'):
                    parts.append(f"- **Anomalies**: {', '.join(rec['anomalies'])}")
                parts.append("")
        return "\n".join(parts)

