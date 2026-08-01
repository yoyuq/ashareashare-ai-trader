"""
可视化进度追踪器 — 全市场AI预筛专用

Pipeline stages:
  [1] 数据加载 → [2] 规则预筛 → [3] LLM精筛 → [4] DeepSeek分析 → [✅] 完成

用法:
  tracker = ProgressTracker(total_batches=10, pipeline_name="AI预筛")
  for i, result in enumerate(process_batches()):
      tracker.update(batch_idx=i, stocks_done=n, errors=e, stage=3)
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PipelineStage:
    name: str
    emoji: str
    total: int = 0
    done: int = 0
    active: bool = False


class ProgressTracker:
    """全市场预筛可视化进度追踪器 — 生成Streamlit HTML组件"""

    def __init__(
        self,
        pipeline_name: str = "AI预筛",
        stages: Optional[List[PipelineStage]] = None,
        total_batches: int = 0,
        total_stocks: int = 0,
    ):
        self.pipeline_name = pipeline_name
        self.stages = stages or [
            PipelineStage("数据加载", "📡"),
            PipelineStage("规则预筛", "🔍"),
            PipelineStage("LLM精筛", "🤖"),
            PipelineStage("深度分析", "🔬"),
        ]
        self.total_batches = total_batches
        self.total_stocks = total_stocks
        self.start_time = time.time()

        # 实时指标
        self.batch_idx = 0
        self.stocks_done = 0
        self.errors = 0
        self.active_stage = 0
        self.current_model = ""
        self.activity_log: List[str] = []

    def set_stage_total(self, stage_idx: int, total: int):
        """设置某阶段的总工作量"""
        if stage_idx < len(self.stages):
            self.stages[stage_idx].total = total

    def update(
        self,
        batch_idx: int = -1,
        stocks_done: int = -1,
        errors: int = -1,
        stage: int = -1,
        model: str = "",
        log_msg: str = "",
    ):
        """更新进度"""
        if batch_idx >= 0:
            self.batch_idx = batch_idx
        if stocks_done >= 0:
            self.stocks_done = stocks_done
        if errors >= 0:
            self.errors = errors
        if stage >= 0:
            self.active_stage = stage
            # Mark previous stages as complete
            for s in self.stages[:stage]:
                s.done = s.total if s.total > 0 else 1
                s.active = False
            if stage < len(self.stages):
                self.stages[stage].active = True
        if model:
            self.current_model = model
        if log_msg:
            self.activity_log.append(f"[{time.strftime('%H:%M:%S')}] {log_msg}")
            if len(self.activity_log) > 8:
                self.activity_log = self.activity_log[-8:]

    def _get_pct(self) -> float:
        """计算整体完成百分比"""
        if self.total_batches > 0:
            return min(100.0, self.batch_idx / self.total_batches * 100)
        if self.total_stocks > 0:
            return min(100.0, self.stocks_done / self.total_stocks * 100)
        return 0.0

    def _get_eta(self) -> str:
        """估算剩余时间"""
        pct = self._get_pct()
        if pct < 1:
            return "计算中..."
        elapsed = time.time() - self.start_time
        remaining = elapsed / (pct / 100) - elapsed
        if remaining < 60:
            return f"{remaining:.0f}s"
        elif remaining < 3600:
            return f"{remaining/60:.0f}m"
        return f"{remaining/3600:.1f}h"

    def _get_speed(self) -> str:
        """计算处理速度"""
        elapsed = max(1, time.time() - self.start_time)
        if self.stocks_done > 0:
            sps = self.stocks_done / elapsed
            if sps > 100:
                return f"{sps:.0f}只/s"
            return f"{sps:.1f}只/s"
        if self.batch_idx > 0:
            bps = self.batch_idx / elapsed
            return f"{bps:.2f}批/s"
        return "—"

    def render_html(self) -> str:
        """生成完整进度HTML (嵌入Streamlit)"""
        pct = self._get_pct()
        elapsed = time.time() - self.start_time
        elapsed_str = f"{elapsed:.0f}s" if elapsed < 60 else f"{elapsed/60:.1f}min"

        # Stage tracker
        stage_html = ""
        for i, s in enumerate(self.stages):
            if s.done > 0:
                icon = '<span style="color:#3fb950">✓</span>'
                style = "color:#3fb950;font-weight:600"
            elif s.active:
                icon = f'<span class="pulse">{s.emoji}</span>'
                style = "color:#58a6ff;font-weight:600"
            else:
                icon = s.emoji
                style = "color:#484f58"
            stage_html += (
                f'<div style="flex:1;text-align:center;padding:6px 4px;'
                f'border-bottom:3px solid {"#3fb950" if s.done>0 else "#58a6ff" if s.active else "#30363d"};'
                f'margin:0 2px">'
                f'<div style="font-size:18px">{icon}</div>'
                f'<div style="font-size:11px;{style}">{s.name}</div>'
                f'<div style="font-size:10px;color:#8b949e">{"{}/{}".format(s.done,s.total) if s.total>0 else "—"}</div>'
                f'</div>'
            )

        # Progress bar
        bar_color = "#3fb950" if pct >= 99 else "#58a6ff"
        bar_html = (
            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:2px;margin:10px 0">'
            f'<div style="background:{bar_color};height:8px;width:{pct:.1f}%;border-radius:4px;'
            f'transition:width 0.3s ease;box-shadow:0 0 8px {bar_color}44"></div>'
            f'</div>'
        )

        # Activity log
        log_lines = ""
        for line in self.activity_log[-5:]:
            log_lines += f'<div style="font-size:11px;color:#8b949e;font-family:monospace;padding:1px 0">{line}</div>'

        html = f"""
        <style>
        @keyframes pulse {{ 0%,100% {{ opacity:1 }} 50% {{ opacity:0.4 }} }}
        .pulse {{ animation: pulse 0.8s ease-in-out infinite }}
        .tracker-container {{
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 14px 18px;
            margin: 8px 0;
        }}
        </style>
        <div class="tracker-container">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
                <div style="font-size:15px;font-weight:700;color:#f0f6fc">🚀 {self.pipeline_name}</div>
                <div style="font-size:12px;color:#8b949e">⏱ {elapsed_str} | ⚡ {self._get_speed()} | 🕐 ETA: {self._get_eta()}</div>
            </div>

            <!-- Pipeline Stages -->
            <div style="display:flex;margin:8px 0 12px 0">{stage_html}</div>

            <!-- Progress Bar -->
            <div style="display:flex;align-items:center;gap:12px">
                <div style="flex:1">{bar_html}</div>
                <div style="font-size:28px;font-weight:800;color:#58a6ff;min-width:60px;text-align:right">{pct:.0f}%</div>
            </div>

            <!-- Metrics -->
            <div style="display:flex;gap:8px;margin:10px 0;flex-wrap:wrap">
                <div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:6px 14px;flex:1;min-width:100px">
                    <div style="font-size:10px;color:#8b949e">📊 批次</div>
                    <div style="font-size:16px;font-weight:700;color:#f0f6fc">{self.batch_idx}<span style="font-size:11px;color:#8b949e">/{self.total_batches}</span></div>
                </div>
                <div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:6px 14px;flex:1;min-width:100px">
                    <div style="font-size:10px;color:#8b949e">📈 已分析</div>
                    <div style="font-size:16px;font-weight:700;color:#3fb950">{self.stocks_done:,}<span style="font-size:11px;color:#8b949e"> 只</span></div>
                </div>
                <div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:6px 14px;flex:1;min-width:100px">
                    <div style="font-size:10px;color:#8b949e">🤖 模型</div>
                    <div style="font-size:14px;font-weight:600;color:#58a6ff">{self.current_model or '—'}</div>
                </div>
                <div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:6px 14px;flex:1;min-width:100px">
                    <div style="font-size:10px;color:#8b949e">⚠️ 错误</div>
                    <div style="font-size:16px;font-weight:700;color:{'#f85149' if self.errors>0 else '#3fb950'}">{self.errors}</div>
                </div>
                <div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:6px 14px;flex:1;min-width:100px">
                    <div style="font-size:10px;color:#8b949e">💰 剩余</div>
                    <div style="font-size:16px;font-weight:700;color:#f0883e">{self.total_stocks-self.stocks_done:,}<span style="font-size:11px;color:#8b949e"> 只</span></div>
                </div>
            </div>

            <!-- Activity Log -->
            <div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:6px 10px;margin-top:6px;max-height:90px;overflow-y:auto">
                {log_lines if log_lines else '<div style="font-size:11px;color:#484f58;font-family:monospace">等待启动...</div>'}
            </div>
        </div>
        """
        return html
