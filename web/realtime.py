"""
实时行情组件 — JS DOM-patch 丝滑更新

架构: 服务端渲染初始HTML + 内嵌JS fetch轮询 + 仅更新变化的数字
不再使用 st.fragment(run_every) — 无闪烁、无整页重载
"""

from datetime import datetime
import os

import streamlit as st


def realtime_section():
    """Render real-time market data — HTML+JS, polling API every 3 seconds"""

    now = datetime.now().strftime("%H:%M:%S")
    api_url = os.getenv("API_BASE_URL", "http://localhost:8000") + "/api/v1/realtime/market"

    st.components.v1.html(f"""
    <style>
    /* 实时行情容器 */
    .rt-container {{ background:#0d1117; border:1px solid #30363d; border-radius:10px; padding:14px 18px; margin:4px 0 }}
    .rt-header {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:10px }}
    .rt-title {{ font-size:15px; font-weight:700; color:#f0f6fc }}
    .rt-status {{ font-size:11px; color:#8b949e }}
    .rt-pulse-dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; background:#3fb950; margin-right:6px; animation:rt-blink 1s infinite }}
    @keyframes rt-blink {{ 0%,100% {{ opacity:1 }} 50% {{ opacity:0.3 }} }}

    /* 指数条 */
    .rt-indices {{ display:flex; gap:6px; margin-bottom:10px; flex-wrap:wrap }}
    .rt-idx-card {{ flex:1; min-width:100px; background:#161b22; border:1px solid #30363d; border-radius:6px; padding:8px 10px; text-align:center }}
    .rt-idx-name {{ font-size:10px; color:#8b949e }}
    .rt-idx-price {{ font-size:17px; font-weight:700; color:#f0f6fc; transition:color 0.3s }}
    .rt-idx-pct {{ font-size:12px; font-weight:600; margin-top:2px; transition:all 0.3s }}
    .rt-flash-up {{ animation:rt-flash-up 0.6s ease }}
    .rt-flash-dn {{ animation:rt-flash-dn 0.6s ease }}
    @keyframes rt-flash-up {{ 0% {{ background:#3f1a1a;color:#f85149 }} 100% {{ background:transparent;color:inherit }} }}
    @keyframes rt-flash-dn {{ 0% {{ background:#1a3f1a;color:#3fb950 }} 100% {{ background:transparent;color:inherit }} }}

    /* 市场脉冲 */
    .rt-pulse-bar {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:10px 14px; margin-bottom:10px }}
    .rt-pulse-label {{ font-size:14px; font-weight:700 }}
    .rt-pulse-track {{ background:#161b22; border-radius:4px; height:6px; margin-top:8px; position:relative }}
    .rt-pulse-fill {{ position:absolute; height:6px; border-radius:4px; transition:left 0.8s cubic-bezier(0.4,0,0.2,1),background 0.5s }}
    .rt-pulse-marker {{ position:absolute; width:4px; height:14px; border-radius:2px; top:-4px; transition:left 0.8s cubic-bezier(0.4,0,0.2,1); box-shadow:0 0 8px currentColor }}
    .rt-pulse-ticks {{ display:flex; justify-content:space-between; font-size:9px; color:#484f58; margin-top:3px }}

    /* Top异动 */
    .rt-movers {{ display:flex; gap:10px; margin-bottom:10px }}
    .rt-mover-col {{ flex:1; background:#161b22; border:1px solid #30363d; border-radius:6px; padding:8px 10px; max-height:200px; overflow-y:auto }}
    .rt-mover-title {{ font-size:12px; font-weight:600; margin-bottom:4px }}
    .rt-mover-row {{ display:flex; justify-content:space-between; font-size:11px; padding:1px 0; border-bottom:1px solid #21262d }}
    .rt-mover-name {{ color:#c9d1d9; flex:2 }}
    .rt-mover-code {{ color:#8b949e; flex:1; text-align:center }}
    .rt-mover-pct {{ font-weight:600; flex:1; text-align:right; transition:color 0.3s }}

    /* 自选列表 */
    .rt-watch {{ background:#161b22; border:1px solid #30363d; border-radius:6px; padding:6px; max-height:300px; overflow-y:auto }}
    .rt-watch table {{ width:100%; border-collapse:collapse }}
    .rt-watch th {{ font-size:10px; color:#8b949e; text-align:left; padding:2px 6px; border-bottom:2px solid #30363d }}
    .rt-watch td {{ font-size:11px; padding:2px 6px; border-bottom:1px solid #21262d }}
    .rt-watch .name {{ color:#c9d1d9 }}
    .rt-watch .code {{ color:#8b949e }}
    .rt-watch .price {{ color:#f0f6fc; text-align:right; transition:color 0.3s }}
    .rt-watch .pct {{ text-align:right; font-weight:600; transition:all 0.3s }}
    </style>

    <div class="rt-container" id="rt-root">
        <div class="rt-header">
            <div class="rt-title">⚡ 实时行情</div>
            <div class="rt-status"><span class="rt-pulse-dot" id="rt-pulse-dot"></span> <span id="rt-status-text">连接中...</span> | <span id="rt-ts">{now}</span></div>
        </div>

        <!-- 大盘指数 -->
        <div class="rt-indices" id="rt-indices">
            <div class="rt-idx-card"><div class="rt-idx-name">加载中...</div></div>
        </div>

        <!-- 市场脉冲 + Top异动 -->
        <div style="display:flex;gap:10px">
            <div style="flex:1">
                <div class="rt-pulse-bar" id="rt-pulse">
                    <div class="rt-pulse-label" style="color:#8b949e">— 加载中</div>
                    <div class="rt-pulse-track"><div class="rt-pulse-fill" style="left:50%;width:50%;background:#30363d"></div><div class="rt-pulse-marker" style="left:50%;color:#8b949e"></div></div>
                    <div class="rt-pulse-ticks"><span>-3%</span><span>-1.5%</span><span>0%</span><span>+1.5%</span><span>+3%</span></div>
                </div>
            </div>
            <div style="flex:2">
                <div class="rt-movers" id="rt-movers">
                    <div class="rt-mover-col"><div class="rt-mover-title" style="color:#3fb950">🚀 涨幅TOP5</div><span style="color:#8b949e">加载中...</span></div>
                    <div class="rt-mover-col"><div class="rt-mover-title" style="color:#f85149">📉 跌幅TOP5</div><span style="color:#8b949e">加载中...</span></div>
                </div>
            </div>
        </div>

        <!-- 自选监控 -->
        <div class="rt-watch" id="rt-watch">
            <span style="color:#8b949e;font-size:11px">加载中...</span>
        </div>
    </div>

    <script>
    (function() {{
        var api = '{api_url}';
        var prevPrices = {{}};
        var pollTimer = null;

        function fmt(n, d) {{ return Number(n || 0).toFixed(d || 0); }}
        function pctColor(v) {{ return v > 0 ? '#f85149' : v < 0 ? '#3fb950' : '#8b949e'; }}
        function pctSign(v) {{ return v > 0 ? '+' : ''; }}
        function arrow(v) {{ return v > 0 ? '▲' : v < 0 ? '▼' : '—'; }}

        // 检测价格变化并添加闪烁
        function priceEl(id, val) {{
            var prev = prevPrices[id];
            var flash = '';
            if (prev !== undefined && prev !== val) {{
                flash = val > prev ? ' rt-flash-up' : ' rt-flash-dn';
            }}
            prevPrices[id] = val;
            return flash;
        }}

        function update(data) {{
            if (!data) return;

            // 时间戳 + 市场状态
            var ts = document.getElementById('rt-ts');
            if (ts) ts.textContent = (data.ts || '').slice(11,19) || new Date().toLocaleTimeString();

            var ms = data.market_status || {{is_open: false, status: 'unknown', detail: ''}};
            var dot = document.getElementById('rt-pulse-dot');
            var stText = document.getElementById('rt-status-text');
            if (dot && stText) {{
                if (ms.is_open) {{
                    dot.style.background = '#3fb950'; dot.style.animation = 'rt-blink 1s infinite';
                    stText.textContent = ms.detail || '实时';
                    stText.style.color = '#3fb950';
                }} else {{
                    dot.style.background = '#f0883e'; dot.style.animation = 'none';
                    stText.textContent = ms.detail || '已收盘';
                    stText.style.color = '#f0883e';
                }}
            }}

            // === 大盘指数 ===
            var idxHtml = '';
            var idxOrder = ['上证指数','深证成指','创业板指','科创50','沪深300'];
            for (var i=0; i<idxOrder.length; i++) {{
                var d = data.indices && data.indices[idxOrder[i]];
                if (!d) continue;
                var p = d.price || 0;
                var c = d.pct || 0;
                idxHtml += '<div class="rt-idx-card">' +
                    '<div class="rt-idx-name">' + idxOrder[i].slice(0,4) + '</div>' +
                    '<div class="rt-idx-price">' + fmt(p, p<10?2:0) + '</div>' +
                    '<div class="rt-idx-pct" style="color:' + pctColor(c) + '">' + arrow(c) + ' ' + pctSign(c) + fmt(c,2) + '%</div>' +
                    '</div>';
            }}
            document.getElementById('rt-indices').innerHTML = idxHtml || '<div class="rt-idx-card">—</div>';

            // === 市场脉冲 ===
            var totalPct = 0, count = 0;
            var weights = {{'上证指数':0.25,'深证成指':0.2,'创业板指':0.2,'科创50':0.15,'沪深300':0.2}};
            for (var k in data.indices) {{
                totalPct += (data.indices[k].pct || 0) * (weights[k] || 0.1);
                count++;
            }}
            var wp = totalPct;
            var emoji, label, color;
            if (wp > 1.5) {{ emoji='🔴'; label='强势上攻'; color='#f85149'; }}
            else if (wp > 0.5) {{ emoji='🟠'; label='温和偏多'; color='#f0883e'; }}
            else if (wp > -0.5) {{ emoji='⚪'; label='震荡盘整'; color='#8b949e'; }}
            else if (wp > -1.5) {{ emoji='🟡'; label='弱势回调'; color='#f0883e'; }}
            else {{ emoji='🟢'; label='恐慌下跌'; color='#3fb950'; }}
            var barPos = Math.max(0, Math.min(100, 50 + (wp / 3) * 50));
            document.getElementById('rt-pulse').innerHTML =
                '<div style="display:flex;align-items:center;justify-content:space-between">' +
                '<div style="font-size:22px">' + emoji + '</div>' +
                '<div class="rt-pulse-label" style="color:' + color + '">' + label + '</div>' +
                '<div style="font-size:13px;font-weight:600;color:' + color + '">加权 ' + (wp>0?'+':'') + fmt(wp,2) + '%</div></div>' +
                '<div class="rt-pulse-track">' +
                '<div class="rt-pulse-fill" style="left:0;width:100%;background:#21262d"></div>' +
                '<div class="rt-pulse-marker" style="left:' + barPos + '%;color:' + color + ';background:' + color + '"></div></div>' +
                '<div class="rt-pulse-ticks"><span>-3%</span><span>-1.5%</span><span>0%</span><span>+1.5%</span><span>+3%</span></div>';

            // === 涨跌TOP5 ===
            function moverRows(items, isUp) {{
                if (!items || !items.length) return '<span style="color:#8b949e">暂无数据</span>';
                var rows = '';
                for (var i=0; i<items.length; i++) {{
                    var s = items[i];
                    rows += '<div class="rt-mover-row">' +
                        '<span class="rt-mover-name">' + (s.name||'').slice(0,6) + '</span>' +
                        '<span class="rt-mover-code">' + (s.code||'') + '</span>' +
                        '<span class="rt-mover-pct" style="color:' + pctColor(s.pct) + '">' + pctSign(s.pct) + fmt(s.pct,2) + '%</span></div>';
                }}
                return rows;
            }}
            var moversHtml = '<div class="rt-mover-col"><div class="rt-mover-title" style="color:#3fb950">🚀 涨幅TOP5</div>' + moverRows(data.top_up, true) + '</div>' +
                             '<div class="rt-mover-col"><div class="rt-mover-title" style="color:#f85149">📉 跌幅TOP5</div>' + moverRows(data.top_down, false) + '</div>';
            document.getElementById('rt-movers').innerHTML = moversHtml;

            // === 自选监控 ===
            var wl = data.watchlist || [];
            if (wl.length) {{
                wl.sort(function(a,b) {{ return (b.pct||0) - (a.pct||0); }});
                var th = '<thead><tr><th>名称</th><th>代码</th><th style="text-align:right">现价</th><th style="text-align:right">涨跌</th></tr></thead>';
                var tb = '';
                for (var j=0; j<wl.length; j++) {{
                    var r = wl[j];
                    tb += '<tr>' +
                        '<td class="name">' + (r.name||'') + '</td>' +
                        '<td class="code">' + (r.code||'') + '</td>' +
                        '<td class="price">¥' + fmt(r.price,2) + '</td>' +
                        '<td class="pct" style="color:' + pctColor(r.pct) + '">' + pctSign(r.pct) + fmt(r.pct,2) + '%</td></tr>';
                }}
                document.getElementById('rt-watch').innerHTML = '<table>' + th + '<tbody>' + tb + '</tbody></table>';
            }}
        }}

        function poll() {{
            fetch(api)
                .then(function(r) {{ return r.json(); }})
                .then(update)
                .catch(function(e) {{ console.debug('rt poll:', e.message); }});
        }}

        // 首次加载 + 每3秒轮询
        poll();
        pollTimer = setInterval(poll, 3000);

        // Streamlit路由切换时清理定时器 (页面隐藏时暂停)
        document.addEventListener('visibilitychange', function() {{
            if (document.hidden) {{ clearInterval(pollTimer); }}
            else {{ poll(); pollTimer = setInterval(poll, 3000); }}
        }});
    }})();
    </script>
    """, height=540)
