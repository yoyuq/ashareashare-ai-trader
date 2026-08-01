#!/usr/bin/env python3
"""规则预筛 Top100 → DeepSeek 深度分析  (GLM免费额度不足时的降级方案)"""
import asyncio, sys, json, os
from dotenv import load_dotenv; load_dotenv()
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import akshare as ak
import pandas as pd
import numpy as np
from openai import AsyncOpenAI

async def main():
    # 1. 加载全市场数据
    print('加载全市场数据...')
    df = ak.stock_zh_a_spot_em()
    col_map = {
        '代码':'code','名称':'name','最新价':'price','涨跌幅':'pct_change','成交量':'volume',
        '成交额':'amount','换手率':'turnover','市盈率-动态':'pe_ttm','市净率':'pb',
        '总市值':'total_mv','量比':'vol_ratio','60日涨跌幅':'pct_60d','振幅':'amplitude'
    }
    df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})

    # 2. 规则预筛 (三层漏斗)
    initial = len(df)
    df = df.dropna(subset=['code','name','price','pct_change'])
    df = df[df['amount'] > df['amount'].median() * 0.1]
    df = df[df['turnover'] >= 0.1]
    df = df[df['price'] > 2.0]
    df = df[df['pct_change'] > -5.0]
    print(f'L1流动性过滤: {initial} -> {len(df)}')

    # 多因子打分
    score = pd.Series(50.0, index=df.index)
    if 'pct_change' in df.columns: score += df['pct_change'].clip(-5, 10) * 3
    if 'pct_60d' in df.columns: score += df['pct_60d'].clip(-20, 50) * 0.5
    if 'turnover' in df.columns: score += df['turnover'].clip(0, 20) * 1.5
    if 'vol_ratio' in df.columns: score += (df['vol_ratio'].clip(0.3, 5) - 1) * 5
    if 'pe_ttm' in df.columns:
        pe_score = 10 - abs(df['pe_ttm'].clip(0, 100) - 20) / 8
        score += pe_score.clip(-10, 10)
    if 'total_mv' in df.columns: score += df['total_mv'].rank(pct=True) * 5
    df['rule_score'] = score.clip(0, 100)

    top100 = df.nlargest(100, 'rule_score')
    print(f'规则打分 Top100: score [{top100["rule_score"].min():.0f} - {top100["rule_score"].max():.0f}]')

    # 3. DeepSeek 深度分析 (5批 x 20只)
    client = AsyncOpenAI(
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url=os.getenv('DEEPSEEK_BASE_URL','https://api.deepseek.com/v1'),
        timeout=120.0,
    )

    all_results = []
    for batch_i in range(0, 100, 20):
        batch = top100.iloc[batch_i:batch_i+20]
        lines = []
        for _, r in batch.iterrows():
            mv = r.get('total_mv', 0) / 1e8
            pe = r.get('pe_ttm', 0)
            lines.append(
                f"{r['code']} {r['name']} price={r['price']:.2f} "
                f"chg={r['pct_change']:+7.1f}% PE={pe:6.0f} PB={r.get('pb',0):.1f} "
                f"MV={mv:.0f}亿 score={r['rule_score']:.0f}"
            )

        prompt = f"""对以下规则预筛的A股进行深度分析,结合技术面和估值给出最终评分(0-100)和操作建议(BUY/HOLD/SELL)。

每只返回: final_score, action(BUY/HOLD/SELL), conviction(0-1), technical(技术面一句话), fundamental(基本面一句话), risk(主要风险)

候选股票:
{chr(10).join(lines)}

返回JSON数组,按final_score降序。只返回JSON,不要其他文字。"""

        print(f'DeepSeek batch {batch_i//20+1}/5 ({len(lines)} stocks)...')
        resp = await client.chat.completions.create(
            model=os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"),
            messages=[{'role':'system','content':'你是资深A股分析师。只返回JSON,不要markdown包裹。'},
                      {'role':'user','content':prompt}],
            temperature=0.3, max_tokens=4000,
        )
        content = resp.choices[0].message.content.strip()
        # strip markdown code fences
        if content.startswith('```'):
            content = content.split('\n', 1)[1]
            if content.endswith('```'):
                content = content[:-3]
        data = json.loads(content)
        items = data if isinstance(data, list) else data.get('stocks', data.get('results', []))
        # Merge code/name from input batch (DeepSeek may omit them)
        batch_map = {r['code']: r for _, r in batch.iterrows()}
        for idx, item in enumerate(items):
            if not item.get('code') and idx < len(batch):
                # Match by index since order is preserved
                row = batch.iloc[idx]
                item['code'] = str(row['code'])
                item['name'] = str(row['name'])
            elif item.get('code') and item['code'] in batch_map:
                item['name'] = str(batch_map[item['code']]['name'])
        all_results.extend(items)
        print(f'  -> {len(items)} analyzed | tokens={resp.usage.total_tokens}')

    # 4. 排序输出
    all_results.sort(key=lambda x: x.get('final_score', x.get('score', 0)), reverse=True)

    print(f'\n{"="*80}')
    print(f'DeepSeek深度分析 Top30 (from {len(all_results)} stocks)')
    print(f'{"排名":<4} {"代码":<12} {"名称":<10} {"评分":<5} {"操作":<6} {"确信":<5} {"技术面":<30} {"风险"}')
    print('-'*80)
    for i, r in enumerate(all_results[:30]):
        code = r.get('code','')
        name = r.get('name','')
        score = r.get('final_score', r.get('score',0))
        action = r.get('action','')
        conv = f"{r.get('conviction',0):.0%}"
        tech = r.get('technical','')[:28]
        risk = r.get('risk','')[:20]
        print(f'{i+1:<4} {code:<12} {name:<10} {score:<5} {action:<6} {conv:<5} {tech:<30} {risk}')

    # 保存
    outpath = 'reports/deep_analysis_top100.json'
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump({
            'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'total_screened': initial,
            'top100_rule_score_range': [float(top100['rule_score'].min()), float(top100['rule_score'].max())],
            'results': all_results[:100]
        }, f, ensure_ascii=False, indent=2)
    print(f'\nSaved to {outpath}')
    return all_results

if __name__ == '__main__':
    asyncio.run(main())
