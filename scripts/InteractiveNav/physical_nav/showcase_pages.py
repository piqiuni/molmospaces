"""Presentation-only pages for the physical interactive-navigation demo."""

from __future__ import annotations


def build_showcase_page(theme: str) -> str:
    light = theme == "light"
    title = "具身智能交互导航" if light else "具身智能交互导航展示平台"
    subtitle = "交互改变环境的可达性与可见性" if light else "Embodied AI Interactive Navigation Platform for Unitree Go2"
    theme_class = "light" if light else "dark"
    # The dark page follows the supplied 16:9 reference literally: compact
    # header, a 65/35 split, large perception panel over two equal map/graph
    # panels, and a tall right rail with Go2 status and Agent timeline.  Every
    # rule is scoped to body.dark so the light variant remains independent.
    dark_extra_css = "" if light else """
body.dark { overflow:hidden; }
body.dark .page { width:100vw; height:100vh; min-height:0; max-height:100vh; padding:18px 22px 14px; gap:10px; }
body.dark header { min-height:48px; padding:0 1px 3px; }
body.dark .logo { display:none; }
body.dark .brand { gap:0; }
body.dark h1 { font-size:27px; line-height:1.05; text-shadow:0 0 18px #58baff28; }
body.dark .subtitle { font-size:11px; letter-spacing:.01em; }
body.dark .top-tags { gap:11px; }
body.dark .tag { padding:8px 14px; border-radius:7px; background:linear-gradient(180deg,#102b47,#0b1828); box-shadow:inset 0 1px #ffffff0b,0 0 9px #0c6aa522; }
body.dark .layout { grid-template-columns:minmax(0,1.86fr) minmax(390px,1fr); gap:12px; }
body.dark .visuals { grid-template-rows:minmax(0,1.08fr) minmax(0,1fr); gap:10px; }
body.dark .bottom { gap:10px; }
body.dark .panel { border-color:#2b5f91; border-radius:12px; background:linear-gradient(145deg,#0d2035,#081421 78%); box-shadow:0 7px 25px #0008; }
body.dark .panel-title { height:40px; padding:8px 15px; color:#70c5ff; font-size:17px; letter-spacing:.015em; border-bottom-color:#214a72; background:linear-gradient(100deg,#102d4dcc 0%,#0d233900 70%); position:relative; }
body.dark .panel-title:after { content:""; position:absolute; left:0; bottom:-1px; width:205px; height:2px; background:linear-gradient(90deg,#35b5ff,#35b5ff00); }
body.dark .panel-title small { font-size:10px; }
body.dark .visuals > .panel:first-child { box-shadow:0 0 0 1px #58baff1c,0 14px 42px #0009; }
body.dark .visuals > .panel:first-child .panel-title { color:#79caff; }
body.dark .bottom .panel:first-child .panel-title { color:#5ee0c4; }
body.dark .bottom .panel:last-child .panel-title { color:#f3c45b; }
body.dark canvas { background:#050a12; }
body.dark .rail { grid-template-rows:minmax(320px, 3fr) minmax(0,7fr); gap:10px; }
body.dark .robot-card { padding:13px 14px 11px; background:radial-gradient(circle at 82% 20%,#163d6230,transparent 42%),linear-gradient(145deg,#0e2137,#091522); }
body.dark .robot-head { margin-bottom:9px; }
body.dark .robot-head h2, body.dark .agent h2 { font-size:18px; color:#78c7ff; }
body.dark .robot-body { grid-template-columns:1.18fr .82fr; gap:8px; }
body.dark .metrics { grid-template-columns:repeat(3,minmax(0,1fr)); gap:6px; }
body.dark .metric { padding:7px 8px; background:#0a1a2bcc; border-color:#315d88; }
body.dark .metric .icon { font-size:21px; }
body.dark .metric .name { font-size:11px; }
body.dark .metric .value { font-size:16px; }
body.dark .dog { width:100%; max-width:100%; height:auto; max-height:none; object-fit:contain; object-position:center; }
body.dark .agent { padding:13px 12px 11px; background:linear-gradient(145deg,#0d1e31,#08131f); grid-template-rows:auto minmax(112px,.52fr) minmax(0,1.48fr); }
body.dark .block { padding:9px; background:#091522dd; border-color:#315d88; }
body.dark .block h3 { color:#79caff; font-size:17px; }
body.dark .call { padding:8px; font-size:14px; border:1px solid #315d88; border-radius:8px; background:#091522; }
body.dark .stage { color:#7ecaff; }
body.dark .result { color:#d8e6f4; }
body.dark .latency { color:#63ddc1; }
body.dark .note { font-size:11px; }
body.dark .timeline { padding:2px 3px; }
body.dark .event { grid-template-columns:53px 25px 1fr; min-height:43px; }
body.dark .event:not(:last-child):after { left:65px; }
body.dark .time { font-size:10px; }
body.dark .eventtext { font-size:16px; }
body.dark .eventtext b { font-size:17px; }
body.dark footer { height:84px; margin-top:0; background:linear-gradient(90deg,#0b1a2b,#10243a,#0b1a2b); border-color:#2b5f91; gap:17px; font-size:13px; }
body.dark footer b { color:#eef7ff; }
body.dark footer i { width:34px; background:#5f7692; }
body.dark footer i:after { color:#a5c9e8; }
body.dark .debug { right:16px; font-size:10px; }
body.dark .dog { display:block; width:62%; max-width:62%; height:auto; max-height:none; justify-self:center; align-self:center; }
@media (max-height:800px) {
  body.dark .page { padding-top:10px; padding-bottom:8px; gap:7px; }
  body.dark header { min-height:42px; }
  body.dark h1 { font-size:23px; }
  body.dark .rail { grid-template-rows:minmax(280px, 3fr) minmax(0,7fr); }
  body.dark footer { height:63px; }
}
"""
    # Report-oriented light treatment: stronger hierarchy and generous white
    # surfaces keep the live visualisations readable on projectors.
    light_extra_css = "" if not light else """
/* The report view follows the supplied light reference: a white 16:9
   dashboard with perception/task at left, map/graph in the centre and the
   Go2 + Agent column at right.  It is deliberately scoped to body.light. */
body.light { background: #f5f7fa; overflow: hidden; }
body.light .page { width: 100vw; height: 100vh; min-height: 0; max-height: 100vh;
  padding: 20px 22px 14px; gap: 11px; background: #fff; }
body.light header { min-height: 61px; padding: 0 2px 7px; border-bottom: 1px solid #d8e0ea; }
body.light .logo { width: 48px; height: 48px; border-radius: 11px; background: #f0f7ff; border-color: #b9d0ec; }
body.light h1 { color: #143d78; font-size: 31px; letter-spacing: .055em; line-height: 1.05; }
body.light .subtitle { color: #52647b; font-size: 12px; margin-top: 5px; }
body.light .top-tags { gap: 9px; }
body.light .tag { background: #fff; border-color: #9cb8db; border-radius: 7px; padding: 8px 13px; color: #174f91; font-size: 11px; }
body.light .layout { min-height: 0; display: grid; grid-template-columns: minmax(0,1.02fr) minmax(0,1.18fr) minmax(320px,.98fr);
  grid-template-rows: minmax(0,3fr) minmax(0,2fr); gap: 10px; }
body.light .visuals { display: contents; }
body.light .rail { display: grid; grid-column: 3; grid-row: 1 / span 2; grid-template-rows: minmax(0,3fr) minmax(0,7fr); gap: 10px; min-height: 0; }
body.light .visuals > .panel:first-child { grid-column: 1; grid-row: 1; }
body.light .light-task { display: block; grid-column: 1; grid-row: 2; }
body.light .visuals > .bottom { display: contents; }
body.light .visuals > .bottom > .panel:first-child { grid-column: 2; grid-row: 1; }
body.light .visuals > .bottom > .panel:last-child { grid-column: 2; grid-row: 2; }
body.light .rail > .robot-card { grid-column: 3; grid-row: 1; align-self: stretch; }
body.light .rail > .agent { grid-column: 3; grid-row: 2; align-self: stretch; }
body.light .rail > .robot-card, body.light .rail > .agent { grid-column: auto; grid-row: auto; }
body.light .panel { background: #fff; border-color: #c6d1df; border-radius: 10px; box-shadow: 0 2px 11px rgba(35,64,96,.10); }
body.light .panel-title { height: 39px; padding: 8px 13px; color: #164f98; font-size: 15px; background: linear-gradient(90deg,#f0f5fb,#fff); border-bottom-color: #d5dee9; }
body.light .panel-title small { font-size: 9px; color: #66758a; }
body.light canvas { background: #eef1f5; }
body.light .light-task .task-body { padding: 14px 16px; color: #203b61; font-size: 13px; line-height: 1.6; }
body.light .light-task .task-kicker { color: #174f91; font-weight: 700; font-size: 12px; margin-bottom: 7px; }
body.light .light-task .task-target { font-size: 18px; font-weight: 700; color: #123b73; margin: 4px 0 8px; }
body.light .light-task .task-row { display: flex; gap: 8px; align-items: center; margin-top: 5px; }
body.light .light-task .task-dot { width: 8px; height: 8px; border-radius: 50%; background: #1f9b60; flex: 0 0 auto; }
body.light .robot-card { padding: 12px 14px; }
body.light .robot-head { margin-bottom: 6px; }
body.light .robot-head h2, body.light .agent h2 { color: #164f98; font-size: 17px; }
body.light .robot-body { grid-template-columns: 1.18fr .82fr; gap: 7px; }
body.light .metrics { grid-template-columns:repeat(3,minmax(0,1fr)); gap: 5px; }
body.light .metric { background: #f8fbff; border-color: #cfdae7; border-radius: 7px; padding: 6px 7px; }
body.light .metric .icon { font-size: 20px; }
body.light .metric .name { font-size: 10px; }
body.light .metric .value { color: #173d6b; font-size: 15px; }
body.light .dog { width:100%; max-width:100%; height:auto; max-height:none; object-fit:contain; object-position:center; }
body.light .block { background: #fbfcfe; border-color: #cfdae7; padding: 8px; }
body.light .block h3 { color: #173d6b; font-size: 12px; }
body.light .call { border:1px solid #cfdae7; border-radius:8px; background:#fbfcfe; font-size: 9px; padding: 7px; }
body.light .agent .call { font-size: 15px; }
body.light .agent .call .stage { font-size: 15px; }
body.light .agent .eventtext { font-size: 16px; }
body.light .agent .eventtext b { font-size: 17px; }
body.light .stage { color: #145ca6; }
body.light .result { color: #2c4058; }
body.light .dot { background: #fff; }
body.light footer { height: 42px; background: #f8fafc; border-color: #c2cfde; color: #5f7187; }
body.light footer b { color: #214976; }
body.light .debug { right: 14px; }
body.light .dog { display:block; width:100%; max-width:100%; height:auto; max-height:none; }
@media (max-height: 800px) { body.light .page { padding-top: 11px; padding-bottom: 8px; gap: 7px; }
  body.light header { min-height: 52px; } body.light h1 { font-size: 26px; }
  body.light footer { height: 33px; } body.light .panel-title { height: 34px; padding: 6px 11px; }
  body.light .light-task .task-body { padding: 9px 12px; } }
"""
    return f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{title}</title><style>
*{{box-sizing:border-box}}:root{{--bg:#06101d;--panel:#0b1828;--panel2:#0e1d30;--line:#25486c;--text:#f2f7ff;--muted:#8da7c3;--blue:#58baff;--teal:#42dfc0;--green:#64e88b;--gold:#f2bf55;--red:#ff6d7c;--shadow:0 12px 32px #0005}}
body.light{{--bg:#f4f2ed;--panel:#fffdfa;--panel2:#faf9f5;--line:#c7d0dc;--text:#143c76;--muted:#66758a;--blue:#146ca9;--teal:#078d91;--green:#23995d;--gold:#d99a16;--red:#c94a45;--shadow:0 8px 24px #27466a12}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,"Noto Sans SC","Microsoft YaHei",sans-serif;min-height:100vh}}body.dark{{background:radial-gradient(circle at 15% -20%,#14375d,transparent 34%),var(--bg)}}{dark_extra_css}{light_extra_css}
.page{{width:min(100vw,1920px);min-height:100vh;margin:auto;padding:16px 20px;display:grid;grid-template-rows:auto 1fr auto;gap:12px}}header{{display:flex;align-items:center;justify-content:space-between;gap:18px}}.brand{{display:flex;align-items:center;gap:13px}}.logo{{width:43px;height:43px;border:1px solid var(--line);border-radius:10px;display:grid;place-items:center;background:var(--panel)}}.logo svg{{width:29px;height:29px}}h1{{font-size:25px;margin:0;letter-spacing:.03em}}.subtitle{{font-size:11px;color:var(--muted);margin-top:3px}}.top-tags{{display:flex;gap:8px;align-items:center}}.tag{{padding:7px 12px;border:1px solid var(--line);border-radius:7px;color:var(--blue);font-size:11px;background:var(--panel)}}.live{{color:var(--green);font-size:12px;font-weight:700}}.live:before{{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;background:currentColor;box-shadow:0 0 10px currentColor;margin-right:7px}}
.layout{{min-height:0;display:grid;grid-template-columns:minmax(650px,1.75fr) minmax(390px,.75fr);gap:12px}}.visuals{{min-height:0;display:grid;grid-template-rows:minmax(290px,1.25fr) minmax(250px,1fr);gap:10px}}.bottom{{display:grid;grid-template-columns:1fr 1fr;gap:10px;min-height:0}}.light-task{{display:none}}.panel{{position:relative;min-width:0;min-height:0;border:1px solid var(--line);border-radius:12px;background:var(--panel);overflow:hidden;box-shadow:var(--shadow)}}.panel-title{{height:38px;padding:9px 13px;color:var(--blue);font-weight:750;font-size:14px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between}}.panel-title small{{font-size:9px;color:var(--muted);font-weight:400}}canvas{{display:block;width:100%;height:calc(100% - 38px);object-fit:contain;background:#080c12}}body.light canvas{{background:#eef0ef}}
.rail{{min-height:0;display:grid;grid-template-rows:auto minmax(0,1fr);gap:10px}}.robot-card{{padding:12px 14px}}.robot-head{{display:flex;align-items:center;justify-content:space-between;margin-bottom:9px}}.robot-head h2,.agent h2{{font-size:15px;margin:0;color:var(--blue)}}.robot-body{{display:grid;grid-template-columns:1.2fr .8fr;align-items:center;gap:8px;min-width:0;min-height:0}}.metrics{{display:grid;grid-template-columns:1fr 1fr;gap:6px;min-width:0}}.metric{{border:1px solid var(--line);background:var(--panel2);border-radius:8px;padding:7px 8px;min-width:0;text-align:center}}.metric .icon{{font-size:17px;line-height:1.1;display:block}}.metric .name{{font-size:9px;color:var(--muted);display:block}}.metric .value{{font-size:14px;font-weight:700;display:block;margin-top:3px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.dog{{display:block;width:100%;max-width:100%;height:auto;max-height:none;object-fit:contain;object-position:center;filter:drop-shadow(0 10px 9px #0003)}}.safe{{margin-top:7px;display:flex;gap:5px;flex-wrap:wrap}}.pill{{font-size:9px;padding:4px 7px;border:1px solid var(--line);border-radius:99px;color:var(--muted)}}.pill.ok{{color:var(--green);border-color:color-mix(in srgb,var(--green) 45%,transparent)}}
.agent{{padding:12px;display:grid;grid-template-rows:auto minmax(105px,.55fr) minmax(220px,1.35fr);gap:9px}}.block{{min-height:0;border:1px solid var(--line);border-radius:9px;background:var(--panel2);padding:9px}}.block h3{{font-size:12px;margin:0 0 7px;color:var(--text)}}.calls{{display:grid;gap:7px}}.call{{display:grid;grid-template-columns:32px 1fr auto;gap:7px;align-items:center;padding:7px;border:1px solid var(--line);border-radius:8px;background:var(--panel2);font-size:9px}}.stage{{font-weight:800;color:var(--blue)}}.result{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--text)}}.latency{{color:var(--teal)}}.note{{font-size:8px;color:var(--muted);margin-top:5px}}
.timeline{{height:calc(100% - 20px);overflow:auto;padding:1px 3px}}.event{{display:grid;grid-template-columns:50px 24px 1fr;gap:7px;min-height:39px;position:relative}}.event:not(:last-child):after{{content:"";position:absolute;left:61px;top:24px;bottom:-2px;width:1px;background:var(--line)}}.time{{font-size:8px;color:var(--muted);padding-top:7px}}.dot{{width:23px;height:23px;border-radius:50%;border:1px solid var(--blue);display:grid;place-items:center;color:var(--blue);font-size:9px;background:var(--panel2);z-index:1}}.event.good .dot{{color:var(--green);border-color:var(--green)}}.event.good .eventtext b{{color:var(--green)}}.eventtext{{padding-top:4px;font-size:9px;color:var(--muted);min-width:0}}.eventtext b{{color:var(--text);font-size:10px;margin-right:7px}}.eventtext span{{word-break:break-word}}
footer{{height:48px;border:1px solid var(--line);border-radius:10px;background:var(--panel);display:flex;align-items:center;justify-content:center;gap:9px;color:var(--muted);font-size:10px}}.flow-step{{display:flex;align-items:center;gap:6px;padding:7px 11px;border:1px solid var(--line);border-radius:8px;background:var(--panel2);color:var(--text);white-space:nowrap}}.flow-step .flow-icon{{font-size:18px;line-height:1;color:var(--blue)}}.flow-step b{{color:var(--text);font-weight:650}}.flow-arrow{{font-size:23px;color:var(--blue);line-height:1}}.debug{{position:absolute;right:22px;color:var(--muted);text-decoration:none;font-size:9px}}
@media(max-width:1100px){{.page{{padding:10px}}.layout{{grid-template-columns:1fr}}.rail{{grid-template-columns:1fr 1.5fr;grid-template-rows:auto}}.agent{{min-height:520px}}}}@media(max-width:720px){{.top-tags{{display:none}}.bottom,.rail{{grid-template-columns:1fr}}.visuals{{grid-template-rows:340px auto}}.bottom .panel{{height:300px}}footer{{display:none}}}}
</style><body class='{theme_class}'><div class='page'><header><div class='brand'><div class='logo'><svg viewBox='0 0 32 32' fill='none'><path d='M4 10 16 3l12 7v13l-12 6-12-6Z' stroke='var(--blue)' stroke-width='2'/><path d='m4 10 12 7 12-7M16 17v12' stroke='var(--teal)' stroke-width='2'/><circle cx='16' cy='17' r='3' fill='var(--green)'/></svg></div><div><h1>{title}</h1><div class='subtitle'>{subtitle}</div></div></div><div class='top-tags'><span class='live' id='live'>实时运行中</span><span class='tag'>开放词汇感知</span><span class='tag'>分层交互图</span><span class='tag'>模型驱动决策</span></div></header>
<main class='layout'><section class='visuals'><div class='panel'><div class='panel-title'>01&nbsp; 实时语义感知 <small id='detect-meta'>D435i · YOLOE-26l PF Seg</small></div><canvas id='view1' width='480' height='270'></canvas></div><div class='panel light-task'><div class='panel-title'>02&nbsp; 当前任务与目标 <small>Task · Goal</small></div><div class='task-body'><div class='task-kicker'>任务指令</div><div class='task-target' id='task-target'>等待任务输入</div><div class='task-row'><i class='task-dot'></i><span id='task-goal'>高层目标：等待语义目标</span></div><div class='task-row'><i class='task-dot'></i><span id='task-status'>状态：实时感知与建图</span></div></div></div><div class='bottom'><div class='panel'><div class='panel-title'>03&nbsp; 空间理解 <small>Room · Reachability</small></div><canvas id='view3' width='480' height='270'></canvas></div><div class='panel'><div class='panel-title'>06&nbsp; 交互语义图 <small>Room → Portal → Container → Object</small></div><canvas id='view6' width='480' height='270'></canvas></div></div></section>
<aside class='rail'><section class='panel robot-card'><div class='robot-head'><h2>Go2 当前状态</h2><span class='live' id='robot-live'>在线</span></div><div class='robot-body'><div id='metrics' class='metrics'></div><img id='dog' class='dog' src='/assets/go2-user-reference.png' alt='Unitree Go2 实机图'></div><div id='safe' class='safe'></div></section>
<section class='panel agent'><h2>交互导航 Agent</h2><div class='block'><h3>MLLM 调用汇总</h3><div id='calls' class='calls'></div><div class='note'>M3 为规则验证，不调用模型</div></div><div class='block'><h3>Agent 当前行为时间线</h3><div id='timeline' class='timeline'></div></div></section></aside></main>
<footer><span class='flow-step'><span class='flow-icon'>◉</span><b>真实环境输入</b></span><span class='flow-arrow'>➜</span><span class='flow-step'><span class='flow-icon'>◈</span><b>开放词汇感知</b></span><span class='flow-arrow'>➜</span><span class='flow-step'><span class='flow-icon'>◇</span><b>语义交互图</b></span><span class='flow-arrow'>➜</span><span class='flow-step'><span class='flow-icon'>◎</span><b>MLLM 决策</b></span><span class='flow-arrow'>➜</span><span class='flow-step'><span class='flow-icon'>✓</span><b>状态验证</b></span><a href='/' class='debug'>Debug 页面</a></footer></div>
<script>
const $=id=>document.getElementById(id),txt=v=>String(v??'').replace(/\\s+/g,' ').trim(),clip=(v,n=54)=>{{v=txt(v);return v.length>n?v.slice(0,n)+'…':v}},num=(v,d=1)=>Number.isFinite(Number(v))?Number(v).toFixed(d):'--';setTimeout(()=>{{draw=function(b,c){{const v=$('view1').getContext('2d');v.drawImage(c,0,0,480,270);[['view3',960,0],['view6',960,270]].forEach(([id,x,y])=>{{const q=$(id).getContext('2d');q.drawImage(b,x,y,480,270,0,0,480,270)}});$('live').textContent='实时运行中'}}}},0);
function result(e){{if(!e)return '等待调用';if(e.error)return '调用失败';let v=e.raw_text??e.response?.raw_text??e.payload?.result??'';if(typeof v==='object')v=JSON.stringify(v);try{{const o=JSON.parse(v);return clip([o.candidate_id,Array.isArray(o.ranked_ids)?o.ranked_ids.join(' → '):'',o.label,o.state,o.reason].filter(Boolean).join(' · '))}}catch(_){{return clip(v||'调用完成')}}}}
function renderMetrics(s){{const t=s.telemetry||{{}},b=t.battery||{{}},v=Array.isArray(t.velocity)?Math.hypot(...t.velocity.slice(0,3).map(Number)):Number(t.speed||0),yaw=Number(t.yaw),data=[['🔋','电量',num(b.soc??t.battery_soc,0)+' %'],['↗','速度',num(v,2)+' m/s'],['⟳','航向',Number.isFinite(yaw)?num(yaw*180/Math.PI,1)+'°':'--'],['◉','动作模式',t.mode===undefined?'--':'模式 '+t.mode],['📷','D435i',s.link?.connected===false?'离线':'在线'],['◇','语义节点',String(s.graph?.node_count??0)]];const box=$('metrics');box.replaceChildren();data.forEach(([icon,n,v])=>{{const m=document.createElement('div');m.className='metric';m.innerHTML='<span class="icon"></span><span class="name"></span><span class="value"></span>';m.querySelector('.icon').textContent=icon;m.querySelector('.name').textContent=n;m.querySelector('.value').textContent=v;box.append(m)}});const safe=$('safe');safe.replaceChildren();[['D435i 在线',s.link?.connected!==false],['实物动作安全阻断',true],['感知目标 '+(s.detections?.length??0),false]].forEach(([v,ok])=>{{const p=document.createElement('span');p.className='pill '+(ok?'ok':'');p.textContent=v;safe.append(p)}});$('robot-live').textContent=s.link?.connected===false?'离线':'在线';$('detect-meta').textContent='D435i · YOLOE-26l PF Seg · '+(s.detections?.length??0)+' targets'}}
function renderCalls(s){{const all=[...(s.mllm?.M1||[]),...(s.mllm?.M2||[])].sort((a,b)=>(b.timestamp||0)-(a.timestamp||0));const latest={{M1:all.find(x=>x.stage==='M1'),M2:all.find(x=>x.stage==='M2')}};const box=$('calls');box.replaceChildren();['M1','M2'].forEach(stage=>{{const e=latest[stage],r=document.createElement('div');r.className='call';r.innerHTML='<span class="stage"></span><span class="result"></span><span class="latency"></span>';r.children[0].textContent=stage;r.children[1].textContent=e?result(e):(stage==='M1'?'等待交互属性识别':'等待子目标选择');r.children[2].textContent=e?.latency_s!=null?num(e.latency_s,2)+' s':'--';box.append(r)}})}}
let lastSig='';function renderTimeline(s){{const n=s.navigation||{{}},d=n.decision_trace||{{}},e=n.execution_state||{{}},f=n.behavior_feedback||{{}},r=n.interaction_result||{{}},candidate=d.executed_candidate_id||d.model_selected_candidate_id||d.active_candidate_id||e.candidate_id||'等待候选',state=e.state||'IDLE',behavior=e.behavior_type||'',status=f.status||r.status||'',transition=(r.pre_state&&r.post_state)?r.pre_state+' → '+r.post_state:'';const events=[['◉','感知','更新 RGB-D、检测与语义地图'],['AI','推理',d.model_reason||d.model_error||'评估交互可达性'],['◎','子目标',candidate],['→','当前指令',behavior?state+' · '+behavior:state==='IDLE'?'保持待机':state],['✓','执行反馈',status||'等待行为反馈'],['M3','结果验证',transition||'等待状态/图一致性验证']];const sig=JSON.stringify(events);if(sig===lastSig)return;lastSig=sig;const box=$('timeline');box.replaceChildren();events.forEach((x,i)=>{{const row=document.createElement('div');row.className='event '+(i===5&&transition?'good':'');const now=new Date();row.innerHTML='<span class="time"></span><span class="dot"></span><span class="eventtext"><b></b><span></span></span>';row.children[0].textContent=now.toLocaleTimeString().slice(0,8);row.children[1].textContent=x[0];row.children[2].children[0].textContent=x[1];row.children[2].children[1].textContent=clip(x[2],70);box.append(row)}})}}
function renderTask(s){{const n=s.navigation||{{}},d=n.decision_trace||{{}},e=n.execution_state||{{}},r=s.interaction_result||{{}},target=n.task_target||n.goal||d.task_target||s.task_target||'等待任务输入',candidate=d.model_selected_candidate_id||d.active_candidate_id||e.candidate_id||'等待语义目标',status=(r.pre_state&&r.post_state)?r.pre_state+' → '+r.post_state:(e.state||'实时感知与建图');const a=$('task-target'),b=$('task-goal'),c=$('task-status');if(a)a.textContent=clip(target,42);if(b)b.textContent='高层目标：'+clip(candidate,42);if(c)c.textContent='状态：'+clip(status,42)}}
function darkMap(){{const q=$("view3").getContext('2d'),w=480,h=270;q.fillStyle='#091321';q.fillRect(0,0,w,h);q.strokeStyle='#173554';q.lineWidth=1;for(let x=0;x<w;x+=16){{q.beginPath();q.moveTo(x,0);q.lineTo(x,h);q.stroke()}}for(let y=0;y<h;y+=16){{q.beginPath();q.moveTo(0,y);q.lineTo(w,y);q.stroke()}}q.fillStyle='#1b2c42';[[30,25,175,32],[280,20,160,38],[30,180,125,52],[230,170,205,60]].forEach(a=>q.fillRect(...a));q.strokeStyle='#42dfc0';q.lineWidth=2;q.strokeRect(190,105,35,45);q.fillStyle='#58baff';q.beginPath();q.arc(210,150,8,0,7);q.fill();q.strokeStyle='#64e88b';q.setLineDash([5,4]);q.beginPath();q.moveTo(210,150);q.lineTo(330,95);q.stroke();q.setLineDash([]);q.fillStyle='#f2bf55';q.font='12px sans-serif';q.fillText('ROOM · REACHABILITY',18,22);q.fillStyle='#8da7c3';q.fillText('Go2 pose',220,168);q.fillStyle='#64e88b';q.fillText('candidate portal',300,90)}}function darkGraph(){{const q=$("view6").getContext('2d'),w=480,h=270;q.fillStyle='#091321';q.fillRect(0,0,w,h);q.font='12px sans-serif';q.fillStyle='#8da7c3';q.fillText('ROOM',18,28);q.fillText('PORTAL',18,96);q.fillText('CONTAINER',18,164);q.fillText('OBJECT',18,232);const nodes=[[95,22,'Office','#42dfc0'],[95,90,'door · open','#64e88b'],[95,158,'fridge · closed','#58baff'],[95,226,'target · unknown','#8da7c3'],[270,90,'lobby door','#f2bf55'],[270,158,'cabinet · open','#64e88b'],[400,226,'object','#8da7c3']];q.strokeStyle='#31547a';q.lineWidth=2;[[0,1],[1,2],[2,3],[0,4],[4,5],[5,6]].forEach(([a,b])=>{{q.beginPath();q.moveTo(nodes[a][0]+55,nodes[a][1]+16);q.lineTo(nodes[b][0],nodes[b][1]+16);q.stroke()}});nodes.forEach(([x,y,t,c])=>{{q.fillStyle='#10233a';q.strokeStyle=c;q.strokeRect(x,y,105,32);q.fillStyle=c;q.fillText(t,x+8,y+21)}});q.fillStyle='#f2bf55';q.fillText('SELECTED SUBGOAL',330,24)}}let busy=false;function draw(b,c){{$('view1').getContext('2d').drawImage(c,0,0,480,270);darkMap();darkGraph();$('live').textContent='实时运行中'}}async function video(){{if(busy)return;busy=true;try{{const ts=Date.now(),rs=await Promise.all([fetch('/snapshot.jpg?t='+ts,{{cache:'no-store'}}),fetch('/camera-overlay.jpg?t='+ts,{{cache:'no-store'}})]),b=await createImageBitmap(await rs[0].blob()),c=await createImageBitmap(await rs[1].blob());draw(b,c);b.close();c.close()}}catch(_){{$('live').textContent='画面重连中'}}finally{{busy=false}}}}async function state(){{try{{const r=await fetch('/api/state-summary?t='+Date.now(),{{cache:'no-store'}}),s=await r.json();renderMetrics(s);renderCalls(s);renderTimeline(s);renderTask(s)}}catch(_){{$('robot-live').textContent='重连中'}}}}setInterval(video,200);setInterval(state,1000);video();state();
/* Redraw the spatial and interaction panels from raw map/graph receipts. */
function renderRawSpatial(v){{const q=$('view3').getContext('2d'),w=480,h=270,g=v?.occupancy;if(!g||!Array.isArray(g.data)||!g.width||!g.height){{q.fillStyle='#eef1f5';q.fillRect(0,0,w,h);q.fillStyle='#60758e';q.font='13px sans-serif';q.fillText('等待 OCC 原始数据',16,26);return}}q.fillStyle=document.body.classList.contains('dark')?'#091321':'#eef1f5';q.fillRect(0,0,w,h);const gw=Number(g.width),gh=Number(g.height),res=Number(g.resolution)||.05,step=Math.max(1,Math.ceil(Math.max(gw,gh)/150)),cell=Math.min(w/gw,h/gh),ox=(w-gw*cell)/2,oy=(h-gh*cell)/2;for(let y=0;y<gh;y+=step)for(let x=0;x<gw;x+=step){{const z=Number(g.data[y*gw+x]??-1);q.fillStyle=z<0?(document.body.classList.contains('dark')?'#29384a':'#c4cbd3'):z>=50?(document.body.classList.contains('dark')?'#f1f4f7':'#30343a'):(document.body.classList.contains('dark')?'#0e2033':'#fafafa');q.fillRect(ox+x*cell,oy+(gh-y-step)*cell,Math.max(1,cell*step+.3),Math.max(1,cell*step+.3))}}const origin=g.origin||{{}},p=v.telemetry?.map_position||v.telemetry?.position||[0,0,0],px=ox+(Number(p[0])-Number(origin.x||0))/res*cell,py=oy+(gh-(Number(p[1])-Number(origin.y||0))/res)*cell;if(Number.isFinite(px)&&Number.isFinite(py)){{q.fillStyle='#1586ff';q.beginPath();q.arc(px,py,6,0,Math.PI*2);q.fill();const yaw=Number(v.telemetry?.yaw||0),tx=px+18*Math.cos(yaw),ty=py-18*Math.sin(yaw);q.strokeStyle='#1586ff';q.lineWidth=2;q.beginPath();q.moveTo(px,py);q.lineTo(tx,ty);q.stroke()}}for(const d of (v.mapped_detections||[])){{const a=d.position||{{}},dx=ox+(Number(a.x)-Number(origin.x||0))/res*cell,dy=oy+(gh-(Number(a.y)-Number(origin.y||0))/res)*cell;if(Number.isFinite(dx)&&Number.isFinite(dy)){{q.fillStyle='#d89a18';q.beginPath();q.arc(dx,dy,3,0,Math.PI*2);q.fill()}}}}q.fillStyle=document.body.classList.contains('dark')?'#9db1c8':'#425770';q.font='11px sans-serif';q.fillText(`OCC · ${{gw}}×${{gh}} · ${{res.toFixed(2)}} m/cell`,10,16)}}
function renderRawGraph(v){{const q=$('view6').getContext('2d'),w=480,h=270,g=v?.graph||{{}},nodes=Array.isArray(g.nodes)?g.nodes:[],edges=Array.isArray(g.edges)?g.edges:[];q.fillStyle=document.body.classList.contains('dark')?'#091321':'#f8f9fa';q.fillRect(0,0,w,h);if(!nodes.length){{q.fillStyle='#60758e';q.font='13px sans-serif';q.fillText('等待语义 Graph 原始数据',16,26);return}}const pts=nodes.map((n,i)=>{{const c=n.aabb_center||n.centroid||n.position||[],x=Number(c[0]),y=Number(c[1]);return {{n,x:Number.isFinite(x)?x:(i%5),y:Number.isFinite(y)?y:Math.floor(i/5)}}}}),xs=pts.map(p=>p.x),ys=pts.map(p=>p.y),minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys),sx=400/Math.max(1,maxx-minx),sy=220/Math.max(1,maxy-miny),sc=Math.min(sx,sy),mx=x=>40+(x-minx)*sc,my=y=>235-(y-miny)*sc,byId=new Map();pts.forEach((p,i)=>{{const id=String(p.n.id??p.n.instance_id??p.n.name??i);byId.set(id,p)}});q.strokeStyle=document.body.classList.contains('dark')?'#476887':'#9aa9b8';q.lineWidth=1.3;for(const e of edges){{const a=byId.get(String(e.source??e.from??e.u??'')),b=byId.get(String(e.target??e.to??e.v??''));if(a&&b){{q.beginPath();q.moveTo(mx(a.x),my(a.y));q.lineTo(mx(b.x),my(b.y));q.stroke()}}}}for(const p of pts){{const n=p.n,type=String(n.type||n.node_type||'object').toLowerCase(),inter=n.interaction||{{}},state=String(inter.state||n.state||'').toLowerCase(),color=type.includes('room')?'#4abf87':type.includes('portal')||type.includes('door')?'#d89a18':type.includes('container')||/fridge|cabinet|drawer/.test(String(n.label||''))?'#42a5d5':'#8b96a3';q.fillStyle=color;q.beginPath();q.arc(mx(p.x),my(p.y),4,0,Math.PI*2);q.fill();q.fillStyle=document.body.classList.contains('dark')?'#d7e5f2':'#33465c';q.font='10px sans-serif';q.fillText(clip(n.label||n.name||n.id||type,20),mx(p.x)+6,my(p.y)-5)}}q.fillStyle=document.body.classList.contains('dark')?'#9db1c8':'#425770';q.font='11px sans-serif';q.fillText(`Graph · ${{nodes.length}} nodes · ${{edges.length}} edges`,10,16)}}
function renderVisualization(v){{renderRawSpatial(v);renderRawGraph(v)}}
let visualizationBusy=false;async function refreshVisualization(){{if(visualizationBusy)return;visualizationBusy=true;try{{const r=await fetch('/api/visualization-data?t='+Date.now(),{{cache:'no-store'}});if(r.ok)renderVisualization(await r.json())}}catch(_){{}}finally{{visualizationBusy=false}}}}
/* The showcase timeline records only the three runtime behavior modes.  A new
   row is added when navigation, exploration, or interaction becomes active. */
let behaviorHistory=[];
let lastBehaviorMode='';
function normalizeBehavior(s){{const e=s.navigation?.execution_state||{{}};const raw=String(e.behavior_type||e.state||'IDLE').toUpperCase();if(raw.includes('INTERACT')||raw.includes('OPEN')||raw.includes('CLOSE'))return ['INTERACT','交互'];if(raw.includes('EXPLORE')||raw.includes('EXPLOR'))return ['EXPLORE','探索'];if(raw.includes('NAVIGAT')||raw.includes('MOVE'))return ['NAVIGATE','导航'];return ['IDLE','待机']}}
function renderTimeline(s){{const [mode,label]=normalizeBehavior(s),n=s.navigation||{{}},e=n.execution_state||{{}},d=n.decision_trace||{{}},candidate=d.executed_candidate_id||d.model_selected_candidate_id||d.active_candidate_id||e.candidate_id||'';if(mode!==lastBehaviorMode){{lastBehaviorMode=mode;behaviorHistory.unshift({{mode,label,detail:String(e.behavior_type||e.state||candidate||'状态更新'),at:new Date()}});behaviorHistory=behaviorHistory.slice(0,8)}}const box=$('timeline');box.replaceChildren();if(!behaviorHistory.length){{box.innerHTML='<div class="empty">等待导航状态</div>';return}}behaviorHistory.forEach((item,index)=>{{const row=document.createElement('div');row.className='event '+(index===0?'good':'');row.innerHTML='<span class="time"></span><span class="dot"></span><span class="eventtext"><b></b><span></span></span>';row.children[0].textContent=item.at.toLocaleTimeString().slice(0,8);row.children[1].textContent=item.mode==='NAVIGATE'?'→':item.mode==='EXPLORE'?'◎':item.mode==='INTERACT'?'⚙':'·';row.children[2].children[0].textContent=item.label;row.children[2].children[1].textContent=clip(item.detail,70);box.append(row)}})}}
/* Dark-only panel renderers. Source pixels are preserved without a color mask. */
function renderDarkSpatial(source){{const q=$('view3').getContext('2d');q.clearRect(0,0,480,270);q.drawImage(source,960,0,480,270,0,0,480,270)}}
function renderDarkInteraction(source){{const q=$('view6').getContext('2d');q.clearRect(0,0,480,270);q.drawImage(source,960,270,480,270,0,0,480,270)}}
function draw(b,c){{$('view1').getContext('2d').drawImage(c,0,0,480,270);$('live').textContent='实时运行中'}}
setTimeout(()=>{{const v=$('view1');v.width=640;v.height=480;draw=function(b,c){{$('view1').getContext('2d').drawImage(c,0,0,640,480);$('live').textContent='实时运行中'}}}},0);setInterval(refreshVisualization,1000);refreshVisualization();
</script></body></html>"""


DARK_SHOWCASE_HTML = build_showcase_page("dark")
LIGHT_SHOWCASE_HTML = build_showcase_page("light")


def build_academic_page() -> str:
    """Minimal four-panel academic figure page.

    This page intentionally shares only the read-only runtime endpoints with the
    other showcases.  It does not alter the algorithm lifecycle and can be
    reloaded independently while the navigation process keeps running.
    """
    return r"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>Interactive Navigation · Academic View</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#fff;color:#17243a;font-family:Georgia,"Noto Serif SC","Times New Roman",serif}header{height:74px;border-bottom:2px solid #17243a;display:flex;align-items:center;justify-content:space-between;padding:0 28px}h1{font-size:25px;margin:0;font-weight:700;letter-spacing:.02em}header p{margin:3px 0 0;font-size:13px;color:#526176}.live{color:#b4212c;font:700 13px Inter,sans-serif}.live:before{content:"";display:inline-block;width:9px;height:9px;border-radius:50%;background:currentColor;margin-right:6px}.grid{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:minmax(330px,1fr) minmax(330px,1fr);gap:0;padding:14px 22px 22px;max-width:1920px;margin:auto}.cell{min-width:0;min-height:0;border:1px solid #c8ced7;padding:13px 15px;display:flex;flex-direction:column}.cell:nth-child(1),.cell:nth-child(2){border-top:0}.cell:nth-child(odd){border-left:0}.cell:nth-child(even){border-right:0}.cell:nth-child(3),.cell:nth-child(4){border-bottom:0}.title{display:flex;align-items:baseline;gap:10px;margin-bottom:8px}.letter{font:bold 24px Arial;color:#174b91}.title b{font-size:19px}.title small{font:12px Inter,sans-serif;color:#66758b;margin-left:auto}.canvas-wrap{flex:1;min-height:0;background:#f6f7f8;border:1px solid #e2e5e9;position:relative}.canvas-wrap canvas{display:block;width:100%;height:100%;object-fit:contain}.legend{display:flex;gap:15px;flex-wrap:wrap;padding-top:7px;font:11px Inter,sans-serif;color:#4c5a6d}.key{display:inline-flex;align-items:center;gap:5px}.sw{width:11px;height:11px;border:2px solid currentColor;display:inline-block}.red{color:#d62c2c}.blue{color:#1e5bb5}.green{color:#198344}.gold{color:#c58a00}.decision{display:grid;grid-template-columns:1.05fr .95fr;gap:16px;flex:1;min-height:0}.stack{display:flex;flex-direction:column;gap:8px;justify-content:center}.step{border:1px solid #1d5ca8;border-radius:5px;padding:9px 11px;font:13px/1.35 Inter,"Noto Sans SC",sans-serif;background:#f8fbff}.step strong{display:block;color:#174b91;font-size:14px;margin-bottom:3px}.step.m3{border-color:#7046a3;background:#fbf9ff}.step.m3 strong{color:#7046a3}.timeline{border-left:2px solid #8795a8;margin:18px 4px 12px 12px;padding-left:17px;display:flex;flex-direction:column;justify-content:space-around;min-height:230px}.event{position:relative;font:12px/1.35 Inter,"Noto Sans SC",sans-serif;color:#34445a}.event:before{content:"";position:absolute;left:-24px;top:4px;width:9px;height:9px;background:#fff;border:2px solid #1e5bb5;border-radius:50%}.event.good:before{border-color:#198344}.event b{display:block;color:#174b91;font-weight:600}.event span{color:#627189}.foot{font:11px Inter,sans-serif;color:#6c7888;text-align:center;padding:0 22px 12px}@media(max-width:900px){header{padding:0 14px}.grid{grid-template-columns:1fr;padding:8px 12px}.cell{min-height:380px;border:1px solid #c8ced7!important}.decision{grid-template-columns:1fr 1fr}}@media(max-width:600px){h1{font-size:18px}.decision{grid-template-columns:1fr}.timeline{min-height:180px}}
</style><header><div><h1>Interactive Navigation on Physical Go2</h1><p>交互性作为导航图中的状态变化因素</p></div><div class='live' id='live'>LIVE</div></header>
<main class='grid'><section class='cell'><div class='title'><span class='letter'>A</span><b>开放词汇感知</b><small id='detect-meta'>Intel RealSense D435i · YOLOE-26l PF Seg</small></div><div class='canvas-wrap'><canvas id='a' width='720' height='405'></canvas></div><div class='legend'><span class='key red'><i class='sw'></i>door</span><span class='key blue'><i class='sw'></i>container</span><span class='key green'><i class='sw'></i>candidate interaction</span></div></section>
<section class='cell'><div class='title'><span class='letter'>B</span><b>在线语义地图</b><small>space · depth · reachability</small></div><div class='canvas-wrap'><canvas id='b' width='720' height='405'></canvas></div><div class='legend'><span class='key blue'><i class='sw'></i>Go2 pose</span><span class='key red'><i class='sw'></i>portal</span><span class='key green'><i class='sw'></i>interaction target</span></div></section>
<section class='cell'><div class='title'><span class='letter'>C</span><b>分层交互 Graph</b><small>Room → Portal → Container → Object</small></div><div class='canvas-wrap'><canvas id='c' width='720' height='405'></canvas></div><div class='legend'><span class='key green'><i class='sw'></i>open</span><span class='key red'><i class='sw'></i>closed</span><span class='key gold'><i class='sw'></i>selected subgoal</span></div></section>
<section class='cell'><div class='title'><span class='letter'>D</span><b>Agent 决策</b><small>M1 / M2 / NAVIGATE / INTERACT / M3</small></div><div class='decision'><div class='stack' id='steps'><div class='step'><strong>M1 · 交互属性识别</strong>等待感知输入</div><div class='step'><strong>M2 · 子目标选择</strong>等待候选目标</div><div class='step'><strong>NAVIGATE / INTERACT</strong>只读动作记录</div><div class='step m3'><strong>M3 · 状态验证</strong>规则一致性检查</div></div><div class='timeline' id='timeline'></div></div></section></main><div class='foot'>A 感知 → B 地图 → C 交互图 → D Agent 决策；页面仅读取统一运行时状态。</div>
<script>
const $=id=>document.getElementById(id),clip=(v,n=62)=>String(v??'').replace(/\s+/g,' ').trim().slice(0,n),ctx=id=>$(id).getContext('2d');
function draw(b,c){ctx('a').drawImage(c,0,0,720,405)}
function drawAcademicRaw(v){const q=ctx('b'),w=q.canvas.width,h=q.canvas.height,g=v?.occupancy;q.fillStyle='#f8f9fa';q.fillRect(0,0,w,h);if(g&&Array.isArray(g.data)&&g.width&&g.height){const gw=Number(g.width),gh=Number(g.height),cell=Math.min(w/gw,h/gh),ox=(w-gw*cell)/2,oy=(h-gh*cell)/2,step=Math.max(1,Math.ceil(Math.max(gw,gh)/180));for(let y=0;y<gh;y+=step)for(let x=0;x<gw;x+=step){const z=Number(g.data[y*gw+x]??-1);q.fillStyle=z<0?'#c4cbd3':z>=50?'#30343a':'#fafafa';q.fillRect(ox+x*cell,oy+(gh-y-step)*cell,Math.max(1,cell*step+.3),Math.max(1,cell*step+.3))}const o=g.origin||{},p=v.telemetry?.map_position||v.telemetry?.position||[0,0,0],r=Number(g.resolution)||.05,px=ox+(Number(p[0])-Number(o.x||0))/r*cell,py=oy+(gh-(Number(p[1])-Number(o.y||0))/r)*cell;q.fillStyle='#1765d1';q.beginPath();q.arc(px,py,9,0,Math.PI*2);q.fill()}else{q.fillStyle='#60758e';q.font='16px Arial';q.fillText('等待 OCC 原始数据',20,30)}const c=ctx('c'),nodes=Array.isArray(v?.graph?.nodes)?v.graph.nodes:[],edges=Array.isArray(v?.graph?.edges)?v.graph.edges:[];c.fillStyle='#f8f9fa';c.fillRect(0,0,w,h);if(!nodes.length){c.fillStyle='#60758e';c.font='16px Arial';c.fillText('等待语义 Graph 原始数据',20,30);return}const pts=nodes.map((n,i)=>{const a=n.aabb_center||n.centroid||n.position||[];return {n,x:Number.isFinite(Number(a[0]))?Number(a[0]):i%8,y:Number.isFinite(Number(a[1]))?Number(a[1]):Math.floor(i/8)}}),xs=pts.map(p=>p.x),ys=pts.map(p=>p.y),minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys),sc=Math.min(820/Math.max(1,maxx-minx),470/Math.max(1,maxy-miny)),mx=x=>70+(x-minx)*sc,my=y=>500-(y-miny)*sc,idmap=new Map();pts.forEach((p,i)=>idmap.set(String(p.n.id??p.n.instance_id??i),p));c.strokeStyle='#9aa9b8';for(const e of edges){const a=idmap.get(String(e.source??e.from??'')),b=idmap.get(String(e.target??e.to??''));if(a&&b){c.beginPath();c.moveTo(mx(a.x),my(a.y));c.lineTo(mx(b.x),my(b.y));c.stroke()}}for(const p of pts){const n=p.n,t=String(n.type||n.node_type||'object').toLowerCase();c.fillStyle=t.includes('room')?'#198344':t.includes('portal')||t.includes('door')?'#c18400':t.includes('container')?'#1e5bb5':'#7f8b98';c.beginPath();c.arc(mx(p.x),my(p.y),7,0,Math.PI*2);c.fill();c.fillStyle='#24364c';c.font='12px Arial';c.fillText(clip(n.label||n.name||n.id||t,22),mx(p.x)+10,my(p.y)-7)}}
async function academicVisualization(){try{const r=await fetch('/api/visualization-data?t='+Date.now(),{cache:'no-store'});if(r.ok)drawAcademicRaw(await r.json())}catch(_){} }
async function video(){try{const ts=Date.now(),rs=await Promise.all([fetch('/snapshot.jpg?t='+ts,{cache:'no-store'}),fetch('/camera-overlay.jpg?t='+ts,{cache:'no-store'})]);draw(await createImageBitmap(await rs[0].blob()),await createImageBitmap(await rs[1].blob()));$('live').textContent='LIVE'}catch(_){$('live').textContent='RECONNECTING'}}
function textOf(e){if(!e)return '等待调用';return clip(e.raw_text??e.response?.raw_text??e.payload?.result??'调用完成')}
function state(s){const n=s.navigation||{},d=n.decision_trace||{},e=n.execution_state||{},r=s.interaction_result||{},m1=(s.mllm?.M1||[])[0],m2=(s.mllm?.M2||[])[0];$('detect-meta').textContent='D435i · YOLOE-26l PF Seg · '+(s.detections?.length??0)+' targets';$('steps').innerHTML='<div class="step"><strong>M1 · 交互属性识别</strong>'+clip(textOf(m1))+'</div><div class="step"><strong>M2 · 子目标选择</strong>'+clip(textOf(m2))+'</div><div class="step"><strong>NAVIGATE / INTERACT</strong>'+clip(e.behavior_type||e.state||'只读动作记录')+'</div><div class="step m3"><strong>M3 · 状态验证</strong>'+clip((r.pre_state&&r.post_state)?r.pre_state+' → '+r.post_state:'规则一致性检查')+'</div>';const ev=[['感知','RGB-D + YOLOE'],['M1','交互属性'],['M2','子目标 '+(d.model_selected_candidate_id||'等待')],['NAV',''+(e.state||'IDLE')],['M3',''+((r.pre_state&&r.post_state)?r.pre_state+' → '+r.post_state:'等待验证')]];$('timeline').innerHTML=ev.map(x=>'<div class="event '+(x[0]==='M3'?'good':'')+'"><b>'+x[0]+'</b><span>'+clip(x[1])+'</span></div>').join('')}
async function poll(){try{const r=await fetch('/api/state-summary?t='+Date.now(),{cache:'no-store'});state(await r.json())}catch(_){}}setInterval(video,200);setInterval(poll,1000);setInterval(academicVisualization,1000);video();poll();academicVisualization();
</script></html>"""


ACADEMIC_SHOWCASE_HTML = build_academic_page()


def build_academic_page_strict() -> str:
    """Paper-like 16:9 academic view (A/B/C/D, equal weight).

    Kept as a separate template so the three showcase routes can be reloaded
    independently.  Only the shared read-only snapshot/state endpoints are
    consumed here; no algorithm process is started by this page.
    """
    return r"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>Interactive Navigation on Physical Go2 · Academic View</title>
<style>
*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden}body{background:#fff;color:#162238;font-family:Arial,"Noto Sans SC","Microsoft YaHei",sans-serif}.academic{width:100vw;height:100vh;display:grid;grid-template-rows:12% 85% 3%;padding:0 1.7vw}.header{border-bottom:2px solid #1d2634;display:flex;flex-direction:column;align-items:center;justify-content:center;position:relative}.header h1{margin:0;font-family:Georgia,"Times New Roman",serif;font-size:clamp(20px,2.1vw,38px);font-weight:700;letter-spacing:.01em}.header p{margin:.35vh 0 0;color:#2d3b50;font-family:Georgia,"Noto Serif SC",serif;font-size:clamp(11px,1.15vw,21px)}.live{position:absolute;right:1.2vw;top:50%;transform:translateY(-50%);font-weight:700;font-size:clamp(12px,1.1vw,20px);color:#dd2525}.live:before{content:"";display:inline-block;width:.7em;height:.7em;margin-right:.4em;border-radius:50%;background:#dd2525;vertical-align:middle}.grid{min-height:0;display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;padding:1.25vh 0 0}.cell{min-width:0;min-height:0;border:1px solid #c4cbd4;padding:1.05vh 1.0vw;display:flex;flex-direction:column;background:#fff}.cell:nth-child(1),.cell:nth-child(2){border-top:0}.cell:nth-child(odd){border-left:0}.cell:nth-child(even){border-right:0}.cell:nth-child(3),.cell:nth-child(4){border-bottom:0}.title{display:flex;align-items:baseline;gap:.6vw;margin-bottom:.7vh;white-space:nowrap}.letter{font-weight:700;font-size:clamp(22px,1.85vw,34px);color:#174c97}.title b{font-family:Georgia,"Noto Serif SC",serif;font-size:clamp(16px,1.35vw,26px);color:#17283e}.title small{font-size:clamp(9px,.75vw,14px);color:#5b6675;margin-left:auto}.canvas-wrap{flex:1;min-height:0;border:1px solid #e1e5e9;background:#f8f9fa;display:flex;align-items:stretch;justify-content:stretch}.canvas-wrap canvas{display:block;width:100%;height:100%;object-fit:contain}.legend{display:flex;gap:1.0vw;flex-wrap:wrap;padding-top:.6vh;font-size:clamp(9px,.72vw,14px);color:#4d5b6d}.key{display:inline-flex;align-items:center;gap:.3em}.sw{display:inline-block;width:.8em;height:.8em;border:2px solid currentColor}.red{color:#d92d2d}.blue{color:#1958ad}.green{color:#178342}.gold{color:#c18400}.decision{display:grid;grid-template-columns:1.04fr .96fr;gap:1.2vw;flex:1;min-height:0}.stack{display:flex;flex-direction:column;justify-content:center;gap:.85vh}.step{border:1.5px solid #1c5dad;border-radius:5px;padding:1.0vh .75vw;font-size:clamp(10px,.8vw,15px);line-height:1.35;background:#fbfdff}.step strong{display:block;color:#174b91;font-size:clamp(11px,.9vw,17px);margin-bottom:.25vh}.step.interact{border-color:#d8332f}.step.interact strong{color:#cf2929}.step.m3{border-color:#7048a5;background:#fcfaff}.step.m3 strong{color:#7048a5}.timeline{border-left:2px solid #8793a2;margin:1.3vh .25vw 1vh .5vw;padding-left:1.0vw;display:flex;flex-direction:column;justify-content:space-around;min-height:0}.event{position:relative;font-size:clamp(9px,.72vw,14px);line-height:1.3;color:#24364c}.event:before{content:"";position:absolute;left:calc(-1.0vw - 7px);top:.25em;width:.58em;height:.58em;background:#fff;border:2px solid #1e5bb5;border-radius:50%}.event.good:before{border-color:#178342}.event b{display:block;color:#174b91;font-weight:700}.event span{color:#5b6b7e}.foot{text-align:center;color:#667486;font-size:clamp(8px,.65vw,12px);padding-top:.3vh}
@media(max-aspect-ratio:4/3){.academic{padding:0 1vw}.header h1{font-size:clamp(18px,2.7vw,30px)}.title small{display:none}.decision{grid-template-columns:1fr 1fr}}
</style><div class='academic'><header class='header'><h1>Interactive Navigation on Physical Go2</h1><p>交互性作为导航图中的状态变化因素</p><span class='live' id='live'>LIVE</span></header>
<main class='grid'><section class='cell'><div class='title'><span class='letter'>A</span><b>开放词汇感知</b><small id='detect-meta'>Intel RealSense D435i · YOLOE-26l PF Seg</small></div><div class='canvas-wrap'><canvas id='a' width='960' height='540'></canvas></div><div class='legend'><span class='key red'><i class='sw'></i>door</span><span class='key blue'><i class='sw'></i>container</span><span class='key green'><i class='sw'></i>candidate interaction</span></div></section>
<section class='cell'><div class='title'><span class='letter'>B</span><b>在线语义地图</b><small>space · depth · reachability</small></div><div class='canvas-wrap'><canvas id='b' width='960' height='540'></canvas></div><div class='legend'><span class='key blue'><i class='sw'></i>Go2 pose</span><span class='key red'><i class='sw'></i>portal</span><span class='key green'><i class='sw'></i>interaction target</span></div></section>
<section class='cell'><div class='title'><span class='letter'>C</span><b>分层交互 Graph</b><small>Room → Portal → Container → Object</small></div><div class='canvas-wrap'><canvas id='c' width='960' height='540'></canvas></div><div class='legend'><span class='key green'><i class='sw'></i>open / reachable</span><span class='key red'><i class='sw'></i>closed / blocked</span><span class='key gold'><i class='sw'></i>selected subgoal</span></div></section>
<section class='cell'><div class='title'><span class='letter'>D</span><b>Agent 决策</b><small>M1 / M2 / NAVIGATE / INTERACT / M3</small></div><div class='decision'><div class='stack' id='steps'><div class='step'><strong>M1 · 交互属性识别</strong>等待感知输入</div><div class='step'><strong>M2 · 子目标选择</strong>等待候选目标</div><div class='step interact'><strong>NAVIGATE / INTERACT</strong>只读动作记录</div><div class='step m3'><strong>M3 · 状态验证</strong>规则一致性检查</div></div><div class='timeline' id='timeline'></div></div></section></main><footer class='foot'>A 感知 → B 地图 → C 交互图 → D Agent 决策　·　页面仅读取统一运行时状态</footer></div>
<script>
const $=id=>document.getElementById(id),clip=(v,n=62)=>String(v??'').replace(/\s+/g,' ').trim().slice(0,n),ctx=id=>$(id).getContext('2d');
function draw(b,c){const a=ctx('a').canvas;a.width=960;a.height=720;ctx('a').drawImage(c,0,0,960,720);b.close();c.close()}
function drawAcademicRaw(v){const q=ctx('b'),w=q.canvas.width,h=q.canvas.height,g=v?.occupancy;q.fillStyle='#f8f9fa';q.fillRect(0,0,w,h);if(g&&Array.isArray(g.data)&&g.width&&g.height){const gw=Number(g.width),gh=Number(g.height),cell=Math.min(w/gw,h/gh),ox=(w-gw*cell)/2,oy=(h-gh*cell)/2,step=Math.max(1,Math.ceil(Math.max(gw,gh)/180));for(let y=0;y<gh;y+=step)for(let x=0;x<gw;x+=step){const z=Number(g.data[y*gw+x]??-1);q.fillStyle=z<0?'#c4cbd3':z>=50?'#30343a':'#fafafa';q.fillRect(ox+x*cell,oy+(gh-y-step)*cell,Math.max(1,cell*step+.3),Math.max(1,cell*step+.3))}const o=g.origin||{},p=v.telemetry?.map_position||v.telemetry?.position||[0,0,0],r=Number(g.resolution)||.05,px=ox+(Number(p[0])-Number(o.x||0))/r*cell,py=oy+(gh-(Number(p[1])-Number(o.y||0))/r)*cell;q.fillStyle='#1765d1';q.beginPath();q.arc(px,py,10,0,Math.PI*2);q.fill()}const c=ctx('c'),nodes=Array.isArray(v?.graph?.nodes)?v.graph.nodes:[],edges=Array.isArray(v?.graph?.edges)?v.graph.edges:[];c.fillStyle='#f8f9fa';c.fillRect(0,0,w,h);if(!nodes.length){c.fillStyle='#60758e';c.font='18px Arial';c.fillText('等待语义 Graph 原始数据',22,34);return}const pts=nodes.map((n,i)=>{const a=n.aabb_center||n.centroid||n.position||[];return {n,x:Number.isFinite(Number(a[0]))?Number(a[0]):i%8,y:Number.isFinite(Number(a[1]))?Number(a[1]):Math.floor(i/8)}}),xs=pts.map(p=>p.x),ys=pts.map(p=>p.y),minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys),sc=Math.min(820/Math.max(1,maxx-minx),470/Math.max(1,maxy-miny)),mx=x=>70+(x-minx)*sc,my=y=>500-(y-miny)*sc,idmap=new Map();pts.forEach((p,i)=>idmap.set(String(p.n.id??p.n.instance_id??i),p));c.strokeStyle='#9aa9b8';for(const e of edges){const a=idmap.get(String(e.source??e.from??'')),b=idmap.get(String(e.target??e.to??''));if(a&&b){c.beginPath();c.moveTo(mx(a.x),my(a.y));c.lineTo(mx(b.x),my(b.y));c.stroke()}}for(const p of pts){const n=p.n,t=String(n.type||n.node_type||'object').toLowerCase();c.fillStyle=t.includes('room')?'#198344':t.includes('portal')||t.includes('door')?'#c18400':t.includes('container')?'#1e5bb5':'#7f8b98';c.beginPath();c.arc(mx(p.x),my(p.y),8,0,Math.PI*2);c.fill();c.fillStyle='#24364c';c.font='13px Arial';c.fillText(clip(n.label||n.name||n.id||t,24),mx(p.x)+11,my(p.y)-8)}}
async function academicVisualization(){try{const r=await fetch('/api/visualization-data?t='+Date.now(),{cache:'no-store'});if(r.ok)drawAcademicRaw(await r.json())}catch(_){} }
async function video(){try{const ts=Date.now(),rs=await Promise.all([fetch('/snapshot.jpg?t='+ts,{cache:'no-store'}),fetch('/camera-overlay.jpg?t='+ts,{cache:'no-store'})]);draw(await createImageBitmap(await rs[0].blob()),await createImageBitmap(await rs[1].blob()));$('live').textContent='LIVE'}catch(_){$('live').textContent='RECONNECTING'}}
function textOf(e){if(!e)return '等待调用';return clip(e.raw_text??e.response?.raw_text??e.payload?.result??'调用完成')}
function state(s){const n=s.navigation||{},d=n.decision_trace||{},e=n.execution_state||{},r=s.interaction_result||{},m1=(s.mllm?.M1||[])[0],m2=(s.mllm?.M2||[])[0];$('detect-meta').textContent='Intel RealSense D435i · YOLOE-26l PF Seg · '+(s.detections?.length??0)+' targets';$('steps').innerHTML='<div class="step"><strong>M1 · 交互属性识别</strong>'+clip(textOf(m1))+'</div><div class="step"><strong>M2 · 子目标选择</strong>'+clip(textOf(m2))+'</div><div class="step interact"><strong>NAVIGATE / INTERACT</strong>'+clip(e.behavior_type||e.state||'只读动作记录')+'</div><div class="step m3"><strong>M3 · 状态验证</strong>'+clip((r.pre_state&&r.post_state)?r.pre_state+' → '+r.post_state:'规则一致性检查')+'</div>';const ev=[['感知','RGB-D + YOLOE'],['M1','交互属性'],['M2','子目标 '+(d.model_selected_candidate_id||'等待')],['NAV',''+(e.state||'IDLE')],['M3',''+((r.pre_state&&r.post_state)?r.pre_state+' → '+r.post_state:'等待验证')]];$('timeline').innerHTML=ev.map(x=>'<div class="event '+(x[0]==='M3'?'good':'')+'"><b>'+x[0]+'</b><span>'+clip(x[1])+'</span></div>').join('')}
async function poll(){try{const r=await fetch('/api/state-summary?t='+Date.now(),{cache:'no-store'});state(await r.json())}catch(_){} }setInterval(video,200);setInterval(poll,1000);setInterval(academicVisualization,1000);video();poll();academicVisualization();
</script></html>"""


# Export the strict 16:9 academic template used by /showcase-academic.
ACADEMIC_SHOWCASE_HTML = build_academic_page_strict()


def _use_original_renderer_panels(html: str, spatial_id: str, graph_id: str) -> str:
    """Display canonical panel 3/6 rasters instead of browser redraws.

    The live six-panel renderer already calls OfflineSixPanelRenderer's
    render_room_panel() and render_topology().  Presentation pages consume
    those native outputs directly so their geometry, labels, candidates and
    interaction states stay identical to the debug six-panel definition.
    """
    script = r"""<script>
/* Panel 03/06 are native OfflineSixPanelRenderer outputs. */
if (typeof renderVisualization === 'function') renderVisualization = function(){};
if (typeof drawAcademicRaw === 'function') drawAcademicRaw = function(){};
let originalRendererPanelsBusy = false;
async function refreshOriginalRendererPanels() {
  if (document.hidden || originalRendererPanelsBusy) return;
  originalRendererPanelsBusy = true;
  try {
    const stamp = Date.now();
    const responses = await Promise.all([
      fetch('/original-panel3.jpg?t=' + stamp, {cache: 'no-store'}),
      fetch('/original-panel6.jpg?t=' + stamp, {cache: 'no-store'})
    ]);
    if (!responses[0].ok || !responses[1].ok) throw new Error('original panel unavailable');
    const images = await Promise.all(responses.slice(0, 2).map(async response =>
      createImageBitmap(await response.blob())
    ));
    const visualization = navigationOverlayData;
    const spatial = document.getElementById('__SPATIAL_ID__');
    const graph = document.getElementById('__GRAPH_ID__');
    if (spatial) {
      spatial.width = images[0].width;
      spatial.height = images[0].height;
      spatial.getContext('2d').drawImage(images[0], 0, 0);
      drawNavigationOverlay(spatial, visualization);
    }
    if (graph) {
      graph.width = images[1].width;
      graph.height = images[1].height;
      graph.getContext('2d').drawImage(images[1], 0, 0);
    }
    images.forEach(image => image.close());
  } catch (_) {
    /* Camera/state polling owns the visible connection indicator. */
  } finally {
    originalRendererPanelsBusy = false;
  }
}
let navigationOverlayData={};
let navigationOverlayDataBusy=false;
async function refreshNavigationOverlayData(){
  if(document.hidden||navigationOverlayDataBusy)return;
  navigationOverlayDataBusy=true;
  try{const response=await fetch('/api/visualization-data?t='+Date.now(),{cache:'no-store'});if(response.ok)navigationOverlayData=await response.json()}catch(_){}finally{navigationOverlayDataBusy=false}
}
function navigationPoint(value) {
  if (Array.isArray(value) && value.length >= 2) return {x:Number(value[0]), y:Number(value[1])};
  if (!value || typeof value !== 'object') return null;
  if (Array.isArray(value.point)) return navigationPoint(value.point);
  if (value.position) return navigationPoint(value.position);
  if (value.pose) return navigationPoint(value.pose);
  const x=Number(value.x), y=Number(value.y);
  return Number.isFinite(x)&&Number.isFinite(y) ? {x,y} : null;
}
function navigationPath(value) {
  if (Array.isArray(value)) return value.map(navigationPoint).filter(Boolean);
  if (!value || typeof value !== 'object') return [];
  for (const key of ['poses','path','points','plan']) {
    if (Array.isArray(value[key])) return navigationPath(value[key]);
  }
  return [];
}
function drawNavigationOverlay(canvas, visualization) {
  const g=visualization?.occupancy, nav=visualization?.navigation||{};
  if (!g || !g.width || !g.height) return;
  const ctx=canvas.getContext('2d'), width=canvas.width, height=canvas.height;
  const resolution=Number(g.resolution)||0.1, origin=g.origin||{};
  const ox=Number(origin.x)||0, oy=Number(origin.y)||0;
  const toCanvas=p=>({x:(p.x-ox)/(Number(g.width)*resolution)*width,y:height-(p.y-oy)/(Number(g.height)*resolution)*height});
  const path=navigationPath(nav.global_plan);
  if (path.length>=2) {
    ctx.save(); ctx.strokeStyle='#35d6ff'; ctx.lineWidth=Math.max(3,width/320); ctx.shadowColor='#001b2b'; ctx.shadowBlur=4;
    ctx.beginPath(); path.forEach((point,index)=>{const p=toCanvas(point); index?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y)}); ctx.stroke();
    const end=toCanvas(path[path.length-1]); ctx.fillStyle='#35d6ff'; ctx.beginPath(); ctx.arc(end.x,end.y,5,0,Math.PI*2); ctx.fill();
    ctx.font='600 12px sans-serif'; ctx.fillText('GLOBAL PLAN',Math.min(width-105,Math.max(8,end.x+8)),Math.max(16,end.y-8)); ctx.restore();
  }
  const target=navigationPoint(nav.current_subgoal);
  const robot=navigationPoint(visualization.telemetry?.map_position||visualization.telemetry?.position);
  if (target) {
    const p=toCanvas(target); ctx.save(); ctx.strokeStyle='#ffb52e'; ctx.fillStyle='#ffb52e'; ctx.lineWidth=Math.max(3,width/320); ctx.setLineDash([8,5]);
    if (robot) {const r=toCanvas(robot);ctx.beginPath();ctx.moveTo(r.x,r.y);ctx.lineTo(p.x,p.y);ctx.stroke()}
    ctx.setLineDash([]);ctx.beginPath();ctx.arc(p.x,p.y,8,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.arc(p.x,p.y,3,0,Math.PI*2);ctx.fill();
    ctx.font='600 12px sans-serif'; ctx.fillText('SELECTED SUBGOAL',Math.min(width-155,Math.max(8,p.x+10)),Math.max(16,p.y-10)); ctx.restore();
  }
}
setInterval(refreshOriginalRendererPanels, 200);
setInterval(refreshNavigationOverlayData, 1000);
refreshOriginalRendererPanels();
refreshNavigationOverlayData();
</script>""".replace("__SPATIAL_ID__", spatial_id).replace("__GRAPH_ID__", graph_id)
    marker = "</body>" if "</body>" in html else "</html>"
    return html.replace(marker, script + marker)


def _use_debug_style_mllm_cards(html: str) -> str:
    """Give the dark showcase the Debug page's real M1/M2 call cards."""
    css = r"""<style>
body.dark .agent { grid-template-rows:auto minmax(190px,.9fr) minmax(0,1.1fr); }
body.dark .agent > .block:nth-of-type(1) { display:flex; flex-direction:column; overflow:hidden; }
body.dark .agent > .block:nth-of-type(1) .calls { flex:1; min-height:0; overflow:auto; align-content:start; padding-right:2px; }
body.dark .debug-call { display:grid; grid-template-columns:minmax(82px,.78fr) minmax(105px,1.08fr) minmax(98px,1fr); gap:7px; padding:8px; border:1px solid #315d88; border-radius:8px; background:#091522; font-size:11px; }
body.dark .debug-call-col { min-width:0; overflow:hidden; }
body.dark .debug-call-label { color:#7893ad; font-size:9px; text-transform:uppercase; letter-spacing:.055em; margin-bottom:5px; }
body.dark .debug-call-stage { color:#79caff; font-weight:800; margin-right:5px; }
body.dark .debug-call-col:first-child { position:relative; }
body.dark .debug-call-thumb { display:block; width:100%; height:auto; aspect-ratio:4/3; object-fit:contain; border:1px solid #35516d; border-radius:5px; background:#070d15; }
body.dark .debug-call-col:first-child > .debug-call-chips { position:absolute; right:5px; bottom:5px; justify-content:flex-end; }
body.dark .debug-call-col:first-child > .debug-call-chips .debug-call-chip { background:#1d324be8; border:1px solid #4a6d92; }
body.dark .debug-call-chips { display:flex; flex-wrap:wrap; gap:3px; }
body.dark .debug-call-chip { display:inline-block; max-width:100%; padding:2px 5px; border-radius:5px; background:#1d324b; color:#cfe1ff; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
body.dark .debug-call-prompt { color:#d9e5f2; line-height:1.35; word-break:break-word; }
body.dark .debug-call-result { color:#8ce9c1; line-height:1.35; word-break:break-word; }
body.dark .debug-call-meta { grid-column:1/-1; color:#7893ad; font-size:9px; padding-top:1px; }
body.dark .debug-call-empty { min-height:74px; display:grid; place-items:center; border:1px dashed #315d88; border-radius:8px; color:#7893ad; }
</style>"""
    script = r"""<script>
/* Same compact information model as Debug page M1/M2 cards. */
function darkCallCandidates(event) {
  const candidates = event?.context?.candidates || event?.payload?.candidates || [];
  return Array.isArray(candidates) ? candidates : [];
}
function darkCallObjectName(event) {
  const value = String(event?.m1_input_label || event?.context?.semantic_class || event?.target_kind || event?.object_id || '目标物体');
  const normalized = value.replace(/^physical_/, '').replace(/_[-\d.]+$/, '').replaceAll('_', ' ');
  const names = {door:'门', fridge:'冰箱', refrigerator:'冰箱', cabinet:'柜子', drawer:'抽屉', 'drawer cabinet':'抽屉柜'};
  return names[normalized.toLowerCase()] || normalized;
}
function darkCallQuestion(event, stage) {
  return stage === 'M1'
    ? '判断图中的' + darkCallObjectName(event) + '是否可交互，并识别其开合状态。'
    : '结合历史、候选目标和任务目标，选择下一个导航子目标。';
}
function darkCallResult(event) {
  if (event?.error) return '调用失败：' + clip(event.error, 90);
  let raw = event?.raw_text ?? event?.response?.raw_text ?? event?.payload?.result ?? '';
  if (typeof raw === 'object') raw = JSON.stringify(raw);
  try {
    const value = JSON.parse(raw);
    const choice = value.ranked_ids || value.candidate_id || value.label || value.interaction_class || value.coarse_state || value.state || '';
    return clip([Array.isArray(choice) ? choice.join(' → ') : choice, value.coarse_state !== choice ? value.coarse_state : '', value.reason, value.confidence != null ? '置信度 ' + value.confidence : ''].filter(Boolean).join(' · '), 110);
  } catch (_) {
    return clip(raw || '已完成（无文本结果）', 110);
  }
}
function darkCallChips(parent, values, maximum=4) {
  const wrap = document.createElement('div'); wrap.className = 'debug-call-chips';
  values.slice(0, maximum).forEach(value => { const chip=document.createElement('span'); chip.className='debug-call-chip'; chip.textContent=clip(value,22); wrap.append(chip); });
  if (values.length > maximum) { const chip=document.createElement('span'); chip.className='debug-call-chip'; chip.textContent='+'+(values.length-maximum); wrap.append(chip); }
  parent.append(wrap);
}
function darkM1Thumb(event, index) {
  const canvas=document.createElement('canvas'); canvas.className='debug-call-thumb'; canvas.width=320; canvas.height=240;
  const key=String(event?.m1_input_image_key || '');
  if (!key) return canvas;
  const image=new Image();
  image.onload=()=>{const context=canvas.getContext('2d'),scale=Math.min(canvas.width/image.width,canvas.height/image.height),drawWidth=image.width*scale,drawHeight=image.height*scale,offsetX=(canvas.width-drawWidth)/2,offsetY=(canvas.height-drawHeight)/2;context.fillStyle='#070d15';context.fillRect(0,0,canvas.width,canvas.height);context.drawImage(image,offsetX,offsetY,drawWidth,drawHeight);const box=event?.m1_input_bbox||[];if(box.length>=4){context.strokeStyle='#ffe04b';context.lineWidth=3;context.strokeRect(offsetX+Number(box[0])*scale,offsetY+Number(box[1])*scale,(Number(box[2])-Number(box[0]))*scale,(Number(box[3])-Number(box[1]))*scale)}};
  image.src='/api/m1-input-image?key='+encodeURIComponent(key)+'&v='+(event.timestamp||index);
  return canvas;
}
let darkCallSignature='';
renderCalls = function(state) {
  const events=[...(state?.mllm?.M1||[]),...(state?.mllm?.M2||[])].sort((a,b)=>{
    const timestampOrder=Number(b.timestamp||0)-Number(a.timestamp||0);
    if(timestampOrder!==0)return timestampOrder;
    return Number(b.request_sequence||0)-Number(a.request_sequence||0);
  }).slice(0,6);
  const signature=events.map(event=>[event.stage||event.module,event.timestamp,event.request_sequence,event.raw_text,event.error].join('|')).join('||');
  if(signature===darkCallSignature)return;
  darkCallSignature=signature;
  const container=$('calls'); container.replaceChildren();
  if (!events.length) { const empty=document.createElement('div'); empty.className='debug-call-empty'; empty.textContent='等待真实 M1 / M2 调用'; container.append(empty); return; }
  events.forEach((event,index)=>{
    const stage=String(event.stage||event.module||'MLLM').toUpperCase(),row=document.createElement('article');row.className='debug-call';
    const left=document.createElement('div'),middle=document.createElement('div'),right=document.createElement('div');[left,middle,right].forEach(element=>element.className='debug-call-col');
    const leftLabel=document.createElement('div');leftLabel.className='debug-call-label';leftLabel.textContent=stage==='M1'?'真实 M1 输入':'候选 subgoal';left.append(leftLabel);
    if(stage==='M1'){left.append(darkM1Thumb(event,index));darkCallChips(left,[darkCallObjectName(event)],1)}else{darkCallChips(left,darkCallCandidates(event).map(candidate=>candidate.id||candidate.candidate_id||candidate.target_name||'candidate'))}
    const middleLabel=document.createElement('div');middleLabel.className='debug-call-label';middleLabel.textContent=stage==='M2'?'历史 / 目标 / 简化请求':'简化问题';middle.append(middleLabel);
    if(stage==='M2'){const mission=event?.context?.mission||{},history=Number(event?.context?.recent_decision_count||0);darkCallChips(middle,['历史 '+history,'候选 '+darkCallCandidates(event).length,'目标 '+(mission.target_name||mission.target||mission.mode||'探索')],3)}
    const prompt=document.createElement('div');prompt.className='debug-call-prompt';prompt.textContent=darkCallQuestion(event,stage);middle.append(prompt);
    const resultLabel=document.createElement('div');resultLabel.className='debug-call-label';resultLabel.textContent='MLLM 输出';const output=document.createElement('div');output.className='debug-call-result';output.textContent=darkCallResult(event);right.append(resultLabel,output);
    const meta=document.createElement('div');meta.className='debug-call-meta';const at=event.timestamp?new Date(Number(event.timestamp)*1000).toLocaleTimeString():'--:--:--';meta.textContent=stage+' · '+at+' · '+(event.latency_s!=null?num(event.latency_s,2)+' s':'等待耗时')+' · '+(event.model||event.role||'');
    row.append(left,middle,right,meta);container.append(row);
  });
  container.scrollTop=0;
};
</script>"""
    html = html.replace("<div class='note'>M3 为规则验证，不调用模型</div>", "")
    return html.replace("</head>", css + "</head>").replace("</body>", script + "</body>") if "</head>" in html else html.replace("</style>", "</style>" + css, 1).replace("</body>", script + "</body>")


def _use_phone_stream_and_navigation_task(html: str) -> str:
    """Add the LAN phone camera selector and a concise active-task card."""
    canvas = "<canvas id='view1' width='480' height='270'></canvas>"
    stream_ui = """<div id='perception-stream-grid' class='perception-stream-grid'>""" + canvas + """<div id='phone-stage' class='phone-stage' hidden><img id='phone-feed' alt='手机实时视频'><div id='phone-waiting' class='phone-waiting'><img src='/phone-qr.png' alt='手机推流二维码'><strong>手机视频未连接</strong><span>扫码进入内网 HTTPS 推流页面</span><code>10.100.5.3:8767/phone-stream</code></div><span id='phone-live-badge' class='phone-live-badge'>等待连接</span></div></div><div class='perception-switch'><button id='source-phone'>＋ 手机视频</button></div>"""
    html = html.replace(canvas, stream_ui, 1)
    html = html.replace(
        "<div id='safe' class='safe'></div>",
        "<div id='nav-task' class='nav-task'><span class='nav-task-icon'>◎</span><span><small>当前导航任务</small><strong>交互导航探索</strong></span></div><div id='safe' class='safe'></div>",
        1,
    )
    html = html.replace(
        "<div class='top-tags'>",
        "<div class='top-tags'><button id='record-page' class='record-page' title='录制完整展示页面'><span>●</span><b>录制完整页面</b></button>",
        1,
    )
    css = r"""<style>
body.dark .visuals > .panel:first-child .panel-title small { margin-right:125px; }
body.dark .perception-switch { position:absolute; right:10px; top:6px; z-index:5; display:flex; padding:2px; border:1px solid #315d88; border-radius:7px; background:#071321e8; }
body.dark .perception-switch button { height:24px; padding:0 9px; border:0; border-radius:5px; background:transparent; color:#8da7c3; font-size:10px; cursor:pointer; }
body.dark .perception-switch button.active { background:#176b9c; color:#f1f9ff; }
body.dark .perception-switch button.connected:not(.active) { color:#64e88b; }
body.dark .perception-stream-grid { position:absolute; z-index:2; inset:40px 0 0; display:grid; grid-template-columns:minmax(0,1fr); align-items:stretch; justify-items:stretch; gap:2px; overflow:hidden; background:#03080e; }
body.dark .perception-stream-grid.dual { grid-template-columns:repeat(2,minmax(0,1fr)); }
body.dark .perception-stream-grid #view1 { width:100%; height:100%; min-width:0; min-height:0; object-fit:contain; justify-self:center; align-self:center; background:#000; }
body.dark .phone-stage { position:relative; z-index:3; min-width:0; min-height:0; background:#03080e; display:grid; place-items:center; overflow:hidden; border-left:1px solid #315d88; }
body.dark .phone-stage[hidden] { display:none; }
body.dark #phone-feed { display:block; width:100%; height:100%; object-fit:contain; background:#000; }
body.dark .phone-waiting { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:6px; color:#dcecff; background:radial-gradient(circle,#102b45,#03080e 66%); }
body.dark .phone-waiting[hidden] { display:none; }
body.dark .phone-waiting img { width:min(168px,28vh); aspect-ratio:1; padding:6px; border-radius:8px; background:#fff; }
body.dark .phone-waiting strong { color:#79caff; font-size:14px; }
body.dark .phone-waiting span, body.dark .phone-waiting code { color:#8da7c3; font-size:10px; }
body.dark .phone-live-badge { position:absolute; left:11px; top:10px; padding:5px 9px; border-radius:99px; background:#0a1725dc; color:#ffd46c; font-size:10px; }
body.dark .phone-live-badge.live { color:#65e99a; }
body.dark .nav-task { margin-top:7px; min-height:42px; display:flex; align-items:center; gap:9px; padding:7px 10px; border:1px solid #315d88; border-radius:8px; background:#0a1a2bcc; }
body.dark .nav-task-icon { width:27px; height:27px; display:grid; place-items:center; border:1px solid #58baff; border-radius:7px; color:#58baff; font-size:17px; }
body.dark .nav-task > span:last-child { min-width:0; display:flex; flex-direction:column; gap:2px; }
body.dark .nav-task small { color:#8da7c3; font-size:9px; }
body.dark .nav-task strong { color:#eef7ff; font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
body.dark .record-page { min-width:116px; height:34px; display:inline-flex; align-items:center; justify-content:center; gap:6px; padding:0 11px; border:1px solid #315d88; border-radius:7px; background:linear-gradient(180deg,#102b47,#0b1828); color:#dcecff; font:700 11px Inter,'Noto Sans SC',system-ui,sans-serif; cursor:pointer; box-shadow:inset 0 1px #ffffff0b; }
body.dark .record-page span { color:#ff6879; font-size:12px; }
body.dark .record-page:hover { border-color:#58baff; }
body.dark .record-page.recording { border-color:#ff6879; background:#45151f; color:#fff; }
body.dark .record-page.recording span { animation:recordPulse 1s infinite; }
@keyframes recordPulse { 50% { opacity:.25; } }
</style>"""
    script = r"""<script>
let phoneMode=false,phoneConnected=false,phoneAudioConnected=false,phoneFrameBusy=false,phoneFrameSeq=0,phoneFrameUrl='',phoneTargetFps=10,phoneFrameTimer=0,phoneAudioBusy=false,phoneAudioContext=null,phoneAudioAfter=0,phoneAudioNextAt=0,pageRecorder=null,pageRecordStream=null,pageRecordChunks=[],pageRecordStartedAt=0,pageRecordTimer=0;
const streamGrid=$('perception-stream-grid'),phoneButton=$('source-phone'),phoneStage=$('phone-stage'),phoneFeed=$('phone-feed'),phoneWaiting=$('phone-waiting'),phoneBadge=$('phone-live-badge');
const pageRecordButton=$('record-page');
function pageRecordingTypes(){return ['video/mp4;codecs=avc1.42E01E,mp4a.40.2','video/mp4','video/webm;codecs=vp9,opus','video/webm;codecs=vp8,opus','video/webm']}
function createPageRecorder(mediaStream){for(const mimeType of pageRecordingTypes()){if(!MediaRecorder.isTypeSupported(mimeType))continue;try{return new MediaRecorder(mediaStream,{mimeType,videoBitsPerSecond:3500000,audioBitsPerSecond:192000})}catch(_){}}return new MediaRecorder(mediaStream,{videoBitsPerSecond:3500000,audioBitsPerSecond:192000})}
function downloadPageBlob(blob,name){const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download=name;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(link.href),30000)}
async function savePageMp4(blob){const stamp=new Date().toISOString().replace(/[:.]/g,'-'),name='go2-showcase-dark-'+stamp;if(blob.type.includes('mp4')){downloadPageBlob(blob,name+'.mp4');return}pageRecordButton.querySelector('b').textContent='正在转换 MP4…';try{const response=await fetch('/api/recording-to-mp4',{method:'POST',headers:{'Content-Type':blob.type||'video/webm'},body:blob});if(!response.ok)throw new Error((await response.json()).error||('HTTP '+response.status));downloadPageBlob(await response.blob(),name+'.mp4')}catch(error){downloadPageBlob(blob,name+'.webm');alert('MP4 转换失败，已保存 WebM：'+error.message)}}
function pageRecordingClock(){const elapsed=Math.max(0,Math.floor((Date.now()-pageRecordStartedAt)/1000)),minutes=String(Math.floor(elapsed/60)).padStart(2,'0'),seconds=String(elapsed%60).padStart(2,'0');pageRecordButton.querySelector('b').textContent='停止并保存 '+minutes+':'+seconds}
function resetPageRecorder(){if(pageRecordTimer)clearInterval(pageRecordTimer);pageRecordTimer=0;if(pageRecordStream)pageRecordStream.getTracks().forEach(track=>track.stop());pageRecordStream=null;pageRecorder=null;pageRecordButton.classList.remove('recording');pageRecordButton.querySelector('b').textContent='录制完整页面'}
function stopPageRecording(){if(pageRecorder&&pageRecorder.state!=='inactive')pageRecorder.stop();else resetPageRecorder()}
async function startPageRecording(){
  if(!window.isSecureContext||!navigator.mediaDevices?.getDisplayMedia){const secure='https://'+location.hostname+':8767/showcase-dark';if(location.protocol!=='https:'){location.assign(secure);return}alert('当前浏览器不支持页面录制，请使用最新版 Chrome 或 Edge。');return}
  try{
    pageRecordStream=await navigator.mediaDevices.getDisplayMedia({video:{frameRate:{ideal:10,max:12},width:{ideal:1600,max:1920},height:{ideal:900,max:1080},displaySurface:'browser'},audio:{echoCancellation:false,noiseSuppression:false,autoGainControl:false,sampleRate:48000},preferCurrentTab:true,selfBrowserSurface:'include',systemAudio:'include',surfaceSwitching:'exclude'});
    const displayTrack=pageRecordStream.getVideoTracks()[0];try{await displayTrack.applyConstraints({frameRate:{ideal:10,max:12},width:{ideal:1600,max:1920},height:{ideal:900,max:1080}})}catch(_){}
    pageRecordChunks=[];pageRecorder=createPageRecorder(pageRecordStream);
    pageRecorder.ondataavailable=event=>{if(event.data?.size)pageRecordChunks.push(event.data)};
    pageRecorder.onerror=event=>{alert('页面录制失败：'+(event.error?.message||'未知错误'));resetPageRecorder()};
    pageRecorder.onstop=async()=>{const type=pageRecorder?.mimeType||'video/webm',blob=new Blob(pageRecordChunks,{type});pageRecordChunks=[];if(blob.size)await savePageMp4(blob);resetPageRecorder()};
    displayTrack.onended=()=>stopPageRecording();pageRecorder.start(4000);pageRecordStartedAt=Date.now();pageRecordButton.classList.add('recording');pageRecordingClock();pageRecordTimer=setInterval(pageRecordingClock,1000);
  }catch(error){resetPageRecorder();if(error.name!=='NotAllowedError')alert('无法开始页面录制：'+error.message)}
}
pageRecordButton.onclick=()=>pageRecorder&&pageRecorder.state==='recording'?stopPageRecording():startPageRecording();
async function startPhoneAudio(){const AudioContextClass=window.AudioContext||window.webkitAudioContext;if(!AudioContextClass)return;if(!phoneAudioContext){try{phoneAudioContext=new AudioContextClass({sampleRate:48000,latencyHint:'interactive'})}catch(_){phoneAudioContext=new AudioContextClass()}}await phoneAudioContext.resume();phoneAudioAfter=0;phoneAudioNextAt=phoneAudioContext.currentTime+.12}
async function stopPhoneAudio(){if(phoneAudioContext?.state==='running')await phoneAudioContext.suspend()}
async function refreshPhoneAudio(){if(document.hidden||!phoneMode||!phoneConnected||!phoneAudioConnected||phoneAudioBusy||!phoneAudioContext||phoneAudioContext.state!=='running')return;phoneAudioBusy=true;try{const response=await fetch('/phone-audio.pcm?after='+phoneAudioAfter+'&t='+Date.now(),{cache:'no-store'});if(response.status===204)return;if(!response.ok)throw new Error('audio '+response.status);const pcm=new Int16Array(await response.arrayBuffer()),rate=Number(response.headers.get('X-Audio-Rate'))||48000,sequence=Number(response.headers.get('X-Audio-Seq'))||phoneAudioAfter;if(!pcm.length)return;phoneAudioAfter=sequence;const buffer=phoneAudioContext.createBuffer(1,pcm.length,rate),output=buffer.getChannelData(0);for(let index=0;index<pcm.length;index++)output[index]=pcm[index]/32768;const source=phoneAudioContext.createBufferSource();source.buffer=buffer;source.connect(phoneAudioContext.destination);if(phoneAudioNextAt<phoneAudioContext.currentTime-.12)phoneAudioNextAt=phoneAudioContext.currentTime+.04;const startAt=Math.max(phoneAudioContext.currentTime+.025,phoneAudioNextAt);source.start(startAt);phoneAudioNextAt=startAt+buffer.duration}catch(_){}finally{phoneAudioBusy=false}}
function setPhonePanel(enabled){phoneMode=Boolean(enabled);streamGrid.classList.toggle('dual',phoneMode);phoneButton.classList.toggle('active',phoneMode);phoneStage.hidden=!phoneMode;phoneButton.textContent=phoneMode?'关闭手机音视频':phoneConnected?'● 手机音视频':'＋ 手机音视频';if(phoneMode){startPhoneAudio();refreshPhoneStatus()}else stopPhoneAudio()}
phoneButton.onclick=()=>setPhonePanel(!phoneMode);
function schedulePhoneFrames(fps){const normalized=Math.min(20,Math.max(5,Number(fps)||10));if(phoneFrameTimer&&normalized===phoneTargetFps)return;phoneTargetFps=normalized;if(phoneFrameTimer)clearInterval(phoneFrameTimer);phoneFrameTimer=setInterval(refreshPhoneFrame,Math.max(42,Math.round(800/phoneTargetFps)))}
async function refreshPhoneStatus(){try{const response=await fetch('/api/phone-status?t='+Date.now(),{cache:'no-store'}),status=await response.json();phoneConnected=Boolean(status.connected);phoneAudioConnected=Boolean(status.audio_connected);schedulePhoneFrames(status.target_fps);phoneButton.classList.toggle('connected',phoneConnected);phoneButton.textContent=phoneMode?'关闭手机音视频':phoneConnected?'● 手机音视频':'＋ 手机音视频';phoneWaiting.hidden=phoneConnected;phoneFeed.hidden=!phoneConnected;phoneBadge.classList.toggle('live',phoneConnected);phoneBadge.textContent=phoneConnected?(phoneAudioConnected?'手机音视频 · '+phoneTargetFps+' FPS':'手机视频 '+phoneTargetFps+' FPS · 等待音频'):'等待手机连接';if(phoneMode&&phoneConnected)refreshPhoneFrame()}catch(_){phoneConnected=false;phoneAudioConnected=false;phoneWaiting.hidden=false;phoneFeed.hidden=true;phoneBadge.textContent='连接检查失败'}}
async function refreshPhoneFrame(){if(document.hidden||!phoneMode||!phoneConnected||phoneFrameBusy)return;phoneFrameBusy=true;try{const response=await fetch('/phone-frame.jpg?after='+phoneFrameSeq+'&t='+Date.now(),{cache:'no-store'});if(response.status===204)return;if(!response.ok)throw new Error('frame '+response.status);phoneFrameSeq=Number(response.headers.get('X-Frame-Seq'))||phoneFrameSeq+1;const nextUrl=URL.createObjectURL(await response.blob()),previousUrl=phoneFrameUrl;await new Promise((resolve,reject)=>{phoneFeed.onload=resolve;phoneFeed.onerror=reject;phoneFeed.src=nextUrl});phoneFrameUrl=nextUrl;if(previousUrl)URL.revokeObjectURL(previousUrl)}catch(_){phoneConnected=false;phoneWaiting.hidden=false;phoneFeed.hidden=true}finally{phoneFrameBusy=false}}
function navigationTargetName(value){let target=String(value||'').trim();if(!target)return '';const names={door:'门',portal:'门',fridge:'冰箱',refrigerator:'冰箱',cabinet:'柜子',drawer:'抽屉'};const lower=target.toLowerCase();for(const [key,name] of Object.entries(names))if(lower.includes(key))return name+' · '+target;return target.replaceAll('_',' ')}
function renderNavigationTask(state){const navigation=state?.navigation||{},selection=navigation.selection||{},execution=navigation.execution_state||{},behavior=String(execution.behavior_type||selection.behavior_type||'').toUpperCase(),rawTarget=selection.target_name||selection.target_id||execution.target_name||execution.target_id||'',isExplore=behavior==='EXPLORE'||String(selection.candidate_id||'').startsWith('frontier:'),target=navigationTargetName(rawTarget),label=isExplore||!target?'交互导航探索':behavior==='INTERACT'?'交互目标：'+target:'导航目标：'+target,box=$('nav-task');if(box)box.querySelector('strong').textContent=label}
const baseRenderMetrics=renderMetrics;
renderMetrics=function(state){baseRenderMetrics(state);renderNavigationTask(state)};
schedulePhoneFrames(10);setInterval(refreshPhoneStatus,1000);setInterval(refreshPhoneAudio,100);refreshPhoneStatus();
</script>"""
    return html.replace("</style>", "</style>" + css, 1).replace("</body>", script + "</body>")


DARK_SHOWCASE_HTML = _use_original_renderer_panels(DARK_SHOWCASE_HTML, "view3", "view6")
LIGHT_SHOWCASE_HTML = _use_original_renderer_panels(LIGHT_SHOWCASE_HTML, "view3", "view6")
ACADEMIC_SHOWCASE_HTML = _use_original_renderer_panels(ACADEMIC_SHOWCASE_HTML, "b", "c")
DARK_SHOWCASE_HTML = _use_debug_style_mllm_cards(DARK_SHOWCASE_HTML)
DARK_SHOWCASE_HTML = _use_phone_stream_and_navigation_task(DARK_SHOWCASE_HTML)
# The presentation view is intentionally box-only.  Debug keeps the separate
# /camera-overlay.jpg endpoint with segmentation fill for diagnosis.
DARK_SHOWCASE_HTML = DARK_SHOWCASE_HTML.replace(
    "fetch('/camera-overlay.jpg?t='",
    "fetch('/camera-box-overlay.jpg?t='",
)
