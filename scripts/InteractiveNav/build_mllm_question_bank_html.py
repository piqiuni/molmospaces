#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
from pathlib import Path


def image_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def render_case(case: dict, index: int) -> str:
    images = []
    for image_path in case.get("image_paths") or []:
        path = Path(image_path).expanduser()
        if path.is_file():
            images.append(
                '<figure><img src="{}" alt="{}"><figcaption>{}</figcaption></figure>'.format(
                    image_data_uri(path),
                    html.escape(path.name),
                    html.escape(path.name),
                )
            )
        else:
            images.append(
                '<div class="missing">Missing image: {}</div>'.format(
                    html.escape(str(path))
                )
            )
    image_html = "".join(images) or '<div class="no-image">Text-only case</div>'
    context = html.escape(json.dumps(case.get("context") or {}, ensure_ascii=False, indent=2))
    expected = html.escape(json.dumps(case.get("expected") or {}, ensure_ascii=False, indent=2))
    instruction = html.escape(str(case.get("instruction") or ""))
    module = html.escape(str(case.get("module") or ""))
    role = html.escape(str(case.get("role") or ""))
    case_id = html.escape(str(case.get("id") or f"case_{index}"))
    return f'''<article class="case" data-module="{module}" data-role="{role}">
  <header><span class="number">#{index}</span><h2>{case_id}</h2><span class="tag module">{module}</span><span class="tag role">{role}</span></header>
  <p class="instruction">{instruction}</p>
  <div class="images">{image_html}</div>
  <div class="columns">
    <section><h3>Input Context</h3><pre>{context}</pre></section>
    <section><h3>Expected GT</h3><pre class="expected">{expected}</pre></section>
  </div>
</article>'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a self-contained HTML MLLM question-bank viewer.")
    parser.add_argument("--question-bank", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bank_path = Path(args.question_bank).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    cases = bank.get("cases") or []
    cards = "\n".join(render_case(case, index) for index, case in enumerate(cases, start=1))
    modules = sorted({str(case.get("module") or "") for case in cases})
    roles = sorted({str(case.get("role") or "") for case in cases})
    module_options = '<option value="">All modules</option>' + "".join(
        f'<option value="{html.escape(value)}">{html.escape(value)}</option>' for value in modules
    )
    role_options = '<option value="">All roles</option>' + "".join(
        f'<option value="{html.escape(value)}">{html.escape(value)}</option>' for value in roles
    )
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MLLM Interactive Navigation Question Bank</title>
<style>
:root {{ color-scheme: dark; --bg:#10141b; --card:#1a212c; --muted:#9aa8ba; --line:#2c3848; --accent:#62b0ff; --gt:#b5efc4; }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family:Inter,system-ui,sans-serif; background:var(--bg); color:#eef4fb; }}
.top {{ position:sticky; top:0; z-index:5; padding:20px 4vw 16px; background:rgba(16,20,27,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(10px); }}
h1 {{ margin:0 0 6px; font-size:25px; }} .summary {{ color:var(--muted); margin-bottom:14px; }}
.filters {{ display:flex; gap:10px; flex-wrap:wrap; }} select {{ color:#eef4fb; background:#202a38; border:1px solid var(--line); padding:8px 10px; border-radius:7px; }}
main {{ max-width:1450px; margin:24px auto; padding:0 4vw 50px; }} .case {{ background:var(--card); border:1px solid var(--line); border-radius:12px; margin:0 0 24px; overflow:hidden; }}
.case header {{ display:flex; align-items:center; gap:10px; padding:14px 18px; border-bottom:1px solid var(--line); }} h2 {{ margin:0; font-size:19px; flex:1; }} .number {{ color:var(--muted); }}
.tag {{ font-size:12px; padding:4px 8px; border-radius:999px; }} .module {{ background:#264967; color:#c5e6ff; }} .role {{ background:#4d3d63; color:#ead8ff; }}
.instruction {{ margin:16px 18px; color:#d5deea; }} .images {{ display:flex; gap:14px; flex-wrap:wrap; padding:0 18px 18px; }} figure {{ margin:0; max-width:48%; }} img {{ display:block; max-width:100%; max-height:430px; object-fit:contain; border-radius:8px; background:#080b0f; border:1px solid #354456; }} figcaption {{ color:var(--muted); font-size:12px; margin-top:5px; }}
.no-image,.missing {{ color:var(--muted); padding:20px; border:1px dashed var(--line); border-radius:8px; }} .missing {{ color:#ffadad; }} .columns {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; padding:0 18px 18px; }} section {{ min-width:0; }} h3 {{ font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; }} pre {{ margin:0; padding:12px; overflow:auto; background:#111720; border:1px solid var(--line); border-radius:8px; font-size:12px; line-height:1.45; }} pre.expected {{ color:var(--gt); }}
@media(max-width:800px) {{ .columns {{ grid-template-columns:1fr; }} figure {{ max-width:100%; }} }}
</style></head><body>
<div class="top"><h1>MLLM Interactive Navigation Question Bank</h1><div class="summary">{len(cases)} cases · modules: {html.escape(', '.join(modules))} · roles: {html.escape(', '.join(roles))}</div>
<div class="filters"><select id="module">{module_options}</select><select id="role">{role_options}</select></div></div>
<main id="cases">{cards}</main>
<script>
const cards=[...document.querySelectorAll('.case')];
function filter() {{ const module=document.querySelector('#module').value, role=document.querySelector('#role').value; cards.forEach(card=>{{card.style.display=(!module||card.dataset.module===module)&&(!role||card.dataset.role===role)?'block':'none';}}); }}
document.querySelector('#module').addEventListener('change',filter); document.querySelector('#role').addEventListener('change',filter);
</script></body></html>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
