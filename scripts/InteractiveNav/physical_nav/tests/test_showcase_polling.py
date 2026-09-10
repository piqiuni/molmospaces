"""Execute presentation polling in JavaScript without ROS or a browser server."""

import json
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from showcase_pages import (
    ACADEMIC_SHOWCASE_HTML, DARK_SHOWCASE_HTML, LIGHT_SHOWCASE_HTML,
    _use_original_renderer_panels,
)


def test_native_polling_retries_failed_decode_without_restarting_legacy_timers():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute browser polling")
    legacy = """<html><body><script>
    function video(){fetch('/snapshot.jpg');}
    function refreshVisualization(){fetch('/api/visualization-data');}
    function academicVisualization(){fetch('/api/visualization-data');}
    setInterval(video,200);video();
    setInterval(refreshVisualization,1000);refreshVisualization();
    setInterval(academicVisualization,1000);academicVisualization();
    </script></body></html>"""
    scripts = re.findall(r"<script>(.*?)</script>",
                         _use_original_renderer_panels(legacy, "spatial", "graph"), re.S)
    page_scripts = [re.findall(r"<script>(.*?)</script>", html, re.S)
                    for html in (DARK_SHOWCASE_HTML, LIGHT_SHOWCASE_HTML, ACADEMIC_SHOWCASE_HTML)]
    harness = r"""
const vm = require('node:vm'), assert = require('node:assert/strict');
const {scripts, pageScripts} = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
// Syntax-check every generated page script, including template rewrites.
for (const page of pageScripts) for (const script of page) new vm.Script(script);
const timers=[], requests=[], draws=[], closed=[];
let round=0;
const context = vm.createContext({URLSearchParams, Date,
  document:{hidden:false, getElementById:id=>({width:640,height:360,
    getContext:()=>({drawImage:image=>draws.push([id,image.rev])})})},
  setInterval:fn=>timers.push(fn),
  fetch:async url=>{
    const parsed=new URL(url,'http://localhost');
    const index=parsed.pathname.includes('panel3')?0:1;
    requests.push([parsed.pathname,parsed.searchParams.get('after')]);
    const rev=round<2?(index===0?10:11):(index===0?12:13);
    const status=Number(parsed.searchParams.get('after'))===rev?204:200;
    return {status,ok:true,headers:{get:()=>String(rev)},blob:async()=>{
      assert.equal(status,200,'204 responses must not be decoded');
      return {rev,fail:round===0&&index===1};
    }};
  },
  createImageBitmap:async blob=>{
    if(blob.fail)throw Error('simulated decode failure');
    return {width:640,height:360,rev:blob.rev,close:()=>closed.push(blob.rev)};
  }
});
(async()=>{
  for(const script of scripts)vm.runInContext(script,context);
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(timers.length,1,'only native panel polling should be registered');
  assert.deepEqual(draws,[['spatial',10]]);
  round=1;await timers[0]();
  assert.deepEqual(requests.slice(-2),[
    ['/original-panel3.jpg','10'],['/original-panel6.jpg',null]]);
  assert.deepEqual(draws,[['spatial',10],['graph',11]]);
  round=2;await timers[0]();
  assert.deepEqual(requests.slice(-2),[
    ['/original-panel3.jpg','10'],['/original-panel6.jpg','11']]);
  round=3;await timers[0]();
  assert.deepEqual(requests.slice(-2),[
    ['/original-panel3.jpg','12'],['/original-panel6.jpg','13']]);
  assert.deepEqual(closed,[10,11,12,13]);
  assert.equal(draws.length,4);
  assert(requests.every(([path])=>path.startsWith('/original-panel')));
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run([node, "-e", harness],
                            input=json.dumps({"scripts": scripts, "pageScripts": page_scripts}),
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
