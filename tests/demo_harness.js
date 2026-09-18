/* Node 검증 하네스: tests/test_demo_js.py 가 호출한다.
 * 사용: node tests/demo_harness.js <model.js> <input.json>
 * input: {"cases": [{"obs": (N,T,2)}...], "risk_case": {"samples": (K,N,T,2), "types": [..], "d_safe": 1.0},
 *         "policy_case": {"frames": [[{"wid","vid","risk"}...]...], "opts": {...}}}
 * 출력(stdout JSON): {"params": [(T,N,5)...], "mean": [(N,T,2)...], "risk": {...}, "alerts": [...]}
 */
const fs = require("fs");
const path = require("path");
const F = require(path.resolve(__dirname, "..", "demo", "foresight.js"));

function loadWindowJs(file, name) {
  const text = fs.readFileSync(file, "utf-8").trim();
  const prefix = "window." + name + " = ";
  if (!text.startsWith(prefix)) throw new Error("unexpected prefix in " + file);
  return JSON.parse(text.slice(prefix.length).replace(/;\s*$/, ""));
}

const model = loadWindowJs(process.argv[2], "FORESIGHT_MODEL");
const input = JSON.parse(fs.readFileSync(process.argv[3], "utf-8"));
const out = { params: [], mean: [], risk: null, alerts: [] };
for (const c of input.cases || []) {
  const r = F.predict(model, c.obs, 0, 0);
  out.params.push(r.params);
  out.mean.push(r.mean);
}
if (input.risk_case) {
  const rc = input.risk_case;
  out.risk = F.pairwiseRisk(rc.samples, rc.types, rc.d_safe, F.STEP_SECONDS);
  out.risk.ttc = out.risk.ttc.map((row) => row.map((v) => (Number.isNaN(v) ? null : v)));
}
if (input.policy_case) {
  const pc = input.policy_case;
  const pol = new F.AlertPolicy(pc.opts || {});
  pc.frames.forEach((obs, i) => {
    for (const a of pol.update(obs, i * F.STEP_SECONDS)) out.alerts.push([a.wid, a.vid, a.tsS, a.consecutive]);
  });
  out.n_suppressed = pol.nSuppressed;
}
process.stdout.write(JSON.stringify(out));
