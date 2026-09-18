/* foresight.js — Social-STGCNN 추론 + 충돌 위험 점수 + 경보 정책을 순수 JavaScript 로 다시 구현한 것.
 *
 * src/foresight 의 PyTorch 구현과 같은 수식을 쓴다 (전처리 float32 규칙, view 축 교환, BatchNorm eval,
 * 이변량 가우시안 촐레스키 샘플링, 같은 샘플 인덱스끼리 짝짓는 위험 추정, 연속 N 프레임 + 히스테리시스 + 쿨다운 경보).
 * tests/test_demo_js.py 가 Node 로 이 파일을 실행해 PyTorch 결과와 수치 비교한다. 브라우저와 Node 양쪽에서 동작.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Foresight = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  const f32 = Math.fround;
  const STEP_SECONDS = 0.4;

  /** (N, T, [x, y]) 절대좌표 → V (2, T, N) float32 상대 변위, A (T, N, N) 정규화 라플라시안 (float64). */
  function preprocess(obs, kernel) {
    kernel = kernel || "velocity";
    const N = obs.length;
    const T = obs[0].length;
    const rel = new Array(N);
    for (let i = 0; i < N; i++) {
      rel[i] = new Array(T);
      rel[i][0] = [0, 0];
      for (let t = 1; t < T; t++) {
        rel[i][t] = [obs[i][t][0] - obs[i][t - 1][0], obs[i][t][1] - obs[i][t - 1][1]];
      }
    }
    const V = new Float32Array(2 * T * N);
    for (let t = 0; t < T; t++) {
      for (let n = 0; n < N; n++) {
        V[t * N + n] = rel[n][t][0];
        V[T * N + t * N + n] = rel[n][t][1];
      }
    }
    const feat = kernel === "velocity" ? rel : obs;
    const A = new Float64Array(T * N * N);
    const a = new Float64Array(N * N);
    const dinv = new Float64Array(N);
    for (let t = 0; t < T; t++) {
      // 역거리 커널: 공식 코드처럼 float32 로 뺀 뒤 제곱합, 제곱근은 float64. 거리 0 → 0, 대각 1.
      for (let i = 0; i < N; i++) {
        const xi = f32(feat[i][t][0]);
        const yi = f32(feat[i][t][1]);
        for (let j = 0; j < N; j++) {
          if (i === j) {
            a[i * N + j] = 1;
            continue;
          }
          const dx = f32(xi - f32(feat[j][t][0]));
          const dy = f32(yi - f32(feat[j][t][1]));
          const sq = f32(f32(dx * dx) + f32(dy * dy));
          const d = Math.sqrt(sq);
          a[i * N + j] = d > 0 ? 1 / d : 0;
        }
      }
      for (let i = 0; i < N; i++) {
        let s = 0;
        for (let j = 0; j < N; j++) s += a[i * N + j];
        dinv[i] = s > 0 ? 1 / Math.sqrt(s) : 0;
      }
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          A[t * N * N + i * N + j] = (i === j ? 1 : 0) - dinv[i] * a[i * N + j] * dinv[j];
        }
      }
    }
    return { V: V, A: A, N: N, T: T };
  }

  function bnApply(x, C, TN, bn) {
    for (let c = 0; c < C; c++) {
      const s = bn.w[c] / Math.sqrt(bn.v[c] + bn.eps);
      const o = bn.b[c] - bn.m[c] * s;
      for (let i = 0; i < TN; i++) x[c * TN + i] = x[c * TN + i] * s + o;
    }
  }

  function prelu(x, a) {
    for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] *= a;
  }

  /** 3x3 Conv2d, padding 1, 입력 (Ci, H, W) 평면 배열 → (Co, H, W). */
  function conv3x3(x, Ci, H, W, layer) {
    const Co = layer.w.length;
    const out = new Float64Array(Co * H * W);
    for (let o = 0; o < Co; o++) {
      const wo = layer.w[o];
      for (let hh = 0; hh < H; hh++) {
        for (let ww = 0; ww < W; ww++) {
          let s = layer.b[o];
          for (let i = 0; i < Ci; i++) {
            const wi = wo[i];
            for (let dh = 0; dh < 3; dh++) {
              const h2 = hh + dh - 1;
              if (h2 < 0 || h2 >= H) continue;
              const row = wi[dh];
              const base = i * H * W + h2 * W;
              for (let dw = 0; dw < 3; dw++) {
                const w2 = ww + dw - 1;
                if (w2 < 0 || w2 >= W) continue;
                s += row[dw] * x[base + w2];
              }
            }
          }
          out[o * H * W + hh * W + ww] = s;
        }
      }
    }
    return out;
  }

  /** 순전파: V (2, T, N), A (T, N, N) → params (T_pred, N, 5) = (μx, μy, log σx, log σy, atanh ρ). */
  function forward(model, V, A, N) {
    const m = model.meta;
    const T = m.obs_len;
    const Tp = m.pred_len;
    const Cin = m.in_channels;
    const C = m.out_channels;
    const st = model.st_gcn;
    const TN = T * N;
    // GraphConv: 1x1 conv (Cin→C) 뒤 프레임별 A_t: z[c,t,w] = Σ_v y[c,t,v] A[t,v,w]
    const y = new Float64Array(C * TN);
    for (let c = 0; c < C; c++) {
      for (let i2 = 0; i2 < TN; i2++) {
        let s = st.gcn_b[c];
        for (let i = 0; i < Cin; i++) s += st.gcn_w[c][i] * V[i * TN + i2];
        y[c * TN + i2] = s;
      }
    }
    const z = new Float64Array(C * TN);
    for (let c = 0; c < C; c++) {
      for (let t = 0; t < T; t++) {
        for (let w = 0; w < N; w++) {
          let s = 0;
          for (let v = 0; v < N; v++) s += y[c * TN + t * N + v] * A[t * N * N + v * N + w];
          z[c * TN + t * N + w] = s;
        }
      }
    }
    // TCN: BN → PReLU → Conv(kt×1, 시간축 패딩) → BN, + 잔차(1x1 conv + BN), PReLU
    bnApply(z, C, TN, st.bn1);
    prelu(z, st.prelu1);
    const kt = m.t_kernel;
    const pad = (kt - 1) >> 1;
    const u = new Float64Array(C * TN);
    for (let o = 0; o < C; o++) {
      for (let t = 0; t < T; t++) {
        for (let n = 0; n < N; n++) {
          let s = st.tcn_b[o];
          for (let i = 0; i < C; i++) {
            for (let k = 0; k < kt; k++) {
              const tt = t + k - pad;
              if (tt >= 0 && tt < T) s += st.tcn_w[o][i][k] * z[i * TN + tt * N + n];
            }
          }
          u[o * TN + t * N + n] = s;
        }
      }
    }
    bnApply(u, C, TN, st.bn2);
    if (st.res_kind === "conv") {
      const r = new Float64Array(C * TN);
      for (let o = 0; o < C; o++) {
        for (let i2 = 0; i2 < TN; i2++) {
          let s = st.res_b[o];
          for (let i = 0; i < Cin; i++) s += st.res_w[o][i] * V[i * TN + i2];
          r[o * TN + i2] = s;
        }
      }
      bnApply(r, C, TN, st.res_bn);
      for (let i = 0; i < u.length; i++) u[i] += r[i];
    } else if (st.res_kind === "identity") {
      for (let i = 0; i < u.length; i++) u[i] += V[i];
    }
    prelu(u, st.prelu_out);
    // (1, C, T, N) --view--> (1, T, C, N): 메모리 재해석. 같은 평면 배열을 [채널 T][높이 C][너비 N] 로 읽는다.
    let v = conv3x3(u, T, C, N, model.tpcnns[0]);
    prelu(v, model.prelus[0]);
    const nTx = model.tpcnns.length;
    for (let k = 1; k < nTx - 1; k++) {
      const w2 = conv3x3(v, Tp, C, N, model.tpcnns[k]);
      prelu(w2, model.prelus[k]);
      for (let i = 0; i < v.length; i++) v[i] += w2[i];
    }
    v = conv3x3(v, Tp, C, N, model.tpcnn_output);
    // (1, Tp, C, N) --view--> (1, C, Tp, N): params[t][n][c] = flat[c·Tp·N + t·N + n]
    const params = new Array(Tp);
    for (let t = 0; t < Tp; t++) {
      params[t] = new Array(N);
      for (let n = 0; n < N; n++) {
        const p = new Array(5);
        for (let c = 0; c < 5; c++) p[c] = v[c * Tp * N + t * N + n];
        params[t][n] = p;
      }
    }
    return params;
  }

  /** 결정적 시드 PRNG (mulberry32) + Box–Muller 표준정규. */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussPair(rng) {
    let u = 0;
    while (u === 0) u = rng();
    const v = rng();
    const r = Math.sqrt(-2 * Math.log(u));
    return [r * Math.cos(2 * Math.PI * v), r * Math.sin(2 * Math.PI * v)];
  }

  /** 평균 궤적: params (T, N, 5) + 마지막 관측 (N, [x, y]) → (N, T, [x, y]) (상대 변위 누적합). */
  function meanAbs(params, last) {
    const T = params.length;
    const N = last.length;
    const out = new Array(N);
    for (let n = 0; n < N; n++) {
      let x = last[n][0];
      let y = last[n][1];
      const tr = new Array(T);
      for (let t = 0; t < T; t++) {
        x += params[t][n][0];
        y += params[t][n][1];
        tr[t] = [x, y];
      }
      out[n] = tr;
    }
    return out;
  }

  /** K 개 샘플 (K, N, T, [x, y]). 공분산 촐레스키 L = [[σx, 0], [ρσy, σy√(1-ρ²)]] — 같은 인덱스 k 가 "하나의 미래". */
  function sampleAbs(params, last, K, rng) {
    const T = params.length;
    const N = last.length;
    const out = new Array(K);
    for (let k = 0; k < K; k++) {
      const per = new Array(N);
      for (let n = 0; n < N; n++) {
        let x = last[n][0];
        let y = last[n][1];
        const tr = new Array(T);
        for (let t = 0; t < T; t++) {
          const p = params[t][n];
          const sx = Math.exp(p[2]);
          const sy = Math.exp(p[3]);
          const rho = Math.tanh(p[4]);
          const g = gaussPair(rng);
          x += p[0] + sx * g[0];
          y += p[1] + sy * (rho * g[0] + Math.sqrt(Math.max(0, 1 - rho * rho)) * g[1]);
          tr[t] = [x, y];
        }
        per[n] = tr;
      }
      out[k] = per;
    }
    return out;
  }

  /** 작업자×차량 위험: samples (K, N, T, 2), types (N; 0 작업자, 1 차량). risk = P(min_t 거리 < dSafe). */
  function pairwiseRisk(samples, types, dSafe, dt) {
    dt = dt || STEP_SECONDS;
    const K = samples.length;
    const N = types.length;
    const T = K ? samples[0][0].length : 0;
    const W = [];
    const Vv = [];
    for (let n = 0; n < N; n++) (types[n] === 0 ? W : Vv).push(n);
    const risk = [];
    const ttc = [];
    const minDist = [];
    for (let wi = 0; wi < W.length; wi++) {
      const w = W[wi];
      const rr = [];
      const tt = [];
      const mm = [];
      for (let vi = 0; vi < Vv.length; vi++) {
        const v = Vv[vi];
        let hits = 0;
        let ttcSum = 0;
        let mdSum = 0;
        for (let k = 0; k < K; k++) {
          const sw = samples[k][w];
          const sv = samples[k][v];
          let md = Infinity;
          let first = 0;
          for (let t = 0; t < T; t++) {
            const dx = sw[t][0] - sv[t][0];
            const dy = sw[t][1] - sv[t][1];
            const d = Math.sqrt(dx * dx + dy * dy);
            if (d < md) md = d;
            if (!first && d < dSafe) first = t + 1;
          }
          if (first) {
            hits++;
            ttcSum += first * dt;
          }
          mdSum += md;
        }
        rr.push(K ? hits / K : 0);
        tt.push(hits ? ttcSum / hits : NaN);
        mm.push(K ? mdSum / K : Infinity);
      }
      risk.push(rr);
      ttc.push(tt);
      minDist.push(mm);
    }
    return { workers: W, vehicles: Vv, risk: risk, ttc: ttc, minDist: minDist, k: K };
  }

  /** 쌍 단위 상태를 가진 경보 정책: 연속 N 프레임 + 히스테리시스 + 쿨다운 (src/foresight/serving/risk.py 와 동일). */
  class AlertPolicy {
    constructor(opts) {
      opts = opts || {};
      this.threshold = opts.threshold == null ? 0.3 : opts.threshold;
      this.cooldownS = opts.cooldownS == null ? 5.0 : opts.cooldownS;
      this.minConsecutive = opts.minConsecutive == null ? 2 : opts.minConsecutive;
      this.clearThreshold = opts.clearThreshold == null ? this.threshold / 2 : opts.clearThreshold;
      this.staleAfterS = opts.staleAfterS == null ? 60 : opts.staleAfterS;
      this.state = new Map();
      this.nAlerts = 0;
      this.nSuppressed = 0;
    }
    /** observations: [{wid, vid, risk, ttc, minDist}] → 새 경보 목록. 관측에 없는 쌍은 건드리지 않는다. */
    update(observations, nowS) {
      const alerts = [];
      for (const o of observations) {
        const key = o.wid + "|" + o.vid;
        let st = this.state.get(key);
        if (!st) {
          st = { consecutive: 0, lastAlert: -Infinity, lastSeen: -Infinity };
          this.state.set(key, st);
        }
        st.lastSeen = nowS;
        if (o.risk >= this.threshold) st.consecutive += 1;
        else if (o.risk < this.clearThreshold) st.consecutive = 0;
        if (st.consecutive >= this.minConsecutive && o.risk >= this.threshold) {
          if (nowS - st.lastAlert >= this.cooldownS) {
            st.lastAlert = nowS;
            this.nAlerts += 1;
            alerts.push(Object.assign({}, o, { tsS: nowS, consecutive: st.consecutive }));
          } else {
            this.nSuppressed += 1;
          }
        }
      }
      return alerts;
    }
    gc(nowS) {
      let n = 0;
      for (const [key, st] of this.state) {
        if (nowS - st.lastSeen > this.staleAfterS) {
          this.state.delete(key);
          n++;
        }
      }
      return n;
    }
  }

  /** 편의 함수: obs (N, T_obs, 2) → {params, mean, samples}. K=0 이면 샘플 없음. */
  function predict(model, obs, K, seed) {
    const pre = preprocess(obs);
    const params = forward(model, pre.V, pre.A, pre.N);
    const last = obs.map(function (tr) {
      return [f32(tr[tr.length - 1][0]), f32(tr[tr.length - 1][1])];
    });
    const mean = meanAbs(params, last);
    const samples = K > 0 ? sampleAbs(params, last, K, mulberry32(seed == null ? 0 : seed)) : null;
    return { params: params, mean: mean, samples: samples };
  }

  return {
    STEP_SECONDS: STEP_SECONDS,
    preprocess: preprocess,
    forward: forward,
    meanAbs: meanAbs,
    sampleAbs: sampleAbs,
    pairwiseRisk: pairwiseRisk,
    AlertPolicy: AlertPolicy,
    predict: predict,
    mulberry32: mulberry32,
  };
});
