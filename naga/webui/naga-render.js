/* Naga 自研消息渲染器（零依赖，不用任何开源库）。
 *
 * 覆盖：Markdown（标题/强调/行内码/代码块/列表/引用/分割线/链接/图片/表格）、
 *       LaTeX 子集（$..$ / $$..$$：上下标、\frac、\sqrt、希腊字母、常见符号）、
 *       工具调用折叠块（🔧 调用 / ↩ 结果 → 默认收起的小字 <details>）。
 *
 * 安全：全程用 DOM API 构建节点、文本走 textContent（绝不 innerHTML），
 *       URL 走协议白名单——从根上杜绝 XSS（渲染的是模型输出，必须防注入）。
 *
 * 入口：NagaRender.render(targetEl, rawText) —— 清空 targetEl 并渲染。
 */
(function () {
  const GREEK = {
    alpha:'α',beta:'β',gamma:'γ',delta:'δ',epsilon:'ε',zeta:'ζ',eta:'η',theta:'θ',
    iota:'ι',kappa:'κ',lambda:'λ',mu:'μ',nu:'ν',xi:'ξ',pi:'π',rho:'ρ',sigma:'σ',
    tau:'τ',phi:'φ',chi:'χ',psi:'ψ',omega:'ω',Gamma:'Γ',Delta:'Δ',Theta:'Θ',
    Lambda:'Λ',Xi:'Ξ',Pi:'Π',Sigma:'Σ',Phi:'Φ',Psi:'Ψ',Omega:'Ω',
  };
  const SYM = {
    times:'×',cdot:'·',div:'÷',pm:'±',mp:'∓',leq:'≤',geq:'≥',neq:'≠',approx:'≈',
    equiv:'≡',sum:'∑',prod:'∏',int:'∫',infty:'∞',partial:'∂',nabla:'∇',
    rightarrow:'→',leftarrow:'←',Rightarrow:'⇒',Leftrightarrow:'⇔',to:'→',
    in:'∈',notin:'∉',subset:'⊂',supset:'⊃',cup:'∪',cap:'∩',forall:'∀',exists:'∃',
    langle:'⟨',rangle:'⟩',ldots:'…',cdots:'⋯',prime:'′',circ:'∘',star:'⋆',
  };

  function safeUrl(u, imageOnly) {
    u = (u || '').trim();
    if (imageOnly) return /^(https?:\/\/|data:image\/)/i.test(u) ? u : '';
    return /^(https?:\/\/|mailto:|\/)/i.test(u) ? u : '';
  }

  // ---------- LaTeX 子集 → DOM ----------
  function latexToNode(src) {
    const span = document.createElement('span');
    span.className = 'math';
    let i = 0;
    function emit(parent, text) { if (text) parent.appendChild(document.createTextNode(text)); }
    function parseGroup(stop) {
      const frag = document.createDocumentFragment();
      while (i < src.length) {
        const ch = src[i];
        if (stop && ch === stop) { i++; break; }
        if (ch === '{') { i++; frag.appendChild(parseGroup('}')); continue; }
        if (ch === '}') { i++; break; }
        if (ch === '^' || ch === '_') {
          i++;
          const el = document.createElement(ch === '^' ? 'sup' : 'sub');
          el.appendChild(nextAtom());
          frag.appendChild(el);
          continue;
        }
        if (ch === '\\') { frag.appendChild(parseCommand()); continue; }
        emit(frag, ch); i++;
      }
      return frag;
    }
    function nextAtom() {
      if (src[i] === '{') { i++; return parseGroup('}'); }
      if (src[i] === '\\') return parseCommand();
      const t = document.createTextNode(src[i] || ''); i++; return t;
    }
    function parseCommand() {
      i++;
      let name = '';
      while (i < src.length && /[a-zA-Z]/.test(src[i])) { name += src[i++]; }
      if (!name) { const t = document.createTextNode(src[i] || ''); i++; return t; }
      if (name === 'frac') {
        const num = document.createElement('span'); num.className = 'frac-n';
        skipWs(); if (src[i] === '{') { i++; num.appendChild(parseGroup('}')); } else num.appendChild(nextAtom());
        const den = document.createElement('span'); den.className = 'frac-d';
        skipWs(); if (src[i] === '{') { i++; den.appendChild(parseGroup('}')); } else den.appendChild(nextAtom());
        const f = document.createElement('span'); f.className = 'frac';
        f.appendChild(num); f.appendChild(den); return f;
      }
      if (name === 'sqrt') {
        const wrap = document.createElement('span'); wrap.className = 'sqrt';
        emit(wrap, '√');
        const rad = document.createElement('span'); rad.className = 'sqrt-r';
        skipWs(); if (src[i] === '{') { i++; rad.appendChild(parseGroup('}')); } else rad.appendChild(nextAtom());
        wrap.appendChild(rad); return wrap;
      }
      if (GREEK[name]) return document.createTextNode(GREEK[name]);
      if (SYM[name]) return document.createTextNode(SYM[name]);
      return document.createTextNode(name);
    }
    function skipWs() { while (src[i] === ' ') i++; }
    span.appendChild(parseGroup(null));
    return span;
  }

  // ---------- 行内 Markdown（+ $..$ 数学）→ 追加到 parent ----------
  function renderInline(parent, text) {
    let i = 0, buf = '';
    const flush = () => { if (buf) { parent.appendChild(document.createTextNode(buf)); buf = ''; } };
    while (i < text.length) {
      const c = text[i];
      if (c === '\\' && i + 1 < text.length) { buf += text[i + 1]; i += 2; continue; }
      if (c === '$') {
        const end = text.indexOf('$', i + 1);
        if (end > i) { flush(); parent.appendChild(latexToNode(text.slice(i + 1, end))); i = end + 1; continue; }
      }
      if (c === '`') {
        const end = text.indexOf('`', i + 1);
        if (end > i) { flush(); const code = document.createElement('code'); code.className = 'inline';
          code.textContent = text.slice(i + 1, end); parent.appendChild(code); i = end + 1; continue; }
      }
      if (c === '!' && text[i + 1] === '[') {
        const m = /^!\[([^\]]*)\]\(([^)]+)\)/.exec(text.slice(i));
        if (m) { flush(); const url = safeUrl(m[2], true);
          if (url) { const im = document.createElement('img'); im.src = url; im.alt = m[1]; parent.appendChild(im); }
          i += m[0].length; continue; }
      }
      if (c === '[') {
        const m = /^\[([^\]]*)\]\(([^)]+)\)/.exec(text.slice(i));
        if (m) { flush(); const url = safeUrl(m[2], false);
          const a = document.createElement(url ? 'a' : 'span');
          if (url) { a.href = url; a.target = '_blank'; a.rel = 'noopener noreferrer'; }
          a.textContent = m[1]; parent.appendChild(a); i += m[0].length; continue; }
      }
      if ((c === '*' && text[i + 1] === '*') || (c === '_' && text[i + 1] === '_')) {
        const mark = c + c; const end = text.indexOf(mark, i + 2);
        if (end > i) { flush(); const b = document.createElement('strong');
          renderInline(b, text.slice(i + 2, end)); parent.appendChild(b); i = end + 2; continue; }
      }
      if (c === '*' || c === '_') {
        const end = text.indexOf(c, i + 1);
        if (end > i && end !== i + 1) { flush(); const em = document.createElement('em');
          renderInline(em, text.slice(i + 1, end)); parent.appendChild(em); i = end + 1; continue; }
      }
      buf += c; i++;
    }
    flush();
  }

  // ---------- 块级 Markdown ----------
  function isTableSep(line) { return /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/.test(line); }
  function splitRow(line) {
    return line.replace(/^\s*\|/, '').replace(/\|\s*$/, '').split('|').map(s => s.trim());
  }

  function renderBlocks(parent, text) {
    const lines = text.replace(/\r\n/g, '\n').split('\n');
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i++; continue; }

      if (/^\s*```/.test(line)) {
        i++; const code = [];
        while (i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i++]);
        i++;
        const pre = document.createElement('pre'); const c = document.createElement('code');
        c.textContent = code.join('\n'); pre.appendChild(c); parent.appendChild(pre); continue;
      }
      if (/^\s*\$\$/.test(line)) {
        const inline = line.trim().slice(2);
        let body = inline.replace(/\$\$\s*$/, '');
        if (!/\$\$\s*$/.test(line) || inline === '') {
          i++; const buf = [inline.replace(/\$\$\s*$/, '')];
          while (i < lines.length && !/\$\$/.test(lines[i])) buf.push(lines[i++]);
          if (i < lines.length) buf.push(lines[i].replace(/\$\$.*/, ''));
          i++; body = buf.join('\n');
        } else i++;
        const div = document.createElement('div'); div.className = 'math-block';
        div.appendChild(latexToNode(body.trim())); parent.appendChild(div); continue;
      }
      const h = /^(#{1,6})\s+(.*)$/.exec(line);
      if (h) { const el = document.createElement('h' + h[1].length); renderInline(el, h[2].trim());
        parent.appendChild(el); i++; continue; }
      if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { parent.appendChild(document.createElement('hr')); i++; continue; }
      if (line.includes('|') && i + 1 < lines.length && isTableSep(lines[i + 1])) {
        const header = splitRow(line); i += 2; const rows = [];
        while (i < lines.length && lines[i].includes('|') && lines[i].trim()) { rows.push(splitRow(lines[i])); i++; }
        const table = document.createElement('table'); table.className = 'md-table';
        const thead = document.createElement('thead'); const htr = document.createElement('tr');
        header.forEach(cell => { const th = document.createElement('th'); renderInline(th, cell); htr.appendChild(th); });
        thead.appendChild(htr); table.appendChild(thead);
        const tbody = document.createElement('tbody');
        rows.forEach(r => { const tr = document.createElement('tr');
          for (let k = 0; k < header.length; k++) { const td = document.createElement('td'); renderInline(td, r[k] || ''); tr.appendChild(td); }
          tbody.appendChild(tr); });
        table.appendChild(tbody); parent.appendChild(table); continue;
      }
      if (/^\s*>/.test(line)) {
        const buf = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) buf.push(lines[i++].replace(/^\s*>\s?/, ''));
        const bq = document.createElement('blockquote'); renderBlocks(bq, buf.join('\n')); parent.appendChild(bq); continue;
      }
      if (/^\s*([-*+]|\d+\.)\s+/.test(line)) {
        const ordered = /^\s*\d+\.\s+/.test(line);
        const listEl = document.createElement(ordered ? 'ol' : 'ul');
        while (i < lines.length && /^\s*([-*+]|\d+\.)\s+/.test(lines[i])) {
          const li = document.createElement('li');
          renderInline(li, lines[i].replace(/^\s*([-*+]|\d+\.)\s+/, '')); listEl.appendChild(li); i++;
        }
        parent.appendChild(listEl); continue;
      }
      const buf = [];
      while (i < lines.length && lines[i].trim() &&
             !/^\s*(#{1,6}\s|```|>|\$\$|([-*+]|\d+\.)\s)/.test(lines[i]) &&
             !(lines[i].includes('|') && i + 1 < lines.length && isTableSep(lines[i + 1]))) {
        buf.push(lines[i++]);
      }
      const p = document.createElement('p'); renderInline(p, buf.join('\n')); parent.appendChild(p);
    }
  }

  // ---------- 工具调用折叠：抽出 🔧/↩ 段，其余走 markdown ----------
  function renderChunk(target, raw) {
    const lines = (raw || '').split('\n');
    let i = 0;
    const isTool = (l) => /^\s*(🔧|↩)/.test(l);
    while (i < lines.length) {
      if (isTool(lines[i])) {
        const tool = [];
        while (i < lines.length && (isTool(lines[i]) || (!lines[i].trim() && i + 1 < lines.length && isTool(lines[i + 1])))) {
          tool.push(lines[i++]);
        }
        const det = document.createElement('details'); det.className = 'toolcall';
        const sum = document.createElement('summary');
        const nCalls = tool.filter(l => /^\s*🔧/.test(l)).length;
        sum.textContent = '工具调用' + (nCalls ? `（${nCalls}）` : '') + ' · 点击展开';
        const body = document.createElement('div'); body.className = 'tcbody';
        body.textContent = tool.join('\n').trim();
        det.appendChild(sum); det.appendChild(body); target.appendChild(det);
        continue;
      }
      const buf = [];
      while (i < lines.length && !isTool(lines[i])) buf.push(lines[i++]);
      const txt = buf.join('\n');
      if (txt.trim()) renderBlocks(target, txt);
    }
  }

  function buildClarify(data) {
    var box = document.createElement('div'); box.className = 'clarify';
    var q = document.createElement('div'); q.className = 'clarify-q';
    q.textContent = data.question || ('请选择 ' + (data.param || ''));
    box.appendChild(q);
    var opts = document.createElement('div'); opts.className = 'clarify-opts';
    (data.options || []).forEach(function (o) {
      var b = document.createElement('button'); b.className = 'clarify-btn';
      b.textContent = o.label || o.value;
      if (o.type) b.title = o.type;
      b.onclick = function () { if (window.nagaClarifyPick) window.nagaClarifyPick(data, o.value, box); };
      opts.appendChild(b);
    });
    box.appendChild(opts);
    return box;
  }

  function render(target, raw) {
    target.innerHTML = '';
    raw = raw || '';
    var re = /<naga-clarify>([\s\S]*?)<\/naga-clarify>/g;
    var last = 0, m, any = false;
    while ((m = re.exec(raw)) !== null) {
      any = true;
      var before = raw.slice(last, m.index);
      if (before.trim()) renderChunk(target, before);
      try { target.appendChild(buildClarify(JSON.parse(m[1]))); } catch (e) {}
      last = re.lastIndex;
    }
    var rest = raw.slice(last);
    if (rest.trim() || !any) renderChunk(target, rest);
  }

  window.NagaRender = { render, renderInline, latexToNode };
})();
