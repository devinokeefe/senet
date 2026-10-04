// The web page in node:vm: a minimal DOM (the parts web/app.js uses), a loader that runs a
// page's scripts on its markup with fake timers, and the user's actions.
// Used by tools/check_web_app.mjs and tools/check_portable.mjs.
import vm from "node:vm";

/** Lets the pending promise jobs run. */
export const settle = () => new Promise((resolve) => setImmediate(resolve));

/** A seeded LCG in [0, 1), so that every run deals the same throws and AI seeds. */
export function seeded(seed) {
  let a = seed >>> 0;
  return () => (a = (Math.imul(a, 1664525) + 1013904223) >>> 0) / 2 ** 32;
}

// ---- A minimal DOM ----
export class Element {
  constructor(tag, attributes = {}) {
    this.tagName = tag.toUpperCase();
    this.attributes = { ...attributes };
    this.children = [];
    this.parentElement = null;
    this.style = {};
    this.dataset = {};
    this.listeners = {};
    this.text = ""; // textContent and innerHTML as last assigned (or parsed)
    this.html = "";
    for (const k of ["hidden", "disabled", "checked", "selected"]) this[k] = k in attributes;
    for (const k of ["min", "max"]) if (k in attributes) this[k] = attributes[k];
    this.value = attributes.value ?? "";
    this.tabIndex = 0;
    this.scrollTop = this.scrollHeight = this.offsetWidth = 0;
  }
  get id() { return this.attributes.id ?? ""; }
  get className() { return this.attributes.class ?? ""; }
  set className(v) { this.attributes.class = String(v); }
  get classList() {
    const names = () => this.className.split(/\s+/).filter(Boolean);
    const has = (c) => names().includes(c);
    const add = (c) => { if (!has(c)) this.className = [...names(), c].join(" "); };
    const remove = (c) => { this.className = names().filter((n) => n !== c).join(" "); };
    return {
      contains: has,
      add: (...cs) => cs.forEach(add),
      remove: (...cs) => cs.forEach(remove),
      toggle: (c, on = !has(c)) => { (on ? add : remove)(c); return on; },
    };
  }
  get textContent() { return this.text; }
  set textContent(v) { this.children = []; this.text = this.html = String(v); }
  get innerHTML() { return this.html; }
  set innerHTML(v) { this.children = []; this.html = String(v); this.text = this.html.replace(/<[^>]*>/g, ""); }
  get value() {
    if (this.tagName !== "SELECT") return this.ownValue;
    const options = this.options;
    return (options.find((o) => o.selected) ?? options[0])?.value ?? "";
  }
  set value(v) {
    if (this.tagName === "SELECT") for (const o of this.options) o.selected = o.value === v;
    else this.ownValue = String(v);
  }
  get options() { return this.children.filter((c) => c.tagName === "OPTION"); }
  get lastElementChild() { return this.children.at(-1) ?? null; }
  getAttribute(k) { return this.attributes[k] ?? null; }
  setAttribute(k, v) { this.attributes[k] = String(v); }
  appendChild(child) {
    child.remove();
    child.parentElement = this;
    this.children.push(child);
    return child;
  }
  append(...children) { for (const c of children) this.appendChild(c); }
  remove() {
    const parent = this.parentElement;
    if (!parent) return;
    parent.children.splice(parent.children.indexOf(this), 1);
    this.parentElement = null;
  }
  // Selectors: comma-separated `tag`, `.class`, `tag.class`, `tag[attribute]`.
  matches(selector) {
    return selector.split(",").some((part) => {
      const m = /^([a-z]*)((?:\.[\w-]+)*)(?:\[([\w-]+)\])?$/i.exec(part.trim());
      if (!m) throw new Error(`DOM stub: unsupported selector "${part}"`);
      const [, tag, classes, attribute] = m;
      return (!tag || this.tagName === tag.toUpperCase())
        && classes.split(".").filter(Boolean).every((c) => this.classList.contains(c))
        && (!attribute || attribute in this.attributes);
    });
  }
  *descendants() {
    for (const c of this.children) {
      yield c;
      yield* c.descendants();
    }
  }
  querySelectorAll(selector) { return [...this.descendants()].filter((e) => e.matches(selector)); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] ?? null; }
  closest(selector) {
    for (let e = this; e; e = e.parentElement) if (e.matches(selector)) return e;
    return null;
  }
  addEventListener(type, listener) { (this.listeners[type] ??= []).push(listener); }
  // Calls this element's listeners (no bubbling) and returns the event.
  fire(type, init = {}) {
    const event = { type, target: this, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, ...init };
    for (const listener of this.listeners[type] ?? []) listener(event);
    return event;
  }
}

const VOID_TAGS = new Set(["meta", "link", "input", "br", "img", "hr"]);
const attributesOf = (text) => Object.fromEntries([...text.matchAll(/([^\s=/]+)(?:="([^"]*)")?/g)].map(([, k, v]) => [k, v ?? ""]));

export class Document extends Element {
  constructor(html) {
    super("#document");
    const open = [this];
    // The text of a script or style element is raw: its "<" opens no tag.
    const token = /<!--[\s\S]*?-->|<!doctype[^>]*>|<(script|style)\b([^>]*)>([\s\S]*?)<\/\1\s*>|<\/([a-z0-9]+)\s*>|<([a-z0-9]+)([^>]*)>|([^<]+)/gi;
    for (const [, rawTag, rawAttrs, raw, closing, tag, attrs, text] of html.matchAll(token)) {
      const parent = open.at(-1);
      if (rawTag) {
        const el = parent.appendChild(new Element(rawTag, attributesOf(rawAttrs)));
        el.text = el.html = raw;
      } else if (text !== undefined) {
        parent.html += text;
        parent.text += text; // entities stay undecoded: the checks never read parsed text
      } else if (tag) {
        const el = parent.appendChild(new Element(tag, attributesOf(attrs)));
        if (!VOID_TAGS.has(tag.toLowerCase())) open.push(el);
      } else if (closing) {
        while (open.length > 1 && open.pop().tagName !== closing.toUpperCase());
      }
    }
    // app.js never gives an id to the elements it creates.
    this.ids = new Map([...this.descendants()].filter((e) => e.id).map((e) => [e.id, e]));
  }
  get body() { return this.querySelector("body"); }
  getElementById(id) { return this.ids.get(id) ?? null; }
  createElement(tag) { return new Element(tag); }
}

// ---- The page ----
const FRAME_MS = 16; // between animation frames

/**
 * Runs `scripts` ([filename, source] pairs, in order, sharing one global scope as a page's
 * scripts do) on the DOM of `html`, with fake timers (animation frames too), `random` as
 * Math.random and `globals` (such as fetch and crypto). Returns the page: its document,
 * $(id), evaluate(expression) in its global scope, and runUntil(done), which fires its
 * timers in order, letting the promise jobs each starts run, until done() holds (false if
 * they run out first).
 */
export async function loadPage(html, scripts, { random, ...globals }) {
  const document = new Document(html);
  const timers = [];
  let now = 0, order = 0;
  const setTimeout = (fn, ms = 0) => {
    timers.push({ at: now + ms, order: ++order, fn });
    return order;
  };
  const context = vm.createContext({
    document, Element, structuredClone, atob, AbortController,
    setTimeout,
    requestAnimationFrame: (fn) => setTimeout(() => fn(now), FRAME_MS),
    clearTimeout: (id) => {
      const i = timers.findIndex((t) => t.order === id);
      if (i >= 0) timers.splice(i, 1);
    },
    ...globals,
  });
  context.window = context;
  vm.runInContext("Math", context).random = random;
  for (const [filename, source] of scripts) vm.runInContext(source, context, { filename });
  await settle();

  async function tick() {
    timers.sort((a, b) => a.at - b.at || a.order - b.order);
    const timer = timers.shift();
    now = timer.at;
    timer.fn();
    await settle();
  }
  return {
    document,
    $: (id) => document.getElementById(id),
    evaluate: (expression) => vm.runInContext(expression, context),
    async runUntil(done, limit = 50000) {
      for (let i = 0; !done(); i++) {
        if (i === limit || !timers.length) return false;
        await tick();
      }
      return true;
    },
  };
}

// ---- User actions ----
export function click(el) {
  if (el.disabled) throw new Error(`clicked the disabled control ${el.id || el.tagName}`);
  el.fire("click");
}
export function choosePlayer(page, color, bot) {
  const sel = page.$("sel-" + color);
  if (!sel.options.some((o) => o.value === bot)) throw new Error(`${bot} is not offered for ${color}`);
  sel.value = bot;
  sel.fire("change");
}
export async function newGame(page, white, black) {
  choosePlayer(page, "white", white);
  choosePlayer(page, "black", black);
  click(page.$("btn-new"));
  await settle();
}
