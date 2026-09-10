/*
 * Element colours, under one rule:
 *
 *     Hue distinguishes species. Saturation marks provenance.
 *
 * Every element below is written in OKLCH at a FIXED chroma and a fixed
 * lightness. Only the hue moves. That is the whole palette. Species are told
 * apart by hue alone, never by being more vivid, so no atom can ever creep up
 * and compete with the accent.
 *
 * The accent (vermilion, OKLCH chroma ~0.17) is roughly four times the chroma
 * used here. It belongs strictly to what CHGNet produced. If it is saturated,
 * a model made it up.
 *
 * Hues follow the Jmol/CPK conventions a chemist already reads: oxygen warm-red,
 * nitrogen blue, carbon neutral, sulfur yellow, halogens green, so the mapping
 * is familiar even though the intensity is not.
 *
 * The guard at the bottom of this file fails loudly if anyone raises an
 * element's chroma to make two species easier to tell apart. That is the
 * failure mode the design brief calls out: if it fires, the honest fix is
 * strict monochrome, not a brighter atom.
 */

// Fixed for every element. Do not raise these per-element.
export const ELEMENT_CHROMA = 0.042;
export const ELEMENT_LIGHTNESS = 0.72;
export const ELEMENT_LIGHTNESS_DARK = 0.62;

// Chroma of the accent, for the guard below.
export const ACCENT_CHROMA = 0.17;

// element -> hue angle in degrees. Hue only.
const HUE = {
  H: 250, D: 250,
  He: 200, Ne: 200, Ar: 200, Kr: 200, Xe: 200, Rn: 200,
  Li: 330, Na: 330, K: 330, Rb: 330, Cs: 330, Fr: 330,
  Be: 140, Mg: 140, Ca: 140, Sr: 140, Ba: 140, Ra: 140,
  B: 30, Al: 25, Ga: 25, In: 25, Tl: 25,
  C: 95, Si: 75, Ge: 75, Sn: 75, Pb: 75,
  N: 265, P: 40, As: 45, Sb: 50, Bi: 55,
  O: 20, S: 100, Se: 105, Te: 110, Po: 110,
  F: 160, Cl: 155, Br: 150, I: 145, At: 145,
  Sc: 190, Ti: 195, V: 200, Cr: 205, Mn: 285, Fe: 15,
  Co: 300, Ni: 130, Cu: 35, Zn: 215,
  Y: 190, Zr: 195, Nb: 200, Mo: 205, Tc: 210, Ru: 220,
  Rh: 225, Pd: 230, Ag: 235, Cd: 240,
  Hf: 195, Ta: 200, W: 205, Re: 210, Os: 220, Ir: 225,
  Pt: 230, Au: 55, Hg: 240,
  La: 175, Ce: 175, Pr: 175, Nd: 175, Pm: 175, Sm: 175, Eu: 175,
  Gd: 175, Tb: 175, Dy: 175, Ho: 175, Er: 175, Tm: 175, Yb: 175, Lu: 175,
  Ac: 320, Th: 320, Pa: 320, U: 320, Np: 320, Pu: 320,
};

const FALLBACK_HUE = 85;

/** Covalent-ish radii in angstroms, for drawing size. Not a colour decision. */
const RADIUS = {
  H: 0.31, D: 0.31, He: 0.28, Li: 1.28, Be: 0.96, B: 0.84, C: 0.76, N: 0.71,
  O: 0.66, F: 0.57, Ne: 0.58, Na: 1.66, Mg: 1.41, Al: 1.21, Si: 1.11, P: 1.07,
  S: 1.05, Cl: 1.02, Ar: 1.06, K: 2.03, Ca: 1.76, Sc: 1.7, Ti: 1.6, V: 1.53,
  Cr: 1.39, Mn: 1.5, Fe: 1.42, Co: 1.38, Ni: 1.24, Cu: 1.32, Zn: 1.22,
  Ga: 1.22, Ge: 1.2, As: 1.19, Se: 1.2, Br: 1.2, Kr: 1.16, Rb: 2.2, Sr: 1.95,
  Y: 1.9, Zr: 1.75, Nb: 1.64, Mo: 1.54, Tc: 1.47, Ru: 1.46, Rh: 1.42,
  Pd: 1.39, Ag: 1.45, Cd: 1.44, In: 1.42, Sn: 1.39, Sb: 1.39, Te: 1.38,
  I: 1.39, Xe: 1.4, Cs: 2.44, Ba: 2.15, La: 2.07, Ce: 2.04, Pr: 2.03,
  Nd: 2.01, Sm: 1.98, Eu: 1.98, Gd: 1.96, Tb: 1.94, Dy: 1.92, Ho: 1.92,
  Er: 1.89, Tm: 1.9, Yb: 1.87, Lu: 1.87, Hf: 1.75, Ta: 1.7, W: 1.62,
  Re: 1.51, Os: 1.44, Ir: 1.41, Pt: 1.36, Au: 1.36, Hg: 1.32, Tl: 1.45,
  Pb: 1.46, Bi: 1.48, Th: 2.06, U: 1.96, Np: 1.9, Pu: 1.87,
};

const DEFAULT_RADIUS = 1.3;

export function elementHue(symbol) {
  return HUE[symbol] ?? FALLBACK_HUE;
}

export function elementRadius(symbol) {
  return RADIUS[symbol] ?? DEFAULT_RADIUS;
}

/** CSS colour for an element, at the one chroma every element shares. */
export function elementColor(symbol, dark = false) {
  const lightness = dark ? ELEMENT_LIGHTNESS_DARK : ELEMENT_LIGHTNESS;
  return `oklch(${lightness} ${ELEMENT_CHROMA} ${elementHue(symbol)})`;
}

/**
 * OKLCH -> linear sRGB, so three.js materials use exactly the colours the CSS
 * shows. Written out rather than pulled from a colour library: it is thirty
 * lines and it keeps the palette rule auditable in one file.
 */
export function elementRgb(symbol, dark = false) {
  const L = dark ? ELEMENT_LIGHTNESS_DARK : ELEMENT_LIGHTNESS;
  const C = ELEMENT_CHROMA;
  const h = (elementHue(symbol) * Math.PI) / 180;
  const a = C * Math.cos(h);
  const b = C * Math.sin(h);

  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;

  const l = l_ * l_ * l_;
  const m = m_ * m_ * m_;
  const s = s_ * s_ * s_;

  return {
    r: clamp01(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
    g: clamp01(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
    b: clamp01(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s),
  };
}

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

/**
 * The palette rule, as an assertion rather than a comment.
 *
 * Species may differ in hue and nothing else. The moment an element is given
 * its own chroma or lightness to make it stand out, "saturation marks
 * provenance" stops being true and the accent no longer means anything.
 */
export function assertPaletteDiscipline() {
  const problems = [];
  if (ELEMENT_CHROMA >= ACCENT_CHROMA / 2) {
    problems.push(
      `element chroma ${ELEMENT_CHROMA} is not clearly below the accent's ${ACCENT_CHROMA}; ` +
        `atoms will compete with model output`
    );
  }
  for (const [symbol, hue] of Object.entries(HUE)) {
    if (typeof hue !== "number" || hue < 0 || hue >= 360) {
      problems.push(`${symbol} has hue ${hue}, which is not an angle`);
    }
  }
  return problems;
}
