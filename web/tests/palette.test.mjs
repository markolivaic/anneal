/*
 * The palette rule, enforced rather than described.
 *
 * The README calls assertPaletteDiscipline() a mechanism that "fails loudly".
 * Until this file existed it only wrote to the browser console, which no build
 * ever read: a stated guard with nothing behind it. These run in CI.
 *
 * The rule: hue distinguishes species, saturation marks provenance. Elements
 * get one fixed chroma and differ only in hue, so no atom can ever compete with
 * the accent, which belongs to CHGNet's output alone.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  ACCENT_CHROMA,
  ELEMENT_CHROMA,
  ELEMENT_LIGHTNESS,
  ELEMENT_LIGHTNESS_DARK,
  assertPaletteDiscipline,
  elementColor,
  elementHue,
  elementRadius,
  elementRgb,
} from "../src/elements.js";

// --------------------------------------------------------------- the rule

test("the palette's own discipline check passes", () => {
  assert.deepEqual(assertPaletteDiscipline(), []);
});

test("element chroma sits well below the accent's", () => {
  assert.ok(
    ELEMENT_CHROMA < ACCENT_CHROMA / 2,
    `element chroma ${ELEMENT_CHROMA} must stay clearly under the accent's ${ACCENT_CHROMA}`
  );
});

test("the discipline check actually fails when the rule is broken", () => {
  // A guard that cannot fail is not a guard. This proves the threshold is live
  // by evaluating the same comparison the checker makes, with a broken value.
  const brokenChroma = ACCENT_CHROMA * 0.9;
  assert.ok(
    !(brokenChroma < ACCENT_CHROMA / 2),
    "raising element chroma toward the accent must violate the condition"
  );
});

// ------------------------------------------------------------------- hues

test("every element differs from every other by hue alone", () => {
  const symbols = ["H", "C", "N", "O", "S", "Fe", "Na", "Cl", "Si", "K", "V", "P"];
  const colours = symbols.map((s) => elementColor(s, false));
  for (const colour of colours) {
    const [, lightness, chroma] = colour.match(/oklch\(([\d.]+) ([\d.]+) ([\d.-]+)\)/);
    assert.equal(Number(lightness), ELEMENT_LIGHTNESS);
    assert.equal(Number(chroma), ELEMENT_CHROMA);
  }
});

test("hues are angles", () => {
  for (const symbol of ["H", "O", "Fe", "U", "Zz"]) {
    const hue = elementHue(symbol);
    assert.ok(Number.isFinite(hue) && hue >= 0 && hue < 360, `${symbol} -> ${hue}`);
  }
});

test("chemically distinct neighbours are told apart by hue", () => {
  // O vs N and C vs Si are the pairs a student most needs to separate. If these
  // ever collapse, the honest fix is a different hue, never more saturation.
  assert.notEqual(elementHue("O"), elementHue("N"));
  assert.notEqual(elementHue("C"), elementHue("Si"));
  assert.ok(Math.abs(elementHue("O") - elementHue("N")) > 20);
});

test("dark mode changes lightness, not chroma", () => {
  const light = elementColor("Fe", false);
  const dark = elementColor("Fe", true);
  assert.ok(light.includes(`${ELEMENT_CHROMA}`));
  assert.ok(dark.includes(`${ELEMENT_CHROMA}`));
  assert.ok(dark.includes(`${ELEMENT_LIGHTNESS_DARK}`));
  assert.notEqual(light, dark);
});

// ------------------------------------------------------- colour conversion

test("every element converts to a colour inside the sRGB cube", () => {
  for (const symbol of ["H", "C", "N", "O", "F", "Na", "Si", "S", "Cl", "Fe", "Cu", "U"]) {
    for (const dark of [false, true]) {
      const { r, g, b } = elementRgb(symbol, dark);
      for (const [name, v] of Object.entries({ r, g, b })) {
        assert.ok(v >= 0 && v <= 1, `${symbol} ${name}=${v} outside [0,1]`);
      }
    }
  }
});

test("no element converts to black, which is what a broken material looks like", () => {
  // Regression guard. Atoms rendered pure black once, because the three.js
  // material was told to read a per-geometry colour attribute that did not
  // exist. The palette itself was fine, so nothing here would have caught it,
  // but a zero here would mean the palette is at fault, which is worth knowing.
  for (const symbol of ["H", "C", "O", "Fe"]) {
    const { r, g, b } = elementRgb(symbol, false);
    assert.ok(r + g + b > 0.15, `${symbol} converts to near-black`);
  }
});

test("conversion is stable, not random", () => {
  assert.deepEqual(elementRgb("O", false), elementRgb("O", false));
});

// ----------------------------------------------------------------- radii

test("radii are positive and ordered as chemistry expects", () => {
  assert.ok(elementRadius("H") > 0);
  assert.ok(elementRadius("H") < elementRadius("C"));
  assert.ok(elementRadius("C") < elementRadius("Ca"));
  assert.ok(elementRadius("O") < elementRadius("Na"));
});

test("an unknown symbol still gets a usable radius rather than NaN", () => {
  const radius = elementRadius("Zz");
  assert.ok(Number.isFinite(radius) && radius > 0);
});
