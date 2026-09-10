/*
 * The geometry column: unit cell, atoms, bonds, and, during a relaxation,
 * the force vectors.
 *
 * Colour discipline lives in elements.js. Everything drawn here is either
 * deposited data (ink and element hues at low chroma) or model output (accent).
 * There is no third category, and nothing in between.
 *
 * Atoms are instanced. A 150-atom cell drawn with its periodic images runs to a
 * few thousand spheres, and one draw call per sphere would cost more than the
 * relaxation it is meant to be showing.
 */

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { elementRadius, elementRgb } from "./elements.js";

const ATOM_SEGMENTS = 20;
const BOND_RADIUS = 0.07;
const BOND_TOLERANCE = 1.25; // fraction of summed radii that still counts as bonded
const MAX_BONDS = 4000;

// Force arrows. Angstroms per eV/A, with a floor so a small force is still
// visible and a ceiling so a large one does not cross the whole cell.
const FORCE_ARROW_SCALE = 2.6;
const FORCE_ARROW_MIN_LENGTH = 0.75;
const FORCE_ARROW_MAX_LENGTH = 3.0;
const FORCE_ARROW_MIN_EV_PER_A = 0.05;

// Slack left around a tight corner fit, so force arrows and the atoms' own
// radii do not clip at the frame edge. The cell fills roughly 80% of the panel.
const FIT_MARGIN = 1.12;

export class LatticeViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.dark = currentThemeIsDark();

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.1, 2000);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;

    // Flat, even light. A crystal diagram is not a product render: no rim
    // lights, no dramatic shadows, nothing that implies a photograph.
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.7));
    const key = new THREE.DirectionalLight(0xffffff, 1.15);
    key.position.set(4, 7, 9);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.5);
    fill.position.set(-6, -3, -5);
    this.scene.add(fill);

    this.root = new THREE.Group();
    this.scene.add(this.root);

    this.pickables = [];
    this.selected = null;
    this.editedSites = new Set();
    this.onPick = null;

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    canvas.addEventListener("pointerdown", (e) => this._handlePick(e));

    // A window listener alone is not enough: the constructor runs before the
    // grid has laid the canvas out, so the first measurement is the 300x150
    // default and nothing ever corrects it. Observe the parent box instead.
    this._resize();
    new ResizeObserver(() => this._resize()).observe(canvas.parentElement);
    this._animate();
  }

  setTheme(dark) {
    this.dark = dark;
    if (this.structure) this.render(this.structure, { keepCamera: true });
  }

  /** Draw a structure. `keepCamera` preserves the user's viewpoint across edits. */
  render(structure, { keepCamera = false } = {}) {
    const previousTarget = this.controls.target.clone();
    const previousPosition = this.camera.position.clone();

    this.structure = structure;
    this._clear();

    const matrix = structure.lattice.matrix;
    const centre = new THREE.Vector3(
      (matrix[0][0] + matrix[1][0] + matrix[2][0]) / 2,
      (matrix[0][1] + matrix[1][1] + matrix[2][1]) / 2,
      (matrix[0][2] + matrix[1][2] + matrix[2][2]) / 2
    );

    this._drawCell(matrix);
    this._drawAtoms(structure);
    this._drawBonds(structure);

    if (keepCamera) {
      this.camera.position.copy(previousPosition);
      this.controls.target.copy(previousTarget);
    } else {
      // Fit the eight cell corners to BOTH axes of the frustum, rather than
      // fitting a bounding sphere to the vertical one.
      //
      // The bounding-sphere version used the body diagonal as the radius and
      // divided by the vertical field of view. For a cell that is long and flat
      // -- most of the catalogue -- the diagonal is dominated by an axis lying
      // across the screen, where the panel is nearly twice as wide as it is
      // tall and has room to spare. The camera pulled back for a height the
      // cell never needed and the crystal sat small in a large empty panel.
      //
      // Projecting the corners into the camera's own basis and asking what
      // distance each one actually requires gives a tight fit in both
      // directions. FIT_MARGIN is the only slack left.
      const direction = new THREE.Vector3(0.75, 0.45, 1).normalize();
      const forward = direction.clone().negate();
      const right = new THREE.Vector3()
        .crossVectors(forward, new THREE.Vector3(0, 1, 0))
        .normalize();
      const up = new THREE.Vector3().crossVectors(right, forward).normalize();

      const halfV = (this.camera.fov * Math.PI) / 360;
      const tanV = Math.tan(halfV);
      const tanH = tanV * this.camera.aspect;

      const cellCorners = cellVertices(matrix);
      let distance = 0;
      for (const corner of cellCorners) {
        const offset = corner.clone().sub(centre);
        const depth = offset.dot(forward);
        distance = Math.max(
          distance,
          Math.abs(offset.dot(up)) / tanV + depth,
          Math.abs(offset.dot(right)) / tanH + depth
        );
      }
      this.camera.position
        .copy(centre)
        .addScaledVector(direction, distance * FIT_MARGIN);
      this.controls.target.copy(centre);
    }
    this.controls.update();
  }

  /** Move atoms to a relaxation step without rebuilding the scene. */
  updatePositions(positions, lattice) {
    if (!this.structure || !this.atomMesh) return;
    for (let i = 0; i < positions.length && i < this.atomCount; i++) {
      const p = positions[i];
      this.dummy.position.set(p[0], p[1], p[2]);
      this.dummy.scale.setScalar(this.atomScales[i]);
      this.dummy.updateMatrix();
      this.atomMesh.setMatrixAt(i, this.dummy.matrix);
      this.pickables[i]?.position.set(p[0], p[1], p[2]);
    }
    this.atomMesh.instanceMatrix.needsUpdate = true;
    // The ring rides along with its atom while the structure relaxes.
    if (this.selectionRing && this.selected != null && positions[this.selected]) {
      this.selectionRing.position.set(...positions[this.selected]);
    }

    if (lattice) {
      this._clearGroup(this.cellGroup);
      this._drawCell(lattice, this.cellGroup);
    }
    // Bonds are rebuilt because relaxation can break and make them, and a stale
    // bond is a lie about the structure.
    this._clearGroup(this.bondGroup);
    this._drawBondsFrom(positions, this.structure.sites, this.bondGroup);
  }

  /** Force vectors, drawn in the accent because CHGNet produced them.
   *
   * Two things this has to get right or the arrows are there and unreadable.
   * They start at the atom's surface, not its centre. A 0.4 eV/A force scaled
   * naively is shorter than the sphere it comes out of, so every arrow sat
   * hidden inside its own atom. And they have a minimum length, because the
   * point of the overlay is to show *where* the model is pushing, which a
   * two-pixel stub does not.
   */
  showForces(forces, positions, scale = FORCE_ARROW_SCALE) {
    this._clearGroup(this.forceGroup);
    if (!forces || !forces.length) return;

    const colour = this._accent();
    for (let i = 0; i < forces.length; i++) {
      const f = forces[i];
      const magnitude = Math.hypot(f[0], f[1], f[2]);
      if (magnitude < FORCE_ARROW_MIN_EV_PER_A) continue;

      const direction = new THREE.Vector3(f[0], f[1], f[2]).normalize();
      const radius = this.atomScales?.[i] ?? 0.4;
      const origin = new THREE.Vector3(...positions[i]).addScaledVector(
        direction,
        radius * 1.05
      );
      const length = Math.min(
        Math.max(magnitude * scale, FORCE_ARROW_MIN_LENGTH),
        FORCE_ARROW_MAX_LENGTH
      );
      this.forceGroup.add(
        new THREE.ArrowHelper(direction, origin, length, colour, length * 0.42, length * 0.26)
      );
    }
  }

  setSelected(index) {
    this.selected = index;
    this._refreshHighlights();
  }

  markEdited(index) {
    if (index != null) this.editedSites.add(index);
    this._refreshHighlights();
  }

  clearEdited() {
    this.editedSites.clear();
    this._refreshHighlights();
  }

  // ------------------------------------------------------------- internals

  _accent() {
    return new THREE.Color(this.dark ? 0xe8623a : 0xc2410c);
  }

  _inkColor() {
    return new THREE.Color(this.dark ? 0x4a483d : 0xa8a294);
  }

  _clear() {
    this._clearGroup(this.root);
    this.pickables = [];
    this.cellGroup = new THREE.Group();
    this.atomGroup = new THREE.Group();
    this.bondGroup = new THREE.Group();
    this.forceGroup = new THREE.Group();
    this.root.add(this.cellGroup, this.atomGroup, this.bondGroup, this.forceGroup);
  }

  _clearGroup(group) {
    if (!group) return;
    while (group.children.length) {
      const child = group.children.pop();
      child.geometry?.dispose?.();
      if (child.material) {
        (Array.isArray(child.material) ? child.material : [child.material]).forEach(
          (m) => m.dispose?.()
        );
      }
    }
  }

  /** The unit cell: twelve edges, hairline, in rule grey. Never the accent. */
  _drawCell(matrix, group = this.cellGroup) {
    const [a, b, c] = matrix.map((row) => new THREE.Vector3(...row));
    const origin = new THREE.Vector3(0, 0, 0);
    const corners = [
      origin,
      a,
      b,
      c,
      a.clone().add(b),
      a.clone().add(c),
      b.clone().add(c),
      a.clone().add(b).add(c),
    ];
    const edges = [
      [0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4],
      [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7],
    ];
    const points = [];
    edges.forEach(([i, j]) => points.push(corners[i], corners[j]));
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color: this._inkColor() });
    group.add(new THREE.LineSegments(geometry, material));
  }

  _drawAtoms(structure) {
    const sites = structure.sites;
    this.atomCount = sites.length;
    this.atomScales = sites.map((s) => Math.max(0.28, elementRadius(s.element) * 0.42));
    this.dummy = new THREE.Object3D();

    const geometry = new THREE.SphereGeometry(1, ATOM_SEGMENTS, ATOM_SEGMENTS);
    // No `vertexColors: true` here. It sounds right and renders every atom
    // black: it defines USE_COLOR, which makes the shader multiply by a
    // per-geometry `color` attribute that SphereGeometry does not have, so the
    // attribute reads as (0,0,0) and wipes out the instance colour applied
    // immediately after it. InstancedMesh picks up `instanceColor` on its own.
    const material = new THREE.MeshLambertMaterial();
    const mesh = new THREE.InstancedMesh(geometry, material, sites.length);

    const colour = new THREE.Color();
    sites.forEach((site, i) => {
      this.dummy.position.set(...site.cart);
      this.dummy.scale.setScalar(this.atomScales[i]);
      this.dummy.updateMatrix();
      mesh.setMatrixAt(i, this.dummy.matrix);

      const rgb = elementRgb(site.element, this.dark);
      colour.setRGB(rgb.r, rgb.g, rgb.b);
      mesh.setColorAt(i, colour);

      // Invisible pick targets: raycasting an InstancedMesh gives an instanceId,
      // but tiny transparent spheres keep hit-testing simple and forgiving.
      const target = new THREE.Mesh(
        new THREE.SphereGeometry(this.atomScales[i] * 1.25, 8, 8),
        new THREE.MeshBasicMaterial({ visible: false })
      );
      target.position.set(...site.cart);
      target.userData.siteIndex = i;
      this.pickables.push(target);
      this.atomGroup.add(target);
    });

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    this.atomMesh = mesh;
    this.atomGroup.add(mesh);
    this._refreshHighlights();
  }

  _drawBonds(structure) {
    this._drawBondsFrom(
      structure.sites.map((s) => s.cart),
      structure.sites,
      this.bondGroup
    );
  }

  _drawBondsFrom(positions, sites, group) {
    const material = new THREE.MeshLambertMaterial({ color: this._inkColor() });
    let drawn = 0;
    for (let i = 0; i < positions.length && drawn < MAX_BONDS; i++) {
      for (let j = i + 1; j < positions.length && drawn < MAX_BONDS; j++) {
        const cutoff =
          (elementRadius(sites[i].element) + elementRadius(sites[j].element)) *
          BOND_TOLERANCE;
        const dx = positions[i][0] - positions[j][0];
        const dy = positions[i][1] - positions[j][1];
        const dz = positions[i][2] - positions[j][2];
        const distance = Math.hypot(dx, dy, dz);
        if (distance > cutoff || distance < 0.1) continue;

        const start = new THREE.Vector3(...positions[i]);
        const end = new THREE.Vector3(...positions[j]);
        const cylinder = new THREE.Mesh(
          new THREE.CylinderGeometry(BOND_RADIUS, BOND_RADIUS, distance, 6),
          material
        );
        cylinder.position.copy(start).add(end).multiplyScalar(0.5);
        cylinder.quaternion.setFromUnitVectors(
          new THREE.Vector3(0, 1, 0),
          end.clone().sub(start).normalize()
        );
        group.add(cylinder);
        drawn++;
      }
    }
  }

  _refreshHighlights() {
    if (!this.atomMesh || !this.structure) return;
    const colour = new THREE.Color();
    const accent = this._accent();
    this.structure.sites.forEach((site, i) => {
      // Only an edited site takes the accent. Selection is deliberately NOT
      // included: a selected atom is still deposited data, and painting it
      // vermilion would claim a model produced it.
      if (this.editedSites.has(i)) {
        colour.copy(accent);
      } else {
        const rgb = elementRgb(site.element, this.dark);
        colour.setRGB(rgb.r, rgb.g, rgb.b);
      }
      this.atomMesh.setColorAt(i, colour);
    });
    if (this.atomMesh.instanceColor) this.atomMesh.instanceColor.needsUpdate = true;
    this._refreshSelectionRing();
  }

  /** Selection is shown as an ink ring around the atom, never as a recolour. */
  _refreshSelectionRing() {
    if (this.selectionRing) {
      this.selectionRing.geometry.dispose();
      this.selectionRing.material.dispose();
      this.selectionRing.parent?.remove(this.selectionRing);
      this.selectionRing = null;
    }
    if (this.selected == null || !this.structure?.sites[this.selected]) return;

    const radius = (this.atomScales?.[this.selected] ?? 0.4) * 1.45;
    const ring = new THREE.Mesh(
      new THREE.SphereGeometry(radius, 16, 12),
      new THREE.MeshBasicMaterial({
        color: this.dark ? 0xeae6da : 0x16150f,
        wireframe: true,
        transparent: true,
        opacity: 0.55,
      })
    );
    const position = this.pickables[this.selected]?.position;
    if (position) ring.position.copy(position);
    this.selectionRing = ring;
    this.atomGroup.add(ring);
  }

  _handlePick(event) {
    const rect = this.canvas.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.pickables, false);
    if (hits.length && this.onPick) {
      this.onPick(hits[0].object.userData.siteIndex);
    }
  }

  _resize() {
    const parent = this.canvas.parentElement;
    if (!parent) return;
    const width = parent.clientWidth;
    const height = parent.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

/** The eight corners of a cell, from its 3x3 lattice matrix. */
function cellVertices(matrix) {
  const [a, b, c] = matrix.map((row) => new THREE.Vector3(...row));
  return [
    new THREE.Vector3(0, 0, 0),
    a,
    b,
    c,
    a.clone().add(b),
    a.clone().add(c),
    b.clone().add(c),
    a.clone().add(b).add(c),
  ];
}

export function currentThemeIsDark() {
  const explicit = document.documentElement.dataset.theme;
  if (explicit) return explicit === "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}
