/* Echo's original pixel artwork, recovered from a80cacb:frontend/index.html.
 * The old 16 x 26 canvas is divided between the current named rig parts.
 * One original pixel is four rig units; transparent cells stay transparent.
 */
(() => {
  'use strict';

  const PALETTE = [null, '#00ff9d', null, '#ff6b9d', '#6b8aff', '#2a2a50'];
  const grid = rows => rows.map(row => Array.from(row, Number));
  const HEAD = grid([
    '0000044404444000',
    '0004444444444000',
    '0004440440444000',
    '0004000000001000',
    '0001000000001000',
    '0001000000001000',
    '0001000000001000',
    '0000111111110000',
  ]);
  // The final row is the old legs' shared waistband, kept on the torso so
  // the two leg bones can move independently without splitting the belt.
  const BODY = grid([
    '0000011111100000',
    '0000100000010000',
    '0001033033001000',
    '0001033333001000',
    '0001003330001000',
    '0001000300001000',
    '0001000000001000',
    '0000100000010000',
    '0000111111110000',
  ]);
  const ARMS = {
    arm_left: grid(['000', '001', '012', '120', '010', '000', '000', '000']),
    arm_right: grid(['000', '100', '210', '021', '010', '000', '000', '000']),
  };
  const LEG = grid(['101', '101', '101', '101', '101', '101', '101']);
  const FOOT = grid(['1001', '1111']);
  const EYES = {
    eye_left: {
      open: grid(['03', '00']),
      closed: grid(['00', '33']),
      happy: grid(['33', '00']),
    },
    eye_right: {
      open: grid(['30', '00']),
      closed: grid(['00', '33']),
      happy: grid(['33', '00']),
    },
  };
  const MOUTHS = {
    neutral: grid(['00000000', '03333000', '00000000']),
    happy: grid(['00000000', '03333300', '00000000']),
    sad: grid(['30000030', '03333000', '30000300']),
    open: grid(['00000000', '30000003', '03333330']),
    talking: grid(['00000000', '30000003', '03333330']),
  };

  function drawGrid(ctx, pixels, w, h) {
    const rows = pixels.length;
    const columns = pixels[0].length;
    // Keep the pixels square even when a slot's dimensions are customized.
    const cell = Math.min(w / columns, h / rows);
    const left = (w - columns * cell) / 2;
    const top = (h - rows * cell) / 2;
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    ctx.shadowBlur = 5;
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < columns; x++) {
        const color = PALETTE[pixels[y][x]];
        if (!color) continue;
        ctx.fillStyle = color;
        ctx.shadowColor = color;
        ctx.fillRect(left + x * cell, top + y * cell, cell, cell);
      }
    }
    ctx.restore();
  }

  const SLOTS = {
    body: [64, 36, ['default']],
    head: [64, 32, ['default']],
    eye_left: [8, 8, ['open', 'closed', 'happy']],
    eye_right: [8, 8, ['open', 'closed', 'happy']],
    mouth: [32, 12, ['neutral', 'happy', 'sad', 'open', 'talking']],
    arm_left: [12, 32, ['default']],
    arm_right: [12, 32, ['default']],
    hand_left: [4, 4, ['default']],
    hand_right: [4, 4, ['default']],
    leg_left: [12, 28, ['default']],
    leg_right: [12, 28, ['default']],
    foot_left: [16, 8, ['default']],
    foot_right: [16, 8, ['default']],
    accessory: [16, 8, ['default']],
  };

  window.TamagiAvatars ||= {};
  window.TamagiAvatars.echo = {
    label: 'Echo',
    pixelated: true,

    createBones(createBone) {
      const bones = [
        createBone('root', null, 0, 0, 0, 0, 0, 0),
        createBone('body', 'root', 0, -26, 0, 18, 36, 2),
        createBone('neck', 'body', 0, -18, 0, 0, 0, 4),
        createBone('head', 'neck', 0, -16, 0, 16, 32, 5),
        createBone('eye_left', 'head', -12, 0, 0, 0, 0, 6),
        createBone('eye_right', 'head', 8, 0, 0, 0, 0, 6),
        createBone('mouth', 'head', 0, 6, 0, 0, 0, 7),
        createBone('arm_left', 'body', -26, -18, 0, 0, 16, 1),
        createBone('hand_left', 'arm_left', 0, 18, 0, 0, 4, 1),
        createBone('arm_right', 'body', 26, -18, 0, 0, 16, 3),
        createBone('hand_right', 'arm_right', 0, 18, 0, 0, 4, 3),
        createBone('leg_left', 'root', -14, -8, 0, 0, 28, 0),
        createBone('foot_left', 'leg_left', -2, 32, 0, 0, 8, -1),
        createBone('leg_right', 'root', 14, -8, 0, 0, 28, 0),
        createBone('foot_right', 'leg_right', 2, 32, 0, 0, 8, -1),
        createBone('accessory', 'head', 0, -16, 0, 0, 0, 8),
      ];
      for (const bone of bones) {
        bone.pivotOffsetX = 0;
        bone.pivotOffsetY = /^(arm|leg)_/.test(bone.name) ? 0.5 : 0;
      }
      return bones;
    },

    manifest(bones) {
      const names = new Set(bones.map(bone => bone.name));
      const parts = {};
      for (const [part, [w, h, variants]] of Object.entries(SLOTS)) {
        if (!names.has(part)) continue;
        parts[part] = {
          bone: part, w, h, ox: 0, oy: 0,
          variants: Object.fromEntries(variants.map(variant => [variant, null])),
        };
      }
      return parts;
    },

    draw(ctx, part, variant, w, h) {
      let pixels;
      if (part === 'head') pixels = HEAD;
      else if (part === 'body') pixels = BODY;
      else if (EYES[part]) pixels = EYES[part][variant] || EYES[part].open;
      else if (part === 'mouth') pixels = MOUTHS[variant] || MOUTHS.neutral;
      else if (ARMS[part]) pixels = ARMS[part];
      else if (part === 'leg_left' || part === 'leg_right') pixels = LEG;
      else if (part === 'foot_left' || part === 'foot_right') pixels = FOOT;
      // The original arm map already includes its tiny hand silhouette.
      // Empty hand/accessory slots remain available for custom PNG uploads.
      if (pixels) drawGrid(ctx, pixels, w, h);
    },
  };
})();
