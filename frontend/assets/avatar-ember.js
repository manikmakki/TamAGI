// Ember: a living spark in the modular skeleton used by the chat and rig editor.
(() => {
  'use strict';
  const avatars = window.TamagiAvatars ||= {};
  const paths = new Map();
  const currents = new Map();
  const outline = '#101825';
  const TAU = Math.PI * 2;
  const radians = degrees => degrees * Math.PI / 180;

  function path(ctx, d, fill, width = 1.5, stroke = outline) {
    if (!paths.has(d)) paths.set(d, new Path2D(d));
    const p = paths.get(d);
    if (fill) { ctx.fillStyle = fill; ctx.fill(p); }
    if (width) {
      ctx.strokeStyle = stroke; ctx.lineWidth = width;
      ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.stroke(p);
    }
  }

  function gradient(ctx, x1, y1, x2, y2, colors) {
    const g = ctx.createLinearGradient(x1, y1, x2, y2);
    colors.forEach(([at, color]) => g.addColorStop(at, color));
    return g;
  }

  const cloth = ctx => gradient(ctx, 5, 5, 90, 105, [[0, '#354155'], [.55, '#263246'], [1, '#1a2332']]);
  const inner = ctx => gradient(ctx, 20, 5, 80, 100, [[0, '#202b3d'], [1, '#111b2a']]);
  const amber = ctx => gradient(ctx, 0, 0, 40, 45, [[0, '#fff1b8'], [.5, '#ffd166'], [1, '#f3ab36']]);

  // A smooth sampled current gives the traveling spark the same path as its
  // gold trace, without a halo surrounding the character or a DOM/SVG dependency.
  function current(ctx, points, time, phase = 0, width = 1.6) {
    const key = points.flat().join(',');
    if (!currents.has(key)) {
      const samples = [], line = new Path2D();
      let distance = 0;
      for (let i = 0; i < points.length - 1; i++) {
        const a = points[Math.max(0, i - 1)], b = points[i];
        const c = points[i + 1], d = points[Math.min(points.length - 1, i + 2)];
        for (let step = 0; step <= 12; step++) {
          if (i && !step) continue;
          const u = step / 12, u2 = u * u, u3 = u2 * u;
          const xy = [0, 1].map(k => .5 * (2 * b[k] + (-a[k] + c[k]) * u
            + (2 * a[k] - 5 * b[k] + 4 * c[k] - d[k]) * u2
            + (-a[k] + 3 * b[k] - 3 * c[k] + d[k]) * u3));
          const previous = samples.at(-1);
          if (previous) distance += Math.hypot(xy[0] - previous.x, xy[1] - previous.y);
          samples.push({x: xy[0], y: xy[1], distance});
          if (samples.length === 1) line.moveTo(...xy); else line.lineTo(...xy);
        }
      }
      currents.set(key, {line, samples, distance});
    }
    const trace = currents.get(key);
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.strokeStyle = '#e6ad4799'; ctx.lineWidth = width * 1.45; ctx.stroke(trace.line);
    ctx.strokeStyle = '#ffd166'; ctx.lineWidth = width; ctx.stroke(trace.line);
    const target = (((time * .11 + phase) % 1 + 1) % 1) * trace.distance;
    const next = trace.samples.findIndex(p => p.distance >= target);
    const b = trace.samples[Math.max(0, next)], a = trace.samples[Math.max(0, next - 1)];
    const blend = b.distance === a.distance ? 0 : (target - a.distance) / (b.distance - a.distance);
    spark(ctx, a.x + (b.x - a.x) * blend, a.y + (b.y - a.y) * blend, 1.9, time);
  }

  function spark(ctx, x, y, radius, time) {
    const r = radius * (1 + Math.sin(time * 2.4) * .09);
    ctx.save();
    ctx.shadowColor = '#ffc75c99'; ctx.shadowBlur = 2.5;
    ctx.fillStyle = '#ffe49b'; ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fill();
    ctx.shadowBlur = 0; ctx.fillStyle = '#fff7cf';
    ctx.beginPath(); ctx.arc(x - r * .12, y - r * .15, r * .48, 0, TAU); ctx.fill();
    ctx.restore();
  }

  function joint(ctx, x, y, rx, ry, time, angle = 0) {
    ctx.save(); ctx.translate(x, y); ctx.rotate(angle);
    ctx.fillStyle = '#1b2638'; ctx.strokeStyle = '#f1ba55'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.ellipse(0, 0, rx, ry, 0, 0, TAU); ctx.fill(); ctx.stroke();
    path(ctx, `M${-rx * .72} ${-ry * .3}Q0 ${ry * .55} ${rx * .75} ${-ry * .15}`, null, 2.3, '#ffd574');
    spark(ctx, -rx * .18, -.1, Math.min(rx, ry) * .72, time);
    ctx.restore();
  }

  function head(ctx, time) {
    // The hood is deliberately soft and asymmetric, joining the tunic at its
    // lower folds rather than reading as a separate helmet or armor shell.
    path(ctx, 'M12 52Q2 44 12 32Q31 5 52 4Q78 2 91 31Q100 51 96 72Q92 85 76 91L65 100L54 89L34 98L24 91Q5 85 8 68Z', cloth(ctx), 1.7);
    path(ctx, 'M20 49Q34 20 60 20Q82 18 91 49L88 76Q77 87 65 92L54 87L37 92L19 79Z', inner(ctx), 1.2);
    path(ctx, 'M25 44Q32 26 56 25Q78 25 84 42L82 67Q78 84 58 87Q39 85 29 72Q20 59 25 44Z', gradient(ctx, 28, 25, 81, 89, [[0, '#394458'], [1, '#263144']]), 1.1);
    path(ctx, 'M13 52Q16 70 30 81L40 90L31 92Q10 85 10 71Z', '#172234', 0);
    path(ctx, 'M80 84L67 97L56 88L62 87Z', '#111c2a88', 0);
    current(ctx, [[12, 70], [22, 47], [44, 28], [61, 19], [76, 24], [89, 43], [94, 65], [87, 80]], time, .12, 1.5);
    current(ctx, [[19, 31], [29, 26], [40, 28], [44, 28]], time, .63, .95);
    path(ctx, 'M17 37Q30 12 50 10', null, 1, '#70829d22');
  }

  function eye(ctx, variant, right) {
    // Brows live in the expression slot, so sadness and blinking can retain
    // their meaning instead of leaving a permanently surprised painted brow.
    if (variant === 'closed' || variant === 'happy') {
      path(ctx, variant === 'happy' ? 'M5 29Q20 11 35 27' : 'M5 25Q20 36 35 24', null, 3, '#ffd77c');
      path(ctx, variant === 'happy' ? 'M7 10Q19 5 31 10' : 'M7 11Q20 6 32 11', null, 1.7, '#e2ae56');
      return;
    }
    path(ctx, right ? 'M7 9Q21-1 32 10' : 'M4 11Q22 14 33 4', null, 2, '#ffd166');
    const d = right ? 'M8 21Q13 10 28 15Q37 25 29 40Q16 47 8 34Z'
      : 'M3 24Q15 10 30 17Q39 27 33 40Q15 46 5 34Z';
    ctx.save(); ctx.shadowColor = '#ffcd6999'; ctx.shadowBlur = 2.5;
    path(ctx, d, amber(ctx), 0); ctx.restore();
    ctx.save(); ctx.clip(paths.get(d));
    ctx.fillStyle = gradient(ctx, 20, 12, 20, 43, [[0, '#6a4524'], [.5, '#ca862a'], [1, '#ffc44c']]);
    ctx.beginPath(); ctx.ellipse(right ? 21 : 24, 27, right ? 8 : 9.5, 15, -.17, 0, TAU); ctx.fill();
    ctx.fillStyle = '#fff9df'; ctx.beginPath(); ctx.ellipse(right ? 17 : 25, 22, 3, 3.7, -.2, 0, TAU); ctx.fill();
    ctx.fillStyle = '#ffebb6bb'; ctx.beginPath(); ctx.arc(22, 36, 1.3, 0, TAU); ctx.fill();
    ctx.restore();
    path(ctx, right ? 'M8 21Q13 10 28 15' : 'M3 24Q15 10 30 17', null, 1.6, '#192436');
  }

  function mouth(ctx, variant) {
    if (variant === 'sad') {
      path(ctx, 'M7 20Q20 9 33 19', null, 2.7, '#ffc96b');
      return;
    }
    const open = variant === 'open' || variant === 'talking';
    const d = open ? 'M8 10Q20 5 33 9Q35 29 22 30Q9 30 8 10Z'
      : variant === 'happy' ? 'M3 10Q20 18 37 6Q31 31 19 29Q8 27 3 10Z'
        : 'M6 12Q21 18 35 7Q31 28 20 27Q10 26 6 12Z';
    path(ctx, d, '#714727', 1.6, '#182333');
    ctx.save(); ctx.clip(paths.get(d));
    path(ctx, 'M5 9Q20 14 35 6L30 15Q19 19 9 14Z', '#ffe9af', 0);
    path(ctx, 'M10 29Q19 16 32 27L31 33H10Z', '#f4ae4c', 0);
    ctx.restore();
  }

  function body(ctx, time) {
    path(ctx, 'M23 5Q46-3 74 6Q89 17 86 38L77 70L93 107Q72 118 53 102Q28 120 8 106L24 69L16 36Q11 17 23 5Z', cloth(ctx), 1.7);
    path(ctx, 'M25 7Q42 8 50 18Q39 31 18 34L18 25Z', '#3b4860', 1);
    path(ctx, 'M53 18Q65 7 75 7L84 28L81 37Q61 30 53 18Z', '#303e53', 1);
    path(ctx, 'M50 19Q47 43 54 60L53 102', null, 1.2, '#182335');
    path(ctx, 'M26 70L12 105Q31 111 45 102Z', '#35425955', 0);
    path(ctx, 'M62 68L87 104L73 107L55 101Z', '#15203144', 0);
    current(ctx, [[52, 7], [55, 29], [61, 48], [45, 69], [36, 91], [19, 107]], time, .35, 1.5);
    // The exposed hip sparks join the leg currents beneath the open hem.
    spark(ctx, 30, 103, 3, time); spark(ctx, 71, 104, 2.7, time + .5);
  }

  function arm(ctx, right, time) {
    if (right) {
      // Shoulder → elbow → upturned wrist: the inviting forearm bend is part of
      // this sprite; its hand attaches at the actual wrist endpoint.
      path(ctx, 'M5 4Q22-7 39 10L51 49L72 11Q83-1 101 19L105 28L64 94Q43 111 23 88L-3 33Q-8 14 5 4Z', cloth(ctx), 1.8);
      path(ctx, 'M38 61Q45 54 54 62L63 73L55 88Q45 99 33 87Z', '#1a2638', 1);
      current(ctx, [[17, 14], [27, 43], [43, 78], [65, 56], [86, 23]], time, .58, 1.6);
      joint(ctx, 18, 16, 10, 7, time, -.3);
      joint(ctx, 43, 78, 10, 6, time + .4, -.3);
      joint(ctx, 86, 23, 9, 5, time + .7, .55);
    } else {
      path(ctx, 'M57 1Q77-10 96 7Q110 24 94 42L75 68L47 106Q26 118 5 96L25 52L49 12Z', cloth(ctx), 1.8);
      path(ctx, 'M39 53Q50 45 65 57L69 65L56 76Q44 75 34 64Z', '#1d2a3c', 1);
      current(ctx, [[76, 14], [65, 34], [51, 59], [41, 77], [28, 94]], time, .17, 1.5);
      joint(ctx, 76, 14, 10, 6, time, .55);
      joint(ctx, 51, 59, 10, 6, time + .3, .55);
      joint(ctx, 28, 94, 10, 5, time + .6, .55);
    }
  }

  function hand(ctx, right) {
    if (right) {
      path(ctx, 'M7 22Q6 13 15 10L29 1Q35-2 38 3Q40 7 35 12L26 20L47 14Q55 14 55 21Q55 28 44 32L20 41Q12 44 7 36Z', cloth(ctx), 1.7);
      path(ctx, 'M18 22Q25 27 31 25', null, .8, '#73829c44');
    } else {
      path(ctx, 'M39 3Q45 8 43 17L42 32Q42 43 34 43Q27 43 28 31L26 25L16 36Q9 42 4 37Q-1 32 7 24L25 7Q31-1 39 3Z', cloth(ctx), 1.7);
    }
  }

  function leg(ctx, right, time) {
    if (right) {
      path(ctx, 'M23 2Q52-8 83 5L94 41Q96 53 84 68L72 99Q53 109 33 98L17 66Q5 53 15 35Z', cloth(ctx), 1.8);
      path(ctx, 'M26 46Q48 39 79 48L77 59Q49 67 26 57Z', '#1e2b3e', 1);
      current(ctx, [[56, 5], [70, 29], [64, 53], [48, 77], [53, 97]], time, .65, 1.7);
      joint(ctx, 56, 5, 12, 5, time);
      joint(ctx, 64, 53, 12, 5.5, time + .3, .12);
      joint(ctx, 53, 97, 11, 4, time + .6);
    } else {
      path(ctx, 'M29 2Q59-8 89 5L74 40Q64 52 73 65L86 96Q64 109 38 99L16 70Q1 54 15 34Z', cloth(ctx), 1.8);
      path(ctx, 'M25 44Q41 38 62 44L64 58Q43 64 23 56Z', '#1c293c', 1);
      current(ctx, [[54, 5], [42, 29], [40, 50], [54, 73], [60, 96]], time, .27, 1.7);
      joint(ctx, 54, 5, 12, 5, time);
      joint(ctx, 40, 50, 12, 5.5, time + .3, -.15);
      joint(ctx, 60, 96, 11, 4, time + .6);
    }
  }

  function foot(ctx, right) {
    if (right) { ctx.translate(60, 0); ctx.scale(-1, 1); }
    path(ctx, 'M19 3Q32-3 43 6L51 36Q59 57 43 64Q23 71 12 57Q6 45 12 31Z', cloth(ctx), 1.8);
    path(ctx, 'M16 57Q29 66 45 57', null, 1, '#151f2f');
  }

  function neck(ctx, time) {
    path(ctx, 'M8 0Q50-5 92 0L92 101Q50 108 8 101Z', inner(ctx), 2.5);
    current(ctx, [[23, 22], [46, 29], [75, 23]], time, .15, 4);
    current(ctx, [[25, 64], [50, 74], [74, 65]], time, .67, 4.5);
    spark(ctx, 51, 70, 7, time);
  }

  avatars.ember = {
    label: 'Ember',
    pixelated: false,
    createBones(createBone) {
      const b = (...args) => createBone(...args);
      const bones = [
        b('root', null, 0, 0, 0, 0, 0, 0),
        b('body', 'root', 0, -26, 0, 13.5, 27, 2),
        b('neck', 'body', 0, -15, 0, 0, 5, 4),
        b('head', 'neck', 0, -19, 0, 0, 36, 5),
        b('eye_left', 'head', -3.5, 1, 0, 0, 0, 6),
        b('eye_right', 'head', 7, -1, 0, 0, 0, 6),
        b('mouth', 'head', 3.3, 9, 0, 0, 0, 7),
        b('arm_left', 'body', -11, -10, 0, 0, 24, 1),
        b('hand_left', 'arm_left', -8.6, 22, 0, 0, 7, 3),
        b('arm_right', 'body', 11, -10, 0, 0, 22, 3),
        b('hand_right', 'arm_right', 18, 1, 0, 0, 7, 4),
        b('leg_left', 'root', -5.5, -15, 0, 0, 29, 0),
        b('foot_left', 'leg_left', .2, 28, 0, 0, 8, 1),
        b('leg_right', 'root', 5.5, -15, 0, 0, 30, 0),
        b('foot_right', 'leg_right', .3, 29, 0, 0, 8, 1),
        b('accessory', 'neck', 0, 0, 0, 0, 12, 4),
      ];
      for (const bone of bones) {
        bone.pivotOffsetX = 0;
        bone.pivotOffsetY = bone.name.startsWith('leg_') ? .5 : bone.name.startsWith('foot_') ? .3 : 0;
      }
      return bones;
    },
    manifest(bones) {
      const lengths = Object.fromEntries(bones.map(b => [b.name, b.length]));
      const def = (bone, w, h, length = 0, variants = ['default'], ox = 0, oy = 0) => {
        const scale = length ? Math.max(.1, lengths[bone] / length) : 1;
        return {bone, w: w * scale, h: h * scale, ox: ox * scale, oy: oy * scale,
          variants: Object.fromEntries(variants.map(v => [v, null]))};
      };
      return {
        body: def('body', 28, 27, 27, ['default'], 0, 1),
        head: def('head', 38, 36, 36),
        eye_left: def('eye_left', 8.5, 9.7, 0, ['open', 'closed', 'happy']),
        eye_right: def('eye_right', 6.7, 9.8, 0, ['open', 'closed', 'happy']),
        mouth: def('mouth', 7.5, 5, 0, ['neutral', 'happy', 'sad', 'open', 'talking']),
        arm_left: def('arm_left', 18, 26, 24, ['default'], -4.7, 10.5),
        arm_right: def('arm_right', 26, 22, 22, ['default'], 8.5, 7),
        hand_left: def('hand_left', 8, 9, 7),
        hand_right: def('hand_right', 11.5, 9.2, 7, ['default'], 2, 0),
        leg_left: def('leg_left', 15.5, 31, 29, ['default'], -1, 0),
        leg_right: def('leg_right', 15.5, 32, 30),
        foot_left: def('foot_left', 9.5, 11, 8),
        foot_right: def('foot_right', 9.5, 11, 8),
        accessory: def('accessory', 7.5, 12, 12),
      };
    },
    draw(ctx, part, variant, w, h, time = 0) {
      ctx.save();
      switch (part) {
        case 'head': ctx.scale(w / 100, h / 100); head(ctx, time); break;
        case 'body': ctx.scale(w / 100, h / 120); body(ctx, time); break;
        case 'eye_left': case 'eye_right':
          ctx.scale(w / 40, h / 46); eye(ctx, variant, part === 'eye_right'); break;
        case 'mouth': ctx.scale(w / 40, h / 34); mouth(ctx, variant); break;
        case 'arm_left': case 'arm_right':
          ctx.scale(w / 100, h / 100); arm(ctx, part === 'arm_right', time); break;
        case 'hand_left': case 'hand_right':
          ctx.scale(w / (part === 'hand_right' ? 56 : 46), h / 46); hand(ctx, part === 'hand_right'); break;
        case 'leg_left': case 'leg_right':
          ctx.scale(w / 100, h / 100); leg(ctx, part === 'leg_right', time); break;
        case 'foot_left': case 'foot_right':
          ctx.scale(w / 60, h / 70); foot(ctx, part === 'foot_right'); break;
        case 'accessory': ctx.scale(w / 100, h / 100); neck(ctx, time); break;
      }
      ctx.restore();
    },
    animate(bones, time, name, variants) {
      const m = Object.fromEntries(bones.map(b => [b.name, b]));
      const gentle = name === 'sad' ? .5 : 1;
      m.root.translateY += -1.8 + Math.sin(time * 1.65) * .95 * gentle;
      m.root.translateX += Math.sin(time * .8) * .35 * gentle;
      m.root.rotation += radians(Math.sin(time * .65) * .8);
      m.leg_left.rotation += radians(6 + Math.sin(time * 1.1) * 2.5);
      m.leg_left.translateY -= 1.2;
      m.leg_right.rotation += radians(-3 + Math.sin(time * 1.1 + 1.3) * 1.8);
      m.foot_left.rotation += radians(-10 + Math.sin(time * 1.3) * 3);
      m.foot_right.rotation += radians(6 + Math.sin(time * 1.3 + 1) * 2);
      if (name === 'idle' || name === 'talking') {
        m.head.rotation += radians(-6 + Math.sin(time * .9) * 1.5);
        m.body.rotation += radians(-1.5);
        m.arm_right.rotation += radians(-8 + Math.sin(time * 1.25) * 4);
        m.hand_right.rotation += radians(-8 + Math.sin(time * 1.4) * 4);
        m.arm_left.rotation += radians(3 + Math.sin(time * 1.1) * 2);
      } else if (name === 'thinking') {
        // The shared pose targets a straight humanoid arm. Ember's forearm is
        // already bent, so bring its attached wrist toward the cheek instead
        // of applying the shared detached-hand translation.
        m.arm_right.rotation += radians(-60);
        m.hand_right.translateX = 0; m.hand_right.translateY = 0;
        m.hand_right.rotation += radians(-30);
      } else if (name === 'sad') {
        m.arm_right.rotation += radians(28);
        m.hand_right.rotation += radians(20);
      }
      return variants;
    },
  };
})();
