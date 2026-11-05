// Vex's vector parts follow the approved docs.
// Each part is drawn locally so the existing skeleton, expression variants,
// and user-uploaded PNG overrides all use the same animation path.
(() => {
  'use strict';
  const avatars = window.TamagiAvatars ||= {};
  const paths = new Map();
  const edge = '#06121b';

  function path(ctx, d, fill, width = 2, stroke = edge) {
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
    colors.forEach(([position, color]) => g.addColorStop(position, color));
    return g;
  }
  const coat = ctx => gradient(ctx, 12, 0, 84, 100, [[0, '#87bcdf'], [.5, '#5799cb'], [1, '#316a99']]);
  const navy = ctx => gradient(ctx, 10, 0, 80, 100, [[0, '#29344e'], [1, '#101827']]);
  const gold = ctx => gradient(ctx, 10, 0, 80, 100, [[0, '#fff0af'], [.55, '#ffd166'], [1, '#efad3d']]);

  function head(ctx) {
    // Ears remain part of the head; the receiver travels with head tilts.
    path(ctx, 'M24 49Q-2 20 7 1Q31 10 47 34Z', coat(ctx), 2.7);
    path(ctx, 'M54 34Q74 9 94 1Q102 24 78 51Z', coat(ctx), 2.7);
    path(ctx, 'M25 39Q11 20 14 12Q29 23 39 39Z', gradient(ctx, 15, 10, 40, 42, [[0, '#5897c5'], [1, '#294a6b']]), 1.9);
    path(ctx, 'M63 39Q76 19 88 11Q90 28 76 44Z', gold(ctx), 1.9);
    ctx.save(); ctx.shadowColor = '#ffd16688'; ctx.shadowBlur = 3;
    path(ctx, 'M67 36L86 14', null, 1.9, '#ffe294');
    ctx.restore();
    path(ctx, 'M80 23Q80 21 83 21H90Q94 21 94 25V37Q94 41 90 41H83Q80 41 80 37Z', navy(ctx), 2);
    path(ctx, 'M84 27H90M84 33H90', null, 1.9, '#ffd166');

    path(ctx, 'M10 61Q12 28 49 27Q86 27 90 61V73Q90 98 50 100Q10 98 10 73Z', coat(ctx), 2.6);
    path(ctx, 'M13 66Q15 91 49 93Q78 92 88 73Q89 97 50 99Q11 97 11 75Z', '#356f9f55', 0);
    path(ctx, 'M22 46Q31 33 45 32', null, 1.9, '#d4efff88');
    path(ctx, 'M23 75Q50 63 77 75L73 87Q50 101 27 87Z', gradient(ctx, 25, 75, 76, 96, [[0, '#f5fbff'], [1, '#c9dfea']]), 1.8);
    path(ctx, 'M18 60Q50 42 82 60L76 82Q50 95 24 82Z', navy(ctx), 2.1);
    path(ctx, 'M20 60Q38 49 57 52L27 65Z', '#ffffff05', 0);
    path(ctx, 'M46 79H54L58 83L50 89L42 83Z', gold(ctx), 1.6);
    ctx.fillStyle = '#ffd166'; ctx.beginPath(); ctx.roundRect(12.4, 59, 4.2, 4.2, 1); ctx.fill();
  }

  function eye(ctx, variant, right) {
    ctx.scale(1 / 40, 1 / 34);
    if (right) { ctx.translate(40, 0); ctx.scale(-1, 1); }
    if (variant === 'closed' || variant === 'happy') {
      ctx.save(); ctx.shadowColor = '#ffd16688'; ctx.shadowBlur = 3;
      path(ctx, variant === 'happy' ? 'M5 21Q20 4 35 21' : 'M5 16Q20 27 35 16', null, 4, '#ffe294');
      ctx.restore();
      return;
    }
    const shape = 'M2 6Q12 12 23 7Q31 10 35 27Q20 36 10 27Q5 22 5 13Z';
    ctx.save(); ctx.shadowColor = '#ffcb6080'; ctx.shadowBlur = 4;
    path(ctx, shape, gradient(ctx, 0, 4, 18, 34, [[0, '#fff1ad'], [1, '#ffd166']]), 0);
    ctx.restore();
    ctx.save(); ctx.clip(paths.get(shape));
    ctx.fillStyle = gradient(ctx, 20, 6, 20, 31, [[0, '#574028'], [.45, '#b77c2a'], [1, '#ffc044']]);
    ctx.beginPath(); ctx.ellipse(23, 17, 10.5, 16, -.18, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.ellipse(27, 14, 3.6, 4, -.15, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#fff3b6aa'; ctx.beginPath(); ctx.arc(21, 25, 1.5, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
    path(ctx, 'M2 6Q12 12 23 7Q30 9 33 18', null, 2.1, '#111b2c');
  }

  function mouth(ctx, variant) {
    ctx.scale(1 / 60, 1 / 26);
    if (variant === 'sad') {
      path(ctx, 'M13 18Q30 4 47 18', null, 3.4, '#77566c');
      return;
    }
    const talking = variant === 'open' || variant === 'talking';
    const d = talking ? 'M10 5Q30 0 50 5Q53 26 30 26Q7 26 10 5Z'
      : variant === 'happy' ? 'M4 4Q30 15 56 4Q47 26 30 26Q13 26 4 4Z'
        : 'M6 5Q30 15 54 5Q44 24 30 24Q16 24 6 5Z';
    path(ctx, d, '#172032', 2.3);
    ctx.save(); ctx.clip(paths.get(d));
    path(ctx, 'M17 25Q30 11 43 25V29H17Z', '#a56c80', 0);
    path(ctx, talking ? 'M13 4L24 5L20 14ZM36 5L47 4L41 14Z'
      : 'M10 7L23 9L18 18ZM37 9L50 7L43 18Z', '#f7fcff', 0);
    ctx.restore();
  }

  function body(ctx) {
    path(ctx, 'M16 5Q50-7 84 5L96 91Q50 108 4 91Z', coat(ctx), 2.8);
    path(ctx, 'M8 78Q50 112 93 82L95 91Q50 105 5 91Z', '#2d649538', 0);
    path(ctx, 'M31 6L50 19L69 6L82 64L50 88L18 64Z', navy(ctx), 2.5);
    path(ctx, 'M32 33H68L78 46L50 69L22 46Z', gold(ctx), 2.2);
    path(ctx, 'M39 22L50 28L61 22L58 37H42Z', '#e5f0f7', 2.1);
    path(ctx, 'M38 46H62', null, 4.1, '#243149');
  }

  function arm(ctx, right) {
    if (right) { ctx.translate(100, 0); ctx.scale(-1, 1); }
    path(ctx, 'M86 4Q7 30 9 123L27 196L92 181L84 49Z', gradient(ctx, 0, 0, 90, 205, [[0, '#8abedf'], [.65, '#548fc0'], [1, '#326b9b']]), 8.5);
    path(ctx, 'M84 49L92 181L72 185Z', '#2c638f77', 0);
    path(ctx, 'M24 100Q16 52 54 29', null, 3, '#d7f1ff55');
  }

  function hand(ctx, right) {
    if (right) { ctx.translate(40, 0); ctx.scale(-1, 1); }
    ctx.fillStyle = gradient(ctx, 5, 2, 35, 40, [[0, '#29354f'], [1, '#121c2b']]);
    ctx.strokeStyle = edge; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.ellipse(20, 20, 18, 18, -.15, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    path(ctx, 'M8 16L14 10M20 19L17 27M29 21L31 28', null, 2.8, '#ffd574');
  }

  function leg(ctx) {
    path(ctx, 'M3 2Q20-1 37 2L33 73H7Z', gradient(ctx, 5, 0, 30, 74, [[0, '#538fc3'], [1, '#336b99']]), 3.3);
    path(ctx, 'M28 3L25 72H33L37 3Z', '#294a6b66', 0);
  }

  function foot(ctx, right) {
    if (right) { ctx.translate(100, 0); ctx.scale(-1, 1); }
    path(ctx, 'M9 24Q20 2 50 2Q83 2 94 25Q104 59 89 62H13Q-3 59 9 24Z', gradient(ctx, 10, 0, 80, 64, [[0, '#2c3851'], [1, '#141e2f']]), 5);
    path(ctx, 'M24 39Q51 41 77 39', null, 5.5, '#ffd16e');
  }

  function tail(ctx) {
    path(ctx, 'M19 117Q86 109 88 63Q91 12 45 8Q30 7 20 15L26 40Q59 27 62 63Q61 91 16 96Z', gradient(ctx, 20, 8, 90, 143, [[0, '#5b95c1'], [1, '#294a6b']]), 6);
    path(ctx, 'M32 24Q76 24 75 64Q72 104 23 108', null, 16, '#273b50');
    ctx.save(); ctx.shadowColor = '#ffd16688'; ctx.shadowBlur = 4;
    path(ctx, 'M32 24Q76 24 75 64Q72 104 23 108', null, 9, '#ffd166');
    ctx.restore();
    path(ctx, 'M12 95L32 103L35 126L14 141L1 120Z', gold(ctx), 4);
  }

  avatars.vex = {
    label: 'Vex',
    pixelated: false,
    createBones(createBone) {
      const b = (...args) => createBone(...args);
      const bones = [
        b('root', null, 0, 0, 0, 0, 0, 0),
        b('body', 'root', 0, -4, 0, 17, 34, 2),
        b('neck', 'body', 0, -16, 0, 2, 4, 4),
        b('head', 'neck', 0, -21, 0, 20, 40, 5),
        b('eye_left', 'head', -9, 4, 0, 0, 0, 6),
        b('eye_right', 'head', 9, 4, 0, 0, 0, 6),
        b('mouth', 'head', 0, 15, 0, 0, 0, 7),
        b('arm_left', 'body', -16, -14, 0, 0, 22, 1),
        b('hand_left', 'arm_left', -1, 20, 0, 0, 8, 3),
        b('arm_right', 'body', 16, -14, 0, 0, 22, 1),
        b('hand_right', 'arm_right', 1, 20, 0, 0, 8, 3),
        b('leg_left', 'root', -7, 10, 0, 0, 13, 0),
        b('foot_left', 'leg_left', -1, 11, 0, 0, 8, 1),
        b('leg_right', 'root', 7, 10, 0, 0, 13, 0),
        b('foot_right', 'leg_right', 1, 11, 0, 0, 8, 1),
        b('accessory', 'body', 25, 7, 0, 0, 30, -1),
      ];
      bones.forEach(bone => {
        if (bone.name.startsWith('arm_') || bone.name.startsWith('leg_')) bone.pivotOffsetY = .5;
      });
      return bones;
    },
    manifest(bones) {
      const lengths = Object.fromEntries(bones.map(b => [b.name, b.length]));
      const def = (bone, w, h, length, variants = ['default'], oy = 0) => ({
        bone, w: w * (length ? lengths[bone] / length : 1),
        h: h * (length ? lengths[bone] / length : 1), ox: 0, oy,
        variants: Object.fromEntries(variants.map(v => [v, null])),
      });
      return {
        body: def('body', 36, 34, 34),
        head: def('head', 56, 58, 40, ['default'], -8),
        eye_left: def('eye_left', 10.5, 9, 0, ['open', 'closed', 'happy']),
        eye_right: def('eye_right', 10.5, 9, 0, ['open', 'closed', 'happy']),
        mouth: def('mouth', 18, 6, 0, ['neutral', 'happy', 'sad', 'open', 'talking']),
        arm_left: def('arm_left', 9, 24, 22),
        arm_right: def('arm_right', 9, 24, 22),
        hand_left: def('hand_left', 11, 11, 8),
        hand_right: def('hand_right', 11, 11, 8),
        leg_left: def('leg_left', 8, 15, 13),
        leg_right: def('leg_right', 8, 15, 13),
        foot_left: def('foot_left', 18, 10, 8),
        foot_right: def('foot_right', 18, 10, 8),
        accessory: def('accessory', 26, 37, 30),
      };
    },
    draw(ctx, part, variant, w, h) {
      ctx.save();
      switch (part) {
        case 'head': ctx.scale(w / 100, h / 100); head(ctx); break;
        case 'body': ctx.scale(w / 100, h / 100); body(ctx); break;
        case 'eye_left': case 'eye_right':
          ctx.scale(w, h); eye(ctx, variant, part === 'eye_right'); break;
        case 'mouth': ctx.scale(w, h); mouth(ctx, variant); break;
        case 'arm_left': case 'arm_right':
          ctx.scale(w / 100, h / 205); arm(ctx, part === 'arm_right'); break;
        case 'hand_left': case 'hand_right':
          ctx.scale(w / 40, h / 40); hand(ctx, part === 'hand_right'); break;
        case 'leg_left': case 'leg_right': ctx.scale(w / 40, h / 75); leg(ctx); break;
        case 'foot_left': case 'foot_right':
          ctx.scale(w / 100, h / 65); foot(ctx, part === 'foot_right'); break;
        case 'accessory': ctx.scale(w / 100, h / 150); tail(ctx); break;
      }
      ctx.restore();
    },
  };
})();
