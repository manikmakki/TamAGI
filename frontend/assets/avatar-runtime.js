// Shared built-in avatar selection and manifest handling for the chat and rig editor.
(() => {
  'use strict';
  const avatars = window.TamagiAvatars = window.TamagiAvatars || {};
  window.TamagiAvatarRuntime = {
    resolve(config) {
      if (config?.avatar === 'custom' || Object.hasOwn(avatars, config?.avatar)) return config.avatar;
      // Existing rigs predate built-in characters. Keep their proportions and art.
      if (config?.skeleton?.length || Object.keys(config?.sprites || {}).length) return 'custom';
      return 'vex';
    },
    createBones(id, createBone, customBones) {
      return avatars[id] ? avatars[id].createBones(createBone) : customBones();
    },
    manifest(id, bones, customManifest) {
      return avatars[id] ? avatars[id].manifest(bones) : customManifest(bones);
    },
    overlay(manifest, bones, sprites) {
      const boneMap = Object.fromEntries(bones.map(b => [b.name, b]));
      for (const [part, saved] of Object.entries(sprites || {})) {
        const entry = manifest[part];
        if (!entry || !saved) continue;
        if (boneMap[saved.bone]) entry.bone = saved.bone;
        const scale = boneMap[entry.bone]?.displayScale || 1;
        if (Number.isFinite(saved.width) && saved.width > 0) entry.w = saved.width / scale;
        if (Number.isFinite(saved.height) && saved.height > 0) entry.h = saved.height / scale;
        if (Number.isFinite(saved.offsetX)) entry.ox = saved.offsetX;
        if (Number.isFinite(saved.offsetY)) entry.oy = saved.offsetY;
      }
      return manifest;
    },
    animate(id, bones, time, name, variants) {
      const overrides = avatars[id]?.animate?.(bones, time, name, variants);
      return overrides ? {...variants, ...overrides} : variants;
    },
    draw(ctx, id, part, variant, width, height, customDraw, time = 0) {
      if (avatars[id]) avatars[id].draw(ctx, part, variant, width, height, time);
      else customDraw(ctx, part, variant, width, height);
    },
    originY(canvas, id) {
      // Leave room above tall ears and pixel heads during bounces and head tilts.
      // Existing custom rigs retain the viewport they were authored against.
      return canvas.height / 2 + (avatars[id] ? 64 : 50);
    },
    style(canvas, id) {
      const pixelated = avatars[id]?.pixelated ?? true;
      canvas.style.imageRendering = pixelated ? 'pixelated' : 'auto';
      canvas.getContext('2d').imageSmoothingEnabled = !pixelated;
    },
  };
})();
